from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import time
import json
from werkzeug.utils import secure_filename
from PIL import Image
from database import db, ImageRecord, init_db, BEARING_TYPES, COATING_STATUSES
from runtime_config import (
    IS_VERCEL,
    get_database_uri,
    get_reference_folder,
    get_static_folder,
    get_upload_folder,
    should_persist_runtime_uploads,
)
from similarity import ImageSimilarityMatcher

STATIC_FOLDER = get_static_folder()
app = Flask(__name__, static_folder=STATIC_FOLDER)
CORS(app)

UPLOAD_FOLDER = get_upload_folder()
REFERENCE_FOLDER = get_reference_folder()
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp', 'bmp', 'tiff', 'tif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['REFERENCE_FOLDER'] = REFERENCE_FOLDER
app.config['SQLALCHEMY_DATABASE_URI'] = get_database_uri()
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

matcher = None


def allowed_file(filename):
    if not filename or '.' not in filename:
        return False
    return filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def ensure_dirs():
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['REFERENCE_FOLDER'], exist_ok=True)
    os.makedirs(os.path.join(app.config['REFERENCE_FOLDER'], 'coatings'), exist_ok=True)
    os.makedirs(os.path.join(app.config['REFERENCE_FOLDER'], 'downloads'), exist_ok=True)


def get_matcher():
    global matcher
    if matcher is None:
        matcher = ImageSimilarityMatcher()
        matcher.load_reference_images(coating_status='coated')
    return matcher


def to_jpg(filepath):
    try:
        img = Image.open(filepath).convert('RGB')
        base, _ = os.path.splitext(filepath)
        new_path = base + '.jpg'
        if new_path != filepath:
            img.save(new_path, 'JPEG', quality=92)
            if os.path.exists(filepath):
                os.remove(filepath)
            return new_path
    except Exception:
        pass
    return filepath


@app.route('/')
def index():
    return send_from_directory(STATIC_FOLDER, 'index.html')


@app.route('/estimate')
def estimate_page():
    return send_from_directory(STATIC_FOLDER, 'estimate.html')


@app.route('/health')
def health():
    return jsonify({'status': 'ok'})


@app.route('/api/estimate', methods=['POST'])
def create_estimate():
    """Generate a coating estimate and related catalog matches."""
    if 'image' not in request.files:
        return jsonify({'error': 'No image'}), 400
    file = request.files['image']
    if not file or not file.filename:
        return jsonify({'error': 'No file'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': 'Bad file type'}), 400
    try:
        active_matcher = get_matcher()
        name = secure_filename(file.filename) or 'estimate'
        unique = str(int(time.time() * 1000)) + '_' + name
        path = os.path.join(app.config['UPLOAD_FOLDER'], unique)
        file.save(path)
        path = to_jpg(path)
        unique = os.path.basename(path)
        from estimate import get_coating_estimate
        previous_work = active_matcher.find_similar_images(
            path,
            top_k=4,
            prefer_coated=True,
            exclude_paths=[path],
        )
        result = get_coating_estimate(path, previous_work=previous_work)
        if should_persist_runtime_uploads():
            result['uploaded_image'] = '/api/uploads/' + unique
        result['previous_work'] = previous_work
        return jsonify({'success': True, **result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image'}), 400
    file = request.files['image']
    if not file or not file.filename:
        return jsonify({'error': 'No file'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': 'Bad file type'}), 400
    try:
        active_matcher = get_matcher()
        name = secure_filename(file.filename) or 'image'
        ext = name.rsplit('.', 1)[-1].lower() if '.' in name else ''
        if ext not in ALLOWED_EXTENSIONS:
            name = name + '.jpg'
        unique = str(int(time.time() * 1000)) + '_' + name
        path = os.path.join(app.config['UPLOAD_FOLDER'], unique)
        file.save(path)
        path = to_jpg(path)
        unique = os.path.basename(path)
        matches = active_matcher.find_similar_images(
            path,
            top_k=3,
            prefer_coated=True,
            exclude_paths=[path],
        )
        response = {
            'success': True,
            'matches': matches,
            'note': 'Your bare/naked bearing photo matched to coated catalog examples.'
        }
        if should_persist_runtime_uploads():
            response['uploaded_image'] = '/api/uploads/' + unique
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/reference/add', methods=['POST'])
def add_reference():
    if IS_VERCEL:
        return jsonify({'error': 'Adding images is disabled on hosted deployments.'}), 501
    if 'image' not in request.files:
        return jsonify({'error': 'No image'}), 400
    file = request.files['image']
    desc = request.form.get('description', '')
    bearing_type = request.form.get('bearing_type', '').strip() or None
    coating_status = request.form.get('coating_status', '').strip() or None
    size_info = request.form.get('size_info', '').strip() or None
    if bearing_type and bearing_type not in BEARING_TYPES:
        bearing_type = None
    if coating_status and coating_status not in COATING_STATUSES:
        coating_status = None
    if not file or not file.filename:
        return jsonify({'error': 'No file'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': 'Bad file type'}), 400
    try:
        name = secure_filename(file.filename)
        subfolder = 'coatings' if coating_status == 'coated' else ''
        dir_path = os.path.join(app.config['REFERENCE_FOLDER'], subfolder) if subfolder else app.config['REFERENCE_FOLDER']
        os.makedirs(dir_path, exist_ok=True)
        unique = str(int(time.time() * 1000)) + '_' + name
        path = os.path.join(dir_path, unique)
        file.save(path)
        url_filename = os.path.join(subfolder, unique) if subfolder else unique
        if not (desc and desc.strip()):
            try:
                from caption import get_caption
                cap = get_caption(path)
                if cap:
                    desc = cap
            except Exception:
                pass
        active_matcher = get_matcher()
        emb = active_matcher.get_image_embedding(path)
        rec = ImageRecord(
            filename=url_filename, filepath=path,
            description=(desc or '').strip() or None, embedding=json.dumps(emb.tolist()),
            bearing_type=bearing_type, coating_status=coating_status, size_info=size_info,
            split='catalog',
        )
        db.session.add(rec)
        db.session.commit()
        active_matcher.load_reference_images(coating_status='coated')
        return jsonify({'success': True, 'id': rec.id, 'image_url': '/api/reference/' + url_filename.replace('\\', '/')})
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500


@app.route('/api/uploads/<path:filename>')
def serve_upload(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/api/reference/<path:filename>')
def serve_reference(filename):
    return send_from_directory(app.config['REFERENCE_FOLDER'], filename)


@app.route('/api/reference/list', methods=['GET'])
def list_reference():
    bearing_type = request.args.get('bearing_type')
    coating_status = request.args.get('coating_status')
    q = ImageRecord.query
    if bearing_type and bearing_type in BEARING_TYPES:
        q = q.filter(ImageRecord.bearing_type == bearing_type)
    if coating_status and coating_status in COATING_STATUSES:
        q = q.filter(ImageRecord.coating_status == coating_status)
    imgs = q.all()
    return jsonify({
        'success': True,
        'images': [{
            'id': i.id, 'filename': i.filename, 'description': i.description,
            'url': '/api/reference/' + (i.filename or '').replace('\\', '/'),
            'bearing_type': i.bearing_type, 'coating_status': i.coating_status, 'size_info': i.size_info
        } for i in imgs]
    })


@app.route('/api/bearing-types', methods=['GET'])
def bearing_types():
    return jsonify({'success': True, 'bearing_types': list(BEARING_TYPES)})


@app.route('/api/coatings/list', methods=['GET'])
def list_coated_catalog():
    """List coated catalog examples."""
    q = ImageRecord.query.filter(ImageRecord.coating_status == 'coated')
    q = q.filter((ImageRecord.split == 'catalog') | (ImageRecord.split.is_(None)))
    imgs = q.all()
    return jsonify({
        'success': True,
        'images': [{
            'id': i.id, 'filename': i.filename, 'description': i.description,
            'url': '/api/reference/' + (i.filename or '').replace('\\', '/'),
            'bearing_type': i.bearing_type, 'size_info': i.size_info
        } for i in imgs]
    })


with app.app_context():
    ensure_dirs()
    init_db()


if __name__ == '__main__':
    with app.app_context():
        get_matcher()
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', '5000')), debug=False)
