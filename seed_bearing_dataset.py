"""
Seed bearing dataset from project image downloads (assets).
50% catalog (our previous work, shown to clients) / 50% test (held out for validation).

Run once: python seed_bearing_dataset.py
"""
import os
import json
import shutil
from catalog_metadata import lookup_reference_profile
from database import db, ImageRecord, init_db, BEARING_TYPES, COATING_STATUSES
from similarity import ImageSimilarityMatcher
from flask import Flask

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///image_database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

REFERENCE_FOLDER = 'reference_images'
COATINGS_SUBFOLDER = os.path.join(REFERENCE_FOLDER, 'coatings')

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
ASSETS_PATHS = [
    os.path.join(PROJECT_ROOT, 'assets'),
    os.path.join(PROJECT_ROOT, 'project_image_downloads'),
]

BEARING_IMAGES = [
    ("IMG-20211001-WA0005-4a365adf-99d0-47ca-a8b8-2c64a34f9c72.png", "bearing_flange_IMG-20211001-WA0005-.png", "flange", "Yellow coated flange assembly on a pipe elbow", "medium assembly"),
    ("IMG-20211001-WA0000-fa9ba8fd-e98a-4962-9adf-023162dacea2.png", "bearing_pillow_block_IMG-20211001-WA0000-.png", "pillow_block", "Blue coated pillow block bearings on conveyor shafts", "medium bearing"),
    ("IMG-20211001-WA0006-8abc96e2-06f5-406d-a5cb-379075d08115.png", "bearing_flange_IMG-20211001-WA0006-.png", "support_pedestal", "White coated structural support pedestal or mount base", "large support base"),
    ("IMG-20211001-WA0001-bb6cc231-cd75-495e-9e27-e00de02aac36.png", "bearing_pillow_block_IMG-20211001-WA0001-.png", "pillow_block", "Blue coated pillow block bearing beside conveyor pulley", "medium bearing"),
    ("IMG-20211001-WA0004-15280a0f-7a5b-4bab-a5ab-c24cdc5cc8e2.png", "bearing_pillow_block_IMG-20211001-WA0004-.png", "gearbox", "Blue coated gearbox or drive reducer housing", "large gearbox"),
    ("IMG-20211001-WA0002-4f600923-526f-4b57-b2bc-ae1d8c94eaf9.png", "bearing_pillow_block_IMG-20211001-WA0002-.png", "bearing_housing", "Blue coated bearing housing mounted on an industrial frame", "medium housing"),
]

CATALOG_SOURCE_FILENAMES = {
    "IMG-20211001-WA0005-4a365adf-99d0-47ca-a8b8-2c64a34f9c72.png",
    "IMG-20211001-WA0000-fa9ba8fd-e98a-4962-9adf-023162dacea2.png",
    "IMG-20211001-WA0006-8abc96e2-06f5-406d-a5cb-379075d08115.png",
    "IMG-20211001-WA0004-15280a0f-7a5b-4bab-a5ab-c24cdc5cc8e2.png",
}


def find_asset_path(filename):
    """Locate image in assets folders."""
    for base in ASSETS_PATHS:
        path = os.path.join(base, filename)
        if os.path.exists(path):
            return path
    return None


def main():
    os.makedirs(COATINGS_SUBFOLDER, exist_ok=True)
    print("Loading CLIP...")
    with app.app_context():
        init_db()
        matcher = ImageSimilarityMatcher()

        added = 0
        skipped = 0
        failed = 0

        for source_filename, dest_filename, bearing_type, description, size_info in BEARING_IMAGES:
            dest_path = os.path.join(COATINGS_SUBFOLDER, dest_filename)
            url_filename = os.path.join("coatings", dest_filename)
            profile = lookup_reference_profile(dest_filename) or {}
            bearing_type = profile.get("bearing_type") or bearing_type
            description = profile.get("description") or description
            size_info = profile.get("size_info") or size_info

            print(f"\n{source_filename} -> {bearing_type}")

            src = find_asset_path(source_filename)
            if not src:
                print(f"  NOT FOUND in assets (checked {ASSETS_PATHS})")
                failed += 1
                continue

            if not os.path.exists(dest_path):
                shutil.copy2(src, dest_path)
                print(f"  Copied to {dest_path}")

            existing = ImageRecord.query.filter(ImageRecord.filepath == dest_path).first()
            split = "catalog" if source_filename in CATALOG_SOURCE_FILENAMES else "test"
            if existing:
                existing.bearing_type = bearing_type
                existing.coating_status = "coated"
                existing.description = description
                existing.size_info = size_info
                existing.split = split
                db.session.commit()
                print(f"  Updated metadata (split={split})")
                skipped += 1
                continue

            try:
                emb = matcher.get_image_embedding(dest_path)
                rec = ImageRecord(
                    filename=url_filename,
                    filepath=dest_path,
                    description=description,
                    embedding=json.dumps(emb.tolist()),
                    bearing_type=bearing_type,
                    coating_status="coated",
                    size_info=size_info,
                    split=split,
                )
                db.session.add(rec)
                db.session.commit()
                print(f"  Added to DB id={rec.id}")
                added += 1
            except Exception as e:
                db.session.rollback()
                print(f"  Error: {e}")
                failed += 1

        total = ImageRecord.query.filter(ImageRecord.bearing_type.isnot(None)).count()
        print(f"\nDone. added={added} skipped={skipped} failed={failed} bearing_total={total}")


if __name__ == "__main__":
    main()
