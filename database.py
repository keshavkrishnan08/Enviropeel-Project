from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

BEARING_TYPES = (
    'pillow_block',
    'flange',
    'gearbox',
    'support_pedestal',
    'bearing_housing',
)
COATING_STATUSES = ('coated', 'naked')
SPLITS = ('catalog', 'test')


class ImageRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    filepath = db.Column(db.String(500), nullable=False)
    description = db.Column(db.Text)
    embedding = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    bearing_type = db.Column(db.String(50), nullable=True)
    coating_status = db.Column(db.String(20), nullable=True)
    size_info = db.Column(db.String(100), nullable=True)
    source_notes = db.Column(db.Text, nullable=True)
    split = db.Column(db.String(20), nullable=True)


def init_db():
    db.create_all()
