"""
Assign catalog/test split to existing ImageRecords that don't have split set.
50% catalog (for matching), 50% test (held out).
Run once after migration: python assign_splits.py
"""
import os
from database import db, ImageRecord, init_db
from flask import Flask

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///image_database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

def main():
    with app.app_context():
        init_db()
        recs = ImageRecord.query.filter(ImageRecord.split.is_(None)).order_by(ImageRecord.id).all()
        n = len(recs)
        if n == 0:
            print("No records need split assignment.")
            return
        for i, rec in enumerate(recs):
            split = "catalog" if (i % 2 == 0) else "test"
            rec.split = split
        db.session.commit()
        print(f"Assigned split to {n} records: {n//2} catalog, {n - n//2} test")

if __name__ == "__main__":
    main()
