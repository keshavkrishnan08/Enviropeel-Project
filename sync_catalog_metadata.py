#!/usr/bin/env python3
"""
Synchronize labeled records with the project's curated image metadata.
"""
from flask import Flask

from catalog_metadata import lookup_reference_profile
from database import ImageRecord, db, init_db

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///image_database.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db.init_app(app)


def main():
    with app.app_context():
        init_db()
        updated = 0
        for rec in ImageRecord.query.all():
            profile = lookup_reference_profile(rec.filename or rec.filepath)
            if not profile:
                continue

            changed = False
            if profile.get("bearing_type") and rec.bearing_type != profile["bearing_type"]:
                rec.bearing_type = profile["bearing_type"]
                changed = True
            if profile.get("description") and rec.description != profile["description"]:
                rec.description = profile["description"]
                changed = True
            if profile.get("size_info") and rec.size_info != profile["size_info"]:
                rec.size_info = profile["size_info"]
                changed = True

            if changed:
                updated += 1

        db.session.commit()
        print(f"Synchronized metadata for {updated} record(s).")


if __name__ == "__main__":
    main()
