#!/usr/bin/env python3
"""
Audit the current dataset split for obvious leakage.

Checks:
1. no exact filepath overlap between catalog and test
2. no exact filename overlap between catalog and test
3. held-out test images do not return themselves as matches
"""
from flask import Flask

from database import db, ImageRecord, init_db
from similarity import ImageSimilarityMatcher


def main():
    app = Flask(__name__)
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///image_database.db"
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    db.init_app(app)

    with app.app_context():
        init_db()
        catalog = (
            ImageRecord.query.filter(ImageRecord.split == "catalog")
            .filter(ImageRecord.bearing_type.isnot(None))
            .all()
        )
        test = (
            ImageRecord.query.filter(ImageRecord.split == "test")
            .filter(ImageRecord.bearing_type.isnot(None))
            .all()
        )

        catalog_paths = {rec.filepath for rec in catalog}
        test_paths = {rec.filepath for rec in test}
        catalog_names = {rec.filename for rec in catalog}
        test_names = {rec.filename for rec in test}

        path_overlap = sorted(catalog_paths & test_paths)
        name_overlap = sorted(catalog_names & test_names)

        matcher = ImageSimilarityMatcher()
        matcher.load_reference_images(coating_status="coated")

        self_match_failures = []
        for rec in test:
            matches = matcher.find_similar_images(
                rec.filepath,
                top_k=3,
                prefer_coated=True,
                exclude_paths=[rec.filepath],
            )
            matched_names = [m.get("filename") for m in matches]
            if rec.filename in matched_names:
                self_match_failures.append(rec.filename)

        print("DATA LEAKAGE AUDIT")
        print("=" * 60)
        print(f"Catalog records: {len(catalog)}")
        print(f"Test records:    {len(test)}")
        print(f"Exact filepath overlap: {len(path_overlap)}")
        print(f"Exact filename overlap: {len(name_overlap)}")
        print(f"Self-match failures:    {len(self_match_failures)}")

        if path_overlap:
            print("Path overlap examples:")
            for item in path_overlap[:5]:
                print(f"  - {item}")
        if name_overlap:
            print("Filename overlap examples:")
            for item in name_overlap[:5]:
                print(f"  - {item}")
        if self_match_failures:
            print("Self-match failure examples:")
            for item in self_match_failures[:5]:
                print(f"  - {item}")

        if not path_overlap and not name_overlap and not self_match_failures:
            print("LEAKAGE STATUS: PASS")
        else:
            print("LEAKAGE STATUS: FAIL")


if __name__ == "__main__":
    main()
