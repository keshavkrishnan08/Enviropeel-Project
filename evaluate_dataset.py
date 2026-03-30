#!/usr/bin/env python3
"""
Evaluate matching and estimate accuracy on the held-out test split.

This focuses on the current dataset:
- catalog split is used as the reference library
- test split is used as held-out evaluation
"""
from flask import Flask

from database import db, ImageRecord, init_db
from estimate import MANUAL_REFERENCE_PROFILES, get_coating_estimate, normalize_component_type
from similarity import ImageSimilarityMatcher


def expected_component_type(rec):
    name = rec.filename or rec.filepath or ""
    for fragment, profile in MANUAL_REFERENCE_PROFILES.items():
        if fragment in name:
            return normalize_component_type(profile["component_type"])
    return normalize_component_type(rec.bearing_type) or "bearing_housing"


def main():
    app = Flask(__name__)
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///image_database.db"
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    db.init_app(app)

    with app.app_context():
        init_db()
        matcher = ImageSimilarityMatcher()
        matcher.load_reference_images(coating_status="coated")

        test_records = (
            ImageRecord.query.filter(ImageRecord.split == "test")
            .filter(ImageRecord.bearing_type.isnot(None))
            .filter(ImageRecord.coating_status == "coated")
            .all()
        )

        if not test_records:
            print("No held-out test records found.")
            return

        top1_family_correct = 0
        top3_family_correct = 0
        estimate_type_correct = 0
        total = 0

        print("Held-out dataset evaluation")
        print("=" * 60)
        for rec in test_records:
            total += 1
            matches = matcher.find_similar_images(
                rec.filepath,
                top_k=3,
                prefer_coated=True,
                exclude_paths=[rec.filepath],
            )
            expected_family = rec.bearing_type
            expected_type = expected_component_type(rec)

            match_families = [m.get("bearing_type") for m in matches]
            top1_ok = bool(matches) and matches[0].get("bearing_type") == expected_family
            top3_ok = expected_family in match_families

            estimate = get_coating_estimate(rec.filepath, previous_work=matches)
            estimate_ok = estimate.get("component_type") == expected_type

            if top1_ok:
                top1_family_correct += 1
            if top3_ok:
                top3_family_correct += 1
            if estimate_ok:
                estimate_type_correct += 1

            print(f"- {rec.filename}")
            print(f"  expected family: {expected_family}")
            print(f"  top matches: {match_families}")
            print(f"  estimate type: {estimate.get('component_type')} (expected {expected_type})")

        print("=" * 60)
        print(f"Top-1 family accuracy: {top1_family_correct}/{total} = {top1_family_correct/total:.0%}")
        print(f"Top-3 family accuracy: {top3_family_correct}/{total} = {top3_family_correct/total:.0%}")
        print(f"Estimate type accuracy: {estimate_type_correct}/{total} = {estimate_type_correct/total:.0%}")


if __name__ == "__main__":
    main()
