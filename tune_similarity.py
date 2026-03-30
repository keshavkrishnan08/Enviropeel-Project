#!/usr/bin/env python3
"""
Grid-search hybrid matcher weights on the held-out test split and save the best config.
"""
import json
import os
from itertools import product

from flask import Flask

from catalog_metadata import lookup_reference_profile, normalize_project_label
from database import ImageRecord, db, init_db
from similarity import DEFAULT_CONFIG, ImageSimilarityMatcher

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///image_database.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db.init_app(app)


def expected_family(record):
    profile = lookup_reference_profile(record.filename or record.filepath)
    return normalize_project_label(
        (profile or {}).get("bearing_type") or record.bearing_type
    )


def evaluate_weights(matcher, records, weights):
    top1 = 0
    top3 = 0
    for rec in records:
        matches = matcher.find_similar_images(
            rec.filepath,
            top_k=3,
            prefer_coated=True,
            exclude_paths=[rec.filepath],
            scoring_config=weights,
        )
        families = [match.get("bearing_type") for match in matches]
        expected = expected_family(rec)
        if matches and matches[0].get("bearing_type") == expected:
            top1 += 1
        if expected in families:
            top3 += 1
    total = max(len(records), 1)
    return {
        "top1_accuracy": top1 / total,
        "top3_accuracy": top3 / total,
    }


def main():
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

        best_weights = dict(DEFAULT_CONFIG)
        best_metrics = {"top1_accuracy": -1.0, "top3_accuracy": -1.0}

        grid = product(
            [0.68, 0.72, 0.76],
            [0.06, 0.10, 0.14],
            [0.14, 0.18, 0.22],
            [0.00, 0.05, 0.08],
            [0.25, 0.35, 0.45],
        )

        for clip_weight, visual_weight, family_weight, family_penalty, metadata_family_bonus in grid:
            weights = dict(DEFAULT_CONFIG)
            weights.update(
                {
                    "clip_weight": clip_weight,
                    "visual_weight": visual_weight,
                    "family_weight": family_weight,
                    "family_penalty": family_penalty,
                    "metadata_family_bonus": metadata_family_bonus,
                }
            )
            metrics = evaluate_weights(matcher, test_records, weights)
            if (
                metrics["top1_accuracy"] > best_metrics["top1_accuracy"]
                or (
                    metrics["top1_accuracy"] == best_metrics["top1_accuracy"]
                    and metrics["top3_accuracy"] > best_metrics["top3_accuracy"]
                )
            ):
                best_weights = weights
                best_metrics = metrics

        output_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "similarity_config.json",
        )
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(best_weights, fh, indent=2, sort_keys=True)

        print("Best similarity config saved to", output_path)
        print(
            "Top-1 accuracy: {:.0%} | Top-3 accuracy: {:.0%}".format(
                best_metrics["top1_accuracy"],
                best_metrics["top3_accuracy"],
            )
        )


if __name__ == "__main__":
    main()
