#!/usr/bin/env python3
import io
import os
import random
import tempfile

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from flask import Flask

from database import ImageRecord, db, init_db
from similarity import ImageSimilarityMatcher


def _safe_crop(img, factor, rng):
    width, height = img.size
    new_w = max(8, int(width * factor))
    new_h = max(8, int(height * factor))
    if new_w >= width or new_h >= height:
        return img.copy()
    left = rng.randint(0, width - new_w)
    top = rng.randint(0, height - new_h)
    cropped = img.crop((left, top, left + new_w, top + new_h))
    return cropped.resize((width, height), Image.Resampling.BICUBIC)


def _jpeg_roundtrip(img, quality):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def make_variants(base_img, seed):
    rng = random.Random(seed)
    variants = []

    variants.append(("clean", base_img.copy()))
    variants.append(("flip_h", base_img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)))
    variants.append(("flip_v", base_img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)))

    for angle in (-25, -15, -10, 10, 15, 25):
        variants.append((f"rot_{angle}", base_img.rotate(angle, expand=False)))

    for factor in (0.65, 0.75, 0.85, 0.92):
        variants.append((f"crop_{factor}", _safe_crop(base_img, factor, rng)))

    for factor in (0.5, 0.7, 1.3, 1.6):
        variants.append((f"bright_{factor}", ImageEnhance.Brightness(base_img).enhance(factor)))
    for factor in (0.6, 0.8, 1.4, 1.8):
        variants.append((f"contrast_{factor}", ImageEnhance.Contrast(base_img).enhance(factor)))
    for factor in (0.4, 0.7, 1.5, 2.0):
        variants.append((f"color_{factor}", ImageEnhance.Color(base_img).enhance(factor)))

    for radius in (1, 2, 3):
        variants.append((f"blur_{radius}", base_img.filter(ImageFilter.GaussianBlur(radius=radius))))

    arr = np.asarray(base_img, dtype=np.float32)
    for sigma in (8, 16, 24):
        noise = np.random.default_rng(seed + sigma).normal(0, sigma, arr.shape)
        noisy = np.clip(arr + noise, 0, 255).astype(np.uint8)
        variants.append((f"noise_{sigma}", Image.fromarray(noisy, mode="RGB")))

    for quality in (15, 30, 45):
        variants.append((f"jpeg_{quality}", _jpeg_roundtrip(base_img, quality)))

    harsh = ImageEnhance.Brightness(base_img).enhance(0.6)
    harsh = ImageEnhance.Contrast(harsh).enhance(1.7)
    harsh = harsh.filter(ImageFilter.GaussianBlur(radius=2))
    harsh = _safe_crop(harsh, 0.72, rng)
    harsh = _jpeg_roundtrip(harsh, 20)
    variants.append(("harsh_combo", harsh))

    return variants


def evaluate_robustness(config_override=None):
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
            return 0.0

        total = 0
        correct = 0
        by_image = {}

        with tempfile.TemporaryDirectory(prefix="robust_eval_") as tmp:
            for rec_idx, rec in enumerate(test_records):
                expected = rec.bearing_type
                img = Image.open(rec.filepath).convert("RGB")
                variants = make_variants(img, seed=1337 + rec_idx)
                image_total = 0
                image_correct = 0
                failures = []

                for var_idx, (name, variant) in enumerate(variants):
                    var_path = os.path.join(tmp, f"query_{rec_idx}_{var_idx}_{name}.jpg")
                    variant.save(var_path, format="JPEG", quality=95)
                    matches = matcher.find_similar_images(
                        var_path,
                        top_k=3,
                        prefer_coated=True,
                        exclude_paths=[rec.filepath],
                        scoring_config=config_override,
                    )
                    predicted = matches[0].get("bearing_type") if matches else None
                    ok = predicted == expected
                    total += 1
                    image_total += 1
                    if ok:
                        correct += 1
                        image_correct += 1
                    else:
                        failures.append((name, predicted))

                by_image[rec.filename] = {
                    "expected": expected,
                    "correct": image_correct,
                    "total": image_total,
                    "failures": failures,
                }

        print("ROBUST IDENTIFICATION EVALUATION")
        print("=" * 60)
        for name, stats in by_image.items():
            ratio = stats["correct"] / max(stats["total"], 1)
            print(
                f"- {name} ({stats['expected']}): "
                f"{stats['correct']}/{stats['total']} = {ratio:.0%}"
            )
            if stats["failures"]:
                preview = ", ".join(
                    [f"{tag}->{pred}" for tag, pred in stats["failures"][:5]]
                )
                print(f"  failures: {preview}")
        print("=" * 60)
        print(f"Overall robust top-1: {correct}/{total} = {correct / max(total, 1):.0%}")
        return correct / max(total, 1)


if __name__ == "__main__":
    evaluate_robustness()
