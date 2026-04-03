#!/usr/bin/env python3
import json
import os

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from flask import Flask

from database import ImageRecord, db, init_db
from similarity import ImageSimilarityMatcher

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///image_database.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db.init_app(app)


def generate_augmented_images(base_img):
    variants = []
    variants.append(("flip_h", base_img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)))
    for angle in (-20, -10, 10, 20):
        variants.append((f"rot_{angle}", base_img.rotate(angle, expand=False)))
    for factor in (0.72, 0.82, 0.92):
        w, h = base_img.size
        nw = max(8, int(w * factor))
        nh = max(8, int(h * factor))
        x0 = (w - nw) // 2
        y0 = (h - nh) // 2
        cropped = base_img.crop((x0, y0, x0 + nw, y0 + nh)).resize((w, h), Image.Resampling.BICUBIC)
        variants.append((f"crop_{factor}", cropped))
    for factor in (0.7, 1.4):
        variants.append((f"bright_{factor}", ImageEnhance.Brightness(base_img).enhance(factor)))
    for factor in (0.75, 1.5):
        variants.append((f"contrast_{factor}", ImageEnhance.Contrast(base_img).enhance(factor)))
    for radius in (1, 2):
        variants.append((f"blur_{radius}", base_img.filter(ImageFilter.GaussianBlur(radius=radius))))

    arr = np.asarray(base_img, dtype=np.float32)
    for sigma in (10, 20):
        noise = np.random.default_rng(1337 + sigma).normal(0, sigma, arr.shape)
        noisy = np.clip(arr + noise, 0, 255).astype(np.uint8)
        variants.append((f"noise_{sigma}", Image.fromarray(noisy, mode="RGB")))
    return variants


def main():
    with app.app_context():
        init_db()
        matcher = ImageSimilarityMatcher()
        output_dir = os.path.join("reference_images", "robust")
        os.makedirs(output_dir, exist_ok=True)

        sources = (
            ImageRecord.query.filter(ImageRecord.bearing_type.isnot(None))
            .filter(ImageRecord.coating_status == "coated")
            .all()
        )

        added = 0
        for rec in sources:
            if not os.path.exists(rec.filepath):
                continue
            base = Image.open(rec.filepath).convert("RGB")
            variants = generate_augmented_images(base)
            stem = os.path.splitext(os.path.basename(rec.filename or rec.filepath))[0]
            for tag, variant in variants:
                out_name = f"robust_{stem}_{tag}.jpg"
                out_path = os.path.join(output_dir, out_name)
                db_filename = os.path.join("robust", out_name)

                existing = ImageRecord.query.filter(ImageRecord.filepath == out_path).first()
                if existing:
                    continue

                variant.save(out_path, format="JPEG", quality=92)
                emb = matcher.get_image_embedding(out_path)
                new_rec = ImageRecord(
                    filename=db_filename,
                    filepath=out_path,
                    description=rec.description,
                    embedding=json.dumps(emb.tolist()),
                    bearing_type=rec.bearing_type,
                    coating_status=rec.coating_status,
                    size_info=rec.size_info,
                    split="catalog",
                )
                db.session.add(new_rec)
                added += 1
        db.session.commit()
        print(f"Added {added} robust catalog records.")


if __name__ == "__main__":
    main()
