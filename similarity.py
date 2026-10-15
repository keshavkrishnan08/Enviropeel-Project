import json
import os

import numpy as np
from PIL import Image

from catalog_metadata import lookup_reference_profile, normalize_project_label
from clip_onnx import ClipImageEncoder, load_family_prompt_embeddings, to_rgb
from database import BEARING_TYPES, ImageRecord


DEFAULT_CONFIG = {
    "clip_weight": 0.72,
    "visual_weight": 0.10,
    "family_weight": 0.18,
    "family_penalty": 0.05,
    "metadata_family_bonus": 0.35,
    "coated_bonus": 0.02,
    "prompt_weight": 0.20,
    "knn_family_weight": 0.80,
    "top_family_confidence_threshold": 0.55,
}

FAMILY_PROMPTS = {
    "pillow_block": "a coated pillow block bearing mounted on industrial equipment",
    "flange": "a coated industrial flange assembly attached to piping",
    "gearbox": "a coated gearbox or drive reducer housing in an industrial setting",
    "support_pedestal": "a coated support pedestal or machine mount base",
    "bearing_housing": "a coated bearing housing mounted on an industrial frame",
}


class ImageSimilarityMatcher:
    def __init__(self, config_path=None):
        print("Loading CLIP (ONNX)...")
        self.encoder = ClipImageEncoder()
        self.config_path = config_path or os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "similarity_config.json",
        )
        self.config = self._load_config()
        self.reference_images = []
        self.reference_embeddings = np.empty((0, 0), dtype=np.float32)
        self.family_prompt_embeddings = self._build_family_prompt_embeddings()

    def _load_config(self):
        config = dict(DEFAULT_CONFIG)
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r", encoding="utf-8") as fh:
                    loaded = json.load(fh)
                for key, value in loaded.items():
                    if key in config:
                        config[key] = float(value)
            except Exception:
                pass
        return config

    def _build_family_prompt_embeddings(self):
        # Text embeddings for the fixed prompts are precomputed by export_onnx.py,
        # so the server never loads the CLIP text encoder.
        prompts = {family: FAMILY_PROMPTS[family] for family in BEARING_TYPES if family in FAMILY_PROMPTS}
        return load_family_prompt_embeddings(prompts)

    def get_image_embedding(self, image_path):
        return self.encoder.embed(Image.open(image_path))

    def get_visual_signature(self, image_path):
        img = to_rgb(Image.open(image_path)).resize((128, 128))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        rgb_mean = arr.mean(axis=(0, 1))
        rgb_std = arr.std(axis=(0, 1))
        gray = arr.mean(axis=2)
        gy, gx = np.gradient(gray)
        edge_ratio = float(np.mean(np.sqrt(gx ** 2 + gy ** 2) > 0.10))
        object_fill = float(np.mean((gray > 0.10) & (gray < 0.95)))
        aspect_ratio = float(img.size[0] / max(img.size[1], 1))
        return np.array(
            [
                rgb_mean[0],
                rgb_mean[1],
                rgb_mean[2],
                rgb_std[0],
                rgb_std[1],
                rgb_std[2],
                edge_ratio,
                object_fill,
                aspect_ratio,
            ],
            dtype=np.float32,
        )

    def _visual_similarity(self, query_signature, ref_signature):
        distance = np.linalg.norm(query_signature - ref_signature)
        return float(1.0 / (1.0 + distance))

    def _normalize_score_map(self, score_map):
        if not score_map:
            return {}
        values = np.array(list(score_map.values()), dtype=np.float32)
        lo = float(values.min())
        hi = float(values.max())
        if hi - lo < 1e-6:
            return {key: 1.0 for key in score_map}
        return {key: float((value - lo) / (hi - lo)) for key, value in score_map.items()}

    def _infer_family_scores(self, query_embedding, metadata=None):
        if metadata and metadata.get("bearing_type"):
            return {metadata["bearing_type"]: 1.0}

        knn_scores = {}
        if self.reference_images:
            for ref in self.reference_images:
                family = normalize_project_label(ref.get("bearing_type"))
                if not family:
                    continue
                sim = float(np.dot(query_embedding, ref["image_embedding"]))
                knn_scores[family] = max(knn_scores.get(family, -1.0), sim)

        prompt_scores = {}
        for family, vector in self.family_prompt_embeddings.items():
            prompt_scores[family] = float(np.dot(query_embedding, vector))

        knn_scores = self._normalize_score_map(knn_scores)
        prompt_scores = self._normalize_score_map(prompt_scores)

        scores = {}
        for family in set(knn_scores) | set(prompt_scores):
            scores[family] = (
                self.config["knn_family_weight"] * knn_scores.get(family, 0.0)
                + self.config["prompt_weight"] * prompt_scores.get(family, 0.0)
            )
        return self._normalize_score_map(scores)

    def _build_query_context(self, query_path):
        metadata = lookup_reference_profile(query_path) or {}
        query_embedding = self.get_image_embedding(query_path)
        query_signature = self.get_visual_signature(query_path)
        family_scores = self._infer_family_scores(query_embedding, metadata=metadata)
        top_family = None
        top_family_confidence = 0.0
        if family_scores:
            top_family, top_family_confidence = max(
                family_scores.items(),
                key=lambda item: item[1],
            )
        return {
            "embedding": query_embedding,
            "signature": query_signature,
            "metadata": metadata,
            "family_scores": family_scores,
            "top_family": top_family,
            "top_family_confidence": float(top_family_confidence),
        }

    def load_reference_images(
        self,
        bearing_type=None,
        coating_status=None,
        catalog_only=True,
        labeled_only=True,
    ):
        self.reference_images = []
        q = ImageRecord.query
        if catalog_only:
            q = q.filter((ImageRecord.split == "catalog") | (ImageRecord.split.is_(None)))
        if labeled_only:
            q = q.filter(ImageRecord.bearing_type.isnot(None))
        if bearing_type:
            q = q.filter(ImageRecord.bearing_type == bearing_type)
        if coating_status:
            q = q.filter(ImageRecord.coating_status == coating_status)

        for rec in q.all():
            if not os.path.exists(rec.filepath):
                continue
            try:
                metadata = lookup_reference_profile(rec.filename) or {}
                if rec.embedding:
                    emb = np.asarray(json.loads(rec.embedding), dtype=np.float32)
                    emb = emb / np.linalg.norm(emb)
                else:
                    emb = self.get_image_embedding(rec.filepath)
                self.reference_images.append(
                    {
                        "id": rec.id,
                        "filename": rec.filename,
                        "description": metadata.get("description") or rec.description,
                        "filepath": rec.filepath,
                        "bearing_type": metadata.get("bearing_type")
                        or normalize_project_label(rec.bearing_type),
                        "coating_status": rec.coating_status,
                        "size_info": metadata.get("size_info") or rec.size_info,
                        "image_embedding": emb,
                        "visual_signature": self.get_visual_signature(rec.filepath),
                    }
                )
            except Exception as e:
                print("skip", rec.filename, e)

        if self.reference_images:
            self.reference_embeddings = np.asarray(
                [ref["image_embedding"] for ref in self.reference_images],
                dtype=np.float32,
            )
        else:
            self.reference_embeddings = np.empty((0, 0), dtype=np.float32)

        filter_desc = []
        if catalog_only:
            filter_desc.append("catalog")
        if labeled_only:
            filter_desc.append("labeled")
        if bearing_type:
            filter_desc.append(f"bearing={bearing_type}")
        if coating_status:
            filter_desc.append(f"coating={coating_status}")
        suffix = f" [{', '.join(filter_desc)}]" if filter_desc else ""
        print("Loaded", len(self.reference_images), "refs" + suffix)

    def _score_reference(self, query_context, ref, prefer_coated, scoring_config=None):
        config = dict(self.config)
        if scoring_config:
            config.update(scoring_config)

        clip_similarity = float(np.dot(query_context["embedding"], ref["image_embedding"]))
        visual_similarity = self._visual_similarity(
            query_context["signature"],
            ref["visual_signature"],
        )
        family_score = query_context["family_scores"].get(ref.get("bearing_type"), 0.0)

        rank_score = (
            config["clip_weight"] * clip_similarity
            + config["visual_weight"] * visual_similarity
            + config["family_weight"] * family_score
        )

        query_metadata_family = query_context["metadata"].get("bearing_type")
        if query_metadata_family and query_metadata_family == ref.get("bearing_type"):
            rank_score += config["metadata_family_bonus"]

        if (
            query_context["top_family"]
            and query_context["top_family_confidence"] >= config["top_family_confidence_threshold"]
            and ref.get("bearing_type") != query_context["top_family"]
        ):
            rank_score -= config["family_penalty"] * query_context["top_family_confidence"]

        if prefer_coated and ref.get("coating_status") == "coated":
            rank_score += config["coated_bonus"]

        return clip_similarity, visual_similarity, family_score, rank_score

    def find_similar_images(
        self,
        query_path,
        top_k=1,
        prefer_coated=True,
        exclude_paths=None,
        scoring_config=None,
    ):
        if not self.reference_images:
            return []

        exclude_paths = {os.path.normpath(path) for path in (exclude_paths or [])}
        query_context = self._build_query_context(query_path)
        scored = []
        for ref in self.reference_images:
            if os.path.normpath(ref.get("filepath", "")) in exclude_paths:
                continue
            clip_similarity, visual_similarity, family_score, rank_score = self._score_reference(
                query_context,
                ref,
                prefer_coated=prefer_coated,
                scoring_config=scoring_config,
            )
            scored.append(
                {
                    "id": ref["id"],
                    "filename": ref["filename"],
                    "description": ref["description"],
                    "similarity_score": clip_similarity,
                    "rank_score": float(rank_score),
                    "visual_score": float(visual_similarity),
                    "family_score": float(family_score),
                    "image_url": "/api/reference/" + ref["filename"].replace("\\", "/"),
                    "bearing_type": ref.get("bearing_type"),
                    "coating_status": ref.get("coating_status"),
                    "size_info": ref.get("size_info"),
                    "predicted_family": query_context.get("top_family"),
                    "predicted_family_confidence": query_context.get("top_family_confidence", 0.0),
                }
            )

        scored.sort(key=lambda item: (item["rank_score"], item["similarity_score"]), reverse=True)

        results = []
        seen_ids = set()
        for match in scored:
            if match["id"] in seen_ids:
                continue
            results.append(match)
            seen_ids.add(match["id"])
            if len(results) >= top_k:
                break
        return results
