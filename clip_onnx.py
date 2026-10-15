"""
CLIP ViT-B/32 image embeddings on ONNX Runtime, with no PyTorch at serving time.

Preprocessing matches transformers' CLIPImageProcessor: shortest side to 224 with bicubic
resampling, a centered 224 x 224 crop, scaling to [0, 1], and CLIP's per-channel normalization.
The encoder is exported and quantized by export_onnx.py, published as a GitHub release asset, and
downloaded with a checksum check on first use (or ahead of time by fetch_model.py).
"""
import hashlib
import json
import os
import tempfile
import threading
import urllib.request

import numpy as np
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_ID = "openai/clip-vit-base-patch32"
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "clip-vit-b32-vision-w8.onnx")
MODEL_URL = (
    "https://github.com/keshavkrishnan08/Enviropeel-Project/releases/download/"
    "clip-w8-v1/clip-vit-b32-vision-w8.onnx"
)
MODEL_SHA256 = "bf76dccbe0902c8623274d03f08fb556bfaee1b66b65526751d347fb9798a617"
FAMILY_PROMPTS_PATH = os.path.join(PROJECT_ROOT, "models", "family_prompts.json")
_download_lock = threading.Lock()

INPUT_SIZE = 224
MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


def to_rgb(image):
    """RGB view of a PIL image, skipping convert()'s full-size copy when it is already RGB."""
    return image if image.mode == "RGB" else image.convert("RGB")


def preprocess(image):
    """Return a (1, 3, 224, 224) float32 array for one PIL image."""
    image = to_rgb(image)
    width, height = image.size
    if width <= height:
        new_w, new_h = INPUT_SIZE, int(INPUT_SIZE * height / width)
    else:
        new_w, new_h = int(INPUT_SIZE * width / height), INPUT_SIZE
    image = image.resize((new_w, new_h), resample=Image.Resampling.BICUBIC, reducing_gap=None)
    left = (new_w - INPUT_SIZE) // 2
    top = (new_h - INPUT_SIZE) // 2
    image = image.crop((left, top, left + INPUT_SIZE, top + INPUT_SIZE))
    pixels = np.asarray(image, dtype=np.float32) / 255.0
    pixels = (pixels - MEAN) / STD
    return pixels.transpose(2, 0, 1)[np.newaxis].astype(np.float32)


def ensure_model(path=MODEL_PATH, url=MODEL_URL, sha256=MODEL_SHA256):
    """Download the encoder from the GitHub release if it is not on disk, verifying its checksum."""
    with _download_lock:
        if os.path.exists(path):
            return path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print(f"Downloading {url} ...")
        digest = hashlib.sha256()
        fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".part")
        try:
            with os.fdopen(fd, "wb") as out, urllib.request.urlopen(url, timeout=60) as response:
                for chunk in iter(lambda: response.read(1 << 20), b""):
                    digest.update(chunk)
                    out.write(chunk)
            if digest.hexdigest() != sha256:
                raise ValueError(f"checksum mismatch for {url}: got {digest.hexdigest()}")
            os.replace(tmp_path, path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        return path


class ClipImageEncoder:
    def __init__(self, model_path=MODEL_PATH, threads=None):
        import onnxruntime as ort

        if model_path == MODEL_PATH:
            ensure_model(model_path)
        elif not os.path.exists(model_path):
            raise FileNotFoundError(f"{model_path} is missing. Build it with `python export_onnx.py`.")
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads or min(4, os.cpu_count() or 1)
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # One photo per request: release activation memory after each run instead of pooling it,
        # which keeps the process inside a 512 MB instance.
        options.enable_cpu_mem_arena = False
        options.enable_mem_pattern = False
        self.session = ort.InferenceSession(model_path, options, providers=["CPUExecutionProvider"])

    def embed(self, image):
        """L2-normalized 512-number embedding for one PIL image."""
        feats = self.session.run(["image_embeds"], {"pixel_values": preprocess(image)})[0][0]
        return (feats / np.linalg.norm(feats)).astype(np.float32)


def load_family_prompt_embeddings(prompts, path=FAMILY_PROMPTS_PATH):
    """Precomputed text embeddings for each family prompt, checked against the prompts in code."""
    with open(path, "r", encoding="utf-8") as fh:
        stored = json.load(fh)
    vectors = {}
    for family, prompt in prompts.items():
        entry = stored.get(family)
        if not entry or entry.get("prompt") != prompt:
            raise ValueError(
                f"{path} is out of date for '{family}'. Run `python export_onnx.py` after editing FAMILY_PROMPTS."
            )
        vector = np.asarray(entry["embedding"], dtype=np.float32)
        vectors[family] = vector / np.linalg.norm(vector)
    return vectors
