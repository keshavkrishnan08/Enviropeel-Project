#!/usr/bin/env python3
"""
Export the CLIP image encoder to ONNX with 8-bit weights, and precompute the family prompt embeddings.

The server only needs these two files, so it runs on ONNX Runtime without PyTorch:

    models/clip-vit-b32-vision-w8.onnx   image encoder, weight-only 8-bit (MatMulNBits, blocks of 128)
    models/family_prompts.json           one L2-normalized text embedding per family prompt

Weight-only quantization keeps activations in float32, so embeddings stay within a cosine of
0.9994 of the original model on every catalog photo while the file shrinks from 352 MB to 98 MB.
Plain dynamic int8 quantization was smaller but drifted to 0.93 on some photos.

Needs the development requirements (PyTorch, Transformers, ONNX): pip install -r requirements-dev.txt
"""
import hashlib
import json
import os
import tempfile

import numpy as np
import onnx
import torch
from onnxruntime.quantization.matmul_nbits_quantizer import DefaultWeightOnlyQuantConfig, MatMulNBitsQuantizer
from transformers import CLIPModel, CLIPProcessor

from clip_onnx import FAMILY_PROMPTS_PATH, MODEL_ID, MODEL_PATH
from similarity import FAMILY_PROMPTS


class ImageEncoder(torch.nn.Module):
    def __init__(self, clip):
        super().__init__()
        self.clip = clip

    def forward(self, pixel_values):
        return self.clip.get_image_features(pixel_values=pixel_values)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    model = CLIPModel.from_pretrained(MODEL_ID).eval()
    processor = CLIPProcessor.from_pretrained(MODEL_ID)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        fp32_path = os.path.join(tmp, "vision-fp32.onnx")
        torch.onnx.export(
            ImageEncoder(model),
            torch.zeros(1, 3, 224, 224),
            fp32_path,
            input_names=["pixel_values"],
            output_names=["image_embeds"],
            dynamic_axes={"pixel_values": {0: "batch"}, "image_embeds": {0: "batch"}},
            opset_version=17,
            do_constant_folding=True,
        )
        config = DefaultWeightOnlyQuantConfig(block_size=128, is_symmetric=True, bits=8)
        quantizer = MatMulNBitsQuantizer(onnx.load(fp32_path), algo_config=config)
        quantizer.process()
        quantizer.model.save_model_to_file(MODEL_PATH, use_external_data_format=False)

    families = list(FAMILY_PROMPTS)
    texts = [FAMILY_PROMPTS[family] for family in families]
    with torch.no_grad():
        inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True)
        vectors = model.get_text_features(**inputs).numpy()
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    with open(FAMILY_PROMPTS_PATH, "w", encoding="utf-8") as fh:
        json.dump(
            {family: {"prompt": FAMILY_PROMPTS[family], "embedding": vectors[i].round(7).tolist()}
             for i, family in enumerate(families)},
            fh,
            indent=1,
        )

    size_mb = os.path.getsize(MODEL_PATH) / 1e6
    print(f"wrote {MODEL_PATH} ({size_mb:.1f} MB, sha256 {sha256(MODEL_PATH)})")
    print(f"wrote {FAMILY_PROMPTS_PATH} ({len(families)} prompts)")


if __name__ == "__main__":
    main()
