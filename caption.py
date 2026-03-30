import re
from PIL import Image

_captioner = None


def _get_captioner():
    global _captioner
    if _captioner is None:
        import torch
        from transformers import pipeline
        dev = 0 if torch.cuda.is_available() else -1
        _captioner = pipeline(
            "image-to-text",
            model="Salesforce/blip-image-captioning-base",
            device=dev,
        )
    return _captioner


def get_caption(image_path, max_len=80):
    """Return a short caption for an image."""
    try:
        img = Image.open(image_path).convert("RGB")
        out = _get_captioner()(img)
        if not out or not isinstance(out, list):
            return None
        first = out[0]
        if isinstance(first, dict):
            text = (first.get("generated_text") or "").strip()
        elif isinstance(first, str):
            text = first.strip()
        else:
            return None
        if not text:
            return None
        text = re.split(r"[.!?]\s+", text)[0].strip()
        if not text:
            return None
        if len(text) > max_len:
            text = text[: max_len - 3].rsplit(" ", 1)[0] + "..."
        return text
    except Exception:
        return None
