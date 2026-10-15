#!/usr/bin/env python3
"""Download the ONNX image encoder from the GitHub release and verify its SHA-256.

The app also does this on first use, so running it is optional; the build step runs it so the
first request after a deploy does not wait on the download.
"""
import os

from clip_onnx import ensure_model

if __name__ == "__main__":
    path = ensure_model()
    print(f"{path} ready ({os.path.getsize(path) / 1e6:.1f} MB)")
