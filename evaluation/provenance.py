"""Reproducibility metadata for benchmark reports."""

from __future__ import annotations

import hashlib
import platform
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def environment_manifest() -> dict[str, object]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "torch": torch.__version__,
    }
