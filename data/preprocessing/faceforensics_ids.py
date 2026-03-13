"""
Stable sample-ID helpers for FaceForensics assets.
"""

from __future__ import annotations

import re
from pathlib import Path


FF_ROOT = Path("data/raw/faceforensics")


def sanitize_id_component(value: str) -> str:
    value = value.strip().replace(" ", "_")
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_.-")
    return value.lower()


def build_faceforensics_sample_id(
    video_path: str | Path,
    *,
    ff_root: str | Path = FF_ROOT,
) -> str:
    video_path = Path(video_path)
    ff_root = Path(ff_root)
    relative_path = video_path.relative_to(ff_root)
    if len(relative_path.parts) < 5:
        raise ValueError(f"Unexpected FaceForensics path: {video_path}")

    source_bucket = sanitize_id_component(relative_path.parts[1])
    video_stem = sanitize_id_component(video_path.stem)
    if not source_bucket or not video_stem:
        raise ValueError(f"Could not build sample ID for: {video_path}")
    return f"{source_bucket}_{video_stem}"


def processed_sample_id_from_stem(stem: str) -> str:
    sample_id, marker, frame_suffix = stem.rpartition("_f")
    if marker and frame_suffix.isdigit():
        return sample_id
    return stem
