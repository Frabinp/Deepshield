"""
extract_frames.py - DeepShield frame extraction pipeline.
Reads the FaceForensics++ folder structure directly.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from data.preprocessing.faceforensics_ids import build_faceforensics_sample_id


FF_ROOT = Path("data/raw/faceforensics")
OUT_ROOT = Path("data/processed/frames")
IMG_SIZE = 224
FRAMES_PER_VIDEO = 16

REAL_DIRS = [
    FF_ROOT / "original_sequences/youtube/c40/videos",
    FF_ROOT / "original_sequences/actors/c40/videos",
]

FAKE_DIRS = [
    FF_ROOT / "manipulated_sequences/Deepfakes/c40/videos",
    FF_ROOT / "manipulated_sequences/Face2Face/c40/videos",
    FF_ROOT / "manipulated_sequences/FaceSwap/c40/videos",
    FF_ROOT / "manipulated_sequences/NeuralTextures/c40/videos",
    FF_ROOT / "manipulated_sequences/FaceShifter/c40/videos",
    FF_ROOT / "manipulated_sequences/DeepFakeDetection/c40/videos",
]

(OUT_ROOT / "real").mkdir(parents=True, exist_ok=True)
(OUT_ROOT / "fake").mkdir(parents=True, exist_ok=True)

print("=" * 55)
print("DEEPSHIELD - Frame Extraction Pipeline")
print(f"  Frames per video : {FRAMES_PER_VIDEO}")
print(f"  Image size       : {IMG_SIZE}x{IMG_SIZE}")
print("=" * 55)

stats = {"videos_processed": 0, "frames_saved": 0, "errors": 0}


def extract_frames(video_path: Path, label: str):
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return

    sample_id = build_faceforensics_sample_id(video_path, ff_root=FF_ROOT)
    frame_indices = np.linspace(0, max(total_frames - 1, 0), FRAMES_PER_VIDEO, dtype=int)
    saved = 0

    for frame_index, source_index in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(source_index))
        ok, frame = cap.read()
        if not ok:
            continue
        frame_resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        out_name = f"{sample_id}_f{frame_index:02d}.jpg"
        cv2.imwrite(str(OUT_ROOT / label / out_name), frame_resized)
        saved += 1

    cap.release()
    stats["frames_saved"] += saved
    stats["videos_processed"] += 1


def collect_videos(directories):
    videos = []
    for directory in directories:
        if directory.exists():
            videos.extend(sorted(directory.glob("*.mp4")))
        else:
            print(f"  Missing: {directory}")
    return videos


def process_dirs(directories, label):
    for video_path in tqdm(collect_videos(directories), desc=label.capitalize()):
        try:
            extract_frames(video_path, label)
        except Exception as exc:
            stats["errors"] += 1
            print(f"  ERROR {video_path.name}: {exc}")


print("\n[1/2] Extracting REAL frames ...")
process_dirs(REAL_DIRS, "real")

print("\n[2/2] Extracting FAKE frames ...")
process_dirs(FAKE_DIRS, "fake")

print("\n" + "=" * 55)
print("EXTRACTION COMPLETE")
print(f"  Videos processed : {stats['videos_processed']}")
print(f"  Frames saved     : {stats['frames_saved']}")
print(f"  Errors           : {stats['errors']}")
print("=" * 55)

stats_path = OUT_ROOT / "frame_stats.json"
with open(stats_path, "w", encoding="utf-8") as handle:
    json.dump(stats, handle, indent=2)
print(f"  Stats -> {stats_path}")
