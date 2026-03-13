"""
extract_faces.py - DeepShield face extraction pipeline.
Reads the FaceForensics++ folder structure directly.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from data.preprocessing.faceforensics_ids import build_faceforensics_sample_id


FF_ROOT = Path("data/raw/faceforensics")
OUT_ROOT = Path("data/processed/faces")
IMG_SIZE = 224
FRAMES_PER_VIDEO = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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

mtcnn = MTCNN(
    image_size=IMG_SIZE,
    margin=20,
    keep_all=False,
    post_process=False,
    device=DEVICE,
)

print("=" * 55)
print("DEEPSHIELD - Face Extraction Pipeline")
print(f"  Device : {DEVICE}")
print("=" * 55)

stats = {"real": 0, "fake": 0, "no_face": 0, "errors": 0}


def sample_frames(video_path: Path, count: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return []

    frame_indices = np.linspace(0, max(total_frames - 1, 0), count, dtype=int)
    frames = []
    for frame_index in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = cap.read()
        if ok:
            frames.append(frame)
    cap.release()
    return frames


def process_video(video_path: Path, label: str):
    sample_id = build_faceforensics_sample_id(video_path, ff_root=FF_ROOT)
    saved = 0
    for frame_index, frame in enumerate(sample_frames(video_path, FRAMES_PER_VIDEO)):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            face = mtcnn(rgb)
        except Exception:
            stats["errors"] += 1
            continue

        if face is None:
            stats["no_face"] += 1
            continue

        face_np = face.permute(1, 2, 0).numpy().astype(np.uint8)
        face_bgr = cv2.cvtColor(face_np, cv2.COLOR_RGB2BGR)
        out_name = f"{sample_id}_f{frame_index:03d}.jpg"
        cv2.imwrite(str(OUT_ROOT / label / out_name), face_bgr)
        stats[label] += 1
        saved += 1
    return saved


def collect_videos(directories):
    videos = []
    for directory in directories:
        if directory.exists():
            videos.extend(sorted(directory.glob("*.mp4")))
        else:
            print(f"  Missing: {directory}")
    return videos


print("\n[1/2] Extracting REAL faces ...")
for video_path in tqdm(collect_videos(REAL_DIRS), desc="Real"):
    try:
        process_video(video_path, "real")
    except Exception:
        stats["errors"] += 1

print("\n[2/2] Extracting FAKE faces ...")
for video_path in tqdm(collect_videos(FAKE_DIRS), desc="Fake"):
    try:
        process_video(video_path, "fake")
    except Exception:
        stats["errors"] += 1

total_faces = stats["real"] + stats["fake"]
print("\n" + "=" * 55)
print("EXTRACTION COMPLETE")
print(f"  Real faces saved : {stats['real']}")
print(f"  Fake faces saved : {stats['fake']}")
print(f"  Total            : {total_faces}")
print(f"  No-face frames   : {stats['no_face']}")
print(f"  Errors           : {stats['errors']}")
print("=" * 55)

stats_path = OUT_ROOT / "extraction_stats.json"
with open(stats_path, "w", encoding="utf-8") as handle:
    json.dump(stats, handle, indent=2)
print(f"  Stats -> {stats_path}")
