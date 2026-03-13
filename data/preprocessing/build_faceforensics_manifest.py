"""
Build a manifest for aligned FaceForensics multimodal samples.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from data.preprocessing.faceforensics_ids import build_faceforensics_sample_id


FF_ROOT = Path("data/raw/faceforensics")
FACES_ROOT = Path("data/processed/faces")
FRAMES_ROOT = Path("data/processed/frames")
AUDIO_ROOT = Path("data/processed/spectrograms_faceforensics")
MANIFEST_DIR = Path("data/processed/manifests")
MANIFEST_PATH = MANIFEST_DIR / "faceforensics_multimodal.json"

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


def collect_videos(directories):
    videos = []
    for directory in directories:
        if directory.exists():
            videos.extend(sorted(directory.glob("*.mp4")))
    return videos


def build_records(label, videos):
    records = []
    for video_path in videos:
        sample_id = build_faceforensics_sample_id(video_path, ff_root=FF_ROOT)
        source_bucket = video_path.relative_to(FF_ROOT).parts[1]
        face_paths = sorted((FACES_ROOT / label).glob(f"{sample_id}_f*.jpg"))
        frame_paths = sorted((FRAMES_ROOT / label).glob(f"{sample_id}_f*.jpg"))
        audio_path = AUDIO_ROOT / label / f"{sample_id}.npy"
        records.append(
            {
                "sample_id": sample_id,
                "label": 0 if label == "real" else 1,
                "class_name": label,
                "source_bucket": source_bucket,
                "source_video": str(video_path.as_posix()),
                "face_paths": [str(path.as_posix()) for path in face_paths],
                "frame_paths": [str(path.as_posix()) for path in frame_paths],
                "audio_path": str(audio_path.as_posix()) if audio_path.exists() else None,
                "has_image": bool(face_paths),
                "has_video": bool(frame_paths),
                "has_audio": audio_path.exists(),
            }
        )
    return records


def main():
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

    real_records = build_records("real", collect_videos(REAL_DIRS))
    fake_records = build_records("fake", collect_videos(FAKE_DIRS))
    records = real_records + fake_records

    with open(MANIFEST_PATH, "w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2)

    visual_records = [
        record for record in records if record["has_image"] and record["has_video"]
    ]
    complete_records = [
        record
        for record in records
        if record["has_image"] and record["has_video"] and record["has_audio"]
    ]

    print("=" * 55)
    print("DEEPSHIELD - FaceForensics Manifest Builder")
    print(f"  Total records    : {len(records)}")
    print(f"  Image+Video      : {len(visual_records)}")
    print(f"  Complete records : {len(complete_records)}")
    print(f"  Manifest         : {MANIFEST_PATH}")
    print("=" * 55)


if __name__ == "__main__":
    main()
