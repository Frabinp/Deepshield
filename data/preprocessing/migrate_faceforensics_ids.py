"""
Rename legacy numeric FaceForensics processed files to stable filename-based IDs.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from data.preprocessing.faceforensics_ids import build_faceforensics_sample_id


FF_ROOT = Path("data/raw/faceforensics")
FACES_ROOT = Path("data/processed/faces")
FRAMES_ROOT = Path("data/processed/frames")
AUDIO_ROOT = Path("data/processed/spectrograms_faceforensics")

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


def rename_frame_family(label, root, legacy_id, stable_id):
    renamed = 0
    for source_path in sorted((root / label).glob(f"{legacy_id}_f*")):
        suffix = source_path.name[len(legacy_id) :]
        target_path = source_path.with_name(f"{stable_id}{suffix}")
        if source_path == target_path:
            continue
        if target_path.exists():
            raise FileExistsError(f"Target already exists: {target_path}")
        source_path.rename(target_path)
        renamed += 1
    return renamed


def rename_audio(label, legacy_id, stable_id):
    source_path = AUDIO_ROOT / label / f"{legacy_id}.npy"
    if not source_path.exists():
        return 0
    target_path = AUDIO_ROOT / label / f"{stable_id}.npy"
    if target_path.exists():
        raise FileExistsError(f"Target already exists: {target_path}")
    source_path.rename(target_path)
    return 1


def migrate_label(label, directories):
    stats = {"videos": 0, "faces": 0, "frames": 0, "audio": 0}
    videos = collect_videos(directories)
    for index, video_path in enumerate(videos):
        legacy_id = f"{index:05d}"
        stable_id = build_faceforensics_sample_id(video_path, ff_root=FF_ROOT)
        stats["faces"] += rename_frame_family(label, FACES_ROOT, legacy_id, stable_id)
        stats["frames"] += rename_frame_family(label, FRAMES_ROOT, legacy_id, stable_id)
        stats["audio"] += rename_audio(label, legacy_id, stable_id)
        stats["videos"] += 1
    return stats


def main():
    real_stats = migrate_label("real", REAL_DIRS)
    fake_stats = migrate_label("fake", FAKE_DIRS)

    print("=" * 60)
    print("DEEPSHIELD - FaceForensics ID Migration")
    print(f"  Real videos : {real_stats['videos']}")
    print(f"  Fake videos : {fake_stats['videos']}")
    print(f"  Face files  : {real_stats['faces'] + fake_stats['faces']}")
    print(f"  Frame files : {real_stats['frames'] + fake_stats['frames']}")
    print(f"  Audio files : {real_stats['audio'] + fake_stats['audio']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
