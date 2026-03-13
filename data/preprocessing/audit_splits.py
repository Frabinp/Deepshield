"""
Audit train/val split hygiene across DeepShield datasets.
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from data.preprocessing.dataset import AudioDataset, DeepShieldDataset, ImageDataset, VideoDataset


REPORT_PATH = Path("data/processed/manifests/split_audit.json")
MANIFEST_PATH = Path("data/processed/manifests/faceforensics_multimodal.json")
OFFICIAL_AUDIO_ROOT = Path("data/processed/spectrograms_asvspoof")


def overlap_summary(train_keys, val_keys):
    overlap = train_keys & val_keys
    return {
        "train_groups": len(train_keys),
        "val_groups": len(val_keys),
        "overlap_groups": len(overlap),
        "passes": len(overlap) == 0,
        "overlap_examples": sorted(list(overlap))[:10],
    }


def audit_image():
    train_ds = ImageDataset(split="train")
    val_ds = ImageDataset(split="val")
    train_groups = {sample[2] for sample in train_ds.samples}
    val_groups = {sample[2] for sample in val_ds.samples}

    report = overlap_summary(train_groups, val_groups)
    report.update(
        {
            "train_samples": len(train_ds),
            "val_samples": len(val_ds),
            "group_strategy": "source_video",
        }
    )
    return report


def audit_video():
    train_ds = VideoDataset(split="train")
    val_ds = VideoDataset(split="val")
    train_groups = {sample[2] for sample in train_ds.samples}
    val_groups = {sample[2] for sample in val_ds.samples}

    report = overlap_summary(train_groups, val_groups)
    report.update(
        {
            "train_samples": len(train_ds),
            "val_samples": len(val_ds),
            "group_strategy": "source_video",
        }
    )
    return report


def audit_audio():
    root_dir = str(OFFICIAL_AUDIO_ROOT) if OFFICIAL_AUDIO_ROOT.exists() else "data/processed/spectrograms"
    train_split = "train+dev" if OFFICIAL_AUDIO_ROOT.exists() else "train"
    val_split = "eval" if OFFICIAL_AUDIO_ROOT.exists() else "val"
    train_ds = AudioDataset(root_dir=root_dir, split=train_split)
    val_ds = AudioDataset(root_dir=root_dir, split=val_split)
    train_groups = {sample[2] for sample in train_ds.samples}
    val_groups = {sample[2] for sample in val_ds.samples}
    metadata_coverage = sum(1 for sample in train_ds.samples + val_ds.samples if sample[3] is not None)
    train_speakers = {
        sample[3]["speaker_id"]
        for sample in train_ds.samples
        if sample[3] is not None
    }
    val_speakers = {
        sample[3]["speaker_id"]
        for sample in val_ds.samples
        if sample[3] is not None
    }
    speaker_overlap = train_speakers & val_speakers
    train_label_counts = {
        "real": sum(1 for sample in train_ds.samples if sample[1] == 0),
        "fake": sum(1 for sample in train_ds.samples if sample[1] == 1),
    }
    val_label_counts = {
        "real": sum(1 for sample in val_ds.samples if sample[1] == 0),
        "fake": sum(1 for sample in val_ds.samples if sample[1] == 1),
    }

    report = overlap_summary(train_groups, val_groups)
    report.update(
        {
            "train_samples": len(train_ds),
            "val_samples": len(val_ds),
            "group_strategy": train_ds.group_by,
            "metadata_path": str(train_ds.metadata_path),
            "root_dir": root_dir,
            "train_split": train_split,
            "val_split": val_split,
            "metadata_covered_samples": metadata_coverage,
            "train_label_counts": train_label_counts,
            "val_label_counts": val_label_counts,
            "speaker_overlap_across_splits": len(speaker_overlap),
            "speaker_overlap_examples": sorted(list(speaker_overlap))[:10],
        }
    )
    return report


def audit_multimodal():
    if not MANIFEST_PATH.exists():
        return {
            "available": False,
            "manifest_path": str(MANIFEST_PATH),
        }

    train_ds = DeepShieldDataset(mode="train", modalities=["image", "video"], manifest_path=str(MANIFEST_PATH))
    val_ds = DeepShieldDataset(mode="val", modalities=["image", "video"], manifest_path=str(MANIFEST_PATH))
    train_groups = {
        f"{record['class_name']}:{record['sample_id']}"
        for record in train_ds.records
    }
    val_groups = {
        f"{record['class_name']}:{record['sample_id']}"
        for record in val_ds.records
    }

    report = overlap_summary(train_groups, val_groups)
    report.update(
        {
            "available": True,
            "train_samples": len(train_ds),
            "val_samples": len(val_ds),
            "group_strategy": "sample_id",
            "manifest_path": str(MANIFEST_PATH),
        }
    )
    return report


def main():
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "image": audit_image(),
        "video": audit_video(),
        "audio": audit_audio(),
        "multimodal": audit_multimodal(),
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print("=" * 60)
    print("DEEPSHIELD - Split Audit")
    for key in ["image", "video", "audio", "multimodal"]:
        section = report[key]
        if not section.get("available", True):
            print(f"  {key:<10} unavailable")
            continue
        status = "PASS" if section["passes"] else "FAIL"
        print(
            f"  {key:<10} {status}  "
            f"train={section['train_samples']}  "
            f"val={section['val_samples']}  "
            f"overlap_groups={section['overlap_groups']}"
        )
        if key == "audio":
            print(
                f"             split={section['train_split']} vs {section['val_split']}  "
                f"speakers_across_splits={section['speaker_overlap_across_splits']}  "
                f"train_real={section['train_label_counts']['real']}  "
                f"train_fake={section['train_label_counts']['fake']}  "
                f"val_real={section['val_label_counts']['real']}  "
                f"val_fake={section['val_label_counts']['fake']}"
            )
    print(f"  Report         : {REPORT_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
