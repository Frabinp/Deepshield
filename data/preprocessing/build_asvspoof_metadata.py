"""
Rebuild metadata for processed ASVspoof spectrograms.
"""

import json
from pathlib import Path


ASV_ROOT = Path("data/raw/asvspoof/LA/LA")
PROTOCOL_PATH = ASV_ROOT / "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt"
FLAC_DIR = ASV_ROOT / "ASVspoof2019_LA_dev/flac"
OUT_ROOT = Path("data/processed/spectrograms")
OUTPUT_PATH = OUT_ROOT / "audio_metadata.json"


def load_protocol_records():
    real_records = []
    fake_records = []

    with open(PROTOCOL_PATH, "r", encoding="utf-8") as handle:
        for line in handle:
            speaker_id, file_id, _, attack_id, label = line.strip().split()
            record = {
                "speaker_id": speaker_id,
                "source_file_id": file_id,
                "attack_id": attack_id,
                "protocol_split": "dev",
                "source_flac": str((FLAC_DIR / f"{file_id}.flac").as_posix()),
            }
            if label == "bonafide":
                real_records.append(record)
            else:
                fake_records.append(record)

    return real_records, fake_records


def build_metadata_for_label(label_name, label_value, protocol_records):
    metadata_records = []
    for spectrogram_path in sorted((OUT_ROOT / label_name).glob("*.npy")):
        sample_index = int(spectrogram_path.stem)
        if sample_index >= len(protocol_records):
            continue

        protocol_record = protocol_records[sample_index]
        metadata_records.append(
            {
                "relative_path": f"{label_name}/{spectrogram_path.name}",
                "label_name": label_name,
                "label": label_value,
                "speaker_id": protocol_record["speaker_id"],
                "source_file_id": protocol_record["source_file_id"],
                "attack_id": protocol_record["attack_id"],
                "protocol_split": protocol_record["protocol_split"],
                "source_flac": protocol_record["source_flac"],
            }
        )
    return metadata_records


def main():
    if not PROTOCOL_PATH.exists():
        raise FileNotFoundError(f"ASVspoof protocol not found: {PROTOCOL_PATH}")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    real_records, fake_records = load_protocol_records()
    metadata_records = []
    metadata_records.extend(build_metadata_for_label("real", 0, real_records))
    metadata_records.extend(build_metadata_for_label("fake", 1, fake_records))

    with open(OUTPUT_PATH, "w", encoding="utf-8") as handle:
        json.dump(metadata_records, handle, indent=2)

    speaker_ids = {record["speaker_id"] for record in metadata_records}
    print("=" * 60)
    print("DEEPSHIELD - ASVspoof Metadata Builder")
    print(f"  Metadata file    : {OUTPUT_PATH}")
    print(f"  Spectrogram rows : {len(metadata_records)}")
    print(f"  Unique speakers  : {len(speaker_ids)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
