"""
Extract balanced ASVspoof spectrograms using the official train/dev/eval protocol.
"""

import json
from pathlib import Path

import librosa
import numpy as np
from PIL import Image
from tqdm import tqdm


ASV_ROOT = Path("data/raw/asvspoof/LA/LA")
OUT_ROOT = Path("data/processed/spectrograms_asvspoof")

PROTOCOLS = {
    "train": ASV_ROOT / "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt",
    "dev": ASV_ROOT / "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt",
    "eval": ASV_ROOT / "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt",
}

FLAC_DIRS = {
    "train": ASV_ROOT / "ASVspoof2019_LA_train/flac",
    "dev": ASV_ROOT / "ASVspoof2019_LA_dev/flac",
    "eval": ASV_ROOT / "ASVspoof2019_LA_eval/flac",
}

SAMPLE_RATE = 16000
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 512
DURATION = 4.0
MAX_PER_CLASS = {
    "train": 2000,
    "dev": 2000,
    "eval": 2000,
}


def load_protocol_records(split_name):
    protocol_path = PROTOCOLS[split_name]
    flac_dir = FLAC_DIRS[split_name]
    if not protocol_path.exists():
        raise FileNotFoundError(f"Protocol not found: {protocol_path}")
    if not flac_dir.exists():
        raise FileNotFoundError(f"FLAC directory not found: {flac_dir}")

    by_label = {"real": [], "fake": []}
    with open(protocol_path, "r", encoding="utf-8") as handle:
        for line in handle:
            speaker_id, file_id, _, attack_id, label = line.strip().split()
            flac_path = flac_dir / f"{file_id}.flac"
            if not flac_path.exists():
                continue

            record = {
                "split": split_name,
                "speaker_id": speaker_id,
                "source_file_id": file_id,
                "attack_id": attack_id,
                "label_name": "real" if label == "bonafide" else "fake",
                "label": 0 if label == "bonafide" else 1,
                "source_flac": str(flac_path.as_posix()),
            }
            by_label[record["label_name"]].append(record)
    return by_label


def select_balanced_records(records_by_label, split_name):
    max_per_class = MAX_PER_CLASS[split_name]
    real_records = records_by_label["real"][:max_per_class]
    fake_records = records_by_label["fake"][: min(len(records_by_label["fake"]), len(real_records), max_per_class)]
    selected = {
        "real": real_records,
        "fake": fake_records,
    }
    return selected


def to_mel_array(flac_path):
    target_length = int(SAMPLE_RATE * DURATION)
    audio, _ = librosa.load(str(flac_path), sr=SAMPLE_RATE, mono=True)
    if len(audio) < target_length:
        audio = np.pad(audio, (0, target_length - len(audio)))
    else:
        audio = audio[:target_length]

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
    mel_uint8 = (mel_norm * 255).astype(np.uint8)
    resized = Image.fromarray(mel_uint8).resize((128, 128))
    return np.array(resized, dtype=np.float32) / 255.0


def ensure_directories():
    for split_name in PROTOCOLS:
        for label_name in ["real", "fake"]:
            (OUT_ROOT / split_name / label_name).mkdir(parents=True, exist_ok=True)


def process_selected_records():
    ensure_directories()

    stats = {
        split_name: {
            "real": 0,
            "fake": 0,
            "skipped": 0,
            "errors": 0,
        }
        for split_name in PROTOCOLS
    }
    metadata_records = []

    for split_name in ["train", "dev", "eval"]:
        selected = select_balanced_records(load_protocol_records(split_name), split_name)

        print("=" * 60)
        print(f"DEEPSHIELD - ASVspoof {split_name.upper()} Extraction")
        print(f"  Real selected : {len(selected['real'])}")
        print(f"  Fake selected : {len(selected['fake'])}")
        print("=" * 60)

        for label_name in ["real", "fake"]:
            for record in tqdm(selected[label_name], desc=f"{split_name}:{label_name}"):
                out_path = OUT_ROOT / split_name / label_name / f"{record['source_file_id']}.npy"
                record["relative_path"] = str(out_path.relative_to(OUT_ROOT).as_posix())

                if out_path.exists():
                    stats[split_name]["skipped"] += 1
                    metadata_records.append(record)
                    continue

                try:
                    mel_array = to_mel_array(Path(record["source_flac"]))
                    np.save(str(out_path), mel_array)
                    stats[split_name][label_name] += 1
                    metadata_records.append(record)
                except Exception:
                    stats[split_name]["errors"] += 1

    with open(OUT_ROOT / "audio_metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata_records, handle, indent=2)

    with open(OUT_ROOT / "audio_stats.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "max_per_class": MAX_PER_CLASS,
                "stats": stats,
            },
            handle,
            indent=2,
        )

    print("\n" + "=" * 60)
    print("ASVSPOOF EXTRACTION COMPLETE")
    for split_name in ["train", "dev", "eval"]:
        split_stats = stats[split_name]
        print(
            f"  {split_name:<5} "
            f"real={split_stats['real']}  "
            f"fake={split_stats['fake']}  "
            f"skipped={split_stats['skipped']}  "
            f"errors={split_stats['errors']}"
        )
    print(f"  Output root : {OUT_ROOT}")
    print("=" * 60)


if __name__ == "__main__":
    process_selected_records()
