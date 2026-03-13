"""
Extract FaceForensics audio to mel-spectrogram arrays.
Output: data/processed/spectrograms_faceforensics/{real,fake}
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

import librosa
import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from data.preprocessing.faceforensics_ids import build_faceforensics_sample_id


warnings.filterwarnings("ignore")

FF_ROOT = Path("data/raw/faceforensics")
OUT_ROOT = Path("data/processed/spectrograms_faceforensics")
SAMPLE_RATE = 16000
N_MELS = 128
HOP_LENGTH = 512
N_FFT = 1024
DURATION = 4.0

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
print("DEEPSHIELD - Audio/Spectrogram Extraction Pipeline")
print(f"  Sample rate : {SAMPLE_RATE} Hz")
print(f"  Duration    : {DURATION}s per video")
print(f"  Mel bins    : {N_MELS}")
print("=" * 55)

stats = {"processed": 0, "spectrograms": 0, "no_audio": 0, "errors": 0}


def video_to_spectrogram(video_path: Path, out_path: Path) -> bool:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
        tmp_wav = tmp_file.name

    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(video_path),
                "-ac",
                "1",
                "-ar",
                str(SAMPLE_RATE),
                "-t",
                str(DURATION),
                tmp_wav,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if result.returncode != 0:
            stats["no_audio"] += 1
            return False

        signal, sample_rate = librosa.load(tmp_wav, sr=SAMPLE_RATE, mono=True)
        if len(signal) < 1024:
            stats["no_audio"] += 1
            return False

        target_len = int(SAMPLE_RATE * DURATION)
        if len(signal) < target_len:
            signal = np.pad(signal, (0, target_len - len(signal)))
        else:
            signal = signal[:target_len]

        mel = librosa.feature.melspectrogram(
            y=signal,
            sr=sample_rate,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
        mel_img = Image.fromarray((mel_norm * 255).astype(np.uint8))
        mel_resized = np.array(
            mel_img.resize((128, 128), Image.BILINEAR),
            dtype=np.float32,
        ) / 255.0

        np.save(str(out_path), mel_resized)
        return True
    except Exception:
        stats["errors"] += 1
        return False
    finally:
        if os.path.exists(tmp_wav):
            os.remove(tmp_wav)


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
        sample_id = build_faceforensics_sample_id(video_path, ff_root=FF_ROOT)
        out_path = OUT_ROOT / label / f"{sample_id}.npy"
        ok = video_to_spectrogram(video_path, out_path)
        stats["processed"] += 1
        if ok:
            stats["spectrograms"] += 1


print("\n[1/2] Extracting REAL spectrograms ...")
process_dirs(REAL_DIRS, "real")

print("\n[2/2] Extracting FAKE spectrograms ...")
process_dirs(FAKE_DIRS, "fake")

print("\n" + "=" * 55)
print("EXTRACTION COMPLETE")
print(f"  Videos processed   : {stats['processed']}")
print(f"  Spectrograms saved : {stats['spectrograms']}")
print(f"  No audio / silent  : {stats['no_audio']}")
print(f"  Errors             : {stats['errors']}")
print("=" * 55)

stats_path = OUT_ROOT / "audio_stats.json"
with open(stats_path, "w", encoding="utf-8") as handle:
    json.dump(stats, handle, indent=2)
print(f"  Stats -> {stats_path}")
