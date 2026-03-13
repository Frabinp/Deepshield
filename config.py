"""
config.py  —  DeepShield centralized configuration.
All paths, hyperparameters, and constants in one place.
"""

import os
from pathlib import Path

# ── Project root ────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent

# ── Data paths ──────────────────────────────────────────
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

FACES_DIR = PROCESSED_DIR / "faces"
FRAMES_DIR = PROCESSED_DIR / "frames"
SPECTROGRAMS_DIR = PROCESSED_DIR / "spectrograms"
SPECTROGRAMS_FF_DIR = PROCESSED_DIR / "spectrograms_faceforensics"
SPECTROGRAMS_ASVSPOOF_DIR = PROCESSED_DIR / "spectrograms_asvspoof"
MANIFESTS_DIR = PROCESSED_DIR / "manifests"
MANIFEST_PATH = MANIFESTS_DIR / "faceforensics_multimodal.json"

# ── Checkpoints ─────────────────────────────────────────
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
IMAGE_CHECKPOINT = CHECKPOINT_DIR / "best_image.pt"
VIDEO_CHECKPOINT = CHECKPOINT_DIR / "best_video.pt"
AUDIO_CHECKPOINT = CHECKPOINT_DIR / "best_audio.pt"
FUSION_CHECKPOINT = CHECKPOINT_DIR / "best_fusion.pt"

# ── Evaluation results ──────────────────────────────────
RESULTS_DIR = PROJECT_ROOT / "eval" / "results"

# ── Training defaults ───────────────────────────────────
DEFAULT_SEED = int(os.environ.get("DEEPSHIELD_SEED", "42"))
DEFAULT_EPOCHS = int(os.environ.get("DEEPSHIELD_EPOCHS", "30"))
DEFAULT_LR = 1e-4
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_PATIENCE = 7

# ── Model constants ─────────────────────────────────────
FEATURE_DIM = 512
N_FRAMES = 16
SAMPLE_RATE = 16000
N_MELS = 128

# ── ImageNet normalization ──────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
