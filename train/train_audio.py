"""
train_audio.py - DeepShield audio model training on official ASVspoof splits.
"""

import os
import sys
from contextlib import nullcontext

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from runtime_env import configure_wandb_environment

configure_wandb_environment()

import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from data.preprocessing.dataset import AudioDataset
from models.audio_model import AudioDetector
from train.training_utils import compute_best_f1_threshold, seed_everything


BATCH_SIZE = 16
EPOCHS = int(os.environ.get("DEEPSHIELD_EPOCHS", "30"))
LR = 1e-4
WEIGHT_DECAY = 1e-4
GRAD_ACCUM = 4
PATIENCE = 7
CHECKPOINT = os.environ.get("DEEPSHIELD_CHECKPOINT", "checkpoints/best_audio.pt")
AUDIO_ROOT = "data/processed/spectrograms_asvspoof"
TRAIN_SPLIT = "train+dev"
VAL_SPLIT = "eval"
SEED = int(os.environ.get("DEEPSHIELD_SEED", "42"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"

os.makedirs("checkpoints", exist_ok=True)
seed_everything(SEED)


def amp_context():
    if USE_AMP:
        return autocast(device_type="cuda", enabled=True)
    return nullcontext()


print("=" * 60)
print("DEEPSHIELD - Audio Model Training")
print(f"  Device      : {DEVICE}")
print(f"  Audio root  : {AUDIO_ROOT}")
print(f"  Train split : {TRAIN_SPLIT}")
print(f"  Val split   : {VAL_SPLIT}")
print(f"  Seed        : {SEED}")
print("=" * 60)

train_ds = AudioDataset(root_dir=AUDIO_ROOT, split=TRAIN_SPLIT)
val_ds = AudioDataset(root_dir=AUDIO_ROOT, split=VAL_SPLIT)

if len(train_ds) == 0 or len(val_ds) == 0:
    raise RuntimeError(
        "Official ASVspoof spectrograms were not found. "
        "Run data/preprocessing/extract_audio_asvspoof.py first."
    )

print(f"  Train samples : {len(train_ds)}")
print(f"  Val samples   : {len(val_ds)}")

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=USE_AMP,
)
val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=USE_AMP,
)

model = AudioDetector().to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
criterion = nn.BCEWithLogitsLoss()
scaler = GradScaler(DEVICE.type, enabled=USE_AMP)

best_auc = 0.0
patience_count = 0

for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss = 0.0
    optimizer.zero_grad()

    for step, (mels, labels) in enumerate(train_loader):
        mels = mels.to(DEVICE)
        labels = labels.to(DEVICE)

        with amp_context():
            _, logits = model(mels)
            loss = criterion(logits.squeeze(), labels) / GRAD_ACCUM

        scaler.scale(loss).backward()
        train_loss += loss.item() * GRAD_ACCUM

        if (step + 1) % GRAD_ACCUM == 0 or (step + 1) == len(train_loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

    train_loss /= len(train_loader)

    model.eval()
    val_loss = 0.0
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for mels, labels in val_loader:
            mels = mels.to(DEVICE)
            labels = labels.to(DEVICE)

            with amp_context():
                _, logits = model(mels)
                loss = criterion(logits.squeeze(), labels)

            val_loss += loss.item()
            probs = torch.sigmoid(logits.squeeze()).cpu().numpy()
            all_probs.extend(probs.tolist() if probs.ndim > 0 else [probs.item()])
            all_labels.extend(labels.cpu().numpy().tolist())

    val_loss /= len(val_loader)
    auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0
    threshold, threshold_f1 = compute_best_f1_threshold(all_labels, all_probs)
    scheduler.step()

    print(
        f"Epoch {epoch:02d}/{EPOCHS}  "
        f"train_loss={train_loss:.4f}  "
        f"val_loss={val_loss:.4f}  "
        f"val_AUC={auc:.4f}  "
        f"thr={threshold:.4f}  "
        f"bestF1={threshold_f1:.4f}"
    )

    if auc > best_auc:
        best_auc = auc
        torch.save(
            {
                "model_state": model.state_dict(),
                "best_auc": best_auc,
                "best_threshold": threshold,
                "best_threshold_metric": "best_f1",
                "best_threshold_f1": threshold_f1,
                "audio_root": AUDIO_ROOT,
                "train_split": TRAIN_SPLIT,
                "val_split": VAL_SPLIT,
                "seed": SEED,
            },
            CHECKPOINT,
        )
        print(f"  Saved best audio checkpoint (AUC={best_auc:.4f})")
        patience_count = 0
    else:
        patience_count += 1
        if patience_count >= PATIENCE:
            print(f"  Early stopping at epoch {epoch}")
            break

print("\n" + "=" * 60)
print(f"TRAINING COMPLETE - Best Eval AUC: {best_auc:.4f}")
print(f"  Checkpoint -> {CHECKPOINT}")
print("=" * 60)
