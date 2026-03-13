"""
train_video.py - DeepShield video model training on grouped clip splits.
"""

import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from runtime_env import configure_wandb_environment

configure_wandb_environment()

import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.amp import GradScaler
from torch.utils.data import DataLoader

from data.preprocessing.dataset import VideoDataset, VIDEO_TRAIN_TRANSFORM, VIDEO_VAL_TRANSFORM
from models.video_model import VideoDetector
from train.training_utils import amp_context, build_balanced_sampler, compute_best_f1_threshold, seed_everything


BATCH_SIZE = 4
EPOCHS = int(os.environ.get("DEEPSHIELD_EPOCHS", "30"))
LR = 1e-4
WEIGHT_DECAY = 1e-4
GRAD_ACCUM = 8
MAX_GRAD_NORM = 1.0
PATIENCE = 7
CHECKPOINT = os.environ.get("DEEPSHIELD_CHECKPOINT", "checkpoints/best_video.pt")
SEED = int(os.environ.get("DEEPSHIELD_SEED", "42"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"

os.makedirs("checkpoints", exist_ok=True)
seed_everything(SEED)



print("=" * 60)
print("DEEPSHIELD - Video Model Training")
print(f"  Device : {DEVICE}")
print(f"  Seed   : {SEED}")
print("=" * 60)

train_ds = VideoDataset(split="train", transform=VIDEO_TRAIN_TRANSFORM)
val_ds = VideoDataset(split="val", transform=VIDEO_VAL_TRANSFORM)
train_labels = [int(sample[1]) for sample in train_ds.samples]
train_balance = Counter(train_labels)
train_sampler = build_balanced_sampler(train_labels)

print(f"  Train clips : {len(train_ds)}")
print(f"  Val clips   : {len(val_ds)}")
print(f"  Train balance : {dict(sorted(train_balance.items()))}")
print("  Train sampler : class-balanced")

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    sampler=train_sampler,
    num_workers=2,
    pin_memory=USE_AMP,
)
val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=USE_AMP,
)

model = VideoDetector().to(DEVICE)
print("  Frame init    : default timm weights")
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
warmup_epochs = min(3, EPOCHS)
warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(EPOCHS - warmup_epochs, 1))
scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])
criterion = nn.BCEWithLogitsLoss()
scaler = GradScaler(DEVICE.type, enabled=USE_AMP)

best_auc = 0.0
patience_count = 0

for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss = 0.0
    optimizer.zero_grad()

    for step, (clips, labels) in enumerate(train_loader):
        clips = clips.to(DEVICE)
        labels = labels.to(DEVICE)

        with amp_context(DEVICE):
            _, logits = model(clips)
            loss = criterion(logits.view(-1), labels.view(-1)) / GRAD_ACCUM

        scaler.scale(loss).backward()
        train_loss += loss.item() * GRAD_ACCUM

        if (step + 1) % GRAD_ACCUM == 0 or (step + 1) == len(train_loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

    train_loss /= len(train_loader)

    model.eval()
    val_loss = 0.0
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for clips, labels in val_loader:
            clips = clips.to(DEVICE)
            labels = labels.to(DEVICE)

            with amp_context(DEVICE):
                _, logits = model(clips)
                loss = criterion(logits.view(-1), labels.view(-1))

            val_loss += loss.item()
            probs = torch.sigmoid(logits.view(-1)).cpu().numpy()
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
                "train_samples": len(train_ds),
                "val_samples": len(val_ds),
                "train_class_balance": {str(label): count for label, count in sorted(train_balance.items())},
                "split_policy": "grouped_source_video",
                "train_sampler": "balanced_weighted_random",
                "augmentations": "clip_consistent_crop_flip_color_blur",
                "seed": SEED,
            },
            CHECKPOINT,
        )
        print(f"  Saved best video checkpoint (AUC={best_auc:.4f})")
        patience_count = 0
    else:
        patience_count += 1
        if patience_count >= PATIENCE:
            print(f"  Early stopping at epoch {epoch}")
            break

print("\n" + "=" * 60)
print(f"TRAINING COMPLETE - Best Val AUC: {best_auc:.4f}")
print(f"  Checkpoint -> {CHECKPOINT}")
print("=" * 60)
