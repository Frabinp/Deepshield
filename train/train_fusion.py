"""
train_fusion.py - DeepShield multimodal fusion training.
"""

import json
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

from data.preprocessing.dataset import (
    DeepShieldDataset,
    IMAGE_TRAIN_TRANSFORM,
    IMAGE_VAL_TRANSFORM,
    VIDEO_TRAIN_TRANSFORM,
    VIDEO_VAL_TRANSFORM,
)
from models.audio_model import AudioDetector
from models.checkpoint_utils import load_module_checkpoint
from models.fusion_model import FusionModel
from models.image_model import ImageDetector
from models.video_model import VideoDetector
from train.training_utils import amp_context, build_balanced_sampler, compute_best_f1_threshold, seed_everything


BATCH_SIZE = 4
EPOCHS = int(os.environ.get("DEEPSHIELD_EPOCHS", "25"))
LR = 1e-4
WEIGHT_DECAY = 1e-4
GRAD_ACCUM = 4
MAX_GRAD_NORM = 1.0
PATIENCE = 7
CM_LOSS_WEIGHT = 0.1
CHECKPOINT = os.environ.get("DEEPSHIELD_CHECKPOINT", "checkpoints/best_fusion.pt")
MANIFEST_PATH = "data/processed/manifests/faceforensics_multimodal.json"
IMG_CKPT = "checkpoints/best_image.pt"
VID_CKPT = os.environ.get("DEEPSHIELD_VIDEO_CHECKPOINT", "checkpoints/best_video.pt")
AUD_CKPT = "checkpoints/best_audio.pt"
SEED = int(os.environ.get("DEEPSHIELD_SEED", "42"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"

os.makedirs("checkpoints", exist_ok=True)
seed_everything(SEED)


def choose_active_modalities(manifest_path):
    with open(manifest_path, "r", encoding="utf-8") as handle:
        records = json.load(handle)

    has_visual = any(record.get("has_image") and record.get("has_video") for record in records)
    has_audio_visual = any(
        record.get("has_image") and record.get("has_video") and record.get("has_audio")
        for record in records
    )

    if has_audio_visual:
        return ("image", "video", "audio")
    if has_visual:
        return ("image", "video")
    raise RuntimeError("Manifest does not contain aligned image/video samples.")


ACTIVE_MODALITIES = choose_active_modalities(MANIFEST_PATH)

print("=" * 55)
print("DEEPSHIELD - Multimodal Fusion Training")
print(f"  Device        : {DEVICE}")
print(f"  Manifest      : {MANIFEST_PATH}")
print(f"  Active mods   : {', '.join(ACTIVE_MODALITIES)}")
print(f"  Seed          : {SEED}")
print("=" * 55)

train_ds = DeepShieldDataset(
    mode="train",
    modalities=list(ACTIVE_MODALITIES),
    manifest_path=MANIFEST_PATH,
    image_transform=IMAGE_TRAIN_TRANSFORM,
    video_transform=VIDEO_TRAIN_TRANSFORM,
)
val_ds = DeepShieldDataset(
    mode="val",
    modalities=list(ACTIVE_MODALITIES),
    manifest_path=MANIFEST_PATH,
    image_transform=IMAGE_VAL_TRANSFORM,
    video_transform=VIDEO_VAL_TRANSFORM,
)
train_labels = [int(record["label"]) for record in train_ds.records]
train_balance = Counter(train_labels)
train_sampler = build_balanced_sampler(train_labels)

if len(train_ds) == 0 or len(val_ds) == 0:
    raise RuntimeError(
        "Aligned multimodal dataset is empty. "
        "Build the manifest and FaceForensics spectrograms first."
    )

print(f"  Train samples : {len(train_ds)}")
print(f"  Val samples   : {len(val_ds)}")
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

image_model = ImageDetector().to(DEVICE)
video_model = VideoDetector().to(DEVICE)
audio_model = AudioDetector().to(DEVICE)
load_module_checkpoint(image_model, IMG_CKPT, map_location=DEVICE)
load_module_checkpoint(video_model, VID_CKPT, map_location=DEVICE)
load_module_checkpoint(audio_model, AUD_CKPT, map_location=DEVICE)

for branch_model in [image_model, video_model, audio_model]:
    branch_model.eval()
    for parameter in branch_model.parameters():
        parameter.requires_grad = False

fusion_model = FusionModel().to(DEVICE)

optimizer = torch.optim.AdamW(
    fusion_model.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY,
)
warmup_epochs = min(3, EPOCHS)
warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(EPOCHS - warmup_epochs, 1))
scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])
criterion = nn.BCEWithLogitsLoss()
scaler = GradScaler(DEVICE.type, enabled=USE_AMP)




def extract_features(batch):
    images = batch["image"].to(DEVICE)
    clips = batch["frames"].to(DEVICE)

    with torch.no_grad():
        with amp_context(DEVICE):
            image_feat, _ = image_model(images)
            video_feat, _ = video_model(clips)
            if "audio" in ACTIVE_MODALITIES:
                audio = batch["audio"].to(DEVICE)
                audio_feat, _ = audio_model(audio)
            else:
                audio_feat = torch.zeros(images.size(0), 512, device=DEVICE)
    return image_feat, video_feat, audio_feat


best_auc = 0.0
patience_count = 0

for epoch in range(1, EPOCHS + 1):
    fusion_model.train()
    train_loss = 0.0
    optimizer.zero_grad()

    for step, batch in enumerate(train_loader):
        labels = batch["label"].to(DEVICE)
        image_feat, video_feat, audio_feat = extract_features(batch)

        with amp_context(DEVICE):
            out = fusion_model(image_feat, video_feat, audio_feat, labels=labels)
            logits = out["logit"].view(-1)
            cls_loss = criterion(logits, labels.view(-1))
            total_loss = cls_loss + CM_LOSS_WEIGHT * out["cross_modal_loss"]
            loss = total_loss / GRAD_ACCUM

        scaler.scale(loss).backward()
        train_loss += total_loss.item()

        if (step + 1) % GRAD_ACCUM == 0 or (step + 1) == len(train_loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(fusion_model.parameters(), MAX_GRAD_NORM)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

    train_loss /= len(train_loader)

    fusion_model.eval()
    val_loss = 0.0
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            labels = batch["label"].to(DEVICE)
            image_feat, video_feat, audio_feat = extract_features(batch)

            with amp_context(DEVICE):
                out = fusion_model(image_feat, video_feat, audio_feat, labels=labels)
                logits = out["logit"].view(-1)
                cls_loss = criterion(logits, labels.view(-1))
                total_loss = cls_loss + CM_LOSS_WEIGHT * out["cross_modal_loss"]

            val_loss += total_loss.item()
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs.tolist())
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
                "fusion_model_state": fusion_model.state_dict(),
                "best_auc": best_auc,
                "best_threshold": threshold,
                "best_threshold_metric": "best_f1",
                "best_threshold_f1": threshold_f1,
                "active_modalities": list(ACTIVE_MODALITIES),
                "manifest_path": MANIFEST_PATH,
                "image_checkpoint": IMG_CKPT,
                "video_checkpoint": VID_CKPT,
                "audio_checkpoint": AUD_CKPT,
                "cm_loss_weight": CM_LOSS_WEIGHT,
                "branches_frozen": True,
                "train_class_balance": {str(label): count for label, count in sorted(train_balance.items())},
                "train_sampler": "balanced_weighted_random",
                "augmentations": {
                    "image": "random_resized_crop_color_affine_blur_erasing",
                    "video": "clip_consistent_crop_flip_color_blur",
                },
                "seed": SEED,
            },
            CHECKPOINT,
        )
        print(f"  Saved best fusion checkpoint (AUC={best_auc:.4f})")
        patience_count = 0
    else:
        patience_count += 1
        if patience_count >= PATIENCE:
            print(f"  Early stopping at epoch {epoch}")
            break

print("\n" + "=" * 55)
print(f"TRAINING COMPLETE - Best Val AUC: {best_auc:.4f}")
print(f"  Checkpoint -> {CHECKPOINT}")
print("=" * 55)
