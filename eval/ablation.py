import json
import os
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from runtime_env import configure_wandb_environment

configure_wandb_environment()

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from data.preprocessing.dataset import DeepShieldDataset
from models.audio_model import AudioDetector
from models.checkpoint_utils import (
    get_active_modalities,
    load_fusion_models,
    load_module_checkpoint,
    mask_fusion_features,
)
from models.fusion_model import FusionModel
from models.image_model import ImageDetector
from models.video_model import VideoDetector


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = Path("checkpoints")
RESULTS_DIR = Path("eval/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_all_models():
    image_model = ImageDetector(pretrained=False).to(DEVICE)
    fusion_image_model = ImageDetector(pretrained=False).to(DEVICE)
    video_model = VideoDetector(pretrained=False).to(DEVICE)
    audio_model = AudioDetector(pretrained=False).to(DEVICE)
    fusion_model = FusionModel().to(DEVICE)

    load_module_checkpoint(image_model, str(CHECKPOINT_DIR / "best_image.pt"), map_location=DEVICE)
    print(" Image loaded")
    load_module_checkpoint(video_model, str(CHECKPOINT_DIR / "best_video.pt"), map_location=DEVICE)
    print(" Video loaded")
    load_module_checkpoint(audio_model, str(CHECKPOINT_DIR / "best_audio.pt"), map_location=DEVICE)
    print(" Audio loaded")

    fusion_checkpoint = load_fusion_models(
        fusion_image_model,
        fusion_model,
        str(CHECKPOINT_DIR / "best_fusion.pt"),
        image_checkpoint_path=str(CHECKPOINT_DIR / "best_image.pt"),
        map_location=DEVICE,
    )
    print(" Fusion loaded")

    active_modalities = get_active_modalities(fusion_checkpoint, default=("image",))

    for model in [image_model, fusion_image_model, video_model, audio_model, fusion_model]:
        model.eval()

    return image_model, fusion_image_model, video_model, audio_model, fusion_model, active_modalities


@torch.no_grad()
def ablation_single_vs_fused(
    image_model,
    fusion_image_model,
    video_model,
    audio_model,
    fusion_model,
    loader,
    fusion_active_modalities,
):
    print("\n-- Ablation 1: Single Modality vs Fusion --")

    results = {
        "image_only": {"probs": [], "labels": []},
        "video_only": {"probs": [], "labels": []},
        "audio_only": {"probs": [], "labels": []},
        "fusion_all": {"probs": [], "labels": []},
    }

    for batch in loader:
        labels = batch["label"].float()
        img_tensor = batch["image"].to(DEVICE)
        vid_tensor = batch["frames"].to(DEVICE)
        aud_tensor = batch["audio"].to(DEVICE)

        img_feat, img_logit = image_model(img_tensor)
        vid_feat, vid_logit = video_model(vid_tensor)
        aud_feat, aud_logit = audio_model(aud_tensor)

        results["image_only"]["probs"].extend(torch.sigmoid(img_logit.squeeze(1)).cpu().numpy())
        results["video_only"]["probs"].extend(torch.sigmoid(vid_logit.squeeze(1)).cpu().numpy())
        results["audio_only"]["probs"].extend(torch.sigmoid(aud_logit.squeeze(1)).cpu().numpy())

        fusion_img_feat, _ = fusion_image_model(img_tensor)
        fusion_img_feat, fusion_vid_feat, fusion_aud_feat = mask_fusion_features(
            fusion_img_feat,
            vid_feat,
            aud_feat,
            fusion_active_modalities,
        )
        out = fusion_model(fusion_img_feat, fusion_vid_feat, fusion_aud_feat)
        results["fusion_all"]["probs"].extend(out["probability"].squeeze(1).cpu().numpy())

        label_list = labels.numpy().tolist()
        for result in results.values():
            result["labels"].extend(label_list)

    auc_results = {}
    for name, result in results.items():
        auc = roc_auc_score(result["labels"], result["probs"])
        auc_results[name] = round(auc, 4)
        print(f"  {name:<15}: AUC = {auc:.4f}")

    return auc_results


@torch.no_grad()
def ablation_cross_modal(
    fusion_image_model,
    video_model,
    audio_model,
    fusion_model,
    loader,
    fusion_active_modalities,
):
    print("\n-- Ablation 2: Cross-Modal Consistency Loss --")

    if not {"video", "audio"}.issubset(set(fusion_active_modalities)):
        message = "fusion checkpoint does not use both video and audio modalities"
        print(f"  Skipped: {message}.")
        return {"skipped": message}

    probs_with = []
    labels_all = []

    for batch in loader:
        labels = batch["label"].float()
        img_feat, _ = fusion_image_model(batch["image"].to(DEVICE))
        vid_feat, _ = video_model(batch["frames"].to(DEVICE))
        aud_feat, _ = audio_model(batch["audio"].to(DEVICE))

        img_feat, vid_feat, aud_feat = mask_fusion_features(
            img_feat,
            vid_feat,
            aud_feat,
            fusion_active_modalities,
        )
        out = fusion_model(img_feat, vid_feat, aud_feat)
        probs_with.extend(out["probability"].squeeze(1).cpu().numpy())
        labels_all.extend(labels.numpy().tolist())

    auc_with = roc_auc_score(labels_all, probs_with)

    probs_without = []
    for batch in loader:
        img_feat, _ = fusion_image_model(batch["image"].to(DEVICE))
        vid_feat, _ = video_model(batch["frames"].to(DEVICE))
        aud_feat_zero = torch.zeros(batch["image"].shape[0], 512, device=DEVICE)
        img_feat, vid_feat, aud_feat_zero = mask_fusion_features(
            img_feat,
            vid_feat,
            aud_feat_zero,
            fusion_active_modalities,
        )
        out = fusion_model(img_feat, vid_feat, aud_feat_zero)
        probs_without.extend(out["probability"].squeeze(1).cpu().numpy())

    auc_without = roc_auc_score(labels_all, probs_without)

    print(f"  With cross-modal loss   : AUC = {auc_with:.4f}")
    print(f"  Without cross-modal loss: AUC = {auc_without:.4f}")
    print(f"  Improvement             : +{(auc_with - auc_without):.4f}")

    return {
        "with_cm_loss": round(auc_with, 4),
        "without_cm_loss": round(auc_without, 4),
        "improvement": round(auc_with - auc_without, 4),
    }


@torch.no_grad()
def ablation_robustness(image_model, loader):
    print("\n-- Ablation 3: Robustness to Noise --")

    noise_levels = [0.0, 0.01, 0.05, 0.1, 0.2]
    auc_results = {}

    for noise in noise_levels:
        probs = []
        labels_all = []

        for batch in loader:
            labels = batch["label"].float()
            images = batch["image"].to(DEVICE)

            if noise > 0:
                images = images + torch.randn_like(images) * noise
                images = torch.clamp(images, -3, 3)

            _, logit = image_model(images)
            probs.extend(torch.sigmoid(logit.squeeze(1)).cpu().numpy())
            labels_all.extend(labels.numpy().tolist())

        auc = roc_auc_score(labels_all, probs)
        auc_results[f"noise_{noise}"] = round(auc, 4)
        print(f"  Noise sigma={noise:.2f}: AUC = {auc:.4f}")

    return auc_results


def plot_ablation_1(auc_results):
    names = ["Image\nOnly", "Video\nOnly", "Audio\nOnly", "All\nFused"]
    keys = ["image_only", "video_only", "audio_only", "fusion_all"]
    values = [auc_results[key] for key in keys]
    colors = ["#7c3aed", "#0891b2", "#16a34a", "#dc2626"]

    plt.figure(figsize=(8, 6))
    bars = plt.bar(names, values, color=colors, width=0.5, edgecolor="white")

    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=12,
        )

    plt.ylim(0.5, 1.0)
    plt.ylabel("AUC Score", fontsize=13)
    plt.title("Ablation: Single Modality vs Fusion", fontsize=14)
    plt.axhline(y=0.9, color="gray", linestyle="--", alpha=0.5, label="Target 90%")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "ablation_single_vs_fused.png", dpi=150)
    plt.close()
    print(" Ablation plot saved")


def plot_robustness(auc_results):
    noise_levels = [float(key.split("_")[1]) for key in auc_results]
    auc_values = list(auc_results.values())

    plt.figure(figsize=(8, 5))
    plt.plot(noise_levels, auc_values, "o-", color="#dc2626", linewidth=2.5, markersize=8)
    plt.xlabel("Gaussian Noise sigma", fontsize=13)
    plt.ylabel("AUC Score", fontsize=13)
    plt.title("Robustness to Input Noise", fontsize=14)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "ablation_robustness.png", dpi=150)
    plt.close()
    print(" Robustness plot saved")


def main():
    print("=" * 55)
    print("DEEPSHIELD - Ablation Studies")
    print("=" * 55)

    (
        image_model,
        fusion_image_model,
        video_model,
        audio_model,
        fusion_model,
        fusion_active_modalities,
    ) = load_all_models()

    test_ds = DeepShieldDataset(mode="test", modalities=["image", "video", "audio"])
    if len(test_ds) == 0:
        print(" No test data found. Run preprocessing first.")
        return

    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=0)
    print(f"Running ablations on {len(test_ds)} test samples...")

    abl1 = ablation_single_vs_fused(
        image_model,
        fusion_image_model,
        video_model,
        audio_model,
        fusion_model,
        test_loader,
        fusion_active_modalities,
    )
    abl2 = ablation_cross_modal(
        fusion_image_model,
        video_model,
        audio_model,
        fusion_model,
        test_loader,
        fusion_active_modalities,
    )
    abl3 = ablation_robustness(image_model, test_loader)

    plot_ablation_1(abl1)
    plot_robustness(abl3)

    all_results = {
        "single_vs_fused": abl1,
        "cross_modal_loss": abl2,
        "robustness": abl3,
    }
    with open(RESULTS_DIR / "ablation_results.json", "w", encoding="utf-8") as handle:
        json.dump(all_results, handle, indent=2)

    print("\n" + "=" * 55)
    print(" All ablation studies complete!")
    print(f"   Results saved to {RESULTS_DIR}/")
    print("=" * 55)


if __name__ == "__main__":
    main()
