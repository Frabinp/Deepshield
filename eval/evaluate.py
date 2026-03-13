"""
evaluate.py - DeepShield evaluation with richer metrics and saved reports.
"""

import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from runtime_env import configure_wandb_environment

configure_wandb_environment()

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from data.preprocessing.dataset import (
    AudioDataset,
    DeepShieldDataset,
    IMAGE_VAL_TRANSFORM,
    ImageDataset,
)
from data.preprocessing.faceforensics_ids import processed_sample_id_from_stem
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
FACES_DIR = Path("data/processed/faces")
RESULTS_DIR = Path("eval/results")
MANIFEST_PATH = Path("data/processed/manifests/faceforensics_multimodal.json")
AUDIO_ROOT = Path("data/processed/spectrograms_asvspoof")
N_FRAMES = 16

TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


class FaceDataset(Dataset):
    def __init__(self, split="val", val_ratio=0.2, seed=42):
        real = sorted((FACES_DIR / "real").glob("*.jpg"))
        fake = sorted((FACES_DIR / "fake").glob("*.jpg"))
        samples = [(path, 0) for path in real] + [(path, 1) for path in fake]

        # Use grouped splits to prevent data leakage:
        # face crops from the same source video stay in the same split
        from data.preprocessing.dataset import _split_samples

        self.samples = _split_samples(
            samples,
            split=split,
            val_ratio=val_ratio,
            seed=seed,
            group_key_fn=lambda s: f"{s[1]}:{processed_sample_id_from_stem(s[0].stem)}",
            stratify_fn=lambda s: s[1],
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        image = TRANSFORM(Image.open(path).convert("RGB"))
        return image, torch.tensor(label, dtype=torch.float32)


class FaceClipDataset(Dataset):
    def __init__(self, split="val", val_ratio=0.2, seed=42):
        def collect(directory, label):
            files = sorted(Path(directory).glob("*.jpg"))
            groups = {}
            for file_path in files:
                groups.setdefault(processed_sample_id_from_stem(file_path.stem), []).append(file_path)
            return [(sorted(paths), label) for paths in groups.values() if len(paths) >= 3]

        samples = collect(FACES_DIR / "real", 0) + collect(FACES_DIR / "fake", 1)
        rng = np.random.default_rng(seed)
        indices = rng.permutation(len(samples))
        val_count = int(len(samples) * val_ratio)
        indices = indices[val_count:] if split == "train" else indices[:val_count]
        self.samples = [samples[index] for index in indices]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        paths, label = self.samples[index]
        if len(paths) >= N_FRAMES:
            chosen = paths[:N_FRAMES]
        else:
            repeats = (N_FRAMES // len(paths)) + 1
            chosen = (paths * repeats)[:N_FRAMES]

        clip = torch.stack([TRANSFORM(Image.open(path).convert("RGB")) for path in chosen])
        image = TRANSFORM(Image.open(paths[0]).convert("RGB"))
        return image, clip, torch.tensor(label, dtype=torch.float32)


def compute_eer(labels, scores):
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    if len(np.unique(labels)) < 2:
        return None

    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1 - tpr
    index = int(np.argmin(np.abs(fpr - fnr)))
    return float((fpr[index] + fnr[index]) / 2)


from train.training_utils import scalarize


def metric_or_none(function, labels, probs):
    labels = np.asarray(labels)
    if len(np.unique(labels)) < 2:
        return None
    return float(function(labels, probs))


def summarize_at_threshold(labels, probs, threshold):
    labels = np.asarray(labels, dtype=int)
    probs = np.asarray(probs, dtype=float)
    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    precision = float(precision_score(labels, preds, zero_division=0))
    recall = float(recall_score(labels, preds, zero_division=0))
    f1 = float((2 * precision * recall) / max(precision + recall, 1e-12))

    return {
        "threshold": float(threshold),
        "accuracy": float((tp + tn) / max(len(labels), 1)),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }


def best_f1_threshold(labels, probs):
    labels = np.asarray(labels, dtype=int)
    probs = np.asarray(probs, dtype=float)

    if len(np.unique(labels)) < 2:
        return 0.5, summarize_at_threshold(labels, probs, 0.5)

    precision, recall, thresholds = precision_recall_curve(labels, probs)
    if len(thresholds) == 0:
        return 0.5, summarize_at_threshold(labels, probs, 0.5)

    precision = precision[:-1]
    recall = recall[:-1]
    f1 = (2 * precision * recall) / np.maximum(precision + recall, 1e-12)
    best_index = int(np.argmax(f1))
    threshold = float(thresholds[best_index])
    summary = summarize_at_threshold(labels, probs, threshold)
    summary["f1"] = float(f1[best_index])
    return threshold, summary


def evaluate_predictions(name, probs, labels):
    labels = np.asarray(labels, dtype=int)
    probs = np.asarray(probs, dtype=float)

    threshold_050 = summarize_at_threshold(labels, probs, 0.5)
    best_threshold, best_threshold_metrics = best_f1_threshold(labels, probs)

    metrics = {
        "model": name,
        "samples": int(len(labels)),
        "positives": int(labels.sum()),
        "negatives": int(len(labels) - labels.sum()),
        "auc": metric_or_none(roc_auc_score, labels, probs),
        "pr_auc": metric_or_none(average_precision_score, labels, probs),
        "eer": compute_eer(labels, probs),
        "threshold_0_50": threshold_050,
        "best_f1_threshold": float(best_threshold),
        "best_f1": best_threshold_metrics,
    }

    print(f"\n{'=' * 60}")
    print(f"{name}")
    print(f"{'=' * 60}")
    print(f"Samples            : {metrics['samples']}")
    print(f"Positives / Negs   : {metrics['positives']} / {metrics['negatives']}")
    print(f"AUC                : {metrics['auc'] if metrics['auc'] is not None else 'n/a'}")
    print(f"PR-AUC             : {metrics['pr_auc'] if metrics['pr_auc'] is not None else 'n/a'}")
    print(f"EER                : {metrics['eer'] if metrics['eer'] is not None else 'n/a'}")
    print(
        "Threshold 0.50     : "
        f"acc={threshold_050['accuracy']:.4f}  "
        f"prec={threshold_050['precision']:.4f}  "
        f"rec={threshold_050['recall']:.4f}  "
        f"f1={threshold_050['f1']:.4f}"
    )
    print(
        "Best F1 threshold  : "
        f"{best_threshold_metrics['threshold']:.4f}  "
        f"acc={best_threshold_metrics['accuracy']:.4f}  "
        f"prec={best_threshold_metrics['precision']:.4f}  "
        f"rec={best_threshold_metrics['recall']:.4f}  "
        f"f1={best_threshold_metrics['f1']:.4f}"
    )
    print(
        "Confusion @0.50    : "
        f"TP={threshold_050['confusion_matrix']['tp']}  "
        f"TN={threshold_050['confusion_matrix']['tn']}  "
        f"FP={threshold_050['confusion_matrix']['fp']}  "
        f"FN={threshold_050['confusion_matrix']['fn']}"
    )

    return metrics


def save_results(results, prediction_rows):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "device": str(DEVICE),
        "manifest_path": str(MANIFEST_PATH),
        "results": results,
    }

    json_paths = [
        RESULTS_DIR / f"evaluation_{timestamp}.json",
        RESULTS_DIR / "latest_evaluation.json",
    ]
    for path in json_paths:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    csv_paths = [
        RESULTS_DIR / f"evaluation_{timestamp}_predictions.csv",
        RESULTS_DIR / "latest_predictions.csv",
    ]
    for path in csv_paths:
        with open(path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["model", "label", "probability"])
            writer.writeheader()
            writer.writerows(prediction_rows)

    return json_paths[0], csv_paths[0]


def run_image_eval():
    dataset = ImageDataset(split="val", transform=IMAGE_VAL_TRANSFORM)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    model = ImageDetector().to(DEVICE)
    load_module_checkpoint(model, "checkpoints/best_image.pt", map_location=DEVICE)
    model.eval()

    probs = []
    labels = []
    with torch.no_grad():
        for images, batch_labels in loader:
            _, logits = model(images.to(DEVICE))
            probs.extend(scalarize(torch.sigmoid(logits.squeeze())))
            labels.extend(scalarize(batch_labels))

    return probs, [int(label) for label in labels]


def run_audio_eval():
    if AUDIO_ROOT.exists():
        dataset = AudioDataset(root_dir=str(AUDIO_ROOT), split="eval")
        split_name = "eval"
    else:
        dataset = AudioDataset(split="val")
        split_name = "val"
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    model = AudioDetector().to(DEVICE)
    load_module_checkpoint(model, "checkpoints/best_audio.pt", map_location=DEVICE)
    model.eval()

    probs = []
    labels = []
    with torch.no_grad():
        for mels, batch_labels in loader:
            _, logits = model(mels.to(DEVICE))
            probs.extend(scalarize(torch.sigmoid(logits.squeeze())))
            labels.extend(scalarize(batch_labels))

    return probs, [int(label) for label in labels], model, split_name


def build_multimodal_loader(active_modalities):
    modalities = sorted(set(active_modalities) | {"image", "video"})
    if not MANIFEST_PATH.exists():
        return None

    dataset = DeepShieldDataset(
        mode="val",
        modalities=modalities,
        manifest_path=str(MANIFEST_PATH),
        image_transform=IMAGE_VAL_TRANSFORM,
        video_transform=IMAGE_VAL_TRANSFORM,
    )
    if len(dataset) == 0:
        return None

    return DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)


def run_video_and_fusion_eval(audio_model):
    video_model = VideoDetector().to(DEVICE)
    load_module_checkpoint(video_model, "checkpoints/best_video.pt", map_location=DEVICE)
    video_model.eval()

    fusion_image_model = ImageDetector().to(DEVICE)
    fusion_model = FusionModel().to(DEVICE)
    fusion_checkpoint = load_fusion_models(
        fusion_image_model,
        fusion_model,
        "checkpoints/best_fusion.pt",
        image_checkpoint_path="checkpoints/best_image.pt",
        map_location=DEVICE,
    )
    fusion_image_model.eval()
    fusion_model.eval()
    fusion_active_modalities = get_active_modalities(fusion_checkpoint, default=("image",))

    multimodal_loader = build_multimodal_loader(fusion_active_modalities)
    fallback_loader = None
    if multimodal_loader is None:
        fallback_loader = DataLoader(FaceClipDataset(split="val"), batch_size=4, shuffle=False, num_workers=0)

    video_probs = []
    fusion_probs = []
    labels = []

    with torch.no_grad():
        if multimodal_loader is not None:
            for batch in multimodal_loader:
                images = batch["image"].to(DEVICE)
                clips = batch["frames"].to(DEVICE)
                clips_flip = torch.flip(clips, dims=[4])
                batch_labels = batch["label"]

                image_features, _ = fusion_image_model(images)
                video_features, video_logits = video_model(clips)
                _, video_logits_flip = video_model(clips_flip)

                if "audio" in batch:
                    mels = batch["audio"].to(DEVICE)
                    audio_features, _ = audio_model(mels)
                else:
                    audio_features = torch.zeros(images.size(0), 512, device=DEVICE)

                image_features, video_features, audio_features = mask_fusion_features(
                    image_features,
                    video_features,
                    audio_features,
                    fusion_active_modalities,
                )
                fusion_out = fusion_model(image_features, video_features, audio_features)

                video_prob = 0.5 * (
                    torch.sigmoid(video_logits.view(-1))
                    + torch.sigmoid(video_logits_flip.view(-1))
                )
                video_probs.extend(scalarize(video_prob))
                fusion_probs.extend(scalarize(torch.sigmoid(fusion_out["logit"].squeeze())))
                labels.extend(int(label) for label in scalarize(batch_labels))
        else:
            for images, clips, batch_labels in fallback_loader:
                images = images.to(DEVICE)
                clips = clips.to(DEVICE)
                clips_flip = torch.flip(clips, dims=[4])

                image_features, _ = fusion_image_model(images)
                video_features, video_logits = video_model(clips)
                _, video_logits_flip = video_model(clips_flip)
                audio_features = torch.zeros(images.size(0), 512, device=DEVICE)
                image_features, video_features, audio_features = mask_fusion_features(
                    image_features,
                    video_features,
                    audio_features,
                    fusion_active_modalities,
                )
                fusion_out = fusion_model(image_features, video_features, audio_features)

                video_prob = 0.5 * (
                    torch.sigmoid(video_logits.view(-1))
                    + torch.sigmoid(video_logits_flip.view(-1))
                )
                video_probs.extend(scalarize(video_prob))
                fusion_probs.extend(scalarize(torch.sigmoid(fusion_out["logit"].squeeze())))
                labels.extend(int(label) for label in scalarize(batch_labels))

    return video_probs, fusion_probs, labels, fusion_active_modalities


def build_prediction_rows(model_name, labels, probs):
    return [
        {
            "model": model_name,
            "label": int(label),
            "probability": float(prob),
        }
        for label, prob in zip(labels, probs)
    ]


def main():
    print("=" * 60)
    print("DEEPSHIELD EVALUATION")
    print(f"Device             : {DEVICE}")
    print(f"Faces directory    : {FACES_DIR}")
    print(f"Manifest available : {MANIFEST_PATH.exists()}")
    print("=" * 60)

    results = {}
    prediction_rows = []

    print("\n[1/3] Evaluating image model")
    image_probs, image_labels = run_image_eval()
    results["image"] = evaluate_predictions("IMAGE MODEL", image_probs, image_labels)
    prediction_rows.extend(build_prediction_rows("image", image_labels, image_probs))

    print("\n[2/3] Evaluating audio model")
    audio_probs, audio_labels, audio_model, audio_split = run_audio_eval()
    results["audio"] = evaluate_predictions("AUDIO MODEL", audio_probs, audio_labels)
    results["audio"]["dataset_split"] = audio_split
    prediction_rows.extend(build_prediction_rows("audio", audio_labels, audio_probs))

    print("\n[3/3] Evaluating video and fusion models")
    video_probs, fusion_probs, fusion_labels, active_modalities = run_video_and_fusion_eval(audio_model)
    results["video"] = evaluate_predictions("VIDEO MODEL", video_probs, fusion_labels)
    results["fusion"] = evaluate_predictions("FUSION MODEL", fusion_probs, fusion_labels)
    results["fusion"]["active_modalities"] = list(active_modalities)
    prediction_rows.extend(build_prediction_rows("video", fusion_labels, video_probs))
    prediction_rows.extend(build_prediction_rows("fusion", fusion_labels, fusion_probs))

    report_path, predictions_path = save_results(results, prediction_rows)

    print(f"\nSaved report       : {report_path}")
    print(f"Saved predictions  : {predictions_path}")
    print("=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    for key in ["image", "audio", "video", "fusion"]:
        model_result = results[key]
        auc = model_result["auc"]
        pr_auc = model_result["pr_auc"]
        f1 = model_result["threshold_0_50"]["f1"]
        print(
            f"{key.title():<10} "
            f"AUC={auc if auc is not None else 'n/a'}  "
            f"PR-AUC={pr_auc if pr_auc is not None else 'n/a'}  "
            f"F1@0.50={f1:.4f}"
        )


if __name__ == "__main__":
    main()
