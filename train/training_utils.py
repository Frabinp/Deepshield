from __future__ import annotations

from collections import Counter
from contextlib import nullcontext
import random

import numpy as np
import torch
from sklearn.metrics import precision_recall_curve
from torch.amp import autocast
from torch.utils.data import WeightedRandomSampler


def scalarize(values) -> list[float]:
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    values = np.asarray(values)
    if values.ndim == 0:
        return [float(values.item())]
    return values.astype(float).reshape(-1).tolist()


def compute_best_f1_threshold(labels, probs, default=0.5) -> tuple[float, float]:
    labels = np.asarray(labels, dtype=int)
    probs = np.asarray(probs, dtype=float)

    if labels.size == 0 or len(np.unique(labels)) < 2:
        return float(default), 0.0

    precision, recall, thresholds = precision_recall_curve(labels, probs)
    if thresholds.size == 0:
        return float(default), 0.0

    precision = precision[:-1]
    recall = recall[:-1]
    f1_scores = (2 * precision * recall) / np.maximum(precision + recall, 1e-12)
    best_index = int(np.argmax(f1_scores))
    return float(thresholds[best_index]), float(f1_scores[best_index])


def build_balanced_sampler(labels) -> WeightedRandomSampler:
    labels = [int(label) for label in labels]
    counts = Counter(labels)
    if not counts:
        raise ValueError("Cannot build a sampler from an empty label list")

    weights = [1.0 / counts[label] for label in labels]
    return WeightedRandomSampler(
        weights=torch.tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
    )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def amp_context(device: torch.device):
    """Returns an AMP autocast context manager, or a no-op on CPU."""
    if device.type == "cuda":
        return autocast(device_type="cuda", enabled=True)
    return nullcontext()
