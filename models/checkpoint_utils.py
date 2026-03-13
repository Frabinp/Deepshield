from __future__ import annotations

from typing import Iterable, Sequence

import torch


def _is_state_dict(candidate: object) -> bool:
    if not isinstance(candidate, dict) or not candidate:
        return False
    return all(torch.is_tensor(value) for value in candidate.values())


def extract_state_dict(checkpoint: object, state_key: str | None = None) -> dict:
    if state_key is not None:
        if not isinstance(checkpoint, dict) or state_key not in checkpoint:
            raise KeyError(f"Checkpoint does not contain '{state_key}'")
        state_dict = checkpoint[state_key]
        if not _is_state_dict(state_dict):
            raise TypeError(f"Checkpoint entry '{state_key}' is not a state dict")
        return state_dict

    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        state_dict = checkpoint["model_state"]
        if _is_state_dict(state_dict):
            return state_dict

    if _is_state_dict(checkpoint):
        return checkpoint

    raise TypeError("Unsupported checkpoint format")


def load_module_checkpoint(
    module: torch.nn.Module,
    checkpoint_path: str,
    *,
    map_location: str | torch.device = "cpu",
    state_key: str | None = None,
    strict: bool = True,
) -> object:
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    module.load_state_dict(extract_state_dict(checkpoint, state_key=state_key), strict=strict)
    return checkpoint


def load_fusion_models(
    image_model: torch.nn.Module,
    fusion_model: torch.nn.Module,
    fusion_checkpoint_path: str,
    *,
    image_checkpoint_path: str,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> object:
    checkpoint = torch.load(fusion_checkpoint_path, map_location=map_location)

    if isinstance(checkpoint, dict) and "fusion_model_state" in checkpoint:
        fusion_model.load_state_dict(
            extract_state_dict(checkpoint, state_key="fusion_model_state"),
            strict=strict,
        )
        if "image_model_state" in checkpoint:
            image_model.load_state_dict(
                extract_state_dict(checkpoint, state_key="image_model_state"),
                strict=strict,
            )
        else:
            load_module_checkpoint(
                image_model,
                image_checkpoint_path,
                map_location=map_location,
                strict=strict,
            )
        return checkpoint

    fusion_model.load_state_dict(extract_state_dict(checkpoint), strict=strict)
    load_module_checkpoint(
        image_model,
        image_checkpoint_path,
        map_location=map_location,
        strict=strict,
    )
    return checkpoint


def get_active_modalities(
    checkpoint: object,
    default: Sequence[str] = ("image",),
) -> tuple[str, ...]:
    if isinstance(checkpoint, dict):
        active_modalities = checkpoint.get("active_modalities")
        if isinstance(active_modalities, Iterable) and not isinstance(active_modalities, (str, bytes)):
            return tuple(str(modality) for modality in active_modalities)
    return tuple(default)


def get_decision_threshold(checkpoint: object, default: float = 0.5) -> float:
    if isinstance(checkpoint, dict):
        for key in ("decision_threshold", "best_threshold"):
            value = checkpoint.get(key)
            if isinstance(value, (int, float)):
                return float(value)
    return float(default)


def mask_fusion_features(
    image_feat: torch.Tensor,
    video_feat: torch.Tensor,
    audio_feat: torch.Tensor,
    active_modalities: Sequence[str],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    active = set(active_modalities)

    if "image" not in active:
        image_feat = torch.zeros_like(image_feat)
    if "video" not in active:
        video_feat = torch.zeros_like(video_feat)
    if "audio" not in active:
        audio_feat = torch.zeros_like(audio_feat)

    return image_feat, video_feat, audio_feat
