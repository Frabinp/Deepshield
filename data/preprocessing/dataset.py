"""
dataset.py - DeepShield dataset classes.
"""

import json
import random
import numpy as np
import torch
from PIL import Image, ImageFilter
from pathlib import Path
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode, functional as F

from data.preprocessing.faceforensics_ids import processed_sample_id_from_stem


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


IMAGE_TRAIN_TRANSFORM = transforms.Compose(
    [
        transforms.RandomResizedCrop(224, scale=(0.82, 1.0), ratio=(0.92, 1.08)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply(
            [
                transforms.ColorJitter(
                    brightness=0.25,
                    contrast=0.25,
                    saturation=0.2,
                    hue=0.03,
                )
            ],
            p=0.7,
        ),
        transforms.RandomApply(
            [
                transforms.RandomAffine(
                    degrees=8,
                    translate=(0.04, 0.04),
                    scale=(0.95, 1.05),
                    interpolation=InterpolationMode.BILINEAR,
                )
            ],
            p=0.35,
        ),
        transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.2))],
            p=0.2,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        transforms.RandomErasing(p=0.15, scale=(0.01, 0.07), ratio=(0.4, 2.5)),
    ]
)

IMAGE_VAL_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)


class ConsistentVideoTransform:
    """
    Applies the same spatial/color augmentation to every frame in a clip.
    """

    def __init__(self, train=True, size=(224, 224)):
        self.train = train
        self.size = size

    def _sample_params(self, image):
        if not self.train:
            return {"crop": None, "flip": False, "jitter": None, "blur_radius": None}

        crop = transforms.RandomResizedCrop.get_params(
            image,
            scale=(0.82, 1.0),
            ratio=(0.92, 1.08),
        )
        flip = random.random() < 0.5
        jitter = None
        if random.random() < 0.65:
            jitter = {
                "brightness": random.uniform(0.82, 1.18),
                "contrast": random.uniform(0.82, 1.18),
                "saturation": random.uniform(0.88, 1.12),
                "hue": random.uniform(-0.03, 0.03),
            }
        blur_radius = random.uniform(0.15, 1.0) if random.random() < 0.15 else None
        return {
            "crop": crop,
            "flip": flip,
            "jitter": jitter,
            "blur_radius": blur_radius,
        }

    def _apply(self, image, params):
        if params["crop"] is not None:
            top, left, height, width = params["crop"]
            image = F.resized_crop(
                image,
                top,
                left,
                height,
                width,
                self.size,
                interpolation=InterpolationMode.BILINEAR,
            )
            if params["flip"]:
                image = F.hflip(image)
            if params["jitter"] is not None:
                image = F.adjust_brightness(image, params["jitter"]["brightness"])
                image = F.adjust_contrast(image, params["jitter"]["contrast"])
                image = F.adjust_saturation(image, params["jitter"]["saturation"])
                image = F.adjust_hue(image, params["jitter"]["hue"])
            if params["blur_radius"] is not None:
                image = image.filter(ImageFilter.GaussianBlur(radius=params["blur_radius"]))
        else:
            image = F.resize(
                image,
                self.size,
                interpolation=InterpolationMode.BILINEAR,
            )

        tensor = F.to_tensor(image)
        return F.normalize(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD)

    def __call__(self, image):
        return self._apply(image, self._sample_params(image))

    def apply_clip(self, images):
        params = self._sample_params(images[0])
        return [self._apply(image, params) for image in images]


VIDEO_TRAIN_TRANSFORM = ConsistentVideoTransform(train=True)
VIDEO_VAL_TRANSFORM = ConsistentVideoTransform(train=False)


def _load_clip_frames(frame_paths, transform):
    images = []
    for frame_path in frame_paths:
        with Image.open(frame_path) as image:
            images.append(image.convert("RGB"))
    if hasattr(transform, "apply_clip"):
        return transform.apply_clip(images)
    return [transform(image) for image in images]


def _select_frame_paths(frame_paths, n_frames, randomize=False):
    if len(frame_paths) <= n_frames:
        return frame_paths + [frame_paths[-1]] * (n_frames - len(frame_paths))

    if randomize:
        max_start = len(frame_paths) - n_frames
        start = random.randint(0, max_start)
        return frame_paths[start : start + n_frames]
    return frame_paths[:n_frames]


def _normalize_split(split):
    return "val" if split in {"val", "test"} else "train"


def _resolve_audio_splits(split, has_official_splits):
    if not has_official_splits:
        return [_normalize_split(split)]

    resolved = []
    for item in split.split("+"):
        part = item.strip()
        if part == "train":
            resolved.append("train")
        elif part in {"val", "dev"}:
            resolved.append("dev")
        elif part in {"test", "eval"}:
            resolved.append("eval")
        else:
            raise ValueError(f"Unsupported audio split: {split}")

    ordered = []
    for part in resolved:
        if part not in ordered:
            ordered.append(part)
    return ordered


def _split_samples(samples, split, val_ratio, seed, group_key_fn=None, stratify_fn=None):
    normalized_split = _normalize_split(split)
    if not samples:
        return []

    if stratify_fn is not None:
        stratified_samples = {}
        for sample in samples:
            stratified_samples.setdefault(stratify_fn(sample), []).append(sample)

        selected_samples = []
        for key, partition in stratified_samples.items():
            selected_samples.extend(
                _split_samples(
                    partition,
                    split=normalized_split,
                    val_ratio=val_ratio,
                    seed=seed,
                    group_key_fn=group_key_fn,
                    stratify_fn=None,
                )
            )
        return selected_samples

    rng = np.random.default_rng(seed)

    if group_key_fn is None:
        indices = rng.permutation(len(samples))
        val_count = int(len(samples) * val_ratio)
        if normalized_split == "train":
            indices = indices[val_count:]
        else:
            indices = indices[:val_count]
        return [samples[index] for index in indices]

    grouped_samples = {}
    for sample in samples:
        grouped_samples.setdefault(group_key_fn(sample), []).append(sample)

    group_keys = list(grouped_samples)
    indices = rng.permutation(len(group_keys))
    val_count = int(len(group_keys) * val_ratio)

    if normalized_split == "train":
        selected_indices = indices[val_count:]
    else:
        selected_indices = indices[:val_count]

    selected_samples = []
    for index in selected_indices:
        selected_samples.extend(grouped_samples[group_keys[index]])
    return selected_samples


class ImageDataset(Dataset):
    """
    Loads face crops from data/processed/faces/{real,fake}.
    label: 0 = real, 1 = fake
    """

    def __init__(
        self,
        root_dir="data/processed/faces",
        split="train",
        val_ratio=0.2,
        transform=None,
        seed=42,
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform

        def collect(label_name, label_value):
            return [
                (
                    path,
                    label_value,
                    f"{label_name}:{processed_sample_id_from_stem(path.stem)}",
                )
                for path in sorted((self.root_dir / label_name).glob("*.jpg"))
            ]

        samples = collect("real", 0) + collect("fake", 1)
        self.samples = _split_samples(
            samples,
            split=split,
            val_ratio=val_ratio,
            seed=seed,
            group_key_fn=lambda sample: sample[2],
            stratify_fn=lambda sample: sample[1],
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, _ = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.float32)


class VideoDataset(Dataset):
    """
    Loads frame clips from data/processed/frames/{real,fake}.
    label: 0 = real, 1 = fake
    """

    def __init__(
        self,
        root_dir="data/processed/frames",
        split="train",
        val_ratio=0.2,
        transform=None,
        seed=42,
        n_frames=16,
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform or VIDEO_VAL_TRANSFORM
        self.n_frames = n_frames
        self.random_temporal_sampling = False

        def collect(label_dir, label):
            files = sorted(label_dir.glob("*.jpg"))
            groups = {}
            for file_path in files:
                video_id = processed_sample_id_from_stem(file_path.stem)
                groups.setdefault(video_id, []).append(file_path)
            return [
                (sorted(paths), label, f"{label_dir.name}:{video_id}")
                for video_id, paths in groups.items()
            ]

        real = collect(self.root_dir / "real", 0)
        fake = collect(self.root_dir / "fake", 1)
        samples = real + fake
        self.samples = _split_samples(
            samples,
            split=split,
            val_ratio=val_ratio,
            seed=seed,
            group_key_fn=lambda sample: sample[2],
            stratify_fn=lambda sample: sample[1],
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frame_paths, label, _ = self.samples[idx]
        chosen = _select_frame_paths(
            frame_paths,
            self.n_frames,
            randomize=self.random_temporal_sampling,
        )
        frames = _load_clip_frames(chosen, self.transform)
        clip = torch.stack(frames, dim=0)
        return clip, torch.tensor(label, dtype=torch.float32)


class AudioDataset(Dataset):
    """
    Loads mel-spectrogram arrays from data/processed/spectrograms/{real,fake}.
    label: 0 = real, 1 = fake
    """

    def __init__(
        self,
        root_dir="data/processed/spectrograms",
        split="train",
        val_ratio=0.2,
        seed=42,
        metadata_path=None,
        group_by="label_and_speaker",
    ):
        self.root_dir = Path(root_dir)
        self.has_official_splits = all(
            (self.root_dir / split_name).exists()
            for split_name in ("train", "dev", "eval")
        )
        self.resolved_splits = _resolve_audio_splits(split, self.has_official_splits)
        self.metadata_path = (
            Path(metadata_path)
            if metadata_path is not None
            else self.root_dir / "audio_metadata.json"
        )
        self.group_by = group_by

        metadata_map = {}
        if self.metadata_path.exists():
            with open(self.metadata_path, "r", encoding="utf-8") as handle:
                metadata_records = json.load(handle)
            metadata_map = {
                record["relative_path"]: record
                for record in metadata_records
            }

        if self.has_official_splits:
            self.samples = self._load_official_samples(metadata_map)
            return

        def collect(label_name, label_value):
            collected = []
            for path in sorted((self.root_dir / label_name).glob("*.npy")):
                relative_path = f"{label_name}/{path.name}"
                metadata = metadata_map.get(relative_path)
                if metadata and group_by == "speaker":
                    group_key = metadata["speaker_id"]
                elif metadata and group_by == "label_and_speaker":
                    group_key = f"{label_name}:{metadata['speaker_id']}"
                elif metadata and group_by == "source_file":
                    group_key = metadata["source_file_id"]
                else:
                    group_key = relative_path
                collected.append((path, label_value, group_key, metadata))
            return collected

        samples = collect("real", 0) + collect("fake", 1)
        self.samples = _split_samples(
            samples,
            split=split,
            val_ratio=val_ratio,
            seed=seed,
            group_key_fn=lambda sample: sample[2],
            stratify_fn=lambda sample: sample[1],
        )

    def _load_official_samples(self, metadata_map):
        samples = []
        for split_name in self.resolved_splits:
            for label_name, label_value in [("real", 0), ("fake", 1)]:
                split_dir = self.root_dir / split_name / label_name
                for path in sorted(split_dir.glob("*.npy")):
                    relative_path = path.relative_to(self.root_dir).as_posix()
                    metadata = metadata_map.get(relative_path)
                    if metadata is not None:
                        group_key = f"{split_name}:{metadata['source_file_id']}"
                    else:
                        group_key = relative_path
                    samples.append((path, label_value, group_key, metadata))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, _, _ = self.samples[idx]
        mel = np.load(str(path)).astype(np.float32)
        if mel.ndim == 2:
            mel = mel[np.newaxis, :, :]
        mel_tensor = torch.from_numpy(mel)
        return mel_tensor, torch.tensor(label, dtype=torch.float32)


class AlignedMultimodalDataset(Dataset):
    """
    Manifest-backed multimodal dataset for aligned FaceForensics samples.
    """

    def __init__(
        self,
        mode="train",
        modalities=None,
        seed=42,
        val_ratio=0.2,
        n_frames=16,
        image_transform=None,
        video_transform=None,
        manifest_path="data/processed/manifests/faceforensics_multimodal.json",
    ):
        if modalities is None:
            modalities = ["image", "video", "audio"]
        if not modalities:
            raise ValueError("At least one modality is required")

        split = "val" if mode in {"val", "test"} else "train"
        self.modalities = tuple(modalities)
        self.image_transform = image_transform or IMAGE_VAL_TRANSFORM
        self.video_transform = video_transform or VIDEO_VAL_TRANSFORM
        self.n_frames = n_frames
        self.random_image_sampling = split == "train"
        self.random_video_sampling = split == "train"
        manifest_file = Path(manifest_path)
        if not manifest_file.exists():
            raise FileNotFoundError(
                f"Manifest not found: {manifest_file}. "
                "Run data/preprocessing/build_faceforensics_manifest.py first."
            )

        with open(manifest_file, "r", encoding="utf-8") as handle:
            records = json.load(handle)

        self.records = [
            record
            for record in records
            if self._has_required_modalities(record)
        ]

        self.records = _split_samples(
            self.records,
            split=split,
            val_ratio=val_ratio,
            seed=seed,
            group_key_fn=lambda record: f"{record['class_name']}:{record['sample_id']}",
            stratify_fn=lambda record: record["label"],
        )

    def _has_required_modalities(self, record):
        checks = {
            "image": bool(record.get("face_paths")),
            "video": bool(record.get("frame_paths")),
            "audio": bool(record.get("audio_path")),
        }
        return all(checks[modality] for modality in self.modalities)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        label = int(record["label"])
        sample = {
            "label": torch.tensor(float(label), dtype=torch.float32),
            "sample_id": record["sample_id"],
        }

        if "image" in self.modalities:
            face_paths = record["face_paths"]
            if self.random_image_sampling and len(face_paths) > 1:
                image_path = Path(random.choice(face_paths))
            else:
                image_path = Path(face_paths[len(face_paths) // 2])
            image = Image.open(image_path).convert("RGB")
            sample["image"] = self.image_transform(image)

        if "video" in self.modalities:
            frame_paths = [Path(path) for path in record["frame_paths"]]
            chosen = _select_frame_paths(
                frame_paths,
                self.n_frames,
                randomize=self.random_video_sampling,
            )
            frames = _load_clip_frames(chosen, self.video_transform)
            sample["frames"] = torch.stack(frames, dim=0)

        if "audio" in self.modalities:
            audio_path = Path(record["audio_path"])
            mel = np.load(str(audio_path)).astype(np.float32)
            if mel.ndim == 2:
                mel = mel[np.newaxis, :, :]
            sample["audio"] = torch.from_numpy(mel)

        return sample


class DeepShieldDataset(AlignedMultimodalDataset):
    pass


if __name__ == "__main__":
    print("Testing ImageDataset ...")
    image_dataset = ImageDataset(transform=IMAGE_TRAIN_TRANSFORM)
    print(f"  Train samples : {len(image_dataset)}")
    image, label = image_dataset[0]
    print(f"  Sample shape  : {image.shape}, label={label}")

    print("Testing VideoDataset ...")
    video_dataset = VideoDataset()
    print(f"  Train samples : {len(video_dataset)}")
    clip, label = video_dataset[0]
    print(f"  Clip shape    : {clip.shape}, label={label}")

    manifest_path = Path("data/processed/manifests/faceforensics_multimodal.json")
    if manifest_path.exists():
        print("Testing DeepShieldDataset ...")
        multimodal_dataset = DeepShieldDataset(mode="test")
        print(f"  Multimodal samples : {len(multimodal_dataset)}")
        if len(multimodal_dataset) > 0:
            sample = multimodal_dataset[0]
            print(f"  Keys           : {sorted(sample.keys())}")
    else:
        print("Skipping DeepShieldDataset test (manifest not built yet).")

    print("All dataset tests passed!")
