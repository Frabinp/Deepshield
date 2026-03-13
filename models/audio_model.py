"""
audio_model.py  —  DeepShield Audio Detector
Uses EfficientNet-B0 on mel-spectrogram images (128x128).
Input : (B, 1, 128, 128) mel-spectrogram
Output: features (B, 512), logit (B, 1)
"""

import torch
import torch.nn as nn
import timm
from typing import Tuple


class AudioDetector(nn.Module):
    def __init__(
        self,
        pretrained: bool = True,
        dropout: float = 0.3,
        freeze_backbone: bool = False,
    ):
        super().__init__()

        self.backbone = timm.create_model(
            "efficientnet_b0", pretrained=pretrained, num_classes=0
        )
        # EfficientNet-B0 expects 3-channel input; convert 1-ch mel to 3-ch
        self.channel_expand = nn.Conv2d(1, 3, kernel_size=1, bias=False)

        in_features = self.backbone.num_features  # 1280 for B0
        self.feature_proj = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(512, 1)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, 1, 128, 128) mel-spectrogram tensor
        Returns:
            features : (B, 512) for fusion layer
            logit    : (B, 1)   raw prediction
        """
        x = self.channel_expand(x)          # (B, 3, 128, 128)
        feats = self.backbone(x)             # (B, 1280)
        feats = self.feature_proj(feats)     # (B, 512)
        logit = self.classifier(feats)       # (B, 1)
        return feats, logit

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Returns probability (0=real, 1=fake)"""
        _, logit = self.forward(x)
        return torch.sigmoid(logit)


if __name__ == "__main__":
    model = AudioDetector()
    x = torch.randn(4, 1, 128, 128)
    feats, logit = model(x)
    print(f"Features : {feats.shape}")   # (4, 512)
    print(f"Logit    : {logit.shape}")   # (4, 1)
    total = sum(p.numel() for p in model.parameters())
    print(f"Params   : {total/1e6:.1f}M")