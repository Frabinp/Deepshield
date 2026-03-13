"""
audio_model.py  —  DeepShield Audio Detector
Uses EfficientNet-B0 on mel-spectrogram images (128x128).
Input : (B, 1, 128, 128) mel-spectrogram
Output: features (B, 512), logit (B, 1)
"""

import torch
import torch.nn as nn
import timm


class AudioDetector(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()

        self.backbone = timm.create_model(
            "efficientnet_b0", pretrained=pretrained, num_classes=0
        )
        # EfficientNet-B0 expects 3-channel input; convert 1-ch mel to 3-ch
        self.channel_expand = nn.Conv2d(1, 3, kernel_size=1, bias=False)

        in_features = self.backbone.num_features  # 1280 for B0
        self.feature_proj = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.classifier = nn.Linear(512, 1)

    def forward(self, x):
        # x: (B, 1, 128, 128)
        x = self.channel_expand(x)          # (B, 3, 128, 128)
        feats = self.backbone(x)             # (B, 1280)
        feats = self.feature_proj(feats)     # (B, 512)
        logit = self.classifier(feats)       # (B, 1)
        return feats, logit


if __name__ == "__main__":
    model = AudioDetector()
    x = torch.randn(4, 1, 128, 128)
    feats, logit = model(x)
    print(f"Features : {feats.shape}")   # (4, 512)
    print(f"Logit    : {logit.shape}")   # (4, 1)
    total = sum(p.numel() for p in model.parameters())
    print(f"Params   : {total/1e6:.1f}M")