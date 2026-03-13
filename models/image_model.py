import torch
import torch.nn as nn
import timm
from typing import Tuple

class ImageDetector(nn.Module):
    """
    EfficientNet-B0 based image deepfake detector.
    Input  : (B, 3, 224, 224) face crop
    Output : feature vector (B, 512) + logit (B, 1)
    """

    def __init__(
        self,
        pretrained: bool = True,
        dropout: float = 0.3,
        freeze_backbone: bool = False
    ):
        super().__init__()

        # ── Backbone ───────────────────────────────────
        self.backbone = timm.create_model(
            "efficientnet_b0",
            pretrained=pretrained,
            num_classes=0,          # remove classifier head
            global_pool="avg"       # global average pooling
        )
        backbone_dim = self.backbone.num_features  # 1280 for B0

        # ── Feature projection → 512 ───────────────────
        self.feature_head = nn.Sequential(
            nn.Linear(backbone_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # ── Classifier head ────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)       # binary: real vs fake
        )

        # ── Optionally freeze backbone ─────────────────
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, 3, 224, 224) image tensor
        Returns:
            features : (B, 512)  for fusion layer
            logit    : (B, 1)    raw prediction
        """
        # Extract backbone features
        backbone_feat = self.backbone(x)          # (B, 1280)

        # Project to 512-dim
        features = self.feature_head(backbone_feat)  # (B, 512)

        # Classify
        logit = self.classifier(features)         # (B, 1)

        return features, logit

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Returns probability (0=real, 1=fake)"""
        _, logit = self.forward(x)
        return torch.sigmoid(logit)


# ── Quick Test ─────────────────────────────────────────
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Testing ImageDetector on {device}...")

    model = ImageDetector(pretrained=True).to(device)

    # Dummy batch of 4 images
    dummy = torch.randn(4, 3, 224, 224).to(device)

    features, logit = model(dummy)
    probs = torch.sigmoid(logit)

    print(f"Input shape    : {dummy.shape}")
    print(f"Features shape : {features.shape}")   # (4, 512)
    print(f"Logit shape    : {logit.shape}")       # (4, 1)
    print(f"Probs          : {probs.detach().cpu().numpy().flatten()}")

    # Count parameters
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal params    : {total:,}")
    print(f"Trainable params: {trainable:,}")
    print("\n✅ ImageDetector ready!")