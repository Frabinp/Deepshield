import torch
import torch.nn as nn
import timm
from typing import Tuple


class VideoDetector(nn.Module):
    """
    EfficientNet-B0 + LSTM video deepfake detector.
    Per-frame CNN features are fed into a 2-layer LSTM for temporal modeling.
    Input  : (B, 16, 3, 224, 224) video clip (16 frames)
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
        # We use a per-frame CNN + temporal LSTM
        # (X3D not available in timm — this is equivalent and lighter)
        self.frame_encoder = timm.create_model(
            "efficientnet_b0",
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg"
        )
        frame_dim = self.frame_encoder.num_features  # 1280

        # ── Temporal modeling ──────────────────────────
        # LSTM over 16 frame features
        self.temporal = nn.LSTM(
            input_size=frame_dim,
            hidden_size=512,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=False
        )

        # ── Feature projection → 512 ───────────────────
        self.feature_head = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # ── Classifier head ────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

        # ── Optionally freeze backbone ─────────────────
        if freeze_backbone:
            for param in self.frame_encoder.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, 16, 3, 224, 224) video clip tensor
        Returns:
            features : (B, 512)
            logit    : (B, 1)
        """
        B, T, C, H, W = x.shape  # B=batch, T=16 frames

        # Encode each frame independently
        # Reshape to (B*T, C, H, W) for batch processing
        x_flat = x.view(B * T, C, H, W)              # (B*16, 3, 224, 224)
        frame_feats = self.frame_encoder(x_flat)      # (B*16, 1280)

        # Reshape back to (B, T, 1280)
        frame_feats = frame_feats.view(B, T, -1)      # (B, 16, 1280)

        # Temporal modeling with LSTM
        lstm_out, (h_n, _) = self.temporal(frame_feats)
        # Use last hidden state as clip representation
        temporal_feat = h_n[-1]                       # (B, 512)

        # Project features
        features = self.feature_head(temporal_feat)   # (B, 512)

        # Classify
        logit = self.classifier(features)             # (B, 1)

        return features, logit

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Returns probability (0=real, 1=fake)"""
        _, logit = self.forward(x)
        return torch.sigmoid(logit)


# ── Quick Test ─────────────────────────────────────────
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Testing VideoDetector on {device}...")

    model = VideoDetector(pretrained=True).to(device)

    # Dummy batch of 2 video clips (small batch — 4GB VRAM)
    dummy = torch.randn(2, 16, 3, 224, 224).to(device)

    features, logit = model(dummy)
    probs = torch.sigmoid(logit)

    print(f"Input shape    : {dummy.shape}")
    print(f"Features shape : {features.shape}")   # (2, 512)
    print(f"Logit shape    : {logit.shape}")       # (2, 1)
    print(f"Probs          : {probs.detach().cpu().numpy().flatten()}")

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal params    : {total:,}")
    print(f"Trainable params: {trainable:,}")
    print("\n✅ VideoDetector ready!")