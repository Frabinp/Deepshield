import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class FusionModel(nn.Module):
    """
    Late fusion model — combines features from all 3 branches.

    Input  : image_feat (B,512) + video_feat (B,512) + audio_feat (B,512)
    Output : deepfake probability (B,1) + uncertainty (B,1)

    Novel contribution: Cross-modal consistency check via
    contrastive loss between video and audio embeddings.
    """

    def __init__(self, dropout: float = 0.4):
        super().__init__()

        # ── Modality-specific normalizers ──────────────
        self.image_norm = nn.LayerNorm(512)
        self.video_norm = nn.LayerNorm(512)
        self.audio_norm = nn.LayerNorm(512)

        # ── Attention weights per modality ─────────────
        # Learns which modality to trust more
        self.modality_attention = nn.Sequential(
            nn.Linear(512 * 3, 3),
            nn.Softmax(dim=1)
        )

        # ── Fusion MLP ─────────────────────────────────
        self.fusion = nn.Sequential(
            nn.Linear(512 * 3, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # ── Main classifier ────────────────────────────
        self.classifier = nn.Linear(256, 1)

        # ── Uncertainty head (Monte Carlo Dropout) ─────
        self.uncertainty_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.5),        # always on for uncertainty
            nn.Linear(64, 1),
            nn.Softplus()           # ensures positive uncertainty
        )

        # ── Cross-modal projectors (novel contribution) ─
        # Projects video + audio into shared space for
        # contrastive consistency check
        self.video_projector = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        self.audio_projector = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

    def cross_modal_loss(
        self,
        video_feat: torch.Tensor,
        audio_feat: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Cross-modal contrastive loss.
        Real samples → video & audio should be CONSISTENT (similar)
        Fake samples → video & audio may be INCONSISTENT (different)
        """
        v_proj = F.normalize(self.video_projector(video_feat), dim=1)
        a_proj = F.normalize(self.audio_projector(audio_feat), dim=1)

        # Cosine similarity between video and audio
        similarity = F.cosine_similarity(v_proj, a_proj, dim=1)  # (B,)

        # Real (0): similarity should be HIGH → loss = 1 - sim
        # Fake (1): similarity should be LOW  → loss = max(0, sim - margin)
        margin = 0.3
        real_mask = (labels == 0).float()
        fake_mask = (labels == 1).float()

        loss_real = real_mask * (1 - similarity)
        loss_fake = fake_mask * torch.clamp(similarity - margin, min=0)

        return (loss_real + loss_fake).mean()

    def forward(
        self,
        image_feat: torch.Tensor,
        video_feat: torch.Tensor,
        audio_feat: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> dict:
        """
        Args:
            image_feat : (B, 512) from ImageDetector
            video_feat : (B, 512) from VideoDetector
            audio_feat : (B, 512) from AudioDetector
            labels     : (B,) optional — for cross-modal loss
        Returns:
            dict with logit, probability, uncertainty,
                 attention_weights, cross_modal_loss
        """
        # Normalize each modality
        img = self.image_norm(image_feat)
        vid = self.video_norm(video_feat)
        aud = self.audio_norm(audio_feat)

        # Concatenate all features
        combined = torch.cat([img, vid, aud], dim=1)  # (B, 1536)

        # Modality attention weights
        attn = self.modality_attention(combined)      # (B, 3)
        # attn[:, 0] = image weight
        # attn[:, 1] = video weight
        # attn[:, 2] = audio weight

        # Weighted features
        img_w = img * attn[:, 0:1]
        vid_w = vid * attn[:, 1:2]
        aud_w = aud * attn[:, 2:3]
        combined_w = torch.cat([img_w, vid_w, aud_w], dim=1)

        # Fusion
        fused = self.fusion(combined_w)               # (B, 256)

        # Main prediction
        logit = self.classifier(fused)                # (B, 1)
        prob  = torch.sigmoid(logit)                  # (B, 1)

        # Uncertainty head follows the module's current train/eval mode.
        uncertainty = self.uncertainty_head(fused)    # (B, 1)

        # Cross-modal loss (only during training)
        cm_loss = torch.tensor(0.0, device=logit.device)
        if labels is not None:
            cm_loss = self.cross_modal_loss(video_feat, audio_feat, labels)

        return {
            "logit"             : logit,
            "probability"       : prob,
            "uncertainty"       : uncertainty,
            "attention_weights" : attn,
            "cross_modal_loss"  : cm_loss,
            "fused_features"    : fused
        }

    def predict(
        self,
        image_feat: torch.Tensor,
        video_feat: torch.Tensor,
        audio_feat: torch.Tensor,
        mc_samples: int = 10
    ) -> dict:
        """
        Monte Carlo Dropout inference for uncertainty estimation.
        Runs forward pass mc_samples times and averages predictions.
        """
        if mc_samples < 2:
            raise ValueError("mc_samples must be at least 2")

        was_training = self.training
        self.train()

        probs = []
        head_uncertainties = []
        try:
            with torch.no_grad():
                for _ in range(mc_samples):
                    out = self.forward(image_feat, video_feat, audio_feat)
                    probs.append(out["probability"])
                    head_uncertainties.append(out["uncertainty"])
        finally:
            self.train(was_training)

        probs = torch.stack(probs, dim=0)                 # (mc, B, 1)
        mean_prob = probs.mean(dim=0)                     # (B, 1)
        uncertainty = probs.std(dim=0, unbiased=False)    # (B, 1)
        mean_head_uncertainty = torch.stack(head_uncertainties, dim=0).mean(dim=0)

        return {
            "probability": mean_prob,
            "uncertainty": uncertainty,
            "head_uncertainty": mean_head_uncertainty,
            "is_fake": (mean_prob > 0.5).float(),
        }


# ── Quick Test ─────────────────────────────────────────
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Testing FusionModel on {device}...")

    model = FusionModel().to(device)

    # Dummy features from each branch
    B = 4
    img_feat   = torch.randn(B, 512).to(device)
    vid_feat   = torch.randn(B, 512).to(device)
    aud_feat   = torch.randn(B, 512).to(device)
    labels     = torch.randint(0, 2, (B,)).to(device)

    out = model(img_feat, vid_feat, aud_feat, labels)

    print(f"Probability       : {out['probability'].detach().cpu().numpy().flatten()}")
    print(f"Uncertainty       : {out['uncertainty'].detach().cpu().numpy().flatten()}")
    print(f"Attention weights : {out['attention_weights'].detach().cpu().numpy()}")
    print(f"Cross-modal loss  : {out['cross_modal_loss'].item():.4f}")

    # MC Dropout inference
    mc_out = model.predict(img_feat, vid_feat, aud_feat, mc_samples=10)
    print(f"\nMC Probability  : {mc_out['probability'].detach().cpu().numpy().flatten()}")
    print(f"MC Uncertainty  : {mc_out['uncertainty'].detach().cpu().numpy().flatten()}")
    print(f"Is fake         : {mc_out['is_fake'].detach().cpu().numpy().flatten()}")

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal params    : {total:,}")
    print(f"Trainable params: {trainable:,}")
    print("\n FusionModel ready!")
