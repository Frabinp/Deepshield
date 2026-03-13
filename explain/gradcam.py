"""
gradcam.py  —  DeepShield Grad-CAM Explainability

Generates Grad-CAM heatmaps for the image and video branches,
showing which spatial regions most influenced the model's prediction.
"""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)


class GradCAM:
    """
    Grad-CAM for any CNN model that uses a `timm` backbone.

    Usage:
        cam = GradCAM(model, target_layer=model.backbone.conv_head)
        heatmap = cam(input_tensor)
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self._gradients = None
        self._activations = None

        # Register hooks
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self._activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self._gradients = grad_output[0].detach()

    @torch.enable_grad()
    def __call__(
        self,
        input_tensor: torch.Tensor,
        target_class: int = 1,
    ) -> np.ndarray:
        """
        Generate a Grad-CAM heatmap.

        Args:
            input_tensor: (1, 3, 224, 224) preprocessed image tensor
            target_class: 1 for fake, 0 for real

        Returns:
            heatmap: (224, 224) numpy array with values in [0, 1]
        """
        self.model.eval()
        input_tensor = input_tensor.requires_grad_(True)

        # Forward pass
        _, logit = self.model(input_tensor)

        # For binary classification, if target_class=1 (fake),
        # we backprop the logit directly; if 0 (real), negate it.
        if target_class == 0:
            score = -logit.squeeze()
        else:
            score = logit.squeeze()

        self.model.zero_grad()
        score.backward()

        # Grad-CAM computation
        gradients = self._gradients  # (1, C, H, W)
        activations = self._activations  # (1, C, H, W)

        # Global average pool the gradients
        weights = gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Weighted combination of activation maps
        cam = (weights * activations).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        cam = F.relu(cam)  # Only positive contributions

        # Normalize
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()

        # Resize to input size
        cam_resized = np.array(
            Image.fromarray((cam * 255).astype(np.uint8)).resize(
                (224, 224), Image.BILINEAR
            )
        )
        return cam_resized.astype(np.float32) / 255.0


def overlay_heatmap(
    original_image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.4,
    colormap: str = "jet",
) -> Image.Image:
    """
    Overlay a Grad-CAM heatmap on the original image.

    Args:
        original_image: PIL Image (any size)
        heatmap: (H, W) numpy array in [0, 1]
        alpha: transparency of heatmap overlay
        colormap: matplotlib colormap name

    Returns:
        PIL Image with heatmap overlay
    """
    import matplotlib.cm as cm

    cmap = cm.get_cmap(colormap)
    heatmap_colored = cmap(heatmap)[:, :, :3]  # Drop alpha channel
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
    heatmap_pil = Image.fromarray(heatmap_colored).resize(
        original_image.size, Image.BILINEAR
    )

    # Blend
    blended = Image.blend(
        original_image.convert("RGB"),
        heatmap_pil,
        alpha=alpha,
    )
    return blended


def explain_image(model, pil_image: Image.Image, device: torch.device = None):
    """
    End-to-end Grad-CAM explanation for an image.

    Args:
        model: ImageDetector instance
        pil_image: input image
        device: torch device

    Returns:
        dict with probability, label, heatmap, overlay
    """
    if device is None:
        device = next(model.parameters()).device

    # Find the last conv layer in EfficientNet backbone
    target_layer = model.backbone.conv_head

    cam = GradCAM(model, target_layer)

    # Preprocessing
    input_tensor = TRANSFORM(pil_image.convert("RGB")).unsqueeze(0).to(device)

    # Get prediction
    with torch.no_grad():
        _, logit = model(input_tensor)
    probability = torch.sigmoid(logit).item()
    is_fake = probability > 0.5

    # Generate heatmap (explain toward predicted class)
    target_class = 1 if is_fake else 0
    heatmap = cam(input_tensor, target_class=target_class)

    # Create overlay
    resized = pil_image.convert("RGB").resize((224, 224))
    overlay = overlay_heatmap(resized, heatmap)

    return {
        "probability": probability,
        "is_fake": is_fake,
        "label": "FAKE" if is_fake else "REAL",
        "heatmap": heatmap,
        "overlay": overlay,
    }


if __name__ == "__main__":
    import sys
    import os

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from models.image_model import ImageDetector
    from models.checkpoint_utils import load_module_checkpoint

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImageDetector().to(device)

    checkpoint_path = "checkpoints/best_image.pt"
    if os.path.exists(checkpoint_path):
        load_module_checkpoint(model, checkpoint_path, map_location=device)
        print("Loaded checkpoint")
    else:
        print("No checkpoint found, using random weights for demo")

    # Quick test with a random tensor
    dummy = torch.randn(1, 3, 224, 224).to(device)
    target_layer = model.backbone.conv_head
    cam = GradCAM(model, target_layer)
    heatmap = cam(dummy)

    print(f"Heatmap shape: {heatmap.shape}")
    print(f"Heatmap range: [{heatmap.min():.3f}, {heatmap.max():.3f}]")
    print("✅ Grad-CAM working!")
