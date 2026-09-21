"""
Grad-CAM (Gradient-weighted Class Activation Mapping) for Concrete Crack Detection.
Enables visual explainability to verify that model attention grounds on fracture lines
rather than concrete surface textures, shadows, or aggregate stones.
"""

from pathlib import Path
from typing import Optional, Tuple, Union
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms


class GradCAM:
    """
    Computes Grad-CAM heatmaps for a specified convolutional target layer.
    """
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            # grad_out[0] contains the gradient w.r.t layer output
            self.gradients = grad_out[0].detach()

        h1 = self.target_layer.register_forward_hook(forward_hook)
        h2 = self.target_layer.register_full_backward_hook(backward_hook)
        self._hook_handles.extend([h1, h2])

    def remove_hooks(self):
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()

    def generate_heatmap(
        self,
        input_tensor: torch.Tensor,
        target_class: int = 1
    ) -> np.ndarray:
        """
        Generates 2D Grad-CAM heatmap normalized to [0, 1].

        Args:
            input_tensor: Shape (1, 3, H, W) on correct device.
            target_class: 1 for cracked, 0 for sound concrete.

        Returns:
            2D numpy array with dimensions matching input (H, W).
        """
        self.model.eval()
        self.model.zero_grad()

        # Forward pass
        logits = self.model(input_tensor)
        score = logits[0, 0] if target_class == 1 else -logits[0, 0]

        # Backward pass to get gradients
        score.backward(retain_graph=True)

        # Global average pooling on gradients across spatial dimensions (H, W)
        # self.gradients: (1, C, H', W')
        alpha = torch.mean(self.gradients, dim=[2, 3], keepdim=True)

        # Weighted combination of forward activation maps
        cam = torch.sum(alpha * self.activations, dim=1, keepdim=True)

        # Apply ReLU to keep only features that have a positive impact on crack prediction
        cam = torch.relu(cam)

        cam = cam.squeeze().cpu().numpy()

        # Normalize to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        # Resize heatmap to match original input size
        h, w = input_tensor.shape[2], input_tensor.shape[3]
        heatmap = cv2.resize(cam, (w, h))
        return heatmap


def overlay_heatmap_on_image(
    original_image: Union[np.ndarray, Image.Image],
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET
) -> np.ndarray:
    """
    Overlays a normalized Grad-CAM heatmap onto an RGB image.

    Args:
        original_image: PIL Image or RGB uint8 numpy array (H, W, 3).
        heatmap: 2D numpy array [0, 1] of shape (H, W).
        alpha: Blending weight for heatmap.
        colormap: OpenCV colormap.

    Returns:
        RGB numpy array with blended visualization.
    """
    if isinstance(original_image, Image.Image):
        img_np = np.array(original_image)
    else:
        img_np = original_image.copy()

    # Ensure RGB
    if img_np.ndim == 2:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)

    h, w = img_np.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))

    # Convert to 8-bit heatmap and apply colormap
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    colored_heatmap = cv2.applyColorMap(heatmap_uint8, colormap)
    colored_heatmap = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB)

    # Blend
    blended = np.uint8((1 - alpha) * img_np + alpha * colored_heatmap)
    return blended


def explain_crack_prediction(
    model: nn.Module,
    image_path_or_pil: Union[str, Image.Image],
    device: Optional[torch.device] = None,
    img_size: int = 224,
    save_path: Optional[str] = None
) -> Tuple[plt.Figure, float, np.ndarray]:
    """
    End-to-end explainability function:
    Loads an image, computes prediction probability, generates Grad-CAM heatmap,
    and displays a 3-panel figure: [Original Concrete, Grad-CAM Attention, Overlay].
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load and preprocess image
    if isinstance(image_path_or_pil, (str, Path)):
        pil_img = Image.open(image_path_or_pil).convert("RGB")
    else:
        pil_img = image_path_or_pil.convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    input_tensor = transform(pil_img).unsqueeze(0).to(device)

    # Target layer for Grad-CAM
    if hasattr(model, "get_target_layer_for_cam"):
        target_layer = model.get_target_layer_for_cam()
    else:
        raise AttributeError("Model does not provide get_target_layer_for_cam()")

    cam_engine = GradCAM(model, target_layer)
    heatmap = cam_engine.generate_heatmap(input_tensor, target_class=1)

    # Inference probability
    with torch.no_grad():
        logits = model(input_tensor)
        prob = torch.sigmoid(logits).item()

    cam_engine.remove_hooks()

    # Create overlay
    resized_pil = pil_img.resize((img_size, img_size))
    overlay = overlay_heatmap_on_image(resized_pil, heatmap, alpha=0.45)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(resized_pil)
    axes[0].set_title("Input Concrete Surface", fontsize=11, fontweight="semibold")
    axes[0].axis("off")

    im1 = axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title("Grad-CAM Class Activation", fontsize=11, fontweight="semibold")
    axes[1].axis("off")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    status = "CRACK DETECTED" if prob >= 0.5 else "SOUND CONCRETE"
    color = "red" if prob >= 0.5 else "green"
    axes[2].imshow(overlay)
    axes[2].set_title(f"Overlay Inspection\nP(Crack) = {prob:.1%} [{status}]", fontsize=11, fontweight="bold", color=color)
    axes[2].axis("off")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig, prob, heatmap
