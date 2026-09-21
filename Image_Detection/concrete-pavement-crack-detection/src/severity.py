"""
Crack severity and width bucketing module for Concrete Infrastructure Health Monitoring.
Grounds crack assessments in civil engineering standards (ACI 224R / AASHTO / BS 8110).
"""

from typing import Dict, Optional, Tuple, Union
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def estimate_crack_severity(
    image_input: Union[str, Image.Image, np.ndarray],
    gradcam_heatmap: Optional[np.ndarray] = None,
    pixel_to_mm_scale: float = 0.5, # Nominal scale: 1 pixel ~ 0.5 mm for typical 256x256 patch
    visualize: bool = False
) -> Dict[str, Union[float, str, np.ndarray, Optional[plt.Figure]]]:
    """
    Analyzes concrete crack geometry to estimate crack width and assign an ACI 224R severity tier.

    Args:
        image_input: Path to image, PIL Image, or RGB numpy array.
        gradcam_heatmap: Optional 2D Grad-CAM heatmap [0, 1] to mask the crack region of interest.
        pixel_to_mm_scale: Estimated spatial resolution in mm per pixel.
        visualize: Whether to generate a diagnostic plot.

    Returns:
        Dictionary containing:
        - 'max_width_mm': Estimated maximum crack width in millimeters.
        - 'mean_width_mm': Estimated mean crack width in millimeters.
        - 'severity_level': 'Low / Hairline', 'Moderate', or 'Severe'.
        - 'action_recommendation': Recommended civil engineering maintenance protocol.
        - 'binary_mask': Segmented crack binary mask.
        - 'figure': Optional matplotlib figure if visualize=True.
    """
    if isinstance(image_input, str):
        img_bgr = cv2.imread(image_input)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    elif isinstance(image_input, Image.Image):
        img_rgb = np.array(image_input.convert("RGB"))
    else:
        img_rgb = image_input.copy()

    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape

    # Apply bilateral filter to smooth aggregate texture while keeping crack edges sharp
    filtered = cv2.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=50)

    # Adaptive thresholding to segment dark fracture lines from light concrete paste
    # Concrete cracks are typically darker than surrounding paste
    thresh = cv2.adaptiveThreshold(
        filtered,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=25,
        C=8
    )

    # If Grad-CAM heatmap is provided, focus on high-attention crack regions (filter out peripheral noise)
    if gradcam_heatmap is not None:
        heatmap_resized = cv2.resize(gradcam_heatmap, (w, h))
        attention_mask = (heatmap_resized > 0.35).astype(np.uint8) * 255
        # Morphological dilation on attention mask
        kernel_att = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        attention_mask = cv2.dilate(attention_mask, kernel_att)
        thresh = cv2.bitwise_and(thresh, thresh, mask=attention_mask)

    # Clean morphological noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter out tiny aggregate noise
    crack_contours = [c for c in contours if cv2.contourArea(c) > 20]

    if not crack_contours:
        # No significant crack segments detected
        return {
            "max_width_mm": 0.0,
            "mean_width_mm": 0.0,
            "severity_level": "None / Sound Concrete",
            "action_recommendation": "No cracking detected. Continue routine inspection schedule.",
            "binary_mask": cleaned,
            "figure": None
        }

    # Estimate crack width using distance transform on the crack mask
    # Distance transform computes the distance of every foreground pixel to the nearest zero pixel
    dist_transform = cv2.distanceTransform(cleaned, cv2.DIST_L2, 5)

    # Maximum crack width (diameter) is 2 * max distance transform value
    max_radius_px = np.max(dist_transform)
    max_width_px = 2.0 * max_radius_px

    # Mean width over non-zero crack skeleton pixels
    foreground_dist = dist_transform[cleaned > 0]
    mean_width_px = 2.0 * np.mean(foreground_dist) if len(foreground_dist) > 0 else 0.0

    max_width_mm = max_width_px * pixel_to_mm_scale
    mean_width_mm = mean_width_px * pixel_to_mm_scale

    # Categorize into ACI 224R / Structural Health Monitoring Severity Buckets:
    # - Low / Hairline: < 0.3 mm (Acceptable shrinkage cracks in dry/moderate exposure)
    # - Moderate: 0.3 mm to 1.0 mm (Exceeds permissible limits for exposed bridges, moisture ingress risk)
    # - Severe: > 1.0 mm (Active structural distress, rebar corrosion risk, immediate intervention)
    if max_width_mm < 0.30:
        severity = "Low / Hairline (<0.3 mm)"
        color_code = "#2ca02c" # Green
        recommendation = "Surface shrinkage or micro-cracking. No immediate structural hazard. Log in SHM database and monitor periodically."
    elif max_width_mm <= 1.00:
        severity = "Moderate (0.3 - 1.0 mm)"
        color_code = "#ff7f0e" # Orange
        recommendation = "Active crack exceeding serviceability threshold. Potential moisture and chloride ingress. Recommend sealing or epoxy injection."
    else:
        severity = "Severe / Structural (>1.0 mm)"
        color_code = "#d62728" # Red
        recommendation = "CRITICAL STRUCTURAL WARNING. Significant crack opening indicating potential overload, settlement, or active rebar corrosion. Immediate structural engineering inspection and load rating review required."

    fig = None
    if visualize:
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        axes[0].imshow(img_rgb)
        axes[0].set_title("Original Concrete Surface", fontsize=11, fontweight="semibold")
        axes[0].axis("off")

        axes[1].imshow(cleaned, cmap="gray")
        axes[1].set_title("Segmented Crack Fracture Mask", fontsize=11, fontweight="semibold")
        axes[1].axis("off")

        # Overlay segmentation contours onto original image
        overlay = img_rgb.copy()
        cv2.drawContours(overlay, crack_contours, -1, (255, 0, 0), 2)

        axes[2].imshow(overlay)
        axes[2].set_title(
            f"Severity: {severity}\nMax Width: {max_width_mm:.2f} mm | Mean: {mean_width_mm:.2f} mm",
            fontsize=10,
            fontweight="bold",
            color=color_code,
            pad=8
        )
        axes[2].axis("off")
        plt.tight_layout()

    return {
        "max_width_mm": float(max_width_mm),
        "mean_width_mm": float(mean_width_mm),
        "severity_level": severity,
        "color_code": color_code,
        "action_recommendation": recommendation,
        "binary_mask": cleaned,
        "figure": fig
    }
