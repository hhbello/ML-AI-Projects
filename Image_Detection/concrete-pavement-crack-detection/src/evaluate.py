"""
Comprehensive evaluation and diagnostic module for Concrete Crack Detection.
Includes confusion matrix plotting, ROC/PR curves, failure analysis (False Positives vs False Negatives),
and domain-specific Structural Health Monitoring (SHM) risk analysis.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from PIL import Image


@torch.no_grad()
def evaluate_model_on_test_set(
    model: nn.Module,
    test_loader: DataLoader,
    device: Optional[torch.device] = None,
    threshold: float = 0.5
) -> Dict[str, Union[float, np.ndarray, List]]:
    """
    Runs full inference on test loader and collects predictions, probabilities, and metadata.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    all_probs = []
    all_preds = []
    all_targets = []
    all_metadata = []

    for images, labels, meta in test_loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()
        preds = (probs >= threshold).astype(int)

        all_probs.extend(probs.tolist())
        all_preds.extend(preds.tolist())
        all_targets.extend(labels.cpu().numpy().astype(int).tolist())

        # Collect metadata per sample
        batch_size = images.size(0)
        for i in range(batch_size):
            all_metadata.append({
                "filepath": meta["filepath"][i],
                "structure": meta["structure"][i],
                "subfolder": meta["subfolder"][i],
                "label_name": meta["label_name"][i]
            })

    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    acc = accuracy_score(all_targets, all_preds)
    prec = precision_score(all_targets, all_preds, zero_division=0)
    rec = recall_score(all_targets, all_preds, zero_division=0)
    f1 = f1_score(all_targets, all_preds, zero_division=0)

    cm = confusion_matrix(all_targets, all_preds)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / max(tn + fp, 1)

    try:
        fpr, tpr, _ = roc_curve(all_targets, all_probs)
        roc_auc = auc(fpr, tpr)
    except Exception:
        fpr, tpr, roc_auc = np.array([0, 1]), np.array([0, 1]), 0.5

    try:
        precision_curve, recall_curve, _ = precision_recall_curve(all_targets, all_probs)
        pr_auc = average_precision_score(all_targets, all_probs)
    except Exception:
        precision_curve, recall_curve, pr_auc = np.array([1, 0]), np.array([0, 1]), 0.0

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "specificity": float(specificity),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "confusion_matrix": cm,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "y_true": all_targets,
        "y_pred": all_preds,
        "y_prob": all_probs,
        "metadata": all_metadata,
        "fpr": fpr,
        "tpr": tpr,
        "precision_curve": precision_curve,
        "recall_curve": recall_curve
    }


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str] = ["Uncracked", "Cracked"],
    title: str = "Concrete Crack Detection Confusion Matrix",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Renders and optionally saves an annotated, publication-quality confusion matrix.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    annotations = np.empty_like(cm).astype(str)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annotations[i, j] = f"{cm[i, j]:,}\n({cm_percent[i, j]:.1f}%)"

    sns.heatmap(
        cm,
        annot=annotations,
        fmt="",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        ax=ax
    )
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Class", fontsize=11, fontweight="semibold")
    ax.set_ylabel("Actual Class (Ground Truth)", fontsize=11, fontweight="semibold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_roc_and_pr_curves(
    results: Dict[str, Union[float, np.ndarray]],
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plots ROC Curve and Precision-Recall Curve side-by-side."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ROC Curve
    ax1 = axes[0]
    ax1.plot(results["fpr"], results["tpr"], color="#1f77b4", lw=2, label=f"ROC (AUC = {results['roc_auc']:.3f})")
    ax1.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--", label="Random Chance")
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=10, fontweight="semibold")
    ax1.set_ylabel("True Positive Rate (Recall)", fontsize=10, fontweight="semibold")
    ax1.set_title("Receiver Operating Characteristic (ROC)", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="lower right", frameon=True)

    # Precision-Recall Curve
    ax2 = axes[1]
    ax2.plot(results["recall_curve"], results["precision_curve"], color="#2ca02c", lw=2, label=f"PR Curve (PR-AUC = {results['pr_auc']:.3f})")
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel("Recall", fontsize=10, fontweight="semibold")
    ax2.set_ylabel("Precision", fontsize=10, fontweight="semibold")
    ax2.set_title("Precision-Recall Curve (Imbalanced Data)", fontsize=11, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="lower left", frameon=True)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def analyze_failure_cases(
    results: Dict[str, Union[float, np.ndarray, List]],
    num_samples: int = 4,
    save_path: Optional[str] = None
) -> Optional[plt.Figure]:
    """
    Visualizes critical failure modes:
    1. False Negatives: Ground truth cracked, model predicted uncracked (High Civil Safety Risk!)
    2. False Positives: Ground truth uncracked, model predicted cracked (False Alarm)
    """
    y_true = np.array(results["y_true"])
    y_pred = np.array(results["y_pred"])
    y_prob = np.array(results["y_prob"])
    meta = results["metadata"]

    fn_indices = np.where((y_true == 1) & (y_pred == 0))[0]
    fp_indices = np.where((y_true == 0) & (y_pred == 1))[0]

    print(f"\n--- Civil Engineering Inspection Failure Analysis ---")
    print(f"Total Test Samples: {len(y_true)}")
    print(f"True Positives (Detected Cracks): {results['tp']} | True Negatives (Sound Concrete): {results['tn']}")
    print(f"False Negatives (MISSED CRACKS): {len(fn_indices)} -> Critical Hazard in Bridges (Risk of Corrosion/Collapse)")
    print(f"False Positives (FALSE ALARMS): {len(fp_indices)} -> Operational Inconvenience (Re-inspection cost)")

    if len(fn_indices) == 0 and len(fp_indices) == 0:
        print("No classification errors on test set!")
        return None

    samples_to_show = []
    # Pick up to num_samples FN
    for idx in fn_indices[:num_samples]:
        samples_to_show.append((idx, "False Negative (MISSED CRACK)", "#d62728"))
    # Pick up to num_samples FP
    for idx in fp_indices[:num_samples]:
        samples_to_show.append((idx, "False Positive (FALSE ALARM)", "#ff7f0e"))

    cols = min(4, len(samples_to_show))
    rows = (len(samples_to_show) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.5))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.flatten()

    for ax_idx, (sample_idx, error_type, color) in enumerate(samples_to_show):
        ax = axes[ax_idx]
        img_info = meta[sample_idx]
        try:
            img = Image.open(img_info["filepath"]).convert("RGB")
            ax.imshow(img)
        except Exception:
            ax.text(0.5, 0.5, "Image Load Error", ha="center")

        prob = y_prob[sample_idx]
        ax.set_title(
            f"{error_type}\nStructure: {img_info['structure']}\nP(Crack): {prob:.2%}",
            fontsize=9,
            fontweight="bold",
            color=color,
            pad=6
        )
        ax.axis("off")

    for empty_idx in range(len(samples_to_show), len(axes)):
        axes[empty_idx].axis("off")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig
