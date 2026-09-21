"""
Training engine for Concrete Crack Detection.
Features mixed-precision training, cost-sensitive loss (BCEWithLogitsLoss with pos_weight),
two-phase transfer learning (warmup head -> fine-tune backbone), and early stopping.
"""

import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tqdm import tqdm


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.amp.GradScaler],
    device: torch.device
) -> Tuple[float, float, float, float, float]:
    """
    Trains model for one epoch.
    Returns: (epoch_loss, accuracy, precision, recall, f1)
    """
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    use_cuda = device.type == "cuda"

    for images, labels, _ in tqdm(dataloader, desc="Training", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).unsqueeze(1)

        optimizer.zero_grad(set_to_none=True)

        if use_cuda and scaler is not None:
            with torch.amp.autocast("cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)

        probs = torch.sigmoid(outputs).detach().cpu().numpy()
        preds = (probs >= 0.5).astype(int)
        all_preds.extend(preds.flatten())
        all_targets.extend(labels.detach().cpu().numpy().astype(int).flatten())

    epoch_loss = running_loss / len(dataloader.dataset)
    acc = accuracy_score(all_targets, all_preds)
    prec = precision_score(all_targets, all_preds, zero_division=0)
    rec = recall_score(all_targets, all_preds, zero_division=0)
    f1 = f1_score(all_targets, all_preds, zero_division=0)

    return epoch_loss, acc, prec, rec, f1


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float, float, float, float, float]:
    """
    Evaluates model on validation or test dataset.
    Returns: (val_loss, accuracy, precision, recall, f1, roc_auc)
    """
    model.eval()
    running_loss = 0.0
    all_probs = []
    all_preds = []
    all_targets = []

    for images, labels, _ in tqdm(dataloader, desc="Evaluating", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).unsqueeze(1)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)

        probs = torch.sigmoid(outputs).cpu().numpy()
        preds = (probs >= 0.5).astype(int)

        all_probs.extend(probs.flatten())
        all_preds.extend(preds.flatten())
        all_targets.extend(labels.cpu().numpy().astype(int).flatten())

    val_loss = running_loss / len(dataloader.dataset)
    acc = accuracy_score(all_targets, all_preds)
    prec = precision_score(all_targets, all_preds, zero_division=0)
    rec = recall_score(all_targets, all_preds, zero_division=0)
    f1 = f1_score(all_targets, all_preds, zero_division=0)

    try:
        auc = roc_auc_score(all_targets, all_probs)
    except Exception:
        auc = 0.5

    return val_loss, acc, prec, rec, f1, auc


def train_crack_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 10,
    lr_head: float = 1e-3,
    lr_backbone: float = 1e-4,
    warmup_epochs: int = 2,
    pos_weight: Optional[float] = None,
    save_path: str = "best_crack_model.pth",
    patience: int = 4,
    device: Optional[torch.device] = None
) -> Dict[str, List[float]]:
    """
    Two-phase training loop:
    1. Warmup: Train classification head with frozen backbone.
    2. Fine-tuning: Unfreeze top layers of the backbone with reduced learning rate.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Set up weighted loss to address class imbalance (prioritize Recall for civil infrastructure safety)
    if pos_weight is not None:
        pw_tensor = torch.tensor([pos_weight], device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pw_tensor)
        print(f"Using BCEWithLogitsLoss with pos_weight={pos_weight:.2f} (Recall priority)")
    else:
        criterion = nn.BCEWithLogitsLoss()

    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    # Phase 1: Warmup classifier head
    print(f"\n--- Phase 1: Warmup Classifier Head ({warmup_epochs} epochs) ---")
    if hasattr(model, "freeze_backbone"):
        model.freeze_backbone()

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr_head,
        weight_decay=1e-4
    )

    history = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": [],
        "train_prec": [], "val_prec": [],
        "train_rec": [], "val_rec": [],
        "train_f1": [], "val_f1": [],
        "val_auc": []
    }

    best_val_f1 = -1.0
    patience_counter = 0

    total_epochs = warmup_epochs + epochs
    for epoch in range(1, total_epochs + 1):
        # Transition to Phase 2 after warmup
        if epoch == warmup_epochs + 1:
            print(f"\n--- Phase 2: Fine-Tuning Backbone Layers ({epochs} epochs) ---")
            if hasattr(model, "unfreeze_top_layers"):
                model.unfreeze_top_layers(num_blocks=2)
            elif hasattr(model, "unfreeze_all"):
                model.unfreeze_all()

            optimizer = torch.optim.AdamW([
                {"params": [p for n, p in model.named_parameters() if "fc" in n or "classifier" in n], "lr": lr_head * 0.5},
                {"params": [p for n, p in model.named_parameters() if "fc" not in n and "classifier" not in n and p.requires_grad], "lr": lr_backbone}
            ], weight_decay=1e-4)

            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", factor=0.5, patience=2
            )

        start_time = time.time()
        t_loss, t_acc, t_prec, t_rec, t_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device
        )
        v_loss, v_acc, v_prec, v_rec, v_f1, v_auc = evaluate_epoch(
            model, val_loader, criterion, device
        )
        elapsed = time.time() - start_time

        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)
        history["train_acc"].append(t_acc)
        history["val_acc"].append(v_acc)
        history["train_prec"].append(t_prec)
        history["val_prec"].append(v_prec)
        history["train_rec"].append(t_rec)
        history["val_rec"].append(v_rec)
        history["train_f1"].append(t_f1)
        history["val_f1"].append(v_f1)
        history["val_auc"].append(v_auc)

        print(
            f"Epoch [{epoch:02d}/{total_epochs:02d}] ({elapsed:.1f}s) | "
            f"Train Loss: {t_loss:.4f} Acc: {t_acc:.3f} Rec: {t_rec:.3f} F1: {t_f1:.3f} | "
            f"Val Loss: {v_loss:.4f} Acc: {v_acc:.3f} Rec: {v_rec:.3f} F1: {v_f1:.3f} AUC: {v_auc:.3f}"
        )

        # Checkpoint based on validation F1 score
        if v_f1 > best_val_f1:
            best_val_f1 = v_f1
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_f1": v_f1,
                "val_rec": v_rec,
                "val_acc": v_acc
            }, save_path)
            print(f"  --> Saved new best checkpoint to {save_path} (Val F1: {v_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience and epoch > warmup_epochs:
                print(f"Early stopping triggered after {patience} epochs without validation F1 improvement.")
                break

    print(f"\nTraining completed. Best validation F1: {best_val_f1:.4f}")
    return history
