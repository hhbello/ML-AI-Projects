"""
Dataset loading and preprocessing module for SDNET2018 concrete crack dataset.
Designed for Structural Health Monitoring (SHM) across Bridge Decks (D), Pavements (P), and Walls (W).
"""

import os
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Union
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


def find_sdnet_root(candidate_path: Optional[str] = None) -> Path:
    """
    Locates the root directory containing SDNET2018 subdirectories (D, P, W).
    Searches candidate paths, parent directory, and standard project locations.
    """
    candidates = []
    if candidate_path:
        candidates.append(Path(candidate_path))
    if os.environ.get("SDNET_DATA_DIR"):
        candidates.append(Path(os.environ["SDNET_DATA_DIR"]))

    # Current directory, parent directory, data/ subfolder
    curr = Path.cwd()
    candidates.extend([
        curr.parent,                      
        curr / "data",
        curr / "data" / "SDNET2018",
        curr
    ])

    for p in candidates:
        if p.exists() and (p / "D").exists() and (p / "P").exists() and (p / "W").exists():
            return p.resolve()

    # Fallback to known paths in workspace
    for known in [
        Path(r"D:\HafeezBello\HHB\JUPYTER\CompVis_Projects\SDNET2018"),
        Path(r"D:\HafeezBello\HHB\JUPYTER\Computer Vison Projects"),
    ]:
        if known.exists() and (known / "D").exists():
            return known

    raise FileNotFoundError(
        "Could not automatically locate SDNET2018 dataset (folders D, P, W). "
        "Please provide data_dir explicitly or set SDNET_DATA_DIR environment variable."
    )


def build_sdnet_dataframe(data_dir: Optional[str] = None, structure_type: str = "all") -> pd.DataFrame:
    """
    Scans SDNET2018 directory structure and returns a clean metadata DataFrame.

    Args:
        data_dir: Path to dataset root containing 'D', 'P', 'W'.
        structure_type: 'all', 'deck' ('D'), 'pavement' ('P'), or 'wall' ('W').

    Returns:
        pd.DataFrame with columns: ['filepath', 'filename', 'structure', 'subfolder', 'label', 'label_name']
    """
    root = find_sdnet_root(data_dir)
    records = []

    # Mapping of folder structure in SDNET2018
    # D: CD (cracked deck), UD (uncracked deck)
    # P: CP (cracked pavement), UP (uncracked pavement)
    # W: CW (cracked wall), UW (uncracked wall)
    structures_to_scan = []
    st_lower = structure_type.lower()
    if st_lower in ["all", "both"]:
        structures_to_scan = [("D", "Deck"), ("P", "Pavement"), ("W", "Wall")]
    elif st_lower in ["deck", "d", "bridge"]:
        structures_to_scan = [("D", "Deck")]
    elif st_lower in ["pavement", "p", "road"]:
        structures_to_scan = [("P", "Pavement")]
    elif st_lower in ["wall", "w"]:
        structures_to_scan = [("W", "Wall")]
    else:
        raise ValueError(f"Unknown structure_type: {structure_type}. Choose from 'all', 'deck', 'pavement', 'wall'.")

    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}

    for folder_code, structure_name in structures_to_scan:
        folder_path = root / folder_code
        if not folder_path.exists():
            continue

        for subfolder in folder_path.iterdir():
            if not subfolder.is_dir():
                continue

            sub_name = subfolder.name.upper()
            # Determine crack presence: prefix C is Cracked, U is Uncracked
            is_cracked = sub_name.startswith("C")
            label = 1 if is_cracked else 0
            label_name = "Cracked" if is_cracked else "Uncracked"

            for file_path in subfolder.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in valid_extensions:
                    records.append({
                        "filepath": str(file_path),
                        "filename": file_path.name,
                        "structure": structure_name,
                        "subfolder": sub_name,
                        "label": label,
                        "label_name": label_name
                    })

    df = pd.DataFrame(records)
    if df.empty:
        raise RuntimeError(f"No valid images found in {root} for structure_type='{structure_type}'.")
    return df


class SDNETDataset(Dataset):
    """
    PyTorch Dataset for SDNET2018 concrete surface images.
    """
    def __init__(self, df: pd.DataFrame, transform: Optional[transforms.Compose] = None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Union[str, int]]]:
        row = self.df.iloc[idx]
        img_path = row["filepath"]

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            # Fallback if image corrupted
            raise RuntimeError(f"Error loading image {img_path}: {e}")

        if self.transform is not None:
            image = self.transform(image)

        label = torch.tensor(row["label"], dtype=torch.float32)
        metadata = {
            "filepath": img_path,
            "structure": row["structure"],
            "subfolder": row["subfolder"],
            "label_name": row["label_name"]
        }

        return image, label, metadata


def get_default_transforms(img_size: int = 224) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Returns data transforms for training and validation/testing.
    Training includes domain-informed augmentations: flips, rotations, and lighting adjustments.
    """
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform


def create_stratified_splits(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits DataFrame into train, val, and test subsets stratified by (structure + label)
    to preserve both the class balance and concrete structure domain distributions.
    """
    assert np.isclose(train_ratio + val_ratio + test_ratio, 1.0), "Ratios must sum to 1.0"

    # Stratify by combined key (e.g. 'Deck_Cracked', 'Pavement_Uncracked')
    df = df.copy()
    df["strat_key"] = df["structure"] + "_" + df["label_name"]

    train_df, temp_df = train_test_split(
        df,
        test_size=(val_ratio + test_ratio),
        stratify=df["strat_key"],
        random_state=random_state
    )

    relative_test_ratio = test_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test_ratio,
        stratify=temp_df["strat_key"],
        random_state=random_state
    )

    train_df = train_df.drop(columns=["strat_key"]).reset_index(drop=True)
    val_df = val_df.drop(columns=["strat_key"]).reset_index(drop=True)
    test_df = test_df.drop(columns=["strat_key"]).reset_index(drop=True)

    return train_df, val_df, test_df


def get_dataloaders(
    data_dir: Optional[str] = None,
    structure_type: str = "all",
    batch_size: int = 32,
    img_size: int = 224,
    num_workers: int = 0,
    subsample_frac: Optional[float] = None,
    random_state: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, Union[int, float, pd.DataFrame]]]:
    """
    End-to-end data pipeline builder.

    Args:
        data_dir: Dataset root folder path.
        structure_type: 'all', 'deck', 'pavement', or 'wall'.
        batch_size: Batch size for DataLoader.
        img_size: Target image resolution (default 224 for ResNet/MobileNet).
        num_workers: DataLoader worker threads (0 recommended on Windows).
        subsample_frac: Optional float in (0, 1] to train on a fraction for rapid iteration.
        random_state: Seed for reproducibility.

    Returns:
        (train_loader, val_loader, test_loader, info_dict)
    """
    df = build_sdnet_dataframe(data_dir=data_dir, structure_type=structure_type)

    if subsample_frac is not None and 0.0 < subsample_frac < 1.0:
        df, _ = train_test_split(
            df,
            train_size=subsample_frac,
            stratify=df["structure"] + "_" + df["label_name"],
            random_state=random_state
        )
        df = df.reset_index(drop=True)

    train_df, val_df, test_df = create_stratified_splits(
        df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, random_state=random_state
    )

    train_tf, val_tf = get_default_transforms(img_size=img_size)

    train_dataset = SDNETDataset(train_df, transform=train_tf)
    val_dataset = SDNETDataset(val_df, transform=val_tf)
    test_dataset = SDNETDataset(test_df, transform=val_tf)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    num_cracked = int((train_df["label"] == 1).sum())
    num_uncracked = int((train_df["label"] == 0).sum())
    pos_weight = num_uncracked / max(num_cracked, 1)

    info = {
        "total_images": len(df),
        "train_size": len(train_df),
        "val_size": len(val_df),
        "test_size": len(test_df),
        "train_cracked": num_cracked,
        "train_uncracked": num_uncracked,
        "pos_weight": pos_weight,
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
    }

    return train_loader, val_loader, test_loader, info


if __name__ == "__main__":
    print("Testing SDNET dataset loader...")
    try:
        train_loader, val_loader, test_loader, info = get_dataloaders(
            structure_type="deck", batch_size=16, subsample_frac=0.1
        )
        print("Data loaded successfully!")
        print(f"Total: {info['total_images']}, Train: {info['train_size']}, Val: {info['val_size']}, Test: {info['test_size']}")
        print(f"Calculated pos_weight for loss: {info['pos_weight']:.2f}")
        for imgs, lbls, meta in train_loader:
            print(f"Batch images shape: {imgs.shape}, labels shape: {lbls.shape}")
            break
    except Exception as e:
        print(f"Error during dataset test: {e}")
