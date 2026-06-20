"""NIH ChestX-ray14 multi-label dataset + I/O-optimized DataLoader.

Reads data/metadata/Data_Entry_2017.csv. Builds a filename->path index across
the 12 image folders. No argparse; configured via configs/config.py.

Imported by train_classifier.py, cam_pipeline.py, smoke_test_local.py.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# 14 official pathology classes ("No Finding" excluded from the label vector).
DISEASE_CLASSES = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion",
    "Emphysema", "Fibrosis", "Hernia", "Infiltration", "Mass",
    "Nodule", "Pleural_Thickening", "Pneumonia", "Pneumothorax",
]
CLASS_TO_IDX = {c: i for i, c in enumerate(DISEASE_CLASSES)}
NUM_CLASSES = len(DISEASE_CLASSES)


def build_image_index(data_dir: Path) -> dict[str, str]:
    """Map 'XXXXXXXX_YYY.png' -> absolute path across all images_*/images dirs."""
    index: dict[str, str] = {}
    for sub in sorted(Path(data_dir).glob("images_*/images")):
        for p in sub.glob("*.png"):
            index[p.name] = str(p)
    return index


def labels_to_vector(finding_labels: str) -> np.ndarray:
    """'Cardiomegaly|Emphysema' -> multi-hot float32 vector of length 14."""
    vec = np.zeros(NUM_CLASSES, dtype=np.float32)
    if finding_labels and finding_labels != "No Finding":
        for name in finding_labels.split("|"):
            idx = CLASS_TO_IDX.get(name.strip())
            if idx is not None:
                vec[idx] = 1.0
    return vec


def load_dataframe(data_entry_csv: Path, image_index: dict[str, str]) -> pd.DataFrame:
    """Load Data_Entry_2017.csv, attach resolved path + multi-hot labels.
    Rows whose image file is not found on disk are dropped (robust to partial data).
    """
    df = pd.read_csv(data_entry_csv)
    df = df.rename(columns={"Image Index": "image", "Finding Labels": "labels",
                            "Patient ID": "patient_id"})
    df["path"] = df["image"].map(image_index)
    missing = df["path"].isna().sum()
    if missing:
        df = df.dropna(subset=["path"]).reset_index(drop=True)
    return df


def patient_wise_split(df: pd.DataFrame, val_split: float, seed: int):
    """Split into train/val by Patient ID to avoid leakage across the same patient."""
    rng = np.random.default_rng(seed)
    patients = df["patient_id"].unique()
    rng.shuffle(patients)
    n_val = int(len(patients) * val_split)
    val_patients = set(patients[:n_val].tolist())
    is_val = df["patient_id"].isin(val_patients)
    return df[~is_val].reset_index(drop=True), df[is_val].reset_index(drop=True)


class NIHChestDataset(Dataset):
    """Returns (image_tensor[C,H,W], label_vector[14], image_name)."""

    def __init__(self, df: pd.DataFrame, image_size: int, mean, std,
                 train: bool, in_chans: int = 3):
        self.df = df.reset_index(drop=True)
        self.image_size = image_size
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        self.train = train
        self.in_chans = in_chans

    def __len__(self) -> int:
        return len(self.df)

    def _load_image(self, path: str) -> np.ndarray:
        # Some NIH PNGs are RGBA/L; force consistent channel count.
        mode = "RGB" if self.in_chans == 3 else "L"
        img = Image.open(path).convert(mode)
        if img.size != (self.image_size, self.image_size):
            img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        return np.asarray(img, dtype=np.float32) / 255.0

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        arr = self._load_image(row["path"])           # H,W,C or H,W
        if arr.ndim == 2:
            arr = arr[..., None]
        # light train-time augmentation: random horizontal flip
        if self.train and np.random.rand() < 0.5:
            arr = arr[:, ::-1, :].copy()
        arr = (arr - self.mean) / self.std
        tensor = torch.from_numpy(arr.transpose(2, 0, 1)).contiguous()
        label = torch.from_numpy(labels_to_vector(row["labels"]))
        return tensor, label, row["image"]


def compute_pos_weight(df: pd.DataFrame) -> torch.Tensor:
    """pos_weight = neg/pos per class for BCEWithLogitsLoss imbalance handling."""
    mat = np.stack([labels_to_vector(s) for s in df["labels"].values])
    pos = mat.sum(axis=0)
    neg = len(mat) - pos
    pos = np.clip(pos, 1.0, None)  # avoid div-by-zero
    return torch.from_numpy((neg / pos).astype(np.float32))


def make_dataloader(dataset: NIHChestDataset, batch_size: int, num_workers: int,
                    shuffle: bool, pin_memory: bool = True,
                    persistent_workers: bool = True, prefetch_factor: int = 4,
                    sampler=None) -> DataLoader:
    """I/O-optimized loader: multi-worker, pinned memory, async prefetch.

    persistent_workers/prefetch_factor only valid when num_workers > 0.
    """
    kwargs = dict(
        batch_size=batch_size,
        shuffle=(shuffle and sampler is None),
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=shuffle,
        sampler=sampler,
    )
    if num_workers > 0:
        kwargs["persistent_workers"] = persistent_workers
        kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(dataset, **kwargs)
