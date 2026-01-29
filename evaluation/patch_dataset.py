"""Fast dataset using preprocessed patches."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import lightning.pytorch as pl

from .config import DATA_DIR, BATCH_SIZE, NUM_WORKERS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PATCHES_DIR = DATA_DIR / "patches"


class SeaIcePatchDataset(Dataset):
    """Fast dataset using preprocessed .npz patches."""

    SIC_FILL_VALUE = 255

    def __init__(
        self,
        patches_dir: Path,
        file_list: List[str],
        model_type: str = "unet",
        task: str = "binary",
    ):
        self.patches_dir = Path(patches_dir)
        self.file_list = file_list
        self.model_type = model_type
        self.task = task

        logger.info(f"SeaIcePatchDataset: {len(file_list)} patches, model={model_type}, task={task}")

    def __len__(self) -> int:
        return len(self.file_list)

    def _preprocess_sar(self, sar: np.ndarray) -> np.ndarray:
        """Preprocess SAR data."""
        if self.model_type == "terramind":
            # Map to dB-like range for TerraMind
            sar = sar * 5.0 - 15.0
        return sar.astype(np.float32)

    def _preprocess_sic(self, sic: np.ndarray) -> np.ndarray:
        """Preprocess SIC label (class indices 0-10)."""
        if self.task == "binary":
            label = np.zeros_like(sic, dtype=np.int64)
            label[sic >= 2] = 1  # Ice if SIC class >= 2 (10%+)
            label[sic == self.SIC_FILL_VALUE] = 255
            return label
        else:
            # Regression
            SIC_CLASS_TO_PCT = {0: 0, 1: 5, 2: 15, 3: 25, 4: 35, 5: 45,
                               6: 55, 7: 65, 8: 75, 9: 85, 10: 95}
            sic_float = np.zeros_like(sic, dtype=np.float32)
            valid_mask = sic != self.SIC_FILL_VALUE
            for cls, pct in SIC_CLASS_TO_PCT.items():
                sic_float[sic == cls] = pct / 100.0
            sic_float[~valid_mask] = -1.0
            return sic_float

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        patch_path = self.patches_dir / self.file_list[idx]
        data = np.load(patch_path)

        sar = self._preprocess_sar(data['sar'])
        sic = self._preprocess_sic(data['sic'])

        return {
            "image": torch.from_numpy(sar),
            "mask": torch.from_numpy(sic),
            "filename": self.file_list[idx],
        }


class SeaIcePatchDataModule(pl.LightningDataModule):
    """DataModule using preprocessed patches."""

    def __init__(
        self,
        patches_dir: Path = PATCHES_DIR,
        batch_size: int = BATCH_SIZE,
        num_workers: int = 4,  # Can use more workers with small npz files
        model_type: str = "unet",
        task: str = "binary",
    ):
        super().__init__()
        self.patches_dir = Path(patches_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.model_type = model_type
        self.task = task

        self.train_files = []
        self.val_files = []

    def setup(self, stage: Optional[str] = None):
        """Load train/val split files."""
        train_file = self.patches_dir / "train_files.txt"
        val_file = self.patches_dir / "val_files.txt"

        if train_file.exists():
            with open(train_file) as f:
                self.train_files = [line.strip() for line in f if line.strip()]
        else:
            # Fall back to listing files
            all_files = sorted([f.name for f in self.patches_dir.glob("*.npz")])
            n_val = int(len(all_files) * 0.15)
            np.random.seed(42)
            indices = np.random.permutation(len(all_files))
            self.val_files = [all_files[i] for i in indices[:n_val]]
            self.train_files = [all_files[i] for i in indices[n_val:]]

        if val_file.exists():
            with open(val_file) as f:
                self.val_files = [line.strip() for line in f if line.strip()]

        logger.info(f"PatchDataModule: train={len(self.train_files)}, val={len(self.val_files)}")

    def train_dataloader(self) -> DataLoader:
        dataset = SeaIcePatchDataset(
            patches_dir=self.patches_dir,
            file_list=self.train_files,
            model_type=self.model_type,
            task=self.task,
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        dataset = SeaIcePatchDataset(
            patches_dir=self.patches_dir,
            file_list=self.val_files,
            model_type=self.model_type,
            task=self.task,
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        # Use validation set for testing
        return self.val_dataloader()
