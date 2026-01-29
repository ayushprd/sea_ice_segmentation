"""PyTorch Dataset for AI4Arctic sea ice concentration."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset, DataLoader
import lightning.pytorch as pl

from .config import (
    DATA_DIR, TRAIN_DIR, TEST_DIR,
    SAR_VARIABLES, BATCH_SIZE, NUM_WORKERS, PATCH_SIZE, PATCHES_PER_SCENE,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SeaIceDataset(Dataset):
    """Dataset for AI4Arctic sea ice concentration prediction.

    Loads NetCDF files from the AI4Arctic Ready-to-Train dataset.
    Supports both full scene loading and random patch cropping.

    Note: SIC in AI4Arctic is stored as class indices 0-10:
        0: 0% (open water)
        1: 1-10%
        2: 10-20%
        ...
        10: 90-100%
        255: invalid/land
    """

    # Class fill value for invalid pixels
    SIC_FILL_VALUE = 255

    # SIC class to percentage midpoint mapping
    SIC_CLASS_TO_PCT = {
        0: 0, 1: 5, 2: 15, 3: 25, 4: 35, 5: 45,
        6: 55, 7: 65, 8: 75, 9: 85, 10: 95
    }

    def __init__(
        self,
        data_dir: Path,
        file_list: List[str],
        patch_size: int = PATCH_SIZE,
        patches_per_scene: int = PATCHES_PER_SCENE,
        mode: str = "train",  # "train", "val", or "test"
        model_type: str = "unet",  # "unet" or "terramind"
        use_amsr: bool = False,
        task: str = "sic",  # "sic" (regression) or "binary" (classification)
    ):
        """Initialize dataset.

        Args:
            data_dir: Directory containing NetCDF files
            file_list: List of NetCDF filenames
            patch_size: Size of patches to extract (for training)
            patches_per_scene: Number of patches to extract per scene (for training)
            mode: "train", "val", or "test"
            model_type: "unet" or "terramind"
            use_amsr: Whether to include AMSR2 data
            task: "sic" for regression, "binary" for ice/water classification
        """
        self.data_dir = Path(data_dir)
        self.file_list = file_list
        self.patch_size = patch_size
        self.patches_per_scene = patches_per_scene
        self.mode = mode
        self.model_type = model_type
        self.use_amsr = use_amsr
        self.task = task

        # SAR variables to use
        self.sar_vars = ['nersc_sar_primary', 'nersc_sar_secondary']

        # For TerraMind: map HH/HV to VV/VH-like input
        # TerraMind expects VV, VH but AI4Arctic has HH, HV
        # We'll treat them as equivalent for cross-polarization transfer learning

        logger.info(f"SeaIceDataset initialized: {len(file_list)} scenes, mode={mode}, "
                   f"model={model_type}, task={task}")

    def __len__(self) -> int:
        if self.mode == "train":
            return len(self.file_list) * self.patches_per_scene
        else:
            return len(self.file_list)

    def _load_scene(self, scene_path: Path) -> xr.Dataset:
        """Load a scene from NetCDF."""
        return xr.open_dataset(scene_path)

    def _get_random_patch(
        self,
        scene: xr.Dataset,
        max_attempts: int = 50
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Extract a random valid patch from a scene.

        Returns:
            Tuple of (sar_patch, sic_patch) or None if no valid patch found
        """
        sic = scene['SIC'].values
        h, w = sic.shape

        if h < self.patch_size or w < self.patch_size:
            return None

        for _ in range(max_attempts):
            # Random crop location
            row = np.random.randint(0, h - self.patch_size)
            col = np.random.randint(0, w - self.patch_size)

            # Extract SIC patch
            sic_patch = sic[row:row + self.patch_size, col:col + self.patch_size]

            # Check if patch has enough valid pixels (>30%)
            valid_mask = sic_patch != self.SIC_FILL_VALUE
            if valid_mask.sum() < 0.3 * self.patch_size * self.patch_size:
                continue

            # Check if patch has some ice (class >= 2) - avoid all-water patches
            has_ice = (sic_patch >= 2) & (sic_patch != self.SIC_FILL_VALUE)
            if has_ice.sum() < 0.05 * valid_mask.sum():
                # Less than 5% ice, skip to get more balanced samples
                if np.random.random() > 0.3:  # Keep 30% of water-only patches
                    continue

            # Extract SAR patch
            sar_patches = []
            for var in self.sar_vars:
                sar_data = scene[var].values[row:row + self.patch_size, col:col + self.patch_size]
                sar_patches.append(sar_data)

            sar_patch = np.stack(sar_patches, axis=0)  # (2, H, W)

            # Check for NaN in SAR data
            if np.isnan(sar_patch).any():
                continue

            return sar_patch, sic_patch

        return None

    def _preprocess_sar(self, sar: np.ndarray) -> np.ndarray:
        """Preprocess SAR data for model input.

        AI4Arctic ready-to-train data is already standard scaled.
        For TerraMind, we might need different preprocessing.
        """
        if self.model_type == "terramind":
            # TerraMind expects data in specific range
            # The ready-to-train data is standard scaled (mean=0, std=1)
            # TerraMind S1 was trained on dB values around -12 to -20
            # We'll rescale: assume std-scaled data centered around 0
            # Map to approximate dB range
            sar = sar * 5.0 - 15.0  # Rough mapping to dB-like range

        return sar.astype(np.float32)

    def _preprocess_sic(self, sic: np.ndarray) -> np.ndarray:
        """Preprocess SIC label.

        Args:
            sic: Sea ice concentration CLASS INDEX (0-10), with 255 as fill value
                 0: open water (0%)
                 1: 1-10%, 2: 10-20%, ..., 10: 90-100%
                 255: invalid/land

        Returns:
            Processed label based on task type
        """
        if self.task == "binary":
            # Binary: 0 = water (SIC class 0-1), 1 = ice (SIC class 2+), 255 = ignore
            # Class 0 = 0%, Class 1 = 1-10%
            # Consider ice if SIC class >= 2 (i.e., >= 10% concentration)
            label = np.zeros_like(sic, dtype=np.int64)
            label[sic >= 2] = 1  # Ice if SIC class >= 2 (10%+)
            label[sic == self.SIC_FILL_VALUE] = 255
            return label
        else:
            # Regression: convert class to percentage midpoint, scale to 0-1
            sic_float = np.zeros_like(sic, dtype=np.float32)
            valid_mask = sic != self.SIC_FILL_VALUE
            for cls, pct in self.SIC_CLASS_TO_PCT.items():
                sic_float[sic == cls] = pct / 100.0
            sic_float[~valid_mask] = -1.0  # Mark invalid
            return sic_float

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self.mode == "train":
            # Map idx to scene
            scene_idx = idx // self.patches_per_scene
        else:
            # For val/test, use direct index
            scene_idx = idx

        scene_path = self.data_dir / self.file_list[scene_idx]
        scene = self._load_scene(scene_path)

        # Always use random patches for consistent sizing
        result = self._get_random_patch(scene)

        if result is None:
            # Fallback: try another random scene
            fallback_idx = np.random.randint(0, len(self.file_list))
            scene_path = self.data_dir / self.file_list[fallback_idx]
            scene.close()
            scene = self._load_scene(scene_path)
            result = self._get_random_patch(scene)

        if result is None:
            # Return zeros as last resort
            sar = np.zeros((2, self.patch_size, self.patch_size), dtype=np.float32)
            sic = np.full((self.patch_size, self.patch_size), 255, dtype=np.int64)
        else:
            sar, sic = result
            sar = self._preprocess_sar(sar)
            sic = self._preprocess_sic(sic)

        scene.close()

        # Convert to tensors
        sar_tensor = torch.from_numpy(sar)

        if self.task == "binary":
            sic_tensor = torch.from_numpy(sic)
        else:
            sic_tensor = torch.from_numpy(sic)

        return {
            "image": sar_tensor,
            "mask": sic_tensor,
            "filename": self.file_list[idx % len(self.file_list)],
        }


class SeaIceDataModule(pl.LightningDataModule):
    """Lightning DataModule for sea ice dataset."""

    def __init__(
        self,
        data_dir: Path = DATA_DIR,
        batch_size: int = BATCH_SIZE,
        num_workers: int = NUM_WORKERS,
        patch_size: int = PATCH_SIZE,
        model_type: str = "unet",
        task: str = "binary",
        val_split: float = 0.15,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.patch_size = patch_size
        self.model_type = model_type
        self.task = task
        self.val_split = val_split

        self.train_files = []
        self.val_files = []
        self.test_files = []

    def setup(self, stage: Optional[str] = None):
        """Set up train/val/test splits."""
        # Find all NetCDF files
        train_dir = self.data_dir / "train"
        test_dir = self.data_dir / "test"

        if train_dir.exists():
            all_train_files = sorted([f.name for f in train_dir.glob("*.nc")])
        else:
            all_train_files = []

        if test_dir.exists():
            self.test_files = sorted([f.name for f in test_dir.glob("*.nc")])
        else:
            self.test_files = []

        # Split train into train/val
        n_train = len(all_train_files)
        n_val = int(n_train * self.val_split)

        # Use fixed seed for reproducible splits
        np.random.seed(42)
        indices = np.random.permutation(n_train)

        val_indices = indices[:n_val]
        train_indices = indices[n_val:]

        self.train_files = [all_train_files[i] for i in train_indices]
        self.val_files = [all_train_files[i] for i in val_indices]

        logger.info(f"DataModule setup: train={len(self.train_files)}, "
                   f"val={len(self.val_files)}, test={len(self.test_files)}")

    def train_dataloader(self) -> DataLoader:
        dataset = SeaIceDataset(
            data_dir=self.data_dir / "train",
            file_list=self.train_files,
            patch_size=self.patch_size,
            patches_per_scene=PATCHES_PER_SCENE,
            mode="train",
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
        dataset = SeaIceDataset(
            data_dir=self.data_dir / "train",
            file_list=self.val_files,
            patch_size=self.patch_size,
            mode="val",
            model_type=self.model_type,
            task=self.task,
        )
        return DataLoader(
            dataset,
            batch_size=1,  # Full scenes
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        # For test, we use validation set since test has no labels
        dataset = SeaIceDataset(
            data_dir=self.data_dir / "train",
            file_list=self.val_files,
            patch_size=self.patch_size,
            mode="test",
            model_type=self.model_type,
            task=self.task,
        )
        return DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
