"""Preprocess AI4Arctic scenes into individual patch files for faster training."""

import argparse
import logging
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import xarray as xr
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
PATCH_SIZE = 224
PATCHES_PER_SCENE = 20  # Extract more patches per scene
MIN_VALID_FRACTION = 0.3  # Minimum valid pixels
SIC_FILL_VALUE = 255

DATA_DIR = Path("/mnt/data/benchmark/sea_ice/data")
TRAIN_DIR = DATA_DIR / "train"
PATCHES_DIR = DATA_DIR / "patches"
PATCHES_DIR.mkdir(parents=True, exist_ok=True)


def extract_patches_from_scene(scene_path: Path, output_dir: Path, patches_per_scene: int = PATCHES_PER_SCENE):
    """Extract random patches from a single scene."""
    try:
        scene = xr.open_dataset(scene_path)
        scene_name = scene_path.stem

        # Get dimensions
        sic = scene['SIC'].values
        h, w = sic.shape

        if h < PATCH_SIZE or w < PATCH_SIZE:
            scene.close()
            return 0

        patches_saved = 0
        attempts = 0
        max_attempts = patches_per_scene * 10

        while patches_saved < patches_per_scene and attempts < max_attempts:
            attempts += 1

            # Random crop location
            row = np.random.randint(0, h - PATCH_SIZE)
            col = np.random.randint(0, w - PATCH_SIZE)

            # Extract SIC patch
            sic_patch = sic[row:row + PATCH_SIZE, col:col + PATCH_SIZE]

            # Check validity
            valid_mask = sic_patch != SIC_FILL_VALUE
            if valid_mask.sum() < MIN_VALID_FRACTION * PATCH_SIZE * PATCH_SIZE:
                continue

            # Extract SAR patches
            sar_hh = scene['nersc_sar_primary'].values[row:row + PATCH_SIZE, col:col + PATCH_SIZE]
            sar_hv = scene['nersc_sar_secondary'].values[row:row + PATCH_SIZE, col:col + PATCH_SIZE]

            # Check for NaN
            if np.isnan(sar_hh).any() or np.isnan(sar_hv).any():
                continue

            # Stack SAR channels
            sar_patch = np.stack([sar_hh, sar_hv], axis=0).astype(np.float32)

            # Save patch
            patch_name = f"{scene_name}_patch{patches_saved:03d}"
            patch_path = output_dir / f"{patch_name}.npz"

            np.savez_compressed(
                patch_path,
                sar=sar_patch,
                sic=sic_patch.astype(np.uint8),
            )

            patches_saved += 1

        scene.close()
        return patches_saved

    except Exception as e:
        logger.error(f"Error processing {scene_path}: {e}")
        return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=Path, default=TRAIN_DIR)
    parser.add_argument('--output-dir', type=Path, default=PATCHES_DIR)
    parser.add_argument('--patches-per-scene', type=int, default=PATCHES_PER_SCENE)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Find all scenes
    scene_files = sorted(args.input_dir.glob("*.nc"))
    logger.info(f"Found {len(scene_files)} scene files")

    # Process scenes
    total_patches = 0

    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    extract_patches_from_scene,
                    scene_path,
                    args.output_dir,
                    args.patches_per_scene
                ): scene_path for scene_path in scene_files
            }

            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing scenes"):
                patches = future.result()
                total_patches += patches
    else:
        for scene_path in tqdm(scene_files, desc="Processing scenes"):
            patches = extract_patches_from_scene(
                scene_path, args.output_dir, args.patches_per_scene
            )
            total_patches += patches

    logger.info(f"Extracted {total_patches} patches to {args.output_dir}")

    # Create train/val split file
    patch_files = sorted(args.output_dir.glob("*.npz"))
    n_patches = len(patch_files)
    n_val = int(n_patches * 0.15)

    np.random.seed(42)
    indices = np.random.permutation(n_patches)
    val_indices = set(indices[:n_val])

    train_files = []
    val_files = []
    for i, f in enumerate(patch_files):
        if i in val_indices:
            val_files.append(f.name)
        else:
            train_files.append(f.name)

    # Save split files
    with open(args.output_dir / "train_files.txt", 'w') as f:
        f.write('\n'.join(train_files))
    with open(args.output_dir / "val_files.txt", 'w') as f:
        f.write('\n'.join(val_files))

    logger.info(f"Split: {len(train_files)} train, {len(val_files)} val")


if __name__ == '__main__':
    main()
