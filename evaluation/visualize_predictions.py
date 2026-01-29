#!/usr/bin/env python3
"""Visualize sea ice predictions: input, predictions, and ground truth."""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sea_ice.evaluation.patch_dataset import SeaIcePatchDataset, SeaIcePatchDataModule


def load_unet(checkpoint_path, device='cuda'):
    """Load trained UNet model."""
    from sea_ice.evaluation.models.unet import UNet
    model = UNet(in_channels=2, num_classes=2)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    state_dict = {}
    for key, value in checkpoint['state_dict'].items():
        if key.startswith('model.'):
            state_dict[key[6:]] = value
        else:
            state_dict[key] = value

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def load_terramind(checkpoint_path, device='cuda'):
    """Load trained TerraMind model."""
    from sea_ice.evaluation.models.terramind_s1 import TerraMindS1Segmentation

    freeze_backbone = 'probing' in str(checkpoint_path)
    model = TerraMindS1Segmentation(num_classes=2, freeze_backbone=freeze_backbone)

    checkpoint = torch.load(checkpoint_path, map_location=device)

    state_dict = {}
    for key, value in checkpoint['state_dict'].items():
        if key.startswith('model.'):
            state_dict[key[6:]] = value
        else:
            state_dict[key] = value

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def find_diverse_samples(dataset, num_samples=6):
    """Find samples with good mix of ice and water."""
    good_indices = []

    for idx in range(len(dataset)):
        sample = dataset[idx]
        mask = sample['mask'].numpy()
        valid_mask = mask != 255

        if valid_mask.sum() < 100:
            continue

        ice_fraction = (mask[valid_mask] == 1).mean()

        # Look for samples with 20-80% ice
        if 0.2 < ice_fraction < 0.8:
            good_indices.append((idx, ice_fraction))

    # Sort by how close to 50% ice
    good_indices.sort(key=lambda x: abs(x[1] - 0.5))

    if len(good_indices) < num_samples:
        # Add some pure ice and pure water samples
        for idx in range(len(dataset)):
            if len(good_indices) >= num_samples:
                break
            sample = dataset[idx]
            mask = sample['mask'].numpy()
            valid_mask = mask != 255
            if valid_mask.sum() < 100:
                continue
            ice_fraction = (mask[valid_mask] == 1).mean()
            if ice_fraction > 0.9 or ice_fraction < 0.1:
                if idx not in [x[0] for x in good_indices]:
                    good_indices.append((idx, ice_fraction))

    return [x[0] for x in good_indices[:num_samples]]


def visualize_samples(num_samples=6, output_path='sea_ice/evaluation/results/predictions_viz.png'):
    """Create visualization of predictions."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load models
    print("Loading models...")
    unet = load_unet('sea_ice/evaluation/checkpoints/unet_baseline/best-epoch=17-val/iou_mean=0.8953.ckpt', device)
    terramind = load_terramind('sea_ice/evaluation/checkpoints/terramind_finetune/best-epoch=07-val/iou_mean=0.9237.ckpt', device)

    # Load validation data
    print("Loading data...")
    data_module = SeaIcePatchDataModule(model_type='unet', task='binary')
    data_module.setup()

    # Create datasets for both models
    val_dataset_unet = SeaIcePatchDataset(
        patches_dir=data_module.patches_dir,
        file_list=data_module.val_files,
        model_type='unet',
        task='binary',
    )

    val_dataset_tm = SeaIcePatchDataset(
        patches_dir=data_module.patches_dir,
        file_list=data_module.val_files,
        model_type='terramind',
        task='binary',
    )

    # Find diverse samples (use UNet dataset for this)
    print("Finding diverse samples...")
    indices = find_diverse_samples(val_dataset_unet, num_samples)
    print(f"Selected indices: {indices}")

    # Create figure
    fig, axes = plt.subplots(num_samples, 5, figsize=(15, 3*num_samples))

    if num_samples == 1:
        axes = axes.reshape(1, -1)

    column_titles = ['S1 HH (SAR Input)', 'S1 HV (SAR Input)', 'Ground Truth', 'UNet Prediction', 'TerraMind Prediction']

    for row, idx in enumerate(indices):
        sample_unet = val_dataset_unet[idx]
        sample_tm = val_dataset_tm[idx]

        image_unet = sample_unet['image']
        image_tm = sample_tm['image']
        label = sample_unet['mask']

        # Run inference
        with torch.no_grad():
            # UNet prediction
            unet_logits = unet(image_unet.unsqueeze(0).to(device))
            unet_pred = unet_logits.argmax(dim=1).squeeze().cpu().numpy()

            # TerraMind prediction (with proper preprocessing)
            terramind_logits = terramind(image_tm.unsqueeze(0).to(device))
            terramind_pred = terramind_logits.argmax(dim=1).squeeze().cpu().numpy()

        image_np = image_unet.numpy()  # Use UNet version for display
        label_np = label.numpy()

        valid_mask = label_np != 255

        # Plot S1 HH
        ax = axes[row, 0]
        ax.imshow(image_np[0], cmap='gray', vmin=-2, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        if row == 0:
            ax.set_title(column_titles[0], fontsize=11, fontweight='bold')

        # Plot S1 HV
        ax = axes[row, 1]
        ax.imshow(image_np[1], cmap='gray', vmin=-2, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        if row == 0:
            ax.set_title(column_titles[1], fontsize=11, fontweight='bold')

        # Custom colormap: water=blue, ice=white
        from matplotlib.colors import ListedColormap
        ice_cmap = ListedColormap(['#1E90FF', '#FFFFFF'])

        # Plot Ground Truth
        ax = axes[row, 2]
        gt_display = np.ma.masked_where(~valid_mask, label_np)
        ax.imshow(gt_display, cmap=ice_cmap, vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        if row == 0:
            ax.set_title(column_titles[2], fontsize=11, fontweight='bold')

        # Plot UNet prediction
        ax = axes[row, 3]
        unet_display = np.ma.masked_where(~valid_mask, unet_pred)
        ax.imshow(unet_display, cmap=ice_cmap, vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        if row == 0:
            ax.set_title(column_titles[3], fontsize=11, fontweight='bold')

        # Plot TerraMind prediction
        ax = axes[row, 4]
        tm_display = np.ma.masked_where(~valid_mask, terramind_pred)
        ax.imshow(tm_display, cmap=ice_cmap, vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        if row == 0:
            ax.set_title(column_titles[4], fontsize=11, fontweight='bold')

        # Calculate IoU scores
        valid_gt = label_np[valid_mask]
        valid_unet = unet_pred[valid_mask]
        valid_tm = terramind_pred[valid_mask]

        def calc_miou(pred, target):
            ious = []
            for cls in [0, 1]:
                intersection = ((pred == cls) & (target == cls)).sum()
                union = ((pred == cls) | (target == cls)).sum()
                if union > 0:
                    ious.append(intersection / union)
            return np.mean(ious) if ious else 0

        unet_miou = calc_miou(valid_unet, valid_gt)
        tm_miou = calc_miou(valid_tm, valid_gt)

        axes[row, 3].set_xlabel(f'mIoU: {unet_miou:.3f}', fontsize=10)
        axes[row, 4].set_xlabel(f'mIoU: {tm_miou:.3f}', fontsize=10)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#1E90FF', edgecolor='black', label='Water'),
        Patch(facecolor='#FFFFFF', edgecolor='black', label='Sea Ice'),
    ]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98), fontsize=10)

    plt.suptitle('Sea Ice Segmentation from Sentinel-1 SAR', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
    print(f"Saved visualization to {output_path}")

    plt.close()


if __name__ == '__main__':
    visualize_samples(num_samples=6)
