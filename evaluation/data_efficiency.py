"""Data efficiency experiment: train with different fractions of data."""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import lightning.pytorch as pl
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from sea_ice.evaluation.config import (
    CHECKPOINTS_DIR, RESULTS_DIR, LOGS_DIR,
    LEARNING_RATE, WEIGHT_DECAY, BATCH_SIZE,
)
from sea_ice.evaluation.patch_dataset import SeaIcePatchDataModule, PATCHES_DIR
from sea_ice.evaluation.models import UNet, TerraMindS1Segmentation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_iou(pred: torch.Tensor, target: torch.Tensor, num_classes: int = 2, ignore_index: int = 255):
    """Compute IoU for each class."""
    ious = []
    pred = pred.reshape(-1)
    target = target.reshape(-1)

    for cls in range(num_classes):
        pred_cls = pred == cls
        target_cls = target == cls
        valid = target != ignore_index

        pred_cls = pred_cls & valid
        target_cls = target_cls & valid

        intersection = (pred_cls & target_cls).sum().float()
        union = (pred_cls | target_cls).sum().float()

        if union > 0:
            ious.append((intersection / union).item())
        else:
            ious.append(float('nan'))

    return ious


class SeaIceTask(pl.LightningModule):
    """Lightning module for sea ice segmentation."""

    def __init__(self, model: nn.Module, learning_rate: float = 1e-4):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.criterion = nn.CrossEntropyLoss(ignore_index=255)
        self.validation_outputs = []
        self.test_outputs = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images = batch['image']
        masks = batch['mask']
        logits = self(images)
        loss = self.criterion(logits, masks)
        self.log('train/loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images = batch['image']
        masks = batch['mask']
        logits = self(images)
        loss = self.criterion(logits, masks)
        preds = logits.argmax(dim=1)
        ious = compute_iou(preds, masks, 2)
        self.validation_outputs.append({
            'loss': loss.item(),
            'iou_water': ious[0],
            'iou_ice': ious[1],
        })
        return loss

    def on_validation_epoch_end(self):
        losses = [o['loss'] for o in self.validation_outputs]
        ious_water = [o['iou_water'] for o in self.validation_outputs if not np.isnan(o['iou_water'])]
        ious_ice = [o['iou_ice'] for o in self.validation_outputs if not np.isnan(o['iou_ice'])]

        mean_loss = np.mean(losses)
        mean_iou_water = np.mean(ious_water) if ious_water else 0.0
        mean_iou_ice = np.mean(ious_ice) if ious_ice else 0.0
        mean_iou = (mean_iou_water + mean_iou_ice) / 2

        self.log('val/loss', mean_loss, prog_bar=True)
        self.log('val/iou_mean', mean_iou, prog_bar=True)
        self.validation_outputs.clear()

    def test_step(self, batch, batch_idx):
        images = batch['image']
        masks = batch['mask']
        logits = self(images)
        loss = self.criterion(logits, masks)
        preds = logits.argmax(dim=1)
        ious = compute_iou(preds, masks, 2)

        valid = masks != 255
        correct = ((preds == masks) & valid).sum().item()
        total = valid.sum().item()

        self.test_outputs.append({
            'loss': loss.item(),
            'iou_water': ious[0],
            'iou_ice': ious[1],
            'correct': correct,
            'total': total,
        })
        return loss

    def on_test_epoch_end(self):
        losses = [o['loss'] for o in self.test_outputs]
        ious_water = [o['iou_water'] for o in self.test_outputs if not np.isnan(o['iou_water'])]
        ious_ice = [o['iou_ice'] for o in self.test_outputs if not np.isnan(o['iou_ice'])]

        mean_loss = np.mean(losses)
        mean_iou_water = np.mean(ious_water) if ious_water else 0.0
        mean_iou_ice = np.mean(ious_ice) if ious_ice else 0.0
        mean_iou = (mean_iou_water + mean_iou_ice) / 2

        total_correct = sum(o['correct'] for o in self.test_outputs)
        total_pixels = sum(o['total'] for o in self.test_outputs)
        accuracy = total_correct / total_pixels if total_pixels > 0 else 0.0

        self.log('test/loss', mean_loss)
        self.log('test/iou_water', mean_iou_water)
        self.log('test/iou_ice', mean_iou_ice)
        self.log('test/iou_mean', mean_iou)
        self.log('test/accuracy', accuracy)
        self.test_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=WEIGHT_DECAY,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs, eta_min=1e-6
        )
        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'interval': 'epoch'}}


class SubsetDataModule(SeaIcePatchDataModule):
    """DataModule that uses a fraction of training data."""

    def __init__(self, fraction: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.fraction = fraction

    def setup(self, stage=None):
        super().setup(stage)
        # Subsample training data
        n_train = len(self.train_files)
        n_subset = max(1, int(n_train * self.fraction))

        np.random.seed(42)  # Reproducible subsets
        indices = np.random.permutation(n_train)[:n_subset]
        self.train_files = [self.train_files[i] for i in sorted(indices)]

        logger.info(f"Using {len(self.train_files)}/{n_train} training samples ({self.fraction*100:.0f}%)")


def run_experiment(model_name: str, fraction: float, gpu: int, max_epochs: int = 20):
    """Run a single experiment."""
    seed_everything(42)

    # Build model
    if model_name == 'unet':
        model = UNet(in_channels=2, num_classes=2, base_channels=16)
        lr = LEARNING_RATE
    else:
        model = TerraMindS1Segmentation(
            num_classes=2, hidden_dim=256, dropout=0.1, freeze_backbone=False
        )
        lr = 1e-5  # Lower LR for finetuning

    # Data
    data_module = SubsetDataModule(
        fraction=fraction,
        batch_size=BATCH_SIZE,
        num_workers=4,
        model_type=model_name,
        task='binary',
    )

    # Task
    task = SeaIceTask(model=model, learning_rate=lr)

    # Checkpoint
    exp_name = f"{model_name}_frac{int(fraction*100)}"
    checkpoint_dir = CHECKPOINTS_DIR / "data_efficiency" / exp_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='best-{epoch:02d}-{val/iou_mean:.4f}',
        monitor='val/iou_mean',
        mode='max',
        save_top_k=1,
    )

    early_stop = EarlyStopping(monitor='val/iou_mean', patience=10, mode='max')

    # Trainer
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=[gpu],
        max_epochs=max_epochs,
        callbacks=[checkpoint_callback, early_stop],
        precision='16-mixed',
        gradient_clip_val=1.0,
        enable_progress_bar=True,
        logger=False,
    )

    # Train
    logger.info(f"Training {model_name} with {fraction*100:.0f}% data...")
    trainer.fit(task, datamodule=data_module)

    # Test
    results = trainer.test(task, datamodule=data_module, ckpt_path='best')

    return results[0] if results else {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--fractions', type=str, default='0.05,0.1,0.25,0.5,1.0',
                       help='Comma-separated data fractions')
    args = parser.parse_args()

    fractions = [float(f) for f in args.fractions.split(',')]
    models = ['unet', 'terramind']

    all_results = {}

    for model_name in models:
        all_results[model_name] = {}
        for fraction in fractions:
            logger.info(f"\n{'='*50}")
            logger.info(f"Running {model_name} with {fraction*100:.0f}% data")
            logger.info(f"{'='*50}")

            results = run_experiment(model_name, fraction, args.gpu, args.epochs)
            all_results[model_name][str(fraction)] = results

            # Save intermediate results
            output_path = RESULTS_DIR / "data_efficiency_results.json"
            with open(output_path, 'w') as f:
                json.dump(all_results, f, indent=2)

    # Print summary
    print("\n" + "="*60)
    print("DATA EFFICIENCY RESULTS")
    print("="*60)
    print(f"{'Fraction':<10} {'UNet mIoU':<15} {'TerraMind mIoU':<15} {'Δ':<10}")
    print("-"*60)

    for fraction in fractions:
        unet_iou = all_results['unet'].get(str(fraction), {}).get('test/iou_mean', 0) * 100
        tm_iou = all_results['terramind'].get(str(fraction), {}).get('test/iou_mean', 0) * 100
        delta = tm_iou - unet_iou
        print(f"{fraction*100:>6.0f}%    {unet_iou:>10.1f}%      {tm_iou:>10.1f}%      {delta:>+6.1f}%")

    print("="*60)

    logger.info(f"Results saved to {RESULTS_DIR / 'data_efficiency_results.json'}")


if __name__ == '__main__':
    main()
