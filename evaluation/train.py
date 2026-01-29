"""Training script for sea ice concentration prediction."""

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger

from sea_ice.evaluation.config import (
    CHECKPOINTS_DIR, RESULTS_DIR, LOGS_DIR,
    MAX_EPOCHS_PROBING, MAX_EPOCHS_FINETUNE,
    LEARNING_RATE, WEIGHT_DECAY, BATCH_SIZE,
)
from sea_ice.evaluation.patch_dataset import SeaIcePatchDataModule
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


class SeaIceSegmentationTask(pl.LightningModule):
    """Lightning module for sea ice segmentation (binary classification)."""

    def __init__(
        self,
        model: nn.Module,
        num_classes: int = 2,
        learning_rate: float = LEARNING_RATE,
        weight_decay: float = WEIGHT_DECAY,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.model = model
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.freeze_backbone = freeze_backbone

        # Loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=255)

        # Store outputs for epoch-level metrics
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

        # Handle variable-sized inputs by cropping to 224x224 patches
        B, C, H, W = images.shape
        if H > 224 or W > 224:
            # Center crop
            start_h = (H - 224) // 2
            start_w = (W - 224) // 2
            images = images[:, :, start_h:start_h + 224, start_w:start_w + 224]
            masks = masks[:, start_h:start_h + 224, start_w:start_w + 224]

        logits = self(images)
        loss = self.criterion(logits, masks)

        preds = logits.argmax(dim=1)
        ious = compute_iou(preds, masks, self.num_classes)

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
        self.log('val/iou_water', mean_iou_water)
        self.log('val/iou_ice', mean_iou_ice)
        self.log('val/iou_mean', mean_iou, prog_bar=True)

        self.validation_outputs.clear()

    def test_step(self, batch, batch_idx):
        images = batch['image']
        masks = batch['mask']

        B, C, H, W = images.shape
        if H > 224 or W > 224:
            start_h = (H - 224) // 2
            start_w = (W - 224) // 2
            images = images[:, :, start_h:start_h + 224, start_w:start_w + 224]
            masks = masks[:, start_h:start_h + 224, start_w:start_w + 224]

        logits = self(images)
        loss = self.criterion(logits, masks)

        preds = logits.argmax(dim=1)
        ious = compute_iou(preds, masks, self.num_classes)

        # Accuracy
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
        if self.freeze_backbone and hasattr(self.model, 'backbone'):
            # Only train the head
            params = self.model.head.parameters()
        else:
            params = self.model.parameters()

        optimizer = torch.optim.AdamW(
            params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs if self.trainer else 30,
            eta_min=1e-6,
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
            }
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True,
                       choices=['unet', 'terramind'],
                       help='Model architecture')
    parser.add_argument('--mode', type=str, default='probing',
                       choices=['probing', 'finetune'],
                       help='Training mode (probing=frozen backbone)')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--epochs', type=int, default=None,
                       help='Override default epochs')
    args = parser.parse_args()

    seed_everything(42)

    # Determine freeze settings
    freeze_backbone = args.mode == 'probing'
    max_epochs = args.epochs or (MAX_EPOCHS_PROBING if freeze_backbone else MAX_EPOCHS_FINETUNE)

    # Build model
    if args.model == 'unet':
        model = UNet(in_channels=2, num_classes=2, base_channels=16)
        logger.info("Built UNet model")
    else:
        model = TerraMindS1Segmentation(
            num_classes=2,
            hidden_dim=256,
            dropout=0.1,
            freeze_backbone=freeze_backbone,
        )
        logger.info(f"Built TerraMind S1 model (freeze_backbone={freeze_backbone})")

    # Data
    data_module = SeaIcePatchDataModule(
        batch_size=args.batch_size,
        num_workers=4,
        model_type=args.model,
        task='binary',
    )

    # Task
    task = SeaIceSegmentationTask(
        model=model,
        num_classes=2,
        learning_rate=LEARNING_RATE if args.model == 'unet' else (
            1e-4 if freeze_backbone else 1e-5
        ),
        freeze_backbone=freeze_backbone,
    )

    # Callbacks
    exp_name = f"{args.model}_{args.mode}" if args.model != 'unet' else 'unet_baseline'
    checkpoint_dir = CHECKPOINTS_DIR / exp_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='best-{epoch:02d}-{val/iou_mean:.4f}',
        monitor='val/iou_mean',
        mode='max',
        save_top_k=1,
    )

    early_stop = EarlyStopping(
        monitor='val/iou_mean',
        patience=10,
        mode='max',
    )

    # Logger
    tb_logger = TensorBoardLogger(
        save_dir=LOGS_DIR,
        name=exp_name,
    )

    # Trainer
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=[args.gpu],
        max_epochs=max_epochs,
        callbacks=[checkpoint_callback, early_stop],
        logger=tb_logger,
        precision='16-mixed',
        gradient_clip_val=1.0,
    )

    # Train
    logger.info(f"Training {exp_name}...")
    trainer.fit(task, datamodule=data_module)

    # Test
    logger.info("Testing...")
    test_results = trainer.test(task, datamodule=data_module, ckpt_path='best')

    # Save results
    if test_results:
        results = {
            'model': args.model,
            'mode': args.mode if args.model != 'unet' else None,
            'freeze_backbone': freeze_backbone if args.model != 'unet' else None,
            'metrics': test_results[0],
        }

        results_path = RESULTS_DIR / f"{exp_name}_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {results_path}")


if __name__ == '__main__':
    main()
