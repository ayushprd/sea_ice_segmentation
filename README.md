# Sea Ice Segmentation with GFMs

**Slides** https://github.com/ayushprd/sea_ice_segmentation/blob/main/IntrotoGFMs.pdf

Evaluates geospatial foundation models (currently only TerraMind) against a UNet baseline on the task of classifying SAR pixels as **water** or **sea ice** using the [AI4Arctic Sea Ice Challenge](https://platform.ai4eo.eu/ai4arctic-sea-ice-challenge) dataset. 

## Dataset

**Source**: [AI4Arctic Sea Ice Challenge](https://huggingface.co/datasets/torchgeo/ai4artic-sea-ice-challenge) (HuggingFace)

- **512 training scenes** + **20 test scenes** of Sentinel-1 EW mode SAR
- **Dual polarization**: HH and HV channels (80m pixel spacing)
- **Labels**: Sea Ice Concentration (SIC) from ice charts, binarized at SIC >= 15%
- **Coverage**: Arctic regions (Greenland, Svalbard, Canadian Arctic)

### Download

```bash
python download_data.py
```

This downloads ~57 GB of training data and ~2.4 GB of test data from HuggingFace, then extracts the tar archives into `data/train/` and `data/test/`.

### Preprocessing: Extract Patches

The raw data consists of large NetCDF scenes of variable size. We extract fixed 224x224 patches for training:

```bash
python download/preprocess_patches.py --workers 8
```

This produces ~10,000 `.npz` patch files in `data/patches/`, each containing:
- `sar`: float32 array of shape `(2, 224, 224)` — HH and HV channels
- `sic`: uint8 array of shape `(224, 224)` — SIC values (0-100, 255=invalid)

A 85/15 train/val split is saved as `data/patches/train_files.txt` and `data/patches/val_files.txt`.

## Project Structure

```
sea_ice/
├── README.md
├── download_data.py                    # Download from HuggingFace
├── sea_ice_benchmark.ipynb             # Main tutorial notebook
├── download/
│   ├── preprocess_patches.py           # Extract 224x224 patches from NetCDF
│   ├── extract_scene_metadata.py       # Scene metadata extraction
│   ├── download_s2.py                  # Sentinel-2 auxiliary download
│   ├── search_s2_availability.py       # S2 STAC search
│   └── preprocess_s1_s2_patches.py     # S1+S2 joint patches
├── evaluation/
│   ├── config.py                       # Paths, hyperparameters, normalization
│   ├── patch_dataset.py                # PyTorch Dataset + DataModule
│   ├── train.py                        # Main training script
│   ├── data_efficiency.py              # Data efficiency experiments
│   ├── visualize_predictions.py        # Prediction visualization
│   ├── plot_data_efficiency.py         # Plot data efficiency curves
│   ├── run_single_exp.py              # Single experiment runner
│   ├── models/
│   │   ├── unet.py                     # UNet baseline (1.9M params)
│   │   └── terramind_s1.py             # TerraMind S1 segmentation head
│   ├── results/                        # JSON metrics + plots
└── data/                               # Dataset files (gitignored)
    ├── train/                          # 512 NetCDF scenes (~57 GB)
    ├── test/                           # 20 NetCDF scenes (~2.4 GB)
    └── patches/                        # 10K .npz patches (~3.2 GB)
```

## Training

### UNet Baseline

```bash
python -m sea_ice.evaluation.train --model unet --gpu 0
```

Trains a lightweight UNet (`base_channels=16`, 1.9M parameters) for 50 epochs with cosine annealing.

### TerraMind (Linear Probing)

```bash
python -m sea_ice.evaluation.train --model terramind --mode probing --gpu 0
```

Freezes the TerraMind backbone and trains only a segmentation head (30 epochs).

### TerraMind (Fine-tuning)

```bash
python -m sea_ice.evaluation.train --model terramind --mode finetune --gpu 0
```

Fine-tunes the full TerraMind model with a lower learning rate (1e-5, 50 epochs).

### Data Efficiency Experiment

```bash
python -m sea_ice.evaluation.data_efficiency --gpu 0 --fractions 0.05,0.1,0.25,0.5,1.0
```

Trains both models at each data fraction to measure sample efficiency.

## Results

### Model Comparison (100% training data)

| Model | mIoU | IoU Water | IoU Ice | Accuracy |
|-------|------|-----------|---------|----------|
| UNet Baseline | 89.5% | 91.8% | 87.3% | 96.4% |
| TerraMind (probing) | 91.0% | 93.2% | 88.8% | 96.9% |
| TerraMind (fine-tune) | **92.4%** | **94.2%** | **90.6%** | **97.3%** |

### Data Efficiency

| Training Data | UNet mIoU | TerraMind mIoU | Gap |
|---------------|-----------|----------------|-----|
| 5% (428 patches) | 79.3% | 88.4% | +9.1% |
| 10% | 84.5% | 89.6% | +5.1% |
| 25% | 87.2% | 91.7% | +4.5% |
| 50% | 88.8% | 92.6% | +3.7% |
| 100% (8,561 patches) | 89.5% | 92.4% | +2.9% |

TerraMind's pretrained representations provide the largest advantage in low-data regimes (+9.1% mIoU at 5% data).

## Notebook

The main walkthrough is in `sea_ice_benchmark.ipynb`, which covers:

1. Dataset overview and exploration
2. Patch preprocessing pipeline
3. Model architecture (UNet + TerraMind)
4. Pretrained embedding analysis (PCA, kNN, spatial tokens)
5. Training (baseline, probing, fine-tuning)
6. Results comparison and data efficiency analysis

## References

- [AI4Arctic Sea Ice Challenge](https://platform.ai4eo.eu/ai4arctic-sea-ice-challenge)
- [AI4Arctic GitHub](https://github.com/astokholm/AI4ArcticSeaIceChallenge)
- Stokholm et al., "AI4Arctic Sea Ice Challenge Dataset" (2022)
- TerraMind: Multi-modal foundation model for Earth observation
