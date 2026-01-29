"""Configuration for sea ice concentration evaluation."""

from pathlib import Path

# Paths
BASE_DIR = Path("/mnt/data/benchmark/sea_ice")
DATA_DIR = BASE_DIR / "data"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
CHECKPOINTS_DIR = BASE_DIR / "evaluation" / "checkpoints"
RESULTS_DIR = BASE_DIR / "evaluation" / "results"
LOGS_DIR = BASE_DIR / "evaluation" / "logs"

# Create directories
for d in [CHECKPOINTS_DIR, RESULTS_DIR, LOGS_DIR, TRAIN_DIR, TEST_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# AI4Arctic dataset variables in NetCDF files
# SAR variables (Sentinel-1)
SAR_VARIABLES = [
    'nersc_sar_primary',      # HH polarization (primary)
    'nersc_sar_secondary',    # HV polarization (secondary)
    'sar_incidenceangle',     # Incidence angle
]

# AMSR2 microwave radiometer variables
AMSR2_VARIABLES = [
    'btemp_6.9h', 'btemp_6.9v',
    'btemp_7.3h', 'btemp_7.3v',
    'btemp_10.7h', 'btemp_10.7v',
    'btemp_18.7h', 'btemp_18.7v',
    'btemp_23.8h', 'btemp_23.8v',
    'btemp_36.5h', 'btemp_36.5v',
    'btemp_89.0h', 'btemp_89.0v',
]

# ERA5 weather variables
ERA5_VARIABLES = [
    't2m',           # 2m temperature
    'skt',           # Skin temperature
    'tcwv',          # Total column water vapor
    'tclw',          # Total column liquid water
    'u10', 'v10',    # 10m wind components
]

# Label variables
LABEL_VARIABLES = {
    'SIC': 'SIC',           # Sea Ice Concentration (0-100)
    'SOD': 'SOD',           # Stage of Development (0-6)
    'FLOE': 'FLOE',         # Floe size (0-6)
}

# Class mappings for SIC (regression to classification)
SIC_BINS = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]  # 11 classes

# For binary classification (ice vs water)
SIC_BINARY_THRESHOLD = 15  # SIC >= 15% is ice

# Training configuration
BATCH_SIZE = 16
NUM_WORKERS = 2  # Reduced for large NetCDF file IO
MAX_EPOCHS_PROBING = 30
MAX_EPOCHS_FINETUNE = 50
PATCHES_PER_SCENE = 5  # Patches to extract per scene per epoch
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
PATCH_SIZE = 224  # Crop size for training

# Normalization statistics for SAR (from AI4Arctic ready-to-train)
# Already standard scaled in ready-to-train version
SAR_MEAN = [0.0, 0.0, 0.0]  # HH, HV, incidence angle
SAR_STD = [1.0, 1.0, 1.0]

# TerraMind S1 normalization (expects raw backscatter in dB)
TERRAMIND_S1_MEAN = [-12.54, -20.33]  # VV, VH typical Arctic values
TERRAMIND_S1_STD = [5.25, 5.42]
