"""Download AI4Arctic dataset from HuggingFace."""
import os
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download

DATA_DIR = Path("/mnt/data/benchmark/sea_ice/data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

def download_file(filename: str):
    """Download a single file from HuggingFace."""
    print(f"Downloading {filename}...")
    path = hf_hub_download(
        repo_id="torchgeo/ai4artic-sea-ice-challenge",
        filename=filename,
        repo_type="dataset",
        local_dir=DATA_DIR,
        local_dir_use_symlinks=False,
    )
    print(f"Downloaded to: {path}")
    return path

if __name__ == "__main__":
    # Download metadata first (small)
    download_file("metadata.csv")
    
    # Download test set (smaller, 2.35GB)
    download_file("test.tar.gz")
    
    # Download training set (split into parts, ~55GB total)
    download_file("train.tar.gzaa")
    download_file("train.tar.gzab")
    
    print("\nExtracting test.tar.gz...")
    os.system(f"cd {DATA_DIR} && tar -xzf test.tar.gz")
    
    print("\nCombining and extracting training data...")
    os.system(f"cd {DATA_DIR} && cat train.tar.gzaa train.tar.gzab > train.tar.gz && tar -xzf train.tar.gz")
    
    print("\nDone! Cleaning up...")
    # Keep tar files for now in case extraction fails
    
    print("\nDataset ready!")
