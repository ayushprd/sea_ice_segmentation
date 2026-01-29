#!/usr/bin/env python3
"""Extract metadata from AI4Arctic scenes for S2 co-location."""

import os
import json
import xarray as xr
from pathlib import Path
from datetime import datetime
import numpy as np
from tqdm import tqdm

def extract_scene_metadata(nc_path: str) -> dict:
    """Extract metadata from a single NetCDF file."""
    ds = xr.open_dataset(nc_path)

    # Parse datetime from filename (e.g., 20200930T201936_dmi_prep.nc)
    filename = os.path.basename(nc_path)
    dt_str = filename.split('_')[0]  # 20200930T201936
    dt = datetime.strptime(dt_str, '%Y%m%dT%H%M%S')

    # Get geographic bounds from lat/lon grids
    lat = ds['sar_grid2d_latitude'].values
    lon = ds['sar_grid2d_longitude'].values

    # Get full resolution shape
    sar_lines = ds.dims['sar_lines']
    sar_samples = ds.dims['sar_samples']

    metadata = {
        'filename': filename,
        'filepath': nc_path,
        'datetime': dt.isoformat(),
        'date': dt.strftime('%Y-%m-%d'),
        'year': dt.year,
        'month': dt.month,
        'day': dt.day,
        'lat_min': float(np.nanmin(lat)),
        'lat_max': float(np.nanmax(lat)),
        'lon_min': float(np.nanmin(lon)),
        'lon_max': float(np.nanmax(lon)),
        'lat_center': float(np.nanmean(lat)),
        'lon_center': float(np.nanmean(lon)),
        'sar_lines': sar_lines,
        'sar_samples': sar_samples,
        'ice_service': ds.attrs.get('ice_service', 'unknown'),
        'pixel_spacing': int(ds.attrs.get('pixel_spacing', 80)),
    }

    ds.close()
    return metadata

def main():
    data_dir = Path('sea_ice/data')
    output_file = data_dir / 'scene_metadata.json'

    # Find all NetCDF files
    train_files = list((data_dir / 'train').glob('*.nc'))
    test_files = list((data_dir / 'test').glob('*_prep.nc'))  # Exclude reference files

    print(f"Found {len(train_files)} train scenes, {len(test_files)} test scenes")

    all_metadata = {'train': [], 'test': []}

    # Process train files
    print("\nProcessing train scenes...")
    for nc_path in tqdm(train_files):
        try:
            meta = extract_scene_metadata(str(nc_path))
            meta['split'] = 'train'
            all_metadata['train'].append(meta)
        except Exception as e:
            print(f"Error processing {nc_path}: {e}")

    # Process test files
    print("\nProcessing test scenes...")
    for nc_path in tqdm(test_files):
        try:
            meta = extract_scene_metadata(str(nc_path))
            meta['split'] = 'test'
            all_metadata['test'].append(meta)
        except Exception as e:
            print(f"Error processing {nc_path}: {e}")

    # Sort by datetime
    all_metadata['train'].sort(key=lambda x: x['datetime'])
    all_metadata['test'].sort(key=lambda x: x['datetime'])

    # Save metadata
    with open(output_file, 'w') as f:
        json.dump(all_metadata, f, indent=2)

    print(f"\nSaved metadata to {output_file}")

    # Print summary
    print("\n=== Dataset Summary ===")
    for split in ['train', 'test']:
        scenes = all_metadata[split]
        if scenes:
            dates = [s['date'] for s in scenes]
            lats = [s['lat_center'] for s in scenes]
            lons = [s['lon_center'] for s in scenes]
            print(f"\n{split.upper()}: {len(scenes)} scenes")
            print(f"  Date range: {min(dates)} to {max(dates)}")
            print(f"  Lat range: {min(lats):.2f} to {max(lats):.2f}")
            print(f"  Lon range: {min(lons):.2f} to {max(lons):.2f}")

if __name__ == '__main__':
    main()
