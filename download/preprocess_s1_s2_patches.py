#!/usr/bin/env python3
"""
Preprocess AI4Arctic scenes with co-located Sentinel-2 imagery.
Creates patches from regions where a single S2 tile overlaps with the S1 scene.
"""

import os
import json
import numpy as np
import xarray as xr
from pathlib import Path
from datetime import datetime, timedelta
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from scipy.interpolate import griddata
from pystac_client import Client
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Configuration
STAC_URL = "https://earth-search.aws.element84.com/v1"
COLLECTION = "sentinel-2-l2a"
MAX_CLOUD_COVER = 40
TIME_WINDOW_DAYS = 7
PATCH_SIZE = 224
MAX_PATCHES_PER_SCENE = 10
MIN_S2_COVERAGE = 0.9

# S2 bands - only use 10m resolution bands for speed
S2_BANDS = ['blue', 'green', 'red', 'nir']


def get_scene_s2_availability(s2_availability_file):
    """Load S2 availability from pre-computed search."""
    with open(s2_availability_file) as f:
        data = json.load(f)

    lookup = {}
    for scene in data['scenes']:
        if scene['count'] > 0:
            lookup[scene['filename']] = scene
    return lookup


def search_best_s2_tile(lat_min, lat_max, lon_min, lon_max, date_str):
    """Search for best single S2 tile covering the scene."""
    client = Client.open(STAC_URL)

    center_date = datetime.strptime(date_str, '%Y-%m-%d')
    start_date = center_date - timedelta(days=TIME_WINDOW_DAYS)
    end_date = center_date + timedelta(days=TIME_WINDOW_DAYS)
    date_range = f"{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"

    bbox = [lon_min, lat_min, lon_max, lat_max]

    try:
        search = client.search(
            collections=[COLLECTION],
            bbox=bbox,
            datetime=date_range,
            query={"eo:cloud_cover": {"lt": MAX_CLOUD_COVER}},
            max_items=20,
        )
        items = list(search.items())

        if not items:
            return None

        # Return lowest cloud cover
        return min(items, key=lambda x: x.properties.get('eo:cloud_cover', 100))
    except Exception as e:
        return None


def download_s2_tile(s2_item, lat_grid, lon_grid, target_shape):
    """Download S2 bands reprojected to match S1 scene grid."""
    min_lat, max_lat = float(np.nanmin(lat_grid)), float(np.nanmax(lat_grid))
    min_lon, max_lon = float(np.nanmin(lon_grid)), float(np.nanmax(lon_grid))

    dst_crs = CRS.from_epsg(4326)
    dst_transform = from_bounds(min_lon, min_lat, max_lon, max_lat, target_shape[1], target_shape[0])

    bands_data = {}
    coverage_mask = None

    for band_name in S2_BANDS:
        if band_name not in s2_item.assets:
            continue

        href = s2_item.assets[band_name].href

        try:
            with rasterio.open(href) as src:
                dst_array = np.zeros(target_shape, dtype=np.float32)
                reproject(
                    source=rasterio.band(src, 1),
                    destination=dst_array,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=dst_transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.bilinear,
                )

                if dst_array.max() > 0:
                    bands_data[band_name] = dst_array
                    if coverage_mask is None:
                        coverage_mask = (dst_array > 0).astype(np.float32)

        except Exception as e:
            continue

    if len(bands_data) < 3 or coverage_mask is None:
        return None, None

    return bands_data, coverage_mask


def interpolate_coords(grid_lat, grid_lon, target_shape):
    """Interpolate sparse lat/lon grid to full resolution."""
    grid_y = np.linspace(0, target_shape[0]-1, grid_lat.shape[0])
    grid_x = np.linspace(0, target_shape[1]-1, grid_lat.shape[1])

    target_y = np.arange(target_shape[0])
    target_x = np.arange(target_shape[1])
    target_xx, target_yy = np.meshgrid(target_x, target_y)

    source_xx, source_yy = np.meshgrid(grid_x, grid_y)
    points = np.column_stack([source_yy.ravel(), source_xx.ravel()])

    full_lat = griddata(points, grid_lat.ravel(), (target_yy, target_xx), method='linear')
    full_lon = griddata(points, grid_lon.ravel(), (target_yy, target_xx), method='linear')

    return full_lat, full_lon


def find_valid_patch_locations(coverage_mask, s1_hh, sic, num_patches=10):
    """Find patch locations where both S1 and S2 data are valid."""
    h, w = coverage_mask.shape
    valid_locations = []

    # Use larger stride for speed
    stride = PATCH_SIZE
    for y in range(0, h - PATCH_SIZE, stride):
        for x in range(0, w - PATCH_SIZE, stride):
            s2_cov = coverage_mask[y:y+PATCH_SIZE, x:x+PATCH_SIZE].mean()
            s1_valid = 1 - np.isnan(s1_hh[y:y+PATCH_SIZE, x:x+PATCH_SIZE]).mean()
            sic_valid = 1 - np.isnan(sic[y:y+PATCH_SIZE, x:x+PATCH_SIZE]).mean()

            if s2_cov >= MIN_S2_COVERAGE and s1_valid >= 0.7 and sic_valid >= 0.7:
                valid_locations.append((y, x, s2_cov))

    valid_locations.sort(key=lambda x: -x[2])
    return valid_locations[:num_patches]


def process_scene(nc_path, output_dir, s2_availability=None, verbose=False):
    """Process a single AI4Arctic scene with S2 co-location."""
    filename = os.path.basename(nc_path)

    if s2_availability and filename not in s2_availability:
        return {'filename': filename, 'status': 'no_s2', 'patches': 0}

    try:
        ds = xr.open_dataset(nc_path)

        s1_hh = ds['nersc_sar_primary'].values
        s1_hv = ds['nersc_sar_secondary'].values
        sic = ds['SIC'].values
        lat_grid = ds['sar_grid2d_latitude'].values
        lon_grid = ds['sar_grid2d_longitude'].values

        target_shape = s1_hh.shape

        dt_str = filename.split('_')[0]
        date_str = datetime.strptime(dt_str, '%Y%m%dT%H%M%S').strftime('%Y-%m-%d')

        # Search for best S2 tile
        s2_item = search_best_s2_tile(
            float(np.nanmin(lat_grid)), float(np.nanmax(lat_grid)),
            float(np.nanmin(lon_grid)), float(np.nanmax(lon_grid)),
            date_str
        )

        if s2_item is None:
            ds.close()
            return {'filename': filename, 'status': 'no_s2_found', 'patches': 0}

        # Download S2 tile
        s2_bands, coverage_mask = download_s2_tile(s2_item, lat_grid, lon_grid, target_shape)

        if s2_bands is None:
            ds.close()
            return {'filename': filename, 'status': 's2_download_failed', 'patches': 0}

        coverage_pct = (coverage_mask > 0).mean() * 100

        # Find valid patch locations
        valid_locations = find_valid_patch_locations(coverage_mask, s1_hh, sic, MAX_PATCHES_PER_SCENE)

        if not valid_locations:
            ds.close()
            return {'filename': filename, 'status': 'no_valid_patches', 'patches': 0, 'coverage': coverage_pct}

        # Interpolate coordinates
        full_lat, full_lon = interpolate_coords(lat_grid, lon_grid, target_shape)

        valid_patches = 0

        for patch_idx, (y, x, _) in enumerate(valid_locations):
            s1_patch_hh = s1_hh[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
            s1_patch_hv = s1_hv[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
            sic_patch = sic[y:y+PATCH_SIZE, x:x+PATCH_SIZE]

            s2_patches = {name: data[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
                         for name, data in s2_bands.items()}

            center_y = y + PATCH_SIZE // 2
            center_x = x + PATCH_SIZE // 2
            center_lat = full_lat[center_y, center_x]
            center_lon = full_lon[center_y, center_x]

            patch_filename = f"{filename.replace('.nc', '')}_{patch_idx:02d}.npz"
            output_path = output_dir / patch_filename

            s1_stack = np.stack([s1_patch_hh, s1_patch_hv], axis=0)
            s2_stack = np.stack([s2_patches[b] for b in S2_BANDS if b in s2_patches], axis=0)

            np.savez_compressed(
                output_path,
                s1=s1_stack.astype(np.float32),
                s2=s2_stack.astype(np.float32),
                sic=sic_patch.astype(np.float32),
                s2_bands=[b for b in S2_BANDS if b in s2_patches],
                center_lat=center_lat,
                center_lon=center_lon,
            )

            valid_patches += 1

        ds.close()
        return {'filename': filename, 'status': 'success', 'patches': valid_patches, 'coverage': coverage_pct}

    except Exception as e:
        import traceback
        return {'filename': filename, 'status': 'error', 'error': str(e), 'patches': 0}


def main():
    data_dir = Path('sea_ice/data')
    output_dir = data_dir / 's1_s2_patches'
    output_dir.mkdir(parents=True, exist_ok=True)

    s2_availability_file = data_dir / 's2_availability.json'
    if s2_availability_file.exists():
        s2_availability = get_scene_s2_availability(s2_availability_file)
        print(f"Loaded S2 availability: {len(s2_availability)} scenes with S2")
    else:
        s2_availability = None

    train_files = sorted((data_dir / 'train').glob('*.nc'))
    test_files = sorted([f for f in (data_dir / 'test').glob('*.nc')
                        if '_reference' not in f.name])

    all_files = train_files + test_files
    print(f"Found {len(train_files)} train, {len(test_files)} test scenes")

    if s2_availability:
        all_files = [f for f in all_files if f.name in s2_availability]
        print(f"Filtering to {len(all_files)} scenes with S2 availability")

    results = {'success': 0, 'no_s2': 0, 'no_s2_found': 0, 's2_download_failed': 0,
               'no_valid_patches': 0, 'error': 0}
    total_patches = 0

    for nc_path in tqdm(all_files, desc="Processing scenes"):
        result = process_scene(nc_path, output_dir, s2_availability)
        results[result['status']] = results.get(result['status'], 0) + 1
        total_patches += result.get('patches', 0)

    print(f"\n=== Processing Summary ===")
    for status, count in results.items():
        print(f"{status}: {count}")
    print(f"Total patches created: {total_patches}")
    print(f"Patches saved to: {output_dir}")


if __name__ == '__main__':
    main()
