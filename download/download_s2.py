#!/usr/bin/env python3
"""Download co-located Sentinel-2 imagery for AI4Arctic scenes."""

import os
import json
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.crs import CRS
from rasterio.transform import from_bounds
import xarray as xr
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

try:
    from pystac_client import Client
    import planetary_computer as pc
except ImportError:
    pc = None
    from pystac_client import Client

# Configuration
STAC_URL = "https://earth-search.aws.element84.com/v1"  # Element84 Earth Search
COLLECTION = "sentinel-2-l2a"
MAX_CLOUD_COVER = 50  # Maximum cloud cover percentage
TIME_WINDOW_DAYS = 7  # Search window ±days from S1 acquisition
TARGET_RESOLUTION = 80  # Match AI4Arctic resolution (80m)
S2_BANDS = ['B02', 'B03', 'B04', 'B08', 'B11', 'B12']  # Blue, Green, Red, NIR, SWIR1, SWIR2


def search_s2_scenes(lat_min, lat_max, lon_min, lon_max, date_str, max_cloud=MAX_CLOUD_COVER, days=TIME_WINDOW_DAYS):
    """Search for Sentinel-2 scenes covering the given area and time."""
    client = Client.open(STAC_URL)

    # Create date range
    center_date = datetime.strptime(date_str, '%Y-%m-%d')
    start_date = center_date - timedelta(days=days)
    end_date = center_date + timedelta(days=days)
    date_range = f"{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"

    # Create bounding box [west, south, east, north]
    bbox = [lon_min, lat_min, lon_max, lat_max]

    try:
        search = client.search(
            collections=[COLLECTION],
            bbox=bbox,
            datetime=date_range,
            query={"eo:cloud_cover": {"lt": max_cloud}},
            max_items=50,
        )
        items = list(search.items())
        return items
    except Exception as e:
        print(f"Search error: {e}")
        return []


def get_best_s2_scene(items, target_date_str):
    """Select the best S2 scene based on cloud cover and temporal proximity."""
    if not items:
        return None

    target_date = datetime.strptime(target_date_str, '%Y-%m-%d')

    # Score scenes: lower is better
    def score_scene(item):
        cloud_cover = item.properties.get('eo:cloud_cover', 100)
        scene_date = datetime.fromisoformat(item.properties['datetime'].replace('Z', '+00:00')).replace(tzinfo=None)
        days_diff = abs((scene_date - target_date).days)
        # Combine cloud cover and temporal proximity
        return cloud_cover * 0.5 + days_diff * 10

    items_scored = [(item, score_scene(item)) for item in items]
    items_scored.sort(key=lambda x: x[1])

    return items_scored[0][0] if items_scored else None


def download_s2_bands(item, output_path, target_shape, target_bounds, target_crs='EPSG:4326'):
    """Download and resample S2 bands to match AI4Arctic scene."""
    bands_data = {}

    for band_name in S2_BANDS:
        if band_name.lower() in item.assets:
            asset = item.assets[band_name.lower()]
        elif band_name in item.assets:
            asset = item.assets[band_name]
        else:
            print(f"  Band {band_name} not found in assets")
            continue

        href = asset.href

        try:
            with rasterio.open(href) as src:
                # Calculate output transform
                out_transform = from_bounds(*target_bounds, target_shape[1], target_shape[0])

                # Reproject and resample
                out_array = np.zeros(target_shape, dtype=np.float32)
                reproject(
                    source=rasterio.band(src, 1),
                    destination=out_array,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=out_transform,
                    dst_crs=CRS.from_string(target_crs),
                    resampling=Resampling.bilinear,
                )
                bands_data[band_name] = out_array
        except Exception as e:
            print(f"  Error downloading {band_name}: {e}")

    return bands_data


def process_scene(scene_meta, output_dir, skip_existing=True):
    """Process a single AI4Arctic scene: find and download matching S2 data."""
    filename = scene_meta['filename']
    output_file = output_dir / f"{filename.replace('.nc', '_s2.npz')}"

    if skip_existing and output_file.exists():
        return {'filename': filename, 'status': 'exists'}

    # Search for S2 scenes
    items = search_s2_scenes(
        scene_meta['lat_min'], scene_meta['lat_max'],
        scene_meta['lon_min'], scene_meta['lon_max'],
        scene_meta['date'],
        max_cloud=MAX_CLOUD_COVER,
        days=TIME_WINDOW_DAYS
    )

    if not items:
        return {'filename': filename, 'status': 'no_s2_found', 'message': 'No S2 scenes found'}

    # Select best scene
    best_item = get_best_s2_scene(items, scene_meta['date'])
    if not best_item:
        return {'filename': filename, 'status': 'no_s2_found', 'message': 'No suitable S2 scene'}

    # Get target shape and bounds from original scene
    target_shape = (scene_meta['sar_lines'], scene_meta['sar_samples'])
    target_bounds = (scene_meta['lon_min'], scene_meta['lat_min'],
                     scene_meta['lon_max'], scene_meta['lat_max'])

    # Download bands
    bands_data = download_s2_bands(best_item, output_file, target_shape, target_bounds)

    if not bands_data:
        return {'filename': filename, 'status': 'download_failed', 'message': 'Could not download any bands'}

    # Stack bands and save
    band_names = list(bands_data.keys())
    stacked = np.stack([bands_data[b] for b in band_names], axis=0)

    np.savez_compressed(
        output_file,
        data=stacked,
        bands=band_names,
        s2_datetime=best_item.properties['datetime'],
        s2_cloud_cover=best_item.properties.get('eo:cloud_cover', -1),
        s2_item_id=best_item.id,
    )

    return {
        'filename': filename,
        'status': 'success',
        's2_item_id': best_item.id,
        's2_datetime': best_item.properties['datetime'],
        'cloud_cover': best_item.properties.get('eo:cloud_cover', -1),
    }


def main():
    # Load scene metadata
    metadata_file = Path('sea_ice/data/scene_metadata.json')
    with open(metadata_file) as f:
        metadata = json.load(f)

    # Create output directory
    output_dir = Path('sea_ice/data/s2_colocated')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process all scenes
    all_scenes = metadata['train'] + metadata['test']
    print(f"Processing {len(all_scenes)} scenes...")

    results = {'success': 0, 'no_s2_found': 0, 'download_failed': 0, 'exists': 0}
    failed_scenes = []

    for scene_meta in tqdm(all_scenes):
        result = process_scene(scene_meta, output_dir)
        results[result['status']] = results.get(result['status'], 0) + 1

        if result['status'] not in ['success', 'exists']:
            failed_scenes.append(result)

    # Save results
    results_file = output_dir / 'download_results.json'
    with open(results_file, 'w') as f:
        json.dump({
            'summary': results,
            'failed_scenes': failed_scenes,
        }, f, indent=2)

    print(f"\n=== Download Summary ===")
    print(f"Success: {results['success']}")
    print(f"Already exists: {results['exists']}")
    print(f"No S2 found: {results['no_s2_found']}")
    print(f"Download failed: {results['download_failed']}")
    print(f"\nResults saved to {results_file}")


if __name__ == '__main__':
    main()
