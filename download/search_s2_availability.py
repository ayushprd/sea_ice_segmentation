#!/usr/bin/env python3
"""Quick search to check S2 availability for AI4Arctic scenes."""

import json
from pathlib import Path
from datetime import datetime, timedelta
from pystac_client import Client
from tqdm import tqdm
from collections import defaultdict

STAC_URL = "https://earth-search.aws.element84.com/v1"
COLLECTION = "sentinel-2-l2a"
MAX_CLOUD_COVER = 50
TIME_WINDOW_DAYS = 7


def search_s2_count(lat_min, lat_max, lon_min, lon_max, date_str, max_cloud=MAX_CLOUD_COVER, days=TIME_WINDOW_DAYS):
    """Search for Sentinel-2 scenes and return count."""
    client = Client.open(STAC_URL)

    center_date = datetime.strptime(date_str, '%Y-%m-%d')
    start_date = center_date - timedelta(days=days)
    end_date = center_date + timedelta(days=days)
    date_range = f"{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"

    bbox = [lon_min, lat_min, lon_max, lat_max]

    try:
        search = client.search(
            collections=[COLLECTION],
            bbox=bbox,
            datetime=date_range,
            query={"eo:cloud_cover": {"lt": max_cloud}},
            max_items=10,
        )
        items = list(search.items())

        if items:
            # Get best (lowest cloud cover)
            best = min(items, key=lambda x: x.properties.get('eo:cloud_cover', 100))
            return {
                'count': len(items),
                'best_cloud': best.properties.get('eo:cloud_cover', -1),
                'best_date': best.properties.get('datetime', ''),
            }
        return {'count': 0}
    except Exception as e:
        return {'count': -1, 'error': str(e)}


def main():
    metadata_file = Path('sea_ice/data/scene_metadata.json')
    with open(metadata_file) as f:
        metadata = json.load(f)

    all_scenes = metadata['train'] + metadata['test']
    print(f"Checking S2 availability for {len(all_scenes)} scenes...")
    print(f"Search params: ±{TIME_WINDOW_DAYS} days, <{MAX_CLOUD_COVER}% cloud cover\n")

    results = []
    monthly_stats = defaultdict(lambda: {'total': 0, 'with_s2': 0})

    for scene in tqdm(all_scenes):
        result = search_s2_count(
            scene['lat_min'], scene['lat_max'],
            scene['lon_min'], scene['lon_max'],
            scene['date']
        )
        result['filename'] = scene['filename']
        result['date'] = scene['date']
        result['month'] = scene['month']
        results.append(result)

        # Track monthly stats
        month_key = f"{scene['year']}-{scene['month']:02d}"
        monthly_stats[month_key]['total'] += 1
        if result['count'] > 0:
            monthly_stats[month_key]['with_s2'] += 1

    # Summary
    with_s2 = sum(1 for r in results if r['count'] > 0)
    without_s2 = sum(1 for r in results if r['count'] == 0)
    errors = sum(1 for r in results if r['count'] < 0)

    print(f"\n=== S2 Availability Summary ===")
    print(f"Scenes with S2 coverage: {with_s2} ({100*with_s2/len(all_scenes):.1f}%)")
    print(f"Scenes without S2: {without_s2} ({100*without_s2/len(all_scenes):.1f}%)")
    print(f"Search errors: {errors}")

    print(f"\n=== Monthly Breakdown ===")
    for month in sorted(monthly_stats.keys()):
        stats = monthly_stats[month]
        pct = 100 * stats['with_s2'] / stats['total'] if stats['total'] > 0 else 0
        print(f"{month}: {stats['with_s2']}/{stats['total']} ({pct:.0f}%)")

    # Save full results
    output_file = Path('sea_ice/data/s2_availability.json')
    with open(output_file, 'w') as f:
        json.dump({
            'summary': {'with_s2': with_s2, 'without_s2': without_s2, 'errors': errors},
            'monthly': dict(monthly_stats),
            'scenes': results,
        }, f, indent=2)
    print(f"\nFull results saved to {output_file}")


if __name__ == '__main__':
    main()
