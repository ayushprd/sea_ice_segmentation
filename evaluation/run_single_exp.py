"""Run a single data efficiency experiment."""
import argparse
import json
import sys
sys.path.insert(0, '/mnt/data/benchmark')

from sea_ice.evaluation.data_efficiency import run_experiment
from sea_ice.evaluation.config import RESULTS_DIR

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--fraction', type=float, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--epochs', type=int, default=20)
    args = parser.parse_args()

    results = run_experiment(args.model, args.fraction, args.gpu, args.epochs)

    frac_str = str(args.fraction).replace('.', '_')
    output_path = RESULTS_DIR / f"{args.model}_frac{frac_str}.json"

    with open(output_path, 'w') as f:
        json.dump({
            'model': args.model,
            'fraction': args.fraction,
            'metrics': results
        }, f, indent=2)

    print(f"Results saved to {output_path}")
    print(f"mIoU: {results.get('test/iou_mean', 0)*100:.1f}%")

if __name__ == '__main__':
    main()
