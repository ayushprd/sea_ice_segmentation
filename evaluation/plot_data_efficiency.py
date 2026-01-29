"""Plot data efficiency curves for UNet vs TerraMind."""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path("/mnt/data/benchmark/sea_ice/evaluation/results")

# Load results
fractions = [0.05, 0.1, 0.25, 0.5, 1.0]
unet_iou = []
terramind_iou = []

for frac in fractions:
    frac_str = str(frac).replace('.', '_')

    # UNet
    unet_file = RESULTS_DIR / f"unet_frac{frac_str}.json"
    with open(unet_file) as f:
        data = json.load(f)
        unet_iou.append(data['metrics']['test/iou_mean'] * 100)

    # TerraMind
    tm_file = RESULTS_DIR / f"terramind_frac{frac_str}.json"
    with open(tm_file) as f:
        data = json.load(f)
        terramind_iou.append(data['metrics']['test/iou_mean'] * 100)

# Convert to numpy
fractions = np.array(fractions)
unet_iou = np.array(unet_iou)
terramind_iou = np.array(terramind_iou)

# Calculate improvement
improvement = terramind_iou - unet_iou

# Print table
print("="*70)
print("DATA EFFICIENCY RESULTS: Sea Ice Binary Classification (AI4Arctic)")
print("="*70)
print(f"{'Data %':<10} {'# Samples':<12} {'UNet mIoU':<12} {'TerraMind mIoU':<15} {'Δ mIoU':<10}")
print("-"*70)
n_total = 8561
for i, frac in enumerate(fractions):
    n_samples = int(n_total * frac)
    print(f"{frac*100:>6.0f}%    {n_samples:>8}     {unet_iou[i]:>8.1f}%      {terramind_iou[i]:>10.1f}%       {improvement[i]:>+6.1f}%")
print("="*70)

# Create plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left plot: mIoU vs data fraction
ax1.plot(fractions * 100, unet_iou, 'o-', linewidth=2, markersize=8,
         label='UNet (from scratch)', color='#1f77b4')
ax1.plot(fractions * 100, terramind_iou, 's-', linewidth=2, markersize=8,
         label='TerraMind S1 (finetuned)', color='#d62728')

ax1.set_xlabel('Training Data (%)', fontsize=12)
ax1.set_ylabel('Test mIoU (%)', fontsize=12)
ax1.set_title('Data Efficiency: Sea Ice Classification', fontsize=14, fontweight='bold')
ax1.legend(loc='lower right', fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 105)
ax1.set_ylim(82, 95)
ax1.set_xticks([5, 10, 25, 50, 100])

# Add annotations for key points
ax1.annotate(f'+{improvement[0]:.1f}%', xy=(5, terramind_iou[0]),
             xytext=(12, terramind_iou[0]+1.5), fontsize=10, color='#d62728',
             arrowprops=dict(arrowstyle='->', color='#d62728', lw=1))

# Right plot: Improvement over UNet
ax2.bar(range(len(fractions)), improvement, color='#2ca02c', alpha=0.7, edgecolor='black')
ax2.set_xticks(range(len(fractions)))
ax2.set_xticklabels([f'{f*100:.0f}%' for f in fractions])
ax2.set_xlabel('Training Data (%)', fontsize=12)
ax2.set_ylabel('TerraMind Improvement over UNet (mIoU %)', fontsize=12)
ax2.set_title('Foundation Model Advantage', fontsize=14, fontweight='bold')
ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax2.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for i, v in enumerate(improvement):
    ax2.text(i, v + 0.1, f'+{v:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(RESULTS_DIR / 'data_efficiency_plot.png', dpi=150, bbox_inches='tight')
plt.savefig(RESULTS_DIR / 'data_efficiency_plot.pdf', bbox_inches='tight')
print(f"\nPlot saved to {RESULTS_DIR / 'data_efficiency_plot.png'}")

# Also create a summary JSON
summary = {
    'task': 'Sea Ice Binary Classification',
    'dataset': 'AI4Arctic',
    'total_train_samples': n_total,
    'fractions': fractions.tolist(),
    'unet_miou': unet_iou.tolist(),
    'terramind_miou': terramind_iou.tolist(),
    'improvement': improvement.tolist(),
}
with open(RESULTS_DIR / 'data_efficiency_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
