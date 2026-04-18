#!/usr/bin/env python3
"""
Read metrics.txt from each snapshot folder under a training output directory
and plot metric curves.

Usage:
    python check_metric.py <ckpt_dir> [<ckpt_dir2> ...]
    python check_metric.py ckpt/coords_direct_full ckpt/coords_train

Each ckpt_dir should contain a samples/ subfolder with step directories,
each having a metrics.txt file like:
    iou: 0.2452
    precision: 0.4056
    recall: 0.3827
    f1: 0.3938
"""
import os
import sys
import re
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def parse_metrics(metrics_path):
    """Parse a metrics.txt file into a dict of {metric_name: float_value}."""
    metrics = {}
    with open(metrics_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(':')
            if len(parts) >= 2:
                key = parts[0].strip()
                try:
                    val = float(parts[1].strip())
                    metrics[key] = val
                except ValueError:
                    pass
    return metrics


def collect_metrics(ckpt_dir):
    """Collect all snapshot metrics from ckpt_dir/samples/step*/metrics.txt."""
    samples_dir = os.path.join(ckpt_dir, 'samples')
    if not os.path.isdir(samples_dir):
        print(f"Warning: {samples_dir} not found")
        return [], {}

    step_dirs = sorted(glob.glob(os.path.join(samples_dir, 'step*')))
    steps = []
    all_metrics = {}  # {metric_name: [values]}

    for sd in step_dirs:
        metrics_file = os.path.join(sd, 'metrics.txt')
        if not os.path.isfile(metrics_file):
            continue
        # Extract step number
        dirname = os.path.basename(sd)
        m = re.search(r'step(\d+)', dirname)
        if not m:
            continue
        step = int(m.group(1))
        metrics = parse_metrics(metrics_file)
        if not metrics:
            continue

        steps.append(step)
        for k, v in metrics.items():
            if k not in all_metrics:
                all_metrics[k] = []
            all_metrics[k].append((step, v))

    return steps, all_metrics


def plot_metrics(ckpt_dirs, output_path=None):
    """Plot metric curves for one or more experiments."""
    # Collect data
    experiments = {}
    all_metric_names = set()
    for d in ckpt_dirs:
        name = os.path.basename(d.rstrip('/'))
        steps, metrics = collect_metrics(d)
        if not steps:
            print(f"No metrics found in {d}")
            continue
        experiments[name] = metrics
        all_metric_names.update(metrics.keys())

    if not experiments:
        print("No data to plot.")
        return

    # Focus on key metrics
    key_metrics = ['iou', 'precision', 'recall', 'f1']
    plot_metrics_list = [m for m in key_metrics if m in all_metric_names]
    extra = sorted(all_metric_names - set(key_metrics))
    plot_metrics_list.extend(extra)

    if not plot_metrics_list:
        print("No recognized metrics found.")
        return

    n = len(plot_metrics_list)
    cols = min(2, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 5 * rows), squeeze=False)

    colors = plt.cm.tab10.colors
    for i, metric_name in enumerate(plot_metrics_list):
        ax = axes[i // cols][i % cols]
        for j, (exp_name, metrics) in enumerate(experiments.items()):
            if metric_name not in metrics:
                continue
            data = sorted(metrics[metric_name], key=lambda x: x[0])
            xs = [d[0] for d in data]
            ys = [d[1] for d in data]
            color = colors[j % len(colors)]
            ax.plot(xs, ys, '-o', markersize=2, linewidth=1, label=exp_name, color=color)
        ax.set_title(metric_name, fontsize=14, fontweight='bold')
        ax.set_xlabel('Step')
        ax.set_ylabel(metric_name)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Hide unused axes
    for i in range(n, rows * cols):
        axes[i // cols][i % cols].set_visible(False)

    plt.tight_layout()
    if output_path is None:
        output_path = 'metrics_curve.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved to {output_path}")
    plt.close()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python check_metric.py <ckpt_dir> [<ckpt_dir2> ...]")
        print("Example: python check_metric.py ckpt/coords_direct_full ckpt/coords_train")
        sys.exit(1)

    ckpt_dirs = sys.argv[1:]
    # Output path: next to first ckpt dir
    out = os.path.join(os.path.dirname(ckpt_dirs[0].rstrip('/')), 'metrics_curve.png')
    plot_metrics(ckpt_dirs, output_path=out)
