"""Plot a per-epoch metric (mean over active CV folds, +/- std band) from one or
more LACE/BiomedCLIP pretrain SLURM logs.

Each log is expected to contain one or more "# FOLD <n>" sections, each running
its own "Epoch <e>/<total> [stage <s>] | ..." curriculum with a metric line
somewhere in the following few lines (e.g. "... | mean_r1=0.133 mean_r5=0.367").
Per log, per epoch, the metric is averaged across whichever folds are still
active at that epoch (folds that stopped earlier via early stopping simply stop
contributing). The resulting mean+-std line is cut off once fewer than
--min_folds folds remain, since the std becomes unreliable/undefined below that.

Usage:
    python src/data/plot_pretrain_metric.py \\
        --logs logs/lace/slurm-7842_lacev2_binary_A4.out \\
               logs/lace/slurm-7923_lacev2_binary_A4_4_mask_tokens.out \\
               logs/lace/slurm-7924_lacev2_binary_A4_8_mask_tokens.out \\
        --labels "A4 (1 mask token)" "A4 (4 mask tokens)" "A4 (8 mask tokens)" \\
        --metric mean_r1 \\
        --stage1_epochs 20 \\
        --output results/plots/pretrain_mean_r1_mask_tokens.png
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

FOLD_RE = re.compile(r"^#\s*FOLD\s+(\d+)")
EPOCH_RE = re.compile(r"^Epoch\s+(\d+)/(\d+)")

# dataviz skill default categorical palette, slots 1-3 (validated all-pairs CVD-safe)
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
SURFACE = "#ffffff"
GRIDLINE = "#e1e0d9"
AXIS = "#c3c2b7"
PRIMARY_INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED_INK = "#898781"


def parse_log(path: Path, metric: str) -> dict[int, dict[int, float]]:
    """Returns {fold_id: {epoch: metric_value}}."""
    metric_re = re.compile(re.escape(metric) + r"=([0-9eE.+-]+)")
    folds: dict[int, dict[int, float]] = {}
    fold_id = None
    epoch = None
    for line in path.read_text().splitlines():
        fold_match = FOLD_RE.match(line)
        if fold_match:
            fold_id = int(fold_match.group(1))
            folds.setdefault(fold_id, {})
            epoch = None
            continue
        epoch_match = EPOCH_RE.match(line)
        if epoch_match:
            epoch = int(epoch_match.group(1))
        if fold_id is None or epoch is None:
            continue
        metric_match = metric_re.search(line)
        if metric_match:
            folds[fold_id][epoch] = float(metric_match.group(1))
    return folds


def aggregate(folds: dict[int, dict[int, float]], min_folds: int):
    """Returns (epochs, mean, std, n_folds) arrays, truncated once n_folds < min_folds."""
    max_epoch = max(e for epochs in folds.values() for e in epochs)
    epochs, means, stds, counts = [], [], [], []
    for e in range(1, max_epoch + 1):
        values = [v[e] for v in folds.values() if e in v]
        n = len(values)
        if n < min_folds:
            break
        epochs.append(e)
        means.append(float(np.mean(values)))
        stds.append(float(np.std(values, ddof=1)) if n > 1 else 0.0)
        counts.append(n)
    return np.array(epochs), np.array(means), np.array(stds), np.array(counts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs", nargs="+", required=True, type=Path)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--metric", default="mean_r1")
    parser.add_argument("--ylabel", default=None, help="Defaults to --metric if omitted.")
    parser.add_argument("--min_folds", type=int, default=3)
    parser.add_argument("--stage1_epochs", type=int, default=None)
    parser.add_argument("--stage1_label", default="stage 1 → 2")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    if len(args.logs) != len(args.labels):
        parser.error("--logs and --labels must have the same length")

    fig, ax = plt.subplots(figsize=(9, 5.5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    for log_path, label, color in zip(args.logs, args.labels, SERIES_COLORS):
        folds = parse_log(log_path, args.metric)
        epochs, means, stds, counts = aggregate(folds, args.min_folds)
        n_total = len(folds)
        print(
            f"{label}: {n_total} folds, plotted epochs 1-{epochs[-1]} "
            f"(cut off once fewer than {args.min_folds} folds remained; "
            f"last epoch had n={counts[-1]})"
        )
        ax.plot(epochs, means, color=color, linewidth=2, label=label)
        ax.fill_between(epochs, means - stds, means + stds, color=color, alpha=0.18, linewidth=0)

    if args.stage1_epochs is not None:
        stage_color = "#e34948"  # dataviz skill categorical slot 8 (red)
        ax.axvline(args.stage1_epochs, color=stage_color, linestyle="--", linewidth=1)
        ax.text(
            args.stage1_epochs, ax.get_ylim()[1], f" {args.stage1_label}",
            color=stage_color, fontsize=9, va="top", ha="left",
        )

    ax.set_xlabel("Epoch", color=PRIMARY_INK, labelpad=10)
    ax.set_ylabel(args.ylabel or args.metric, color=PRIMARY_INK, labelpad=10)
    ax.grid(True, color=GRIDLINE, linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(AXIS)
    ax.tick_params(colors=PRIMARY_INK)
    ax.legend(frameon=False, labelcolor=PRIMARY_INK)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output, dpi=args.dpi, facecolor=SURFACE)
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
