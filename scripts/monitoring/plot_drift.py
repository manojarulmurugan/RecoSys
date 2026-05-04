"""Generate drift visualization figures from reports/drift_report.json.

Produces two figures in reports/figures/:
  item_popularity_drift.png  — top-30 item frequency bar chart, Jan vs Mar
  popularity_rank_scatter.png — Jan rank vs Mar rank scatter (off-diagonal = shifted items)

Usage:
    python scripts/monitoring/plot_drift.py
    python scripts/monitoring/plot_drift.py --report reports/drift_report.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


FIGURES_DIR = Path("reports/figures")
REPORT_PATH = Path("reports/drift_report.json")


def _load_report(report_path: Path) -> dict:
    if not report_path.exists():
        raise FileNotFoundError(
            f"Report not found at {report_path}. "
            "Run compute_drift.py first."
        )
    return json.loads(report_path.read_text())


def plot_popularity_bar(report: dict, out_path: Path) -> None:
    """Overlaid bar chart of top-20 item frequencies in train vs test window."""
    n = min(20, len(report["train_top20_items"]), len(report["test_top20_items"]))

    train_items = [str(i) for i in report["train_top20_items"][:n]]
    train_probs = np.array(report["train_top20_probs"][:n]) * 100  # percent

    # Align test probs to train_items order (item may be at a different rank)
    test_item_map = dict(zip(
        [str(i) for i in report["test_top20_items"]],
        report["test_top20_probs"]
    ))
    test_probs = np.array([test_item_map.get(it, 0.0) for it in train_items]) * 100

    x = np.arange(n)
    width = 0.38

    fig, ax = plt.subplots(figsize=(14, 5))
    bars_train = ax.bar(x - width / 2, train_probs, width,
                        label=f"{report['train_window']} (reference)",
                        color="#2196F3", alpha=0.85)
    bars_test  = ax.bar(x + width / 2, test_probs, width,
                        label=f"{report['test_window']} (COVID-shifted)",
                        color="#FF5722", alpha=0.85)

    ax.set_yscale("log")
    ax.set_ylabel("Share of events (%)", fontsize=11)
    ax.set_xlabel("Item ID (top-20 by train popularity)", fontsize=11)
    ax.set_title(
        f"Item Popularity Distribution Shift\n"
        f"JSD={report['jsd_normalized']:.3f}  |  "
        f"Top-50 overlap={report['top50_item_overlap_pct']:.0f}%  |  "
        f"{'ALERT' if report['alert'] else 'Stable'}",
        fontsize=12,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(train_items, rotation=45, ha="right", fontsize=8)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_rank_scatter(report: dict, out_path: Path) -> None:
    """Scatter plot: Jan popularity rank vs Mar popularity rank for top-N items."""
    n = min(len(report["train_top20_items"]), len(report["test_top20_items"]))

    train_items = [str(i) for i in report["train_top20_items"][:n]]
    test_items  = [str(i) for i in report["test_top20_items"][:n]]

    # Rank maps (1-indexed)
    train_rank = {item: rank + 1 for rank, item in enumerate(train_items)}
    test_rank  = {item: rank + 1 for rank, item in enumerate(test_items)}

    # Items that appear in both top lists
    shared = set(train_items) & set(test_items)
    x_ranks = [train_rank[it] for it in shared]
    y_ranks = [test_rank[it]  for it in shared]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(x_ranks, y_ranks, alpha=0.7, s=60, color="#9C27B0")

    # Diagonal = no rank change
    lim = max(n, 1) + 1
    ax.plot([1, lim], [1, lim], "k--", alpha=0.4, linewidth=1, label="No change")

    ax.set_xlabel(f"Popularity rank in {report['train_window']}", fontsize=11)
    ax.set_ylabel(f"Popularity rank in {report['test_window']}", fontsize=11)
    ax.set_title(
        f"Popularity Rank Shift: {report['train_window']} → {report['test_window']}\n"
        f"{len(shared)}/{n} top items shared  |  JSD={report['jsd_normalized']:.3f}",
        fontsize=11,
    )
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default=str(REPORT_PATH))
    args = parser.parse_args()

    report = _load_report(Path(args.report))
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Generating drift figures from {args.report} …")
    plot_popularity_bar(report, FIGURES_DIR / "item_popularity_drift.png")
    plot_rank_scatter(report,   FIGURES_DIR / "popularity_rank_scatter.png")

    print(f"\nDrift summary:")
    print(f"  JSD (normalized)    : {report['jsd_normalized']:.4f}")
    print(f"  Top-50 overlap      : {report['top50_item_overlap_pct']:.1f}%")
    print(f"  Alert               : {'YES — ' + report['narrative'] if report['alert'] else 'No'}")


if __name__ == "__main__":
    main()
