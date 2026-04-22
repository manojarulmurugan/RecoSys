"""Plot training loss vs Recall@10 for the 500k two-tower run.

Hardcoded data from the 30-epoch 500k experiment (eval every 5 epochs).
Produces a dual-axis figure exposing the training/eval decoupling pattern:
loss keeps improving while Recall@10 stagnates or slightly regresses.

Output
──────
  reports/figures/500k_training_curves.png

Usage
─────
  python scripts/two_tower/plot_training_curves.py
"""

import pathlib

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── Data ──────────────────────────────────────────────────────────────────────

EPOCHS  = [5, 10, 15, 20, 25, 30]
LOSSES  = [5.0083, 4.6558, 4.4835, 4.3672, 4.2882, 4.2533]
RECALLS = [0.0088, 0.0086, 0.0080, 0.0082, 0.0083, 0.0083]

BEST_RECALL_EPOCH = EPOCHS[int(np.argmax(RECALLS))]
BEST_RECALL_VAL   = max(RECALLS)

# ── Output path ───────────────────────────────────────────────────────────────

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
OUT_PATH   = _REPO_ROOT / "reports" / "figures" / "500k_training_curves.png"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# ── Pearson correlation ────────────────────────────────────────────────────────

pearson_r = float(np.corrcoef(LOSSES, RECALLS)[0, 1])

print("=" * 58)
print("  500k Two-Tower — Loss vs Recall@10 correlation")
print("=" * 58)
print(f"  Epochs evaluated   : {EPOCHS}")
print(f"  Train losses       : {LOSSES}")
print(f"  Recall@10 values   : {RECALLS}")
print(f"  Best Recall@10     : {BEST_RECALL_VAL:.4f}  (epoch {BEST_RECALL_EPOCH})")
print()
print(f"  Pearson r (loss vs recall) : {pearson_r:+.4f}")
if pearson_r < -0.7:
    print("  Interpretation : STRONG NEGATIVE correlation — training loss")
    print("                   is a reliable proxy for recall improvement.")
elif pearson_r < -0.3:
    print("  Interpretation : WEAK NEGATIVE correlation — partial coupling;")
    print("                   loss improvements do not reliably predict recall.")
else:
    print("  Interpretation : NEAR-ZERO / POSITIVE correlation — DECOUPLING.")
    print("                   Training objective is NOT driving recall gains.")
    print("                   Improving loss further will not help retrieval.")
print("=" * 58)

# ── Figure ────────────────────────────────────────────────────────────────────

BLUE   = "#2166ac"
ORANGE = "#d6604d"
SHADE  = "#f4a582"
VLINE  = "#4dac26"

fig, ax_loss = plt.subplots(figsize=(11, 6))
ax_recall    = ax_loss.twinx()

# ── Loss curve (left axis) ────────────────────────────────────────────────────
loss_line, = ax_loss.plot(
    EPOCHS, LOSSES,
    color=BLUE, linewidth=2.2, marker="o", markersize=7,
    label="Train Loss",
    zorder=3,
)
ax_loss.set_ylabel("Training Loss", color=BLUE, fontsize=12, labelpad=10)
ax_loss.tick_params(axis="y", labelcolor=BLUE)
ax_loss.set_ylim(3.8, 5.3)

# ── Recall curve (right axis) ─────────────────────────────────────────────────
recall_line, = ax_recall.plot(
    EPOCHS, RECALLS,
    color=ORANGE, linewidth=2.2, marker="s", markersize=7,
    label="Recall@10",
    zorder=3,
)
ax_recall.set_ylabel("Recall@10", color=ORANGE, fontsize=12, labelpad=10)
ax_recall.tick_params(axis="y", labelcolor=ORANGE)

# Keep recall y-axis range tight so the (small) variance is visible
recall_min = min(RECALLS)
recall_max = max(RECALLS)
recall_pad = (recall_max - recall_min) * 1.8 or 0.001
ax_recall.set_ylim(recall_min - recall_pad, recall_max + recall_pad)
ax_recall.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))

# ── Divergence shaded region ──────────────────────────────────────────────────
ax_loss.axvspan(
    BEST_RECALL_EPOCH, EPOCHS[-1],
    alpha=0.12, color=SHADE, zorder=1,
    label="Divergence zone\n(loss↓, recall stagnates)",
)

# ── Best-recall vertical dashed line ─────────────────────────────────────────
ax_loss.axvline(
    x=BEST_RECALL_EPOCH,
    color=VLINE, linewidth=1.6, linestyle="--",
    zorder=2, label=f"Best Recall (epoch {BEST_RECALL_EPOCH})",
)
ax_loss.annotate(
    f"Best Recall\nR@10={BEST_RECALL_VAL:.4f}",
    xy=(BEST_RECALL_EPOCH, ax_loss.get_ylim()[0]),
    xytext=(BEST_RECALL_EPOCH + 0.6, 3.95),
    fontsize=9, color=VLINE,
    arrowprops=dict(arrowstyle="->", color=VLINE, lw=1.2),
)

# ── Data-point labels — loss ──────────────────────────────────────────────────
for ep, lo in zip(EPOCHS, LOSSES):
    ax_loss.annotate(
        f"{lo:.4f}", xy=(ep, lo),
        xytext=(0, 9), textcoords="offset points",
        ha="center", fontsize=8, color=BLUE,
    )

# ── Data-point labels — recall ────────────────────────────────────────────────
for ep, rc in zip(EPOCHS, RECALLS):
    ax_recall.annotate(
        f"{rc:.4f}", xy=(ep, rc),
        xytext=(0, -15), textcoords="offset points",
        ha="center", fontsize=8, color=ORANGE,
    )

# ── Pearson annotation box ────────────────────────────────────────────────────
pearson_label = (
    f"Pearson r = {pearson_r:+.3f}\n"
    + ("DECOUPLED: loss ↛ recall" if pearson_r >= -0.3 else "Weak coupling")
)
ax_loss.text(
    0.98, 0.97, pearson_label,
    transform=ax_loss.transAxes,
    ha="right", va="top", fontsize=9,
    bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
               edgecolor="#999999", alpha=0.85),
)

# ── Axes and title ────────────────────────────────────────────────────────────
ax_loss.set_xlabel("Epoch", fontsize=12)
ax_loss.set_xticks(EPOCHS)
ax_loss.set_xlim(2, 33)

plt.title(
    "500k Two-Tower: Loss vs Recall@10 — Training/Eval Decoupling",
    fontsize=13, fontweight="bold", pad=14,
)

# ── Combined legend ───────────────────────────────────────────────────────────
handles_l, labels_l = ax_loss.get_legend_handles_labels()
handles_r, labels_r = ax_recall.get_legend_handles_labels()
ax_loss.legend(
    handles_l + handles_r, labels_l + labels_r,
    loc="lower left", fontsize=9,
    framealpha=0.9, edgecolor="#cccccc",
)

ax_loss.spines["left"].set_color(BLUE)
ax_recall.spines["right"].set_color(ORANGE)

plt.tight_layout()
fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
plt.close(fig)

print(f"\n  Figure saved to: {OUT_PATH}")
