"""Diagnostic: why does GRU4Rec invert across user cohorts?

Reported in ``reports/07_sequence_model_results.md``:
    Cold (3-10 events)    R@10 = 0.0728
    Medium (11-50 events) R@10 = 0.0393
    Warm  (51+ events)    R@10 = 0.0208

This is the OPPOSITE of what should happen — warm users with rich
training history should be EASIER to recommend for, not harder.  This
script tests two hypotheses without requiring a trained model:

  H1: ``filter_seen=True`` removes a much larger fraction of warm users'
      ground-truth items, mechanically capping warm Recall@10.
      Test: compute ``% of GT items already in train_pairs`` per cohort.

  H2: A simple popularity baseline (which has no personalization) ALSO
      shows the cold > warm pattern.  If yes, the inversion is a
      property of the eval setup, not of the model — meaning the
      session-based reframe has to fix the eval setup, not just the
      training objective.
      Test: compute Recall@10 of "global popularity" per cohort with
      filter_seen=True (matches the existing eval protocol exactly).

Outputs (written to ``artifacts/diagnostics/``):
    cold_warm_filter_pct.png         — H1 plot
    cold_warm_popularity_recall.png  — H2 plot
    cold_warm_summary.json           — numeric summary for the writeup
    cold_warm_summary.md             — markdown table for the writeup

Usage:
    python scripts/diagnostics/diagnostic_cold_warm_inversion.py
"""

from __future__ import annotations

import json
import os
import pathlib
import pickle
import sys
import time
from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))


# ── GCP credentials helper ────────────────────────────────────────────────────

def _ensure_gcp_credentials() -> None:
    existing = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if existing and pathlib.Path(existing).expanduser().is_file():
        return
    for candidate in (
        _REPO_ROOT / "secrets" / "recosys-service-account.json",
        _REPO_ROOT / "recosys-service-account.json",
        pathlib.Path("/content/recosys-service-account.json"),
    ):
        if candidate.is_file():
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(candidate.resolve())
            print(f"  GCP credentials: {candidate}")
            return
    print("WARNING: No GCP credentials found; GCS reads may fail.")


# ── Configuration ─────────────────────────────────────────────────────────────

ARTIFACTS_DIR = _REPO_ROOT / "artifacts" / "500k"
DIAG_DIR      = _REPO_ROOT / "artifacts" / "diagnostics"
TEST_GCS_PATH = "gs://recosys-data-bucket/samples/users_sample_500k/test/"

COHORTS: list[tuple[str, int, int]] = [
    ("cold",   3,  10),
    ("medium", 11, 50),
    ("warm",   51, int(1e9)),
]

# Numbers from reports/07_sequence_model_results.md (final epoch GRU4Rec eval).
GRU4REC_REFERENCE: dict[str, float] = {
    "overall": 0.0515,
    "cold":    0.0728,
    "medium":  0.0393,
    "warm":    0.0208,
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _section(title: str) -> None:
    print()
    print("=" * 64)
    print(f"  {title}")
    print("=" * 64)


def _step(msg: str) -> float:
    print(f"\n  >  {msg}")
    return time.time()


def _done(t0: float, label: str = "") -> None:
    elapsed = int(time.time() - t0)
    suffix  = f" — {label}" if label else ""
    print(f"     done in {elapsed // 60}m {elapsed % 60}s{suffix}")


def _cohort_for(count: int) -> str | None:
    for name, lo, hi in COHORTS:
        if lo <= count <= hi:
            return name
    return None  # users with < 3 train events excluded from cohorts


def _plot_save(
    cohort_to_value: dict[str, float],
    out_path: pathlib.Path,
    title: str,
    ylabel: str,
    bar_color: str = "#3b6cb0",
    overlay: dict[str, float] | None = None,
    overlay_label: str | None = None,
) -> None:
    """Save a horizontal bar plot of cold/medium/warm values to ``out_path``."""
    import matplotlib
    matplotlib.use("Agg")  # no display in headless environments
    import matplotlib.pyplot as plt

    cohorts = ["cold", "medium", "warm"]
    values  = [cohort_to_value[c] for c in cohorts]

    fig, ax = plt.subplots(figsize=(7, 3.5))
    bars = ax.bar(cohorts, values, color=bar_color, edgecolor="#1f3d6c", width=0.55)
    ax.set_title(title, fontsize=12, pad=12)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_xlabel("User cohort (training interaction count)", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bar, v in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{v:.3f}" if v < 10 else f"{v:.1f}",
            ha="center", va="bottom", fontsize=10, color="#1f3d6c",
        )

    if overlay is not None:
        overlay_values = [overlay[c] for c in cohorts]
        ax.plot(
            cohorts, overlay_values, marker="o", linestyle="--",
            color="#cc4422", label=overlay_label or "overlay",
        )
        for i, v in enumerate(overlay_values):
            ax.text(
                i, v, f"{v:.3f}", ha="center", va="bottom",
                fontsize=9, color="#cc4422",
            )
        ax.legend(loc="upper right", fontsize=9, frameon=False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"     saved {out_path}")


# ── Recall computation ────────────────────────────────────────────────────────

def _compute_popularity_recall(
    train_pairs: pd.DataFrame,
    ground_truth: dict[int, set[int]],
    seen_items: dict[int, set[int]],
    eval_users: list[int],
    k: int = 10,
    filter_seen: bool = True,
) -> dict[int, float]:
    """Recall@k for the 'recommend most popular items' baseline, per user.

    Mirrors the protocol used in notebooks/07_two_tower_500k_colab.ipynb
    (popularity = sum of confidence_score across train_pairs, with
    per-user seen-item filtering when filter_seen=True).
    """
    item_pop = (
        train_pairs.groupby("item_idx")["confidence_score"]
        .sum()
        .sort_values(ascending=False)
    )
    top_items = item_pop.index.to_numpy(dtype=np.int64)  # most popular first

    per_user_recall: dict[int, float] = {}
    for u in eval_users:
        gt   = ground_truth.get(u, set())
        if not gt:
            continue
        seen = seen_items.get(u, set()) if filter_seen else set()
        hits = 0
        seen_count = 0
        # Walk through items in popularity order, skip seen, take first k.
        for iid in top_items:
            iid = int(iid)
            if iid in seen:
                continue
            if iid in gt:
                hits += 1
            seen_count += 1
            if seen_count >= k:
                break
        per_user_recall[u] = hits / min(len(gt), k)
    return per_user_recall


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    _section("Cold/Warm Inversion Diagnostic")
    print(f"  Artifacts dir : {ARTIFACTS_DIR}")
    print(f"  Test GCS path : {TEST_GCS_PATH}")
    print(f"  Output dir    : {DIAG_DIR}")
    print(f"  Cohorts       : {COHORTS}")

    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    _ensure_gcp_credentials()

    # ── Load vocabs + train_pairs ─────────────────────────────────────────────
    t0 = _step(f"Loading vocabs.pkl + train_pairs.parquet")
    with open(ARTIFACTS_DIR / "vocabs.pkl", "rb") as f:
        vocabs: dict[str, Any] = pickle.load(f)
    user2idx = vocabs["user2idx"]
    item2idx = vocabs["item2idx"]
    train_pairs = pd.read_parquet(ARTIFACTS_DIR / "train_pairs.parquet")
    _done(t0, label=(f"{len(user2idx):,} users / {len(item2idx):,} items / "
                     f"{len(train_pairs):,} train pairs"))

    required = {"user_idx", "item_idx", "confidence_score"}
    if not required.issubset(train_pairs.columns):
        raise RuntimeError(
            f"train_pairs.parquet missing columns: {required - set(train_pairs.columns)}"
        )

    # ── Build seen_items + per-user training count → cohort ───────────────────
    t0 = _step("Building seen_items + per-user cohort assignment")
    seen_items: dict[int, set[int]] = defaultdict(set)
    train_count: dict[int, int]     = defaultdict(int)
    for uidx, iidx in zip(
        train_pairs["user_idx"].to_numpy(dtype=np.int64),
        train_pairs["item_idx"].to_numpy(dtype=np.int64),
    ):
        u = int(uidx); i = int(iidx)
        seen_items[u].add(i)
        train_count[u] += 1

    cohort_of: dict[int, str] = {
        u: _cohort_for(c) for u, c in train_count.items()
    }
    cohort_of = {u: c for u, c in cohort_of.items() if c is not None}
    _done(t0, label=f"{len(seen_items):,} users with seen-items, "
                    f"{len(cohort_of):,} in a cohort")

    cohort_sizes = {name: 0 for name, _, _ in COHORTS}
    for c in cohort_of.values():
        cohort_sizes[c] += 1
    print(f"     cohort sizes: {cohort_sizes}")

    # ── Load Feb test data + build ground truth ──────────────────────────────
    t0 = _step(f"Reading Feb test data from {TEST_GCS_PATH}")
    test_df = pd.read_parquet(
        TEST_GCS_PATH,
        columns=["user_id", "product_id", "event_type"],
    )
    test_df["user_idx"] = test_df["user_id"].map(user2idx)
    test_df["item_idx"] = test_df["product_id"].map(item2idx)
    test_df = test_df.dropna(subset=["user_idx", "item_idx"])
    test_df = test_df[test_df["event_type"].isin({"cart", "purchase"})]
    test_df["user_idx"] = test_df["user_idx"].astype(np.int64)
    test_df["item_idx"] = test_df["item_idx"].astype(np.int64)
    test_df = test_df.drop_duplicates(subset=["user_idx", "item_idx"])
    _done(t0, label=f"{len(test_df):,} (user,item) GT pairs over "
                    f"{test_df['user_idx'].nunique():,} users")

    ground_truth: dict[int, set[int]] = (
        test_df.groupby("user_idx")["item_idx"].apply(set).to_dict()
    )

    # ── Define the eval-user set: cohort-assigned users with GT ──────────────
    eval_users = [u for u in ground_truth.keys() if u in cohort_of]
    print(f"     eval users (have GT AND in a cohort) : {len(eval_users):,}")

    # ── Hypothesis 1: % of GT items in seen_items, per cohort ────────────────
    _section("H1 — How much of warm users' GT is filtered by filter_seen?")

    cohort_filter_total: dict[str, int] = {n: 0 for n, _, _ in COHORTS}
    cohort_filter_hit:   dict[str, int] = {n: 0 for n, _, _ in COHORTS}

    for u in eval_users:
        c    = cohort_of[u]
        gt   = ground_truth.get(u, set())
        seen = seen_items.get(u, set())
        cohort_filter_total[c] += len(gt)
        cohort_filter_hit  [c] += len(gt & seen)

    cohort_filter_pct: dict[str, float] = {
        c: 100.0 * cohort_filter_hit[c] / max(cohort_filter_total[c], 1)
        for c in cohort_filter_total
    }

    print(f"\n  {'cohort':<8} {'GT items':>10} {'in seen':>10} {'% filtered':>12}")
    print("  " + "-" * 44)
    for name, _, _ in COHORTS:
        print(
            f"  {name:<8} "
            f"{cohort_filter_total[name]:>10,} "
            f"{cohort_filter_hit[name]:>10,} "
            f"{cohort_filter_pct[name]:>11.1f}%"
        )

    # ── Hypothesis 2: Popularity baseline R@10 per cohort ────────────────────
    _section("H2 — Does popularity also invert across cohorts?")

    t0 = _step("Computing popularity Recall@10 with filter_seen=True (matches eval protocol)")
    per_user_pop_r10 = _compute_popularity_recall(
        train_pairs   = train_pairs,
        ground_truth  = ground_truth,
        seen_items    = seen_items,
        eval_users    = eval_users,
        k             = 10,
        filter_seen   = True,
    )
    _done(t0)

    cohort_pop_r10:    dict[str, float] = {}
    cohort_pop_users:  dict[str, int]   = {}
    for name, _, _ in COHORTS:
        members = [per_user_pop_r10[u] for u in eval_users
                   if cohort_of[u] == name and u in per_user_pop_r10]
        cohort_pop_users[name] = len(members)
        cohort_pop_r10  [name] = float(np.mean(members)) if members else 0.0

    print(f"\n  {'cohort':<8} {'N users':>10} {'pop R@10':>10}  {'GRU4Rec R@10':>14}")
    print("  " + "-" * 50)
    for name, _, _ in COHORTS:
        print(
            f"  {name:<8} "
            f"{cohort_pop_users[name]:>10,} "
            f"{cohort_pop_r10[name]:>10.4f}  "
            f"{GRU4REC_REFERENCE.get(name, float('nan')):>14.4f}"
        )

    # Also report overall (sanity vs the published 0.0646 popularity number).
    overall_pop_r10 = float(np.mean(list(per_user_pop_r10.values()))) \
        if per_user_pop_r10 else 0.0
    print(f"\n  Overall popularity R@10 = {overall_pop_r10:.4f}  "
          f"(reference from notebooks/07: 0.0646)")

    # ── Plots ────────────────────────────────────────────────────────────────
    _section("Generating plots")

    out_h1 = DIAG_DIR / "cold_warm_filter_pct.png"
    _plot_save(
        cohort_to_value = cohort_filter_pct,
        out_path        = out_h1,
        title           = "H1: % of GT items already seen in training (filtered by filter_seen=True)",
        ylabel          = "% of GT filtered",
        bar_color       = "#3b6cb0",
    )

    out_h2 = DIAG_DIR / "cold_warm_popularity_recall.png"
    _plot_save(
        cohort_to_value = cohort_pop_r10,
        out_path        = out_h2,
        title           = "H2: Popularity Recall@10 by cohort (filter_seen=True)",
        ylabel          = "Recall@10",
        bar_color       = "#3b8a4f",
        overlay         = {c: GRU4REC_REFERENCE[c] for c in ["cold", "medium", "warm"]},
        overlay_label   = "GRU4Rec V7 (reference)",
    )

    # ── Summary artifacts ────────────────────────────────────────────────────
    summary: dict[str, Any] = {
        "n_eval_users":         len(eval_users),
        "cohort_sizes":         cohort_sizes,
        "filter_pct_by_cohort": cohort_filter_pct,
        "popularity_r10_by_cohort": cohort_pop_r10,
        "popularity_r10_overall":   overall_pop_r10,
        "gru4rec_v7_reference": GRU4REC_REFERENCE,
        "popularity_overall_reference_from_notebook_07": 0.0646,
    }
    out_json = DIAG_DIR / "cold_warm_summary.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Wrote {out_json}")

    # Markdown summary for direct paste into the writeup.
    md = (
        "# Cold/Warm Inversion Diagnostic\n\n"
        f"Eval users: {len(eval_users):,} (Feb cart/purchase, in a cohort).\n\n"
        "## H1 — % of GT items already in training (filtered by `filter_seen=True`)\n\n"
        "| Cohort | # GT items | # filtered | % filtered |\n"
        "|---|---|---|---|\n"
    )
    for name, _, _ in COHORTS:
        md += (
            f"| {name} | {cohort_filter_total[name]:,} | "
            f"{cohort_filter_hit[name]:,} | "
            f"{cohort_filter_pct[name]:.1f}% |\n"
        )
    md += (
        "\n## H2 — Popularity baseline R@10 by cohort (vs GRU4Rec V7 reference)\n\n"
        "| Cohort | N users | Popularity R@10 | GRU4Rec V7 R@10 |\n"
        "|---|---|---|---|\n"
    )
    for name, _, _ in COHORTS:
        md += (
            f"| {name} | {cohort_pop_users[name]:,} | "
            f"{cohort_pop_r10[name]:.4f} | "
            f"{GRU4REC_REFERENCE.get(name, float('nan')):.4f} |\n"
        )
    md += (
        f"\n**Overall popularity R@10 = {overall_pop_r10:.4f}** "
        "(reference from notebook 07: 0.0646)\n\n"
        "## Interpretation\n\n"
        "If H1 shows warm ≫ cold (e.g. warm > 50% filtered, cold < 20%), the "
        "filter_seen mechanism is mechanically capping warm users' Recall.\n\n"
        "If H2 shows popularity also follows cold > medium > warm, the cold/warm "
        "inversion is a property of the eval setup, not personalization. The "
        "session-based reframe (next-item-in-session, no filter_seen) breaks "
        "this artifact by construction.\n"
    )
    out_md = DIAG_DIR / "cold_warm_summary.md"
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"  Wrote {out_md}")

    _section("Done — see artifacts/diagnostics/ for plots + summary")


if __name__ == "__main__":
    main()
