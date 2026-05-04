"""Compute item-popularity distribution shift between two time windows.

Uses the REES46 cleaned parquet data. Default comparison:
    train window: 2020-01 (January 2020, reference distribution)
    test window:  2020-03 (March 2020, COVID-shifted distribution)

Jensen-Shannon divergence quantifies how much the item popularity distribution
has shifted. JSD > 0.1 triggers an alert (threshold chosen to be sensitive to
the COVID-scale popularity inversion while ignoring normal month-to-month noise).

Usage:
    python scripts/monitoring/compute_drift.py
    python scripts/monitoring/compute_drift.py --train-window 2020-01 --test-window 2020-04
    python scripts/monitoring/compute_drift.py --data-source local --local-data-path artifacts/
"""

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

GCS_RAW_PATH    = "gs://recosys-data-bucket/raw/"
REPORT_PATH     = Path("reports/drift_report.json")
ALERT_THRESHOLD = 0.1  # JSD_normalized above this → alert
TOP_N_OVERLAP   = 50   # top-N items for overlap metric

_MONTH_ABBR = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
               7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}


def _item_col(df: pd.DataFrame) -> str | None:
    """Return the item column name (item_id or product_id), or None if absent."""
    for col in ("item_id", "product_id"):
        if col in df.columns:
            return col
    return None


def _load_window(source: str, window: str, local_path: Path) -> pd.DataFrame:
    """Load events for a given YYYY-MM window. Returns df with column 'item_id'."""
    year, month = int(window.split("-")[0]), int(window.split("-")[1])

    if source == "local":
        # Only use raw event parquets (have event_time + product_id)
        candidates = list((local_path / "500k" / "train_raw" / "train").glob("*.parquet"))
        if not candidates:
            candidates = list(local_path.glob("**/*.parquet"))
        dfs = []
        for p in candidates:
            try:
                df = pd.read_parquet(p)
                col = _item_col(df)
                if col is None or "event_time" not in df.columns:
                    continue
                df["event_time"] = pd.to_datetime(df["event_time"], errors="coerce")
                mask = (df["event_time"].dt.year == year) & (df["event_time"].dt.month == month)
                df = df[mask]
                if len(df) > 0:
                    dfs.append(df[[col]].rename(columns={col: "item_id"}))
            except Exception:
                pass
        if not dfs:
            raise ValueError(f"No data found for window {window} in {local_path}")
        return pd.concat(dfs, ignore_index=True)
    else:
        # GCS: read monthly raw CSV files (gs://recosys-data-bucket/raw/YYYY-Mon.csv)
        mon_abbr = _MONTH_ABBR[month]
        gcs_uri  = f"{GCS_RAW_PATH}{year}-{mon_abbr}.csv"
        print(f"  Reading {gcs_uri} …")
        try:
            df = pd.read_csv(gcs_uri, usecols=lambda c: c in ("item_id", "product_id", "event_time"))
        except Exception as exc:
            raise ValueError(f"Could not read {gcs_uri}: {exc}")
        col = _item_col(df)
        if col is None:
            raise ValueError(f"No item_id/product_id column in {gcs_uri}. Columns: {list(df.columns)}")
        return df[[col]].rename(columns={col: "item_id"})


def _popularity_distribution(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return (item_ids, probabilities) sorted descending by count."""
    counts = df["item_id"].value_counts()
    items = counts.index.to_numpy()
    probs = counts.values.astype(np.float64)
    probs = probs / probs.sum()
    return items, probs


def _jsd(p_items: np.ndarray, p_probs: np.ndarray,
         q_items: np.ndarray, q_probs: np.ndarray) -> float:
    """Jensen-Shannon divergence (natural log, bounded [0, ln2]).

    Aligns P and Q over the union vocabulary with epsilon smoothing.
    Returns JSD_normalized ∈ [0, 1] (divided by ln(2)).
    """
    eps = 1e-10
    all_items = np.union1d(p_items, q_items)
    p_map = dict(zip(p_items, p_probs))
    q_map = dict(zip(q_items, q_probs))

    P = np.array([p_map.get(i, 0.0) for i in all_items]) + eps
    Q = np.array([q_map.get(i, 0.0) for i in all_items]) + eps
    P /= P.sum()
    Q /= Q.sum()

    M = 0.5 * (P + Q)
    kl_pm = np.sum(P * np.log(P / M))
    kl_qm = np.sum(Q * np.log(Q / M))
    jsd   = 0.5 * (kl_pm + kl_qm)

    return float(jsd / math.log(2))  # normalize to [0, 1]


def _top_n_overlap(p_items: np.ndarray, q_items: np.ndarray, n: int = TOP_N_OVERLAP) -> float:
    """Fraction of top-N train items that appear in top-N test items."""
    p_top = set(p_items[:n])
    q_top = set(q_items[:n])
    return len(p_top & q_top) / n * 100.0


def _coverage(train_items: np.ndarray, test_items: np.ndarray) -> float:
    """Fraction of test items seen during training."""
    train_set = set(train_items)
    covered = sum(1 for i in test_items if i in train_set)
    return covered / len(test_items) * 100.0 if len(test_items) > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(description="Item popularity drift detection")
    parser.add_argument("--train-window", default="2020-01",
                        help="Reference window YYYY-MM (default: 2020-01)")
    parser.add_argument("--test-window",  default="2020-03",
                        help="Shifted window YYYY-MM (default: 2020-03, COVID period)")
    parser.add_argument("--data-source",  choices=["gcs", "local"], default="gcs")
    parser.add_argument("--local-data-path", default="artifacts/",
                        help="Root path for local data files (used when --data-source=local)")
    parser.add_argument("--alert-threshold", type=float, default=ALERT_THRESHOLD)
    args = parser.parse_args()

    local_path = Path(args.local_data_path)
    print(f"Computing drift: {args.train_window} (train) vs {args.test_window} (test)")
    print(f"Data source: {args.data_source}")

    print(f"\nLoading {args.train_window} data …")
    train_df = _load_window(args.data_source, args.train_window, local_path)
    print(f"  {len(train_df):,} events")

    print(f"Loading {args.test_window} data …")
    test_df = _load_window(args.data_source, args.test_window, local_path)
    print(f"  {len(test_df):,} events")

    print("\nComputing distributions …")
    p_items, p_probs = _popularity_distribution(train_df)
    q_items, q_probs = _popularity_distribution(test_df)

    print("Computing metrics …")
    jsd_normalized = _jsd(p_items, p_probs, q_items, q_probs)
    top50_overlap  = _top_n_overlap(p_items, q_items, TOP_N_OVERLAP)
    coverage       = _coverage(p_items, q_items)
    alert          = jsd_normalized > args.alert_threshold

    narrative = (
        f"Jensen-Shannon divergence of {jsd_normalized:.3f} "
        + ("EXCEEDS" if alert else "is below")
        + f" threshold {args.alert_threshold:.2f}"
        + (" — distribution shift detected (COVID-19 period)" if alert else " — distribution is stable")
    )

    report = {
        "run_timestamp":          datetime.utcnow().isoformat() + "Z",
        "train_window":           args.train_window,
        "test_window":            args.test_window,
        "jsd_normalized":         round(jsd_normalized, 4),
        "top50_item_overlap_pct": round(top50_overlap, 1),
        "test_item_coverage_pct": round(coverage, 1),
        "n_train_items":          int(len(p_items)),
        "n_test_items":           int(len(q_items)),
        "n_train_events":         int(len(train_df)),
        "n_test_events":          int(len(test_df)),
        "alert":                  alert,
        "alert_threshold":        args.alert_threshold,
        "narrative":              narrative,
        # Top-20 items for plotting
        "train_top20_items":      p_items[:20].tolist(),
        "train_top20_probs":      p_probs[:20].tolist(),
        "test_top20_items":       q_items[:20].tolist(),
        "test_top20_probs":       q_probs[:20].tolist(),
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, default=str))

    sep = "=" * 56
    print(f"\n{sep}")
    print("DRIFT REPORT")
    print(sep)
    print(f"  Train window          : {args.train_window}")
    print(f"  Test window           : {args.test_window} (COVID period)")
    print(f"  JSD (normalized)      : {jsd_normalized:.4f}  (threshold={args.alert_threshold})")
    print(f"  Top-50 item overlap   : {top50_overlap:.1f}%")
    print(f"  Test item coverage    : {coverage:.1f}%")
    print(f"  Train events          : {len(train_df):,}")
    print(f"  Test events           : {len(test_df):,}")
    print(f"  {'*** ALERT ***' if alert else 'Status: stable'}")
    print(f"  {narrative}")
    print(sep)
    print(f"  Report saved to: {REPORT_PATH}")

    if alert:
        raise SystemExit(1)  # exit code 1 for pipeline / CI integration


if __name__ == "__main__":
    main()
