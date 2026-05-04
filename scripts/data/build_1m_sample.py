"""
build_1m_sample.py

End-to-end pipeline: 1M-user sample → GCS splits → vocabs.pkl → session sequences → GCS upload.

Phases:
  1. BigQuery: create events_sample_1m (deterministic via FARM_FINGERPRINT), train/test splits,
     export to gs://recosys-data-bucket/samples/users_sample_1m/{train,test}/
  2. Local: build artifacts/1m/vocabs.pkl from train parquet (user2idx + item2idx)
  3. Subprocess: scripts/sequence/build_session_sequences.py --sample-name users_sample_1m
                 --artifacts-dir artifacts/1m
  4. GCS upload: artifacts/1m/ → gs://recosys-data-bucket/data/1M/

Usage:
    python scripts/data/build_1m_sample.py [--skip-bq] [--skip-sequences] [--skip-upload]

Auth:
    Reads GOOGLE_APPLICATION_CREDENTIALS env var, falling back to
    secrets/recosys-service-account.json.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import pickle
import subprocess
import sys
import time

import pandas as pd
from google.cloud import bigquery, storage
from google.oauth2 import service_account

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent


# ── GCP credentials ───────────────────────────────────────────────────────────

def _ensure_gcp_credentials() -> None:
    """Set GOOGLE_APPLICATION_CREDENTIALS so PyArrow GCS backend can authenticate.

    pd.read_parquet("gs://...") uses PyArrow's GCS filesystem which reads only
    from the env var — it does not accept credentials passed explicitly.
    """
    existing = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if existing and pathlib.Path(existing).expanduser().is_file():
        print(f"  GCP credentials  : {existing}")
        return
    for candidate in (
        _REPO_ROOT / "secrets" / "recosys-service-account.json",
        _REPO_ROOT / "recosys-service-account.json",
        pathlib.Path.home() / "secrets" / "recosys-service-account.json",
    ):
        if candidate.is_file():
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(candidate.resolve())
            print(f"  GCP credentials  : {candidate}")
            return
    print(
        "  WARNING: No service-account key found. "
        "GCS reads via PyArrow will fail unless ADC is configured."
    )


# ── Config ────────────────────────────────────────────────────────────────────
PROJECT_ID   = "recosys-489001"
DATASET_ID   = "recosys"
BUCKET       = "recosys-data-bucket"
SAMPLE_NAME  = "users_sample_1m"
TARGET_USERS = 1_000_000

TRAIN_MONTHS = ("2019-10", "2019-11", "2019-12", "2020-01")
TEST_MONTH   = "2020-02"

ARTIFACTS_DIR = _REPO_ROOT / "artifacts" / "1m"
SEQ_SCRIPT    = _REPO_ROOT / "scripts" / "sequence" / "build_session_sequences.py"

TRAIN_GCS = f"gs://{BUCKET}/samples/{SAMPLE_NAME}/train/*.parquet"
TEST_GCS  = f"gs://{BUCKET}/samples/{SAMPLE_NAME}/test/*.parquet"

EXPORT_COLUMNS = [
    "CAST(event_time AS TIMESTAMP) AS event_time",
    "event_type",
    "product_id",
    "category_id",
    "category_code",
    "brand",
    "price",
    "user_id",
    "user_session",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _elapsed(start: float) -> str:
    s = int(time.time() - start)
    return f"{s // 60}m {s % 60}s"


def _section(title: str) -> None:
    print()
    print("=" * 64)
    print(f"  {title}")
    print("=" * 64)


def _step(msg: str) -> float:
    print(f"\n  >  {msg}")
    return time.time()


def _done(t0: float, label: str = "") -> None:
    suffix = f" — {label}" if label else ""
    print(f"     done in {_elapsed(t0)}{suffix}")


def _run_bq(bq: bigquery.Client, sql: str) -> bigquery.QueryJob:
    job = bq.query(sql)
    job.result()
    if job.errors:
        raise RuntimeError(f"BigQuery job failed: {job.errors}")
    return job


# ── Auth ──────────────────────────────────────────────────────────────────────

def _build_bq_client() -> bigquery.Client:
    key_path = os.environ.get(
        "GOOGLE_APPLICATION_CREDENTIALS",
        str(_REPO_ROOT / "secrets" / "recosys-service-account.json"),
    )
    creds = service_account.Credentials.from_service_account_file(
        key_path,
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )
    client = bigquery.Client(project=PROJECT_ID, credentials=creds)
    print(f"  Authenticated as : {creds.service_account_email}")
    print(f"  Project          : {client.project}")
    return client


def _build_gcs_client() -> storage.Client:
    key_path = os.environ.get(
        "GOOGLE_APPLICATION_CREDENTIALS",
        str(_REPO_ROOT / "secrets" / "recosys-service-account.json"),
    )
    creds = service_account.Credentials.from_service_account_file(
        key_path,
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )
    return storage.Client(project=PROJECT_ID, credentials=creds)


# ── Phase 1: BigQuery ─────────────────────────────────────────────────────────

def create_sample_table(bq: bigquery.Client) -> None:
    table_ref   = f"{PROJECT_ID}.{DATASET_ID}.events_sample_1m"
    source_ref  = f"{PROJECT_ID}.{DATASET_ID}.events_clean"
    month_list  = ", ".join(f"'{m}'" for m in TRAIN_MONTHS)

    # FARM_FINGERPRINT gives deterministic (reproducible) user selection,
    # equivalent to a fixed random seed. RAND() is non-deterministic in BQ.
    sql = f"""
    CREATE OR REPLACE TABLE `{table_ref}` AS
    WITH ranked AS (
      SELECT
        user_id,
        ROW_NUMBER() OVER (ORDER BY FARM_FINGERPRINT(CAST(user_id AS STRING))) AS rn
      FROM (SELECT DISTINCT user_id FROM `{source_ref}`)
    )
    SELECT e.*
    FROM `{source_ref}` e
    INNER JOIN (SELECT user_id FROM ranked WHERE rn <= {TARGET_USERS}) s
      USING (user_id)
    """

    t0 = _step(f"Creating `{table_ref}` ({TARGET_USERS:,} users, deterministic)")
    _run_bq(bq, sql)
    _done(t0)


def create_splits(bq: bigquery.Client) -> None:
    sample_ref = f"{PROJECT_ID}.{DATASET_ID}.events_sample_1m"
    train_ref  = f"{PROJECT_ID}.{DATASET_ID}.train_1m"
    test_ref   = f"{PROJECT_ID}.{DATASET_ID}.test_1m"
    month_list = ", ".join(f"'{m}'" for m in TRAIN_MONTHS)

    train_sql = f"""
    CREATE OR REPLACE TABLE `{train_ref}` AS
    SELECT * FROM `{sample_ref}`
    WHERE FORMAT_TIMESTAMP('%Y-%m', event_time) IN ({month_list})
    """
    test_sql = f"""
    CREATE OR REPLACE TABLE `{test_ref}` AS
    SELECT * FROM `{sample_ref}`
    WHERE FORMAT_TIMESTAMP('%Y-%m', event_time) = '{TEST_MONTH}'
    """

    t0 = _step(f"Creating train split `{train_ref}`")
    _run_bq(bq, train_sql)
    _done(t0)

    t0 = _step(f"Creating test split  `{test_ref}`")
    _run_bq(bq, test_sql)
    _done(t0)


def export_splits(bq: bigquery.Client) -> None:
    cols = ",\n      ".join(EXPORT_COLUMNS)

    for table_name, gcs_uri in [
        ("train_1m", f"gs://{BUCKET}/samples/{SAMPLE_NAME}/train/*.parquet"),
        ("test_1m",  f"gs://{BUCKET}/samples/{SAMPLE_NAME}/test/*.parquet"),
    ]:
        table_ref = f"{PROJECT_ID}.{DATASET_ID}.{table_name}"
        sql = f"""
        EXPORT DATA
          OPTIONS (
            uri       = '{gcs_uri}',
            format    = 'PARQUET',
            overwrite = true
          )
        AS
        SELECT
          {cols}
        FROM `{table_ref}`
        """
        t0 = _step(f"Exporting `{table_name}` → {gcs_uri}")
        _run_bq(bq, sql)
        _done(t0)


def validate_splits(bq: bigquery.Client) -> None:
    train_ref = f"{PROJECT_ID}.{DATASET_ID}.train_1m"
    test_ref  = f"{PROJECT_ID}.{DATASET_ID}.test_1m"

    sql = f"""
    WITH
      train_users AS (SELECT DISTINCT user_id FROM `{train_ref}`),
      test_users  AS (SELECT DISTINCT user_id FROM `{test_ref}`),
      overlap     AS (
        SELECT user_id FROM test_users
        INNER JOIN train_users USING (user_id)
      )
    SELECT
      (SELECT COUNT(*) FROM `{train_ref}`)  AS train_rows,
      (SELECT COUNT(*) FROM train_users)    AS train_users,
      (SELECT COUNT(*) FROM `{test_ref}`)   AS test_rows,
      (SELECT COUNT(*) FROM test_users)     AS test_users,
      (SELECT COUNT(*) FROM overlap)        AS overlap_users
    """

    t0  = _step("Validating splits")
    row = bq.query(sql).result().to_dataframe().iloc[0]
    _done(t0)

    overlap_pct = row.overlap_users / row.test_users * 100 if row.test_users else 0.0
    W = 52
    print()
    print("  " + "─" * W)
    print("  Validation — 1M splits")
    print("  " + "─" * W)
    print(f"  {'Train rows':<28} {int(row.train_rows):>14,}")
    print(f"  {'Train users':<28} {int(row.train_users):>14,}")
    print(f"  {'Test rows':<28} {int(row.test_rows):>14,}")
    print(f"  {'Test users':<28} {int(row.test_users):>14,}")
    print(f"  {'Overlap users':<28} {int(row.overlap_users):>14,}")
    print(f"  {'Overlap %':<28} {overlap_pct:>13.1f}%")
    status = "PASS" if 65.0 <= overlap_pct <= 85.0 else "WARN (expected 65–85%)"
    print(f"  {'Status':<28} {status:>14}")
    print("  " + "─" * W)


# ── Phase 2: Build vocabs.pkl ─────────────────────────────────────────────────

def build_vocabs() -> None:
    train_gcs_dir = f"gs://{BUCKET}/samples/{SAMPLE_NAME}/train/"

    t0 = _step(f"Reading train parquet from {train_gcs_dir}")
    df = pd.read_parquet(train_gcs_dir, columns=["user_id", "product_id"])
    _done(t0, label=f"{len(df):,} rows")

    t0 = _step("Building user2idx and item2idx")
    user_ids = sorted(df["user_id"].unique().tolist())
    item_ids = sorted(df["product_id"].unique().tolist())
    user2idx = {uid: i for i, uid in enumerate(user_ids)}
    item2idx = {pid: i for i, pid in enumerate(item_ids)}
    vocabs = {
        "user2idx": user2idx,
        "idx2user": {v: k for k, v in user2idx.items()},
        "item2idx": item2idx,
        "idx2item": {v: k for k, v in item2idx.items()},
    }
    _done(t0, label=f"{len(user2idx):,} users  /  {len(item2idx):,} items")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    vocabs_path = ARTIFACTS_DIR / "vocabs.pkl"
    t0 = _step(f"Writing {vocabs_path}")
    with open(vocabs_path, "wb") as f:
        pickle.dump(vocabs, f)
    _done(t0, label=f"{vocabs_path.stat().st_size / 1e6:.1f} MB")


# ── Phase 3: Build session sequences ─────────────────────────────────────────

def build_sequences() -> None:
    t0 = _step(f"Launching build_session_sequences.py for {SAMPLE_NAME}")
    result = subprocess.run(
        [
            sys.executable,
            str(SEQ_SCRIPT),
            "--sample-name", SAMPLE_NAME,
            "--artifacts-dir", f"artifacts/1m",
        ],
        check=True,
        cwd=str(_REPO_ROOT),
    )
    _done(t0)


# ── Phase 4: Upload to GCS ────────────────────────────────────────────────────

def upload_artifacts(gcs: storage.Client) -> None:
    bucket    = gcs.bucket(BUCKET)
    seq_dir   = ARTIFACTS_DIR / "sequences_v2"
    gcs_root  = "data/1M"

    to_upload: list[tuple[pathlib.Path, str]] = []

    # vocabs.pkl
    to_upload.append(
        (ARTIFACTS_DIR / "vocabs.pkl", f"{gcs_root}/vocabs.pkl")
    )

    # sequences_v2/
    for local_path in sorted(seq_dir.iterdir()):
        if local_path.is_file():
            to_upload.append(
                (local_path, f"{gcs_root}/sequences_v2/{local_path.name}")
            )

    t0 = _step(f"Uploading {len(to_upload)} files to gs://{BUCKET}/{gcs_root}/")
    for local_path, blob_name in to_upload:
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(str(local_path))
        size_mb = local_path.stat().st_size / 1e6
        print(f"     uploaded {local_path.name:<40} ({size_mb:.1f} MB) → {blob_name}")
    _done(t0)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build 1M-user sample end-to-end: BQ → vocabs → sequences → GCS"
    )
    p.add_argument(
        "--skip-bq",
        action="store_true",
        help="Skip BigQuery steps (assume GCS splits already exist)",
    )
    p.add_argument(
        "--skip-sequences",
        action="store_true",
        help="Skip session sequence building (assume sequences already exist locally)",
    )
    p.add_argument(
        "--skip-upload",
        action="store_true",
        help="Skip final GCS upload of artifacts",
    )
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args    = _parse_args()
    t_total = time.time()

    _ensure_gcp_credentials()

    _section("RecoSys — Build 1M Sample (Day 5)")
    print(f"  Sample name   : {SAMPLE_NAME}")
    print(f"  Target users  : {TARGET_USERS:,}")
    print(f"  Train months  : {', '.join(TRAIN_MONTHS)}")
    print(f"  Test month    : {TEST_MONTH}")
    print(f"  Artifacts dir : {ARTIFACTS_DIR}")
    print(f"  GCS output    : gs://{BUCKET}/data/1M/")
    print(f"  Skip BQ       : {args.skip_bq}")
    print(f"  Skip seqs     : {args.skip_sequences}")
    print(f"  Skip upload   : {args.skip_upload}")

    # ── Phase 1: BigQuery ─────────────────────────────────────────────────────
    if not args.skip_bq:
        _section("Phase 1 — BigQuery: Sample + Splits + Export")
        bq = _build_bq_client()
        create_sample_table(bq)
        create_splits(bq)
        export_splits(bq)
        validate_splits(bq)
    else:
        _section("Phase 1 — BigQuery: SKIPPED")

    # ── Phase 2: Vocabs ───────────────────────────────────────────────────────
    _section("Phase 2 — Build vocabs.pkl")
    build_vocabs()

    # ── Phase 3: Session sequences ────────────────────────────────────────────
    if not args.skip_sequences:
        _section("Phase 3 — Build session sequences")
        build_sequences()
    else:
        _section("Phase 3 — Session sequences: SKIPPED")

    # ── Phase 4: Upload ───────────────────────────────────────────────────────
    if not args.skip_upload:
        _section("Phase 4 — Upload artifacts to GCS")
        gcs = _build_gcs_client()
        upload_artifacts(gcs)
    else:
        _section("Phase 4 — GCS upload: SKIPPED")

    _section("Done")
    print(f"  Total time : {_elapsed(t_total)}")
    seq_dir = ARTIFACTS_DIR / "sequences_v2"
    if seq_dir.exists():
        for p in sorted(seq_dir.iterdir()):
            print(f"  {p.name:<45} {p.stat().st_size / 1e6:.1f} MB")
    print()


if __name__ == "__main__":
    main()
