# RecoSys

An end-to-end ML portfolio project — session-based recommendation engine on the REES46
eCommerce clickstream dataset (Oct 2019 – Feb 2020, ~280 M events). Covers the full
MLOps stack: BigQuery → Spark → GRU4Rec → Vertex AI → Cloud Run → MLflow → drift monitoring.

---

## Results

| Metric | Value | T4Rec Target | Baseline (popularity) |
|---|---|---|---|
| NDCG@20 | **0.2676** | ≥ 0.22 ✓ | 0.0353 |
| HR@20 | **0.4815** | ≥ 0.44 ✓ | 0.0806 |
| vs. T4Rec XLNet (best published) | +5.1% NDCG@20 | — | — |

Model: GRU4Rec V9 with event-type features, trained on 1M-user REES46 sample on Vertex AI (A100 GPU, 10h 46m).

**Live demo (PowerShell):**
```powershell
# Health
Invoke-RestMethod "$SERVICE_URL/health" | ConvertTo-Json

# Recommendations
$body = '{"session":[{"item_id":"4209538","event_type":"view"},{"item_id":"3622698","event_type":"cart"}],"top_k":20}'
Invoke-RestMethod -Method Post "$SERVICE_URL/recommend" -ContentType "application/json" -Body $body | ConvertTo-Json
```

---

## Project status

| Day | Description | Status |
|---|---|---|
| 1 | Data ingestion — raw CSVs → GCS → BigQuery | ✅ Complete |
| 2 | Exploratory data analysis (BigQuery + Spark) | ✅ Complete |
| 3 | Spark preprocessing pipeline (Dataproc) | ✅ Complete |
| 4 | Sampling, temporal splits, interaction tables | ✅ Complete |
| 1–4 | Two-Tower V1–V6 + GRU4Rec V7 + SASRec V8 (all below pop baseline) | ✅ Complete |
| 4 | GRU4Rec V9 session-based — 500k — NDCG@20=0.2606 | ✅ Complete |
| 5 | 1M-user sample creation (890,736 users, 222,864 items) | ✅ Complete |
| 6–7 | Vertex AI training on 1M sample — NDCG@20=0.2676 | ✅ Complete |
| 8–9 | Cloud Run serving (FastAPI + FAISS) | 🔲 In progress |
| 10–11 | MLflow experiment tracking | 🔲 In progress |
| 12–13 | Distribution drift monitoring (COVID-period shift) | 🔲 In progress |
| 14 | End-to-end demo | 🔲 In progress |

---

## Infrastructure

| Resource | Value |
|---|---|
| GCP project | `recosys-489001` |
| BigQuery dataset | `recosys-489001.recosys` |
| GCS bucket | `gs://recosys-data-bucket` |
| Dataproc cluster | `eda-reco` — `us-central1`, `n4-standard-2` × 3 nodes |
| Service account | `~/secrets/recosys-service-account.json` |

---

## Dataset

**Source:** [REES46 eCommerce Behaviour Data](https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store) (Kaggle)

| Property | Value |
|---|---|
| Raw rows loaded | 288,779,227 |
| Months in scope | Oct 2019 – Jan 2020 (train) + Feb 2020 (test) |
| Months held out | Mar – Apr 2020 (reserved for MLOps evaluation) |
| Event types | `view` 94.1 %, `cart` 4.2 %, `purchase` 1.6 % |
| Feedback type | Implicit only (no explicit ratings) |
| Schema | `event_time, event_type, product_id, category_id, category_code, brand, price, user_id, user_session` |

---

## Repository layout

```
RecoSys/
├── notebooks/
│   ├── 01_setup_and_integration.ipynb   # GCS upload, raw data verification
│   ├── 02_sampling_and_splits.ipynb     # events_clean validation (8/8 checks)
│   ├── 03_EDA_BigQuery.ipynb            # Full BigQuery EDA
│   ├── 04_EDA_DataProc.ipynb            # Spark EDA on Dataproc
│   └── 05_cleaned_sample_BigQuery_validation.ipynb
├── scripts/
│   ├── preprocessing_pipeline.py        # PySpark cleaning pipeline (Dataproc)
│   ├── create_samples.py                # User-based samples from events_clean
│   ├── create_splits.py                 # Temporal train/test splits
│   └── create_interactions.py           # Confidence-weighted interaction matrices
├── reports/
│   ├── 01_eda_report_v1.md              # BigQuery EDA findings
│   ├── 02_eda_report_v2.md              # BigQuery + Spark EDA (reconciled)
│   ├── 03_dataproc_preprocessing_run.md # Cluster config, job output, pipeline results
│   └── 04_sampling_splits_interactions.md  # Sampling, splits, interactions
├── requirements.txt
└── README.md
```

---

## Phase 3 — Preprocessing pipeline

**Script:** `scripts/preprocessing_pipeline.py` (PySpark, runs on Dataproc)

Five cleaning steps applied to `events_raw` in order:

| Step | Operation | Rows removed |
|---|---|---|
| 1 | Fill nulls: `category_code` and `brand` → `"unknown"`, drop null `user_session` | ~2 |
| 2 | Exact deduplication on `(event_time, event_type, product_id, user_id, user_session)` | ~8.8 M |
| 3 | Near-duplicate removal — same user/product/type within 1 second | ~96 k |
| 4 | Price floor — drop events with `price < 1.0` | ~6 k |
| 5 | Bot removal — drop users with avg events/day > 300 | ~2 users |

Followed by **3-core filtering** (iterative): retain only users and items with ≥ 3 interactions each, converges in ~3 rounds.

**Output:** `recosys.events_clean` — **279,937,243 rows**, 7,565,157 users, 284,523 items.

BigQuery validation (8/8 checks passed):
- Total rows: 279,937,243 ✅
- Unique users: 7,565,157 ✅
- Unique items: 284,523 ✅
- Price < 1.0: 0 ✅
- NULL `user_session`: 0 ✅
- NULL `category_code`: 0 ✅
- NULL `brand`: 0 ✅
- Bot users (avg > 300 events/day): 0 ✅

---

## Phase 4 — Sampling, splits, and interaction tables

### User-based samples

`scripts/create_samples.py` — draws N random users from `events_clean` and keeps **all** their events.

| Table | Users | Events | Items |
|---|---|---|---|
| `recosys.events_sample_50k` | 50,000 | 1,860,124 | 121,951 |
| `recosys.events_sample_500k` | 500,000 | 18,506,282 | 231,031 |

### Temporal train/test splits

`scripts/create_splits.py` — splits each dataset at the Jan/Feb 2020 boundary.

| Table | Rows | Users |
|---|---|---|
| `recosys.train_50k` | 1,512,837 | 44,559 |
| `recosys.test_50k` | 347,287 | 20,626 |
| `recosys.train_500k` | 15,054,830 | 445,150 |
| `recosys.test_500k` | 3,451,452 | 206,887 |
| `recosys.train_full` | 227,460,074 | 6,736,214 |
| `recosys.test_full` | 52,477,169 | 3,132,215 |

Train/test user overlap is ~73.5 % across all three sizes — the majority of February users have prior training history.

### Interaction tables

`scripts/create_interactions.py` — collapses events into one row per `(user_id, product_id)` with confidence weighting: `purchase × 4 + cart × 2 + view × 1`.

| Table | Source | Pairs |
|---|---|---|
| `recosys.interactions_train_50k` | `train_50k` | ~1.4 M |
| `recosys.interactions_train_500k` | `train_500k` | ~13.2 M |
| `recosys.interactions_train_full` | `train_full` | ~190 M |

All matrices are >99.99 % sparse — consistent with implicit feedback datasets.

---

## Phase 5 — Two-Tower Model (50k experiments)

**Model:** Neural retrieval with user tower + item tower + FAISS
**Location:** `src/two_tower/`, `scripts/two_tower/`

| Experiment | Config | Best Recall@10 |
|---|---|---|
| Baseline (v2) | batch=1024, temp=0.05, no weighting | 0.0097 |
| Confidence weighted (v3) | batch=1024, temp=0.05, weighted | 0.0098 |

**Key findings:**
- Confidence weighting: neutral on 50k — difference of 0.0001
- 50k ceiling confirmed at ~0.010 regardless of config
- FAISS index must be scoped to trained items only
- Ground truth: cart + purchase events (not purchases only)
- Bottleneck is data scale — moving to 500k GPU training next

---

## Repository layout — Two-Tower

```
src/
├── data/
│   ├── __init__.py
│   └── feature_builder.py       # shared: vocab + feature encoding
└── two_tower/
    ├── __init__.py
    ├── data/
    │   └── dataset.py           # TwoTowerDataset, build_full_item_tensors
    ├── models/
    │   └── two_tower.py         # UserTower, ItemTower, TwoTowerModel
    ├── training/
    │   └── train.py             # in_batch_loss, train_epoch, train
    └── evaluation/
        └── evaluate.py          # build_faiss_index, evaluate

scripts/two_tower/
├── build_item_features.py       # BigQuery: item feature table
├── build_user_features.py       # BigQuery: user feature table (50k)
├── build_user_features_500k.py  # BigQuery: user feature table (500k)
├── build_features_local.py      # encode + validate artifacts (50k)
├── build_features_local_500k.py # encode + validate artifacts (500k)
├── train_two_tower.py           # training entry point (50k)
├── evaluate_two_tower.py        # checkpoint sweep evaluation
├── sanity_check_model.py        # forward-pass + loss sanity check
└── diagnose_evaluation.py       # per-user retrieval diagnostics
```

---

## GCS bucket layout

```
gs://recosys-data-bucket/
├── raw/                            # Original CSVs (52.69 GiB, 7 files)
├── processed/
│   └── events_clean/               # Parquet output from Spark pipeline
├── samples/
│   ├── events_sample_50k/          # 50k-user sample (raw events)
│   ├── events_sample_500k/         # 500k-user sample (raw events)
│   ├── users_sample_50k/
│   │   ├── train/                  # train_50k
│   │   ├── test/                   # test_50k
│   │   └── interactions/           # interactions_train_50k
│   └── users_sample_500k/
│       ├── train/                  # train_500k
│       ├── test/                   # test_500k
│       └── interactions/           # interactions_train_500k
├── splits/
│   ├── train_full/                 # train_full
│   └── test_full/                  # test_full
└── features/
    └── interactions_train_full/    # interactions_train_full (~190 M pairs)
```

All GCS exports are sharded Parquet files written by BigQuery `EXPORT DATA`. Point downstream readers at the folder, not individual shards. All `event_time`, `first_interaction`, and `last_interaction` columns are explicitly cast to `TIMESTAMP` in every export to prevent BigQuery writing them as `STRING`.

---

## Running the scripts

```bash
# Set credentials (or rely on the hardcoded fallback path)
export GOOGLE_APPLICATION_CREDENTIALS=~/secrets/recosys-service-account.json

# Create user-based samples from events_clean
python scripts/create_samples.py

# Create temporal train/test splits for all three sizes
python scripts/create_splits.py

# Build confidence-weighted interaction matrices
python scripts/create_interactions.py
```

Each script prints per-step timing and a validation summary. Re-running is safe — all tables use `CREATE OR REPLACE TABLE` and all exports use `overwrite=true`.

---

## Requirements

```
google-cloud-bigquery>=3.0.0
google-cloud-storage>=3.0.0
pandas>=1.0.0
db-dtypes>=1.0.0
polars>=0.20.0
matplotlib>=3.0.0
seaborn>=0.12.0
```

Install with:

```bash
pip install -r requirements.txt
```
