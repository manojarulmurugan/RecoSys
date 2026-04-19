import os

from google.cloud import bigquery
from google.oauth2 import service_account

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DEFAULT_KEY = os.path.join(_REPO_ROOT, "secrets", "recosys-service-account.json")
KEY_PATH = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", _DEFAULT_KEY)

PROJECT = "recosys-489001"
DATASET = "recosys"
GCS_URI = "gs://recosys-data-bucket/features/item_features/*.parquet"

_credentials = service_account.Credentials.from_service_account_file(
    KEY_PATH,
    scopes=["https://www.googleapis.com/auth/cloud-platform"],
)
client = bigquery.Client(project=PROJECT, credentials=_credentials)
print(f"  Authenticated as : {_credentials.service_account_email}")

# ── 1. Create item_features table in BigQuery ──────────────────────────────────
print("Building item_features table in BigQuery...")

create_sql = f"""
CREATE OR REPLACE TABLE `{PROJECT}.{DATASET}.item_features` AS
WITH interaction_stats AS (
  SELECT
    product_id,
    SUM(confidence_score)                       AS item_total_confidence,
    SAFE_DIVIDE(SUM(n_purchases), SUM(n_views)) AS item_purchase_rate
  FROM `{PROJECT}.{DATASET}.interactions_train_50k`
  GROUP BY product_id
)
SELECT
  b.product_id,

  -- category level 1: first segment of dot-separated code, e.g. 'electronics'
  SPLIT(IFNULL(MAX(b.category_code), 'unknown'), '.')[SAFE_OFFSET(0)]  AS cat_l1,

  -- category level 2: second segment, e.g. 'smartphone'. NULL → 'unknown'
  IFNULL(
    SPLIT(IFNULL(MAX(b.category_code), 'unknown'), '.')[SAFE_OFFSET(1)],
    'unknown'
  )                                                                     AS cat_l2,

  IFNULL(MAX(b.brand), 'unknown')                                       AS brand,

  -- 8 log-spaced price buckets (integers 0–7, easier for nn.Embedding)
  CASE
    WHEN AVG(b.price) < 10   THEN 0
    WHEN AVG(b.price) < 50   THEN 1
    WHEN AVG(b.price) < 100  THEN 2
    WHEN AVG(b.price) < 200  THEN 3
    WHEN AVG(b.price) < 500  THEN 4
    WHEN AVG(b.price) < 1000 THEN 5
    WHEN AVG(b.price) < 2000 THEN 6
    ELSE                          7
  END                                                                   AS price_bucket,

  ROUND(AVG(b.price), 4)                                               AS avg_price,

  IFNULL(ANY_VALUE(s.item_total_confidence), 0)                        AS item_total_confidence,
  IFNULL(ANY_VALUE(s.item_purchase_rate),    0)                        AS item_purchase_rate

FROM `{PROJECT}.{DATASET}.events_clean` b
LEFT JOIN interaction_stats s USING (product_id)
GROUP BY b.product_id
"""

job = client.query(create_sql)
job.result()
print("[ok] item_features table created")

# ── 2. Validate row count ──────────────────────────────────────────────────────
count_sql = f"SELECT COUNT(*) AS n FROM `{PROJECT}.{DATASET}.item_features`"
n = list(client.query(count_sql).result())[0].n
print(f"  Row count: {n:,}  (expected: 284,523)")
assert 280_000 < n < 290_000, f"Unexpected row count: {n}"

# ── 3. Validate no nulls in key columns ───────────────────────────────────────
null_check_sql = f"""
SELECT
  COUNTIF(cat_l1                IS NULL) AS null_cat_l1,
  COUNTIF(cat_l2                IS NULL) AS null_cat_l2,
  COUNTIF(brand                 IS NULL) AS null_brand,
  COUNTIF(avg_price             IS NULL) AS null_price,
  COUNTIF(item_total_confidence IS NULL) AS null_total_confidence,
  COUNTIF(item_purchase_rate    IS NULL) AS null_purchase_rate
FROM `{PROJECT}.{DATASET}.item_features`
"""
row = list(client.query(null_check_sql).result())[0]
print(f"  Null check: cat_l1:{row.null_cat_l1}  cat_l2:{row.null_cat_l2}  "
      f"brand:{row.null_brand}  price:{row.null_price}  "
      f"total_confidence:{row.null_total_confidence}  purchase_rate:{row.null_purchase_rate}")
assert row.null_cat_l1 == 0 and row.null_cat_l2 == 0, "Nulls found - check IFNULL logic"
assert row.null_total_confidence == 0 and row.null_purchase_rate == 0, \
    "Nulls found in interaction stats columns - check IFNULL logic"

# ── 4. Validate price_bucket distribution ─────────────────────────────────────
bucket_sql = f"""
SELECT price_bucket, COUNT(*) AS n
FROM `{PROJECT}.{DATASET}.item_features`
GROUP BY price_bucket
ORDER BY price_bucket
"""
print("\n  Price bucket distribution:")
for row in client.query(bucket_sql).result():
    print(f"    bucket {row.price_bucket}: {row.n:,} items")

# ── 5. Export to GCS ───────────────────────────────────────────────────────────
print(f"\nExporting to {GCS_URI} ...")

export_sql = f"""
EXPORT DATA OPTIONS(
  uri='{GCS_URI}',
  format='PARQUET',
  overwrite=true
) AS
SELECT * FROM `{PROJECT}.{DATASET}.item_features`
ORDER BY product_id
"""

job = client.query(export_sql)
job.result()
print("[ok] item_features exported to GCS")
print("\nDone. Artifacts:")
print(f"  BQ  : {PROJECT}.{DATASET}.item_features")
print(f"  GCS : {GCS_URI}")