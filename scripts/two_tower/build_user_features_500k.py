import os

from google.cloud import bigquery
from google.oauth2 import service_account

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DEFAULT_KEY = os.path.join(_REPO_ROOT, "secrets", "recosys-service-account.json")
KEY_PATH = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", _DEFAULT_KEY)

PROJECT = "recosys-489001"
DATASET = "recosys"
GCS_URI = "gs://recosys-data-bucket/features/user_features_500k/*.parquet"

_credentials = service_account.Credentials.from_service_account_file(
    KEY_PATH,
    scopes=["https://www.googleapis.com/auth/cloud-platform"],
)
client = bigquery.Client(project=PROJECT, credentials=_credentials)
print(f"  Authenticated as : {_credentials.service_account_email}")

# ── 1. Create user_features_500k table ─────────────────────────────────────────
print("Building user_features_500k table in BigQuery...")

create_sql = f"""
CREATE OR REPLACE TABLE `{PROJECT}.{DATASET}.user_features_500k` AS
WITH cat_ranked AS (
  SELECT
    t.user_id,
    f.cat_l1,
    ROW_NUMBER() OVER (PARTITION BY t.user_id ORDER BY COUNT(*) DESC) AS rn
  FROM `{PROJECT}.{DATASET}.train_500k` t
  JOIN `{PROJECT}.{DATASET}.item_features` f USING (product_id)
  GROUP BY t.user_id, f.cat_l1
),
price_sens AS (
  SELECT
    t.user_id,
    AVG(CASE WHEN t.event_type = 'purchase' THEN f.avg_price END) AS avg_purchase_price,
    AVG(f.avg_price)                                               AS avg_viewed_price,
    CAST(COUNTIF(t.event_type = 'purchase') > 0 AS INT64)         AS has_purchase_history
  FROM `{PROJECT}.{DATASET}.train_500k` t
  JOIN `{PROJECT}.{DATASET}.item_features` f USING (product_id)
  GROUP BY t.user_id
)
SELECT
  b.user_id,

  COUNT(*)                                                             AS total_events,
  COUNT(DISTINCT FORMAT_TIMESTAMP('%Y-%m', b.event_time))             AS months_active,
  ROUND(COUNTIF(b.event_type = 'purchase') / COUNT(*), 4)            AS purchase_rate,
  ROUND(COUNTIF(b.event_type = 'cart')     / COUNT(*), 4)            AS cart_rate,
  COUNT(DISTINCT b.user_session)                                      AS n_sessions,

  -- 4 time-of-day buckets: 0=night(0-5), 1=morning(6-11),
  --                        2=afternoon(12-17), 3=evening(18-23)
  CAST(FLOOR(AVG(EXTRACT(HOUR FROM b.event_time)) / 6) AS INT64)     AS peak_hour_bucket,

  -- preferred day-of-week (1=Sun ... 7=Sat in BigQuery)
  -- using AVG as a proxy for mode; cast to INT64 for embedding lookup
  CAST(ROUND(AVG(EXTRACT(DAYOFWEEK FROM b.event_time))) AS INT64)    AS preferred_dow,

  IFNULL(ANY_VALUE(c.cat_l1), 'unknown')                             AS top_category,
  IFNULL(ANY_VALUE(p.avg_purchase_price), ANY_VALUE(p.avg_viewed_price)) AS avg_purchase_price,
  ANY_VALUE(p.has_purchase_history)                                   AS has_purchase_history

FROM `{PROJECT}.{DATASET}.train_500k` b
LEFT JOIN cat_ranked  c ON c.user_id = b.user_id AND c.rn = 1
LEFT JOIN price_sens  p ON p.user_id = b.user_id
GROUP BY b.user_id
"""

job = client.query(create_sql)
job.result()
print("[ok] user_features_500k table created")

# ── 2. Validate row count ──────────────────────────────────────────────────────
count_sql = f"SELECT COUNT(*) AS n FROM `{PROJECT}.{DATASET}.user_features_500k`"
n = list(client.query(count_sql).result())[0].n
print(f"  Row count: {n:,}")
assert 400_000 < n < 500_000, f"Unexpected row count: {n}"

# ── 3. Validate value ranges ───────────────────────────────────────────────────
range_sql = f"""
SELECT
  MIN(total_events)           AS min_events,
  MAX(total_events)           AS max_events,
  ROUND(AVG(total_events), 1) AS avg_events,
  MIN(months_active)          AS min_months,
  MAX(months_active)          AS max_months,
  MIN(purchase_rate)          AS min_pr,
  MAX(purchase_rate)          AS max_pr,
  MIN(peak_hour_bucket)       AS min_hour,
  MAX(peak_hour_bucket)       AS max_hour,
  MIN(preferred_dow)          AS min_dow,
  MAX(preferred_dow)          AS max_dow,
  COUNTIF(total_events     IS NULL) AS null_event_count,
  COUNTIF(top_category     IS NULL) AS null_top_category,
  MIN(avg_purchase_price)     AS min_avg_purchase_price,
  MAX(avg_purchase_price)     AS max_avg_purchase_price,
  MIN(has_purchase_history)   AS min_has_purchase,
  MAX(has_purchase_history)   AS max_has_purchase
FROM `{PROJECT}.{DATASET}.user_features_500k`
"""
row = list(client.query(range_sql).result())[0]
print(f"\n  total_events      : {row.min_events} - {row.max_events}  (avg {row.avg_events})")
print(f"  months_active     : {row.min_months} - {row.max_months}  (expect 1-4)")
print(f"  purchase_rate     : {row.min_pr:.4f} - {row.max_pr:.4f}  (expect 0-1)")
print(f"  peak_hour_bucket  : {row.min_hour} - {row.max_hour}  (expect 0-3)")
print(f"  preferred_dow     : {row.min_dow} - {row.max_dow}  (expect 1-7)")
print(f"  null_event_count  : {row.null_event_count}")
print(f"  top_category nulls: {row.null_top_category}  (expect 0)")
print(f"  avg_purchase_price: {row.min_avg_purchase_price} - {row.max_avg_purchase_price}")
print(f"  has_purchase_hist : {row.min_has_purchase} - {row.max_has_purchase}  (expect 0-1)")

assert row.min_hour >= 0 and row.max_hour <= 3,   "peak_hour_bucket out of range"
assert row.min_dow  >= 1 and row.max_dow  <= 7,   "preferred_dow out of range"
assert row.null_event_count == 0,                  "Unexpected nulls"

# ── 4. Export to GCS ───────────────────────────────────────────────────────────
print(f"\nExporting to {GCS_URI} ...")

export_sql = f"""
EXPORT DATA OPTIONS(
  uri='{GCS_URI}',
  format='PARQUET',
  overwrite=true
) AS
SELECT * FROM `{PROJECT}.{DATASET}.user_features_500k`
ORDER BY user_id
"""

job = client.query(export_sql)
job.result()
print("[ok] user_features_500k exported to GCS")
print("\nDone. Artifacts:")
print(f"  BQ  : {PROJECT}.{DATASET}.user_features_500k")
print(f"  GCS : {GCS_URI}")
