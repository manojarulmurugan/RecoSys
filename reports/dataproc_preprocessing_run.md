# Dataproc preprocessing run

This document records the **Google Cloud Dataproc** environment, **job submission**, and **measured results** for the full Spark preprocessing pipeline (`scripts/preprocessing_pipeline.py`) over the five-month REES46 slice (Oct 2019–Feb 2020). It complements the analytical write-ups in `eda_report_v1.md` (BigQuery EDA) and `eda_report_v2.md` (BigQuery + Spark EDA and cleaning rationale).

---

## 1. Cluster configuration (`eda-reco`)

| Field | Value |
|--------|--------|
| **Project** | `recosys-489001` |
| **Cluster name** | `eda-reco` |
| **Region** | `us-central1` |
| **Zone** | `us-central1-b` |
| **Image version** | `2.2.80-debian12` |
| **Autoscaling** | Off |

### Performance and optional services

- **Advanced optimizations / execution layer / GCS caching:** Off  
- **Dataproc Metastore:** None  
- **Scheduled deletion / scheduled stop:** Off  
- **Confidential computing:** Disabled  

### Master node

| Setting | Value |
|---------|--------|
| Topology | Standard (1 master, N workers) |
| Machine type | `n4-standard-2` (2 vCPUs, 8 GiB RAM) |
| GPUs | 0 |
| Primary disk | `hyperdisk-balanced`, 100 GiB |
| Local SSDs | 0 |

### Worker nodes

| Setting | Value |
|---------|--------|
| Count | 2 |
| Machine type | `n4-standard-2` |
| GPUs | 0 |
| Primary disk | `hyperdisk-balanced`, 200 GiB each |
| Local SSDs | 0 |
| Secondary workers | 0 |

### Security (cluster defaults)

Secure Boot, vTPM, and Integrity Monitoring were **disabled** for this cluster.

### Staging bucket

`gs://dataproc-staging-us-central1-921967012784-mgblzxgl`

---

## 2. Job submission

The job was submitted with the Dataproc PySpark entrypoint; the script lives in the project data bucket (not only in Git).

```bash
gcloud dataproc jobs submit pyspark \
  gs://recosys-data-bucket/scripts/preprocessing_pipeline.py \
  --cluster=eda-reco \
  --region=us-central1 \
  --project=recosys-489001 \
  --properties="spark.sql.shuffle.partitions=480,spark.executor.memory=5g,spark.driver.memory=4g"
```

### Spark overrides (CLI)

These **supplement** settings already coded in `preprocessing_pipeline.py` (e.g. AQE, checkpoint compression):

| Property | Value |
|----------|--------|
| `spark.sql.shuffle.partitions` | 480 |
| `spark.executor.memory` | 5g |
| `spark.driver.memory` | 4g |

### Job metadata

| Field | Value |
|--------|--------|
| **Job ID** | `8441bf61fe234f288e2d9c0eccdf0dc1` |
| **YARN application** | `application_1774937295745_0002` |
| **Spark application name** | `recosys-preprocessing` |
| **Spark version (driver log)** | 3.5.3 |
| **Outcome** | Finished successfully (`state: DONE`) |

### Wall-clock (API)

| Milestone | Time (UTC) |
|-----------|------------|
| Job submitted / pending | `2026-03-31T07:38:25Z` |
| Job completed | `2026-03-31T12:00:11Z` |

Approximate **end-to-end job duration:** about **4 hours 22 minutes** (queue + all phases).

---

## 3. Runtime observations from driver output

- **`spark.sparkContext.defaultParallelism` reported as 2** in the driver banner — consistent with a small worker pool (two `n4-standard-2` workers); shuffle partition count was still set to **480** via configuration.
- **Checkpoint directory:** `gs://recosys-data-bucket/spark_checkpoints`
- **Outputs:** Parquet under `gs://recosys-data-bucket/processed/…` and BigQuery table `recosys-489001.recosys.events_clean` (see below).

---

## 4. Pipeline phases and timings (driver log)

### Phase 1 — Load raw CSVs

| Metric | Value |
|--------|--------|
| Duration | **4m 45s** |
| Rows loaded | **288,779,227** |
| Matches expected row count | Yes |

### Phase 2 — Cleaning (nulls, dedup, price floor, bots)

| Step | Rows remaining | Removed in step | Notes |
|------|----------------|------------------|--------|
| After null handling + drop null `user_session` | 288,779,161 | 66 | |
| After exact dedup | 287,694,778 | 1,084,383 | Key: `event_time`, `event_type`, `product_id`, `user_id`, `user_session` |
| After near-dedup (gap ≤ 1s) | 286,856,207 | 838,571 | Per `user_id`, `product_id`, `event_type` |
| After price floor ($1.00) | 286,230,754 | 625,453 | |
| After bot filter (avg EPD > 300) | 285,616,748 | **285 users**, **614,006** events | |

**Phase 2 total duration:** **95m 50s**

### Pre–k-core Parquet write + reload

| Action | Duration |
|--------|----------|
| Write `gs://recosys-data-bucket/processed/events_pre_kcore` | **51m 4s** |
| Reload row count | 285,616,748 |

### Phase 3 — K-core (k = 3, up to 10 rounds)

Starting rows: **285,616,748**

| Round | Rows | Users | Items | Removed this round | Duration |
|-------|------|-------|-------|--------------------|----------|
| 1 | 279,939,578 | 7,565,531 | 285,476 | 5,677,170 | 18m 40s |
| 2 | 279,937,372 | 7,565,219 | 284,533 | 2,206 | 17m 14s |
| 3 | 279,937,247 | 7,565,158 | 284,524 | 125 | 19m 0s |
| 4 | 279,937,243 | 7,565,157 | 284,523 | 4 | 17m 28s |
| 5 | 279,937,243 | 7,565,157 | 284,523 | 0 | 18m 57s |

**Converged at round 5** (no change in row count between rounds 4 and 5).

**Phase 3 total duration:** **91m 44s**

### Phase 4 — Export

| Action | Duration |
|--------|----------|
| Parquet → `gs://recosys-data-bucket/processed/events_clean` | (included in phase block) |
| BigQuery load → `recosys-489001.recosys.events_clean` | |
| **Phase 4 total** | **17m 26s** |

---

## 5. Final summary

| Metric | Value |
|--------|--------|
| Raw rows | 288,779,227 |
| After cleaning (pre–k-core) | 285,616,748 |
| After k-core (k = 3) | **279,937,243** |
| **Total removed** | **8,841,984** (**3.1%** of raw) |
| **Users retained** | **7,565,157** |
| **Items retained** | **284,523** |
| **GCS — cleaned Parquet** | `gs://recosys-data-bucket/processed/events_clean` |
| **GCS — pre–k-core Parquet** | `gs://recosys-data-bucket/processed/events_pre_kcore` |
| **BigQuery — cleaned table** | `recosys-489001.recosys.events_clean` |

---

## 6. YARN resource summary (job completion)

From the submitted job record:

| Metric | Value |
|--------|--------|
| `memoryMbSeconds` | 176,727,237 |
| `vcoreSeconds` | 31,378 |

---

## 7. References

- **Pipeline source (repo):** `scripts/preprocessing_pipeline.py`  
- **EDA and cleaning decisions:** `reports/eda_report_v2.md` (Section 12 estimated pre–k-core counts; this run supersedes “k-core pending” with the numbers above)  
- **BigQuery EDA (original):** `reports/eda_report_v1.md`  

---

*Document generated from cluster UI details, `gcloud dataproc jobs submit` invocation, and Dataproc driver output — March 2026.*
