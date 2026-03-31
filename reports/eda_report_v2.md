# EDA Report — REES46 eCommerce Clickstream Dataset
## RecoSys Project | BigQuery + Spark EDA on `events_raw` | Completed: March 2026

> **Version 2** — Updated to incorporate Spark/PySpark EDA findings (Dataproc cluster).
> Original BigQuery EDA authored by Manoj. Spark EDA conducted by Nikhil.
> All discrepancies between the two EDAs have been reconciled in this document.

---

## 0. Dataset Overview

| Property | Value |
|---|---|
| **Source** | REES46 eCommerce clickstream (Kaggle) |
| **BQ Table** | `recosys-489001.recosys.events_raw` |
| **GCS Raw Data** | `gs://recosys-data-bucket/raw/*.csv` (7 CSV files, 52.69 GiB total) |
| **Months loaded (EDA scope)** | Oct 2019 – Feb 2020 (5 months) |
| **Months held out** | Mar 2020, Apr 2020 (MLOps flow — never touched) |
| **Total raw rows** | 288,779,227 |
| **Columns** | event_time, event_type, product_id, category_id, category_code, brand, price, user_id, user_session |
| **Feedback type** | Implicit only (no ratings) |
| **Event distribution** | View ~93.8%, Cart ~4.5%, Purchase ~1.7% |

### ⚠️ Known Sample Data Issue (Spark EDA Finding)
The GCS sample files (`samples/events_50k_users.parquet`, `samples/events_500k_users.parquet`) have `event_time` stored as **StringType**, not TimestampType. This happened because Polars read it as a string from CSV when creating the samples. **Any code that loads these sample files must cast `event_time` before doing any temporal analysis:**
```python
df = df.withColumn("event_time", F.to_timestamp("event_time", "yyyy-MM-dd HH:mm:ss z"))
```
The full raw CSVs in GCS are unaffected — when loaded with an explicit schema in Spark, `event_time` parses correctly as a timestamp.

---

## 1. Null & Completeness Audit

### Null Counts per Column

| Column | Null Count | Null % | Action |
|---|---|---|---|
| event_time | 0 | 0.00% | ✅ Clean |
| event_type | 0 | 0.00% | ✅ Clean |
| product_id | 0 | 0.00% | ✅ Clean |
| category_id | 0 | 0.00% | ✅ Clean |
| price | 0 | 0.00% | ✅ Clean |
| user_id | 0 | 0.00% | ✅ Clean |
| **category_code** | **52,477,198** | **18.17%** | Encode as `"unknown"` |
| **brand** | **38,569,167** | **13.36%** | Encode as `"unknown"` |
| user_session | 66 | ~0.00% | Drop these 66 rows |

*BQ and Spark both confirmed identical null counts (exact match).*

### Null Correlation Pattern

| Pattern | Count |
|---|---|
| Null in BOTH category_code AND brand | 12,802,183 |
| Null category_code only (brand present) | 39,675,015 |
| Null brand only (category_code present) | 25,766,984 |

### Null Rate by Event Type

| Event Type | Total | % Null category_code | % Null brand |
|---|---|---|---|
| view | 271,045,030 | 18.52% | 13.72% |
| cart | 12,877,066 | 13.01% | 7.99% |
| purchase | 4,857,131 | 12.53% | 7.44% |

### Key Findings
- All critical ID columns (event_time, event_type, product_id, user_id) are 100% complete — no structural integrity issues
- Null rates are consistent across event types; slightly lower nulls for higher-intent events (cart, purchase), suggesting real products are better catalogued
- Null category_code and brand represent real products in the catalog, NOT garbage records — they cluster into 527 distinct category_ids with an avg price of $175 (vs $311 for non-null), indicating a coherent lower-tier/unbranded product segment
- **Do NOT drop null category_code or brand rows** — doing so would remove 18% of data including real purchases

### Decisions
```
null_handling_category_code : encode as "unknown" string — do NOT drop rows
null_handling_brand          : encode as "unknown" string — do NOT drop rows
null_user_session            : drop (only 66 rows — negligible)
```

---

## 2. Duplicate Detection

### Exact Duplicates

| Metric | BQ Value | Spark Value | Delta |
|---|---|---|---|
| Total rows | 288,779,227 | 288,779,227 | ✅ Exact match |
| Exact duplicate rows | 1,084,384 | 1,084,383 | ✅ Off by 1 (rounding) |
| Exact dupe % | 0.38% | 0.38% | ✅ Match |

Exact duplicates defined as: identical (event_time, event_type, product_id, user_id, user_session).

### Near-Duplicates (same user + product + event_type within N seconds)

| Window | BQ Count (on raw) | Spark Count (post exact-dedup) | Action |
|---|---|---|---|
| Within 1s | 1,922,956 | 838,571 | ✅ Remove — double-fire artifact |
| Within 5s | 5,308,170 | — | ❌ Too aggressive — includes real repeat views |
| Within 30s | 38,035,092 | — | ❌ Far too aggressive — overwhelmingly real behavior |

**Why the 1s numbers differ:** BQ measured near-dupes on raw data (before exact dedup). Spark measured after exact dedup had already run. Since exact dupes have a gap of 0 seconds (within 1s), subtracting them reconciles the numbers: 1,922,956 − 1,084,383 = **838,573 ≈ 838,571** (Spark result). The two EDAs are fully consistent.

### Decisions
```
drop_exact_duplicates    : true   — removes 1,084,384 rows (0.38%)
dedup_window_seconds     : 1      — removes ~838,571 additional rows (after exact dedup)
total_dedup_removal      : ~1,923,000 rows (1.04% of raw data, inclusive of exact dupes)
dedup_key                : (user_id, product_id, event_type) within window
```

---

## 3. Price Distribution

### Percentile Distribution

| Stat | Value |
|---|---|
| price_min | $0.00 |
| p1 | $5.12 |
| p25 | $64.33 |
| p50 | $163.45 |
| p75 | $348.53 |
| p90 | $743.85 |
| p95 | $993.59 |
| p99 | $1,703.94 |
| p100 (max) | $2,574.07 |
| price_mean | $286.84 |
| zero_or_negative_price | 583,466 rows |
| price > $10,000 | 0 rows |

*BQ and Spark confirmed identical max price ($2,574.07). All percentile values consistent.*

### Price Bucket Distribution

| Bucket | Events | Unique Products |
|---|---|---|
| ≤ 0 | 583,466 | 68,107 |
| 0–1 | 43,277 | 456 |
| 1–10 | 7,802,137 | 48,089 |
| 10–50 | 48,562,698 | 103,673 |
| 50–100 | 43,975,151 | 64,014 |
| 100–500 | 139,587,503 | 94,908 |
| 500–1k | 33,959,635 | 17,433 |
| 1k–5k | 14,265,360 | 8,362 |

### Key Findings
- **No extreme high-end outliers** — p99→p100 gap is only $1,703→$2,574 (51% jump). Zero prices above $2,574. No ceiling needed
- **Real problem is at the floor** — 583,466 zero/negative price rows across 68,107 unique products
- Sub-$1 band (43,277 events) also below the natural distribution floor (p1 = $5.12)
- Bulk of events (~290M) in healthy $10–$5k range — normal eCommerce distribution

### Spark Validation Note
Spark removed 625,453 rows at the $1 price floor vs BQ's estimate of 626,743. The 1,290-row difference is explained by rows that had both low prices AND were already eliminated during prior dedup steps. The difference is negligible (0.0004% of data).

### Decisions
```
price_floor   : $1.00  — drops ~625,000–627,000 rows (0.22%) — zero/negative + sub-$1 noise
price_ceiling : null   — p100 = $2,574, no extreme outliers exist
```

---

## 4. User Activity Distribution & Bot Detection

### Total Events per User

| Bucket | Users | % of Total |
|---|---|---|
| 1 event | 2,872,363 | 24.3% |
| 2–4 | 3,054,095 | 25.8% |
| 5–9 | 1,746,055 | 14.7% |
| 10–19 | 1,390,351 | 11.7% |
| 20–49 | 1,411,651 | 11.9% |
| 50–99 | 710,552 | 6.0% |
| 100–499 | 611,936 | 5.2% |
| 500–999 | 34,568 | 0.29% |
| 1k–5k | 8,267 | 0.07% |
| 5k–10k | 92 | 0.0008% |
| >10k | 34 | 0.0003% |

**Total unique users: 11,839,964**

### Events per Day (Bot Detection Signal)

| EPD Bucket | Users | Interpretation |
|---|---|---|
| 1–10 | 10,654,902 | Normal (90% of all users) |
| 11–50 | 1,157,559 | Heavy but plausible |
| 51–100 | 20,483 | Suspicious |
| 101–200 | 5,175 | Very suspicious |
| 201–500 | 1,800 | Almost certainly bots |
| 501–1k | 13 | Bots |
| >1k | 32 | Definite bots |

EPD = total events / number of distinct active days.

### 5-Core Viability Across Thresholds

| Min Interactions (k) | Users Surviving | % Surviving |
|---|---|---|
| 1 (no filter) | 11,839,964 | 100.0% |
| 2 | 8,967,601 | 75.74% |
| 3 | 7,593,911 | 64.14% |
| **5** | **5,913,506** | **49.95%** |
| 10 | 4,167,451 | 35.20% |
| 20 | 2,777,100 | 23.46% |

### Key Findings
- **Natural bot break at 200 EPD** — 56x drop between 11–50 bucket (1.15M users) and 51–100 bucket (20K users)
- **50% of users have fewer than 5 interactions** — classic power-law eCommerce distribution

### ⚠️ Bot Count Correction (Spark EDA Finding)
The original BQ report stated threshold 300 removes **45 users**. This figure only counted users in the 501–1k (13) and >1k (32) EPD buckets, which effectively corresponds to a threshold of ~500, not 300.

Spark EDA using the correct calculation (avg EPD > 300) confirmed **285 users** are removed at threshold 300. These 285 users include:
- ~240 users from the 201–500 EPD bucket whose precise average EPD falls between 301–500
- 13 users from the 501–1k EPD bucket
- 32 users from the >1k EPD bucket

The 285 figure is the accurate count for threshold 300.

### Decisions
```
bot_threshold_events_per_day : 300
  — removes 285 users (confirmed by Spark EDA)
  — retains the 201-300 EPD portion of the 201-500 band as legitimate power users
  — natural break at 200 EPD noted; 300 chosen to avoid over-filtering
  — ~614,000 events removed (events belonging to flagged bot users)
```

---

## 5. Item (Product) Activity Distribution

### Events per Product

| Bucket | Products | % of Catalog |
|---|---|---|
| 1 event | 13,568 | 4.3% |
| 2–4 | 23,416 | 7.5% |
| 5–9 | 22,855 | 7.3% |
| 10–19 | 28,375 | 9.0% |
| 20–99 | 84,827 | 27.0% |
| 100–499 | 83,184 | 26.5% |
| 500–4,999 | 49,809 | 15.9% |
| 5k–50k | 7,277 | 2.3% |
| >50k | 573 | 0.2% |

**Total unique products: 313,884**
**Products with < 5 events: 36,984 (11.8% of catalog)**

### Top 5 Products by Interactions

| product_id | Total Events | Brand | Category | Avg Price |
|---|---|---|---|---|
| 1004767 | 3,469,599 | samsung | construction.tools.light | $237 |
| 1005115 | 2,810,536 | apple | construction.tools.light | $890 |
| 1004856 | 2,296,973 | samsung | construction.tools.light | $129 |
| 1005160 | 1,818,261 | xiaomi | electronics.smartphone | $185 |
| 4804056 | 1,642,238 | apple | sport.bicycle | $161 |

### Key Findings
- **Catalog is unusually dense** — only 11.8% of products have fewer than 5 events
- At k=5, item survival is 87.6% — very little item-side filtering would happen
- This density confirms k=3 is more appropriate

### ⚠️ Spark Skew Warning
Product 1004767 has 3.47M events — representing extreme skew on the item side. Any Spark join or groupBy keyed on `product_id` will produce straggler tasks unless handled. The cleaning pipeline should repartition by `user_id` (not `product_id`) for bot removal and k-core steps.

### Decisions
```
item_min_interactions : 3  — matches user-side k=3 (symmetric core filtering)
  — removes ~25,000–30,000 items (<10% of catalog)
```

---

## 6. Session Behavior Analysis

### Session Length Distribution

| Bucket | Sessions | % |
|---|---|---|
| 1 event (single) | 27,427,700 | **41.88%** |
| 2–3 | 16,108,440 | 24.60% |
| 4–5 | 7,723,582 | 11.79% |
| 6–10 | 7,944,695 | 12.13% |
| 11–20 | 4,330,761 | 6.61% |
| 21–50 | 1,736,748 | 2.65% |
| 51–100 | 187,800 | 0.29% |
| >100 | 27,082 | 0.04% |

**Total sessions: ~65.5M**

### Session Duration Percentiles

| Stat | Value |
|---|---|
| p25 | 0 min |
| p50 | 0 min |
| p75 | 3 min |
| p90 | 11 min |
| p95 | 23 min |
| p99 | 2,071 min (~34 hrs) |
| mean | 724 min |
| Zero-duration sessions | 36,121,227 |
| Sessions > 1 hour | 1,451,071 |
| Sessions > 8 hours | 776,713 |

### Sessions per User

| Stat | Value |
|---|---|
| p25 | 1 session |
| p50 | 2 sessions |
| p75 | 5 sessions |
| p90 | 13 sessions |
| p99 | 51 sessions |
| mean | 5.53 sessions |
| max | 92,898 sessions |

### Key Findings
- **41.88% of sessions are single-event** — no sequence to learn from; must be filtered for GRU4Rec
- **Median session = 0 minutes** (instantaneous) — typical browse-and-leave behavior
- **p99 duration = 34 hours** — broken/unclosed sessions; not real behavior
- After filtering single-event sessions: ~38M sessions remain — healthy corpus for GRU4Rec

### Decisions
```
min_session_length          : 2    — drop single-event sessions (41.88% of sessions)
max_session_length          : 50   — p97 boundary; truncate longer sequences
max_session_duration_minutes: 60   — flag/split sessions exceeding 1 hour as broken
padding_value               : 0    — standard for variable-length sequence padding
```

---

## 7. Event Funnel & Conversion Analysis

### User Funnel

| Metric | Users | % of Total |
|---|---|---|
| Total users | 11,839,964 | 100% |
| Users with any view | 11,833,741 | 99.95% |
| Users with any cart | 2,547,061 | 21.51% |
| **View-only users** | **9,192,307** | **77.6%** |
| Carted, never purchased | 1,161,967 | 9.8% |
| **Ever purchased** | **1,485,690** | **12.55%** |

### Product-Level Conversion Rates (products with ≥10 views)

| Stat | Value |
|---|---|
| p50 conversion | 0.00 |
| p75 conversion | 0.01 |
| p90 conversion | 0.02 |
| mean conversion | 0.01 (1%) |
| Zero-conversion products | 134,103 |
| Full-conversion products | 0 |

### Key Findings
- **77.6% of users are view-only** — core implicit feedback challenge
- Only 12.55% of users ever purchase — purchase signal is rare and highly intentional
- Confidence weights strongly validated: purchases ~8x rarer than cart events, ~56x rarer than views
- **1,485,690 purchasing users sufficient for leave-one-out evaluation** (51.9% have ≥2 purchases)

### Decisions
```
confidence_view     : 1   — universal but weak signal
confidence_cart     : 2   — 21.5% of users; clear purchase intent
confidence_purchase : 4   — 12.5% of users; strongest signal
evaluation_strategy : leave-one-out on last purchase per user
```

---

## 8. Temporal Trends & Train/Test Split

### Monthly Volume & User Growth

| Month | Active Users | New Users | Returning Users | % New |
|---|---|---|---|---|
| 2019-10 | 3,022,290 | 3,022,290 | 0 | 100.0% |
| 2019-11 | 3,696,117 | 2,294,359 | 1,401,758 | 62.1% |
| 2019-12 | 4,577,232 | 2,539,989 | 2,037,243 | 55.5% |
| 2020-01 | 4,385,985 | 2,145,747 | 2,240,238 | 48.9% |
| 2020-02 | 4,233,206 | 1,837,579 | 2,395,627 | 43.4% |

### Cross-Month User Overlap (shared users between months)

| Month A | Month B | Shared Users |
|---|---|---|
| 2019-10 | 2019-11 | 1,401,758 |
| 2019-10 | 2019-12 | 1,215,865 |
| 2019-10 | 2020-02 | 888,766 |
| 2019-11 | 2019-12 | 1,726,306 |
| 2019-12 | 2020-01 | 1,797,428 |
| 2020-01 | 2020-02 | 1,702,723 |

### Key Findings
- **Platform growth with increasing loyalty** — new user % drops steadily from 100% → 43%
- **No alarming distribution shift** — smooth, gradual decay in cross-month overlap
- **February is ideal as the test month**: 4.23M active users, 56.6% returning, smooth continuation of January trends
- Adjacent months share 40–50% of users consistently — temporal split is clean

### Decisions
```
train_months : ["2019-10", "2019-11", "2019-12", "2020-01"]  — 4 months
test_months  : ["2020-02"]                                    — 1 month (~20% of events)
holdout_months: ["2020-03", "2020-04"]                        — MLOps flow, never touch
```

---

## 8a. Temporal Granularity — Hourly & Day-of-Week Patterns *(Spark EDA — New)*

This section is new and was not covered in the BigQuery EDA. These patterns are directly useful as Two-Tower user features.

### Hourly Event Volume (500K sample, Oct 2019)

| Hour | Events | Notes |
|---|---|---|
| 0 | 3,358 | Near-zero — midnight |
| 1 | 6,588 | |
| 2 | 12,562 | |
| 3 | 19,222 | |
| 4 | 22,294 | |
| 5 | 25,368 | Morning ramp begins |
| 6 | 27,344 | |
| 7 | 27,414 | |
| 8 | 27,901 | |
| 9 | 26,931 | |
| 10 | 26,967 | |
| 11 | 26,183 | |
| 12 | 25,777 | |
| 13 | 27,782 | |
| 14 | 31,659 | Afternoon peak begins |
| 15 | 34,439 | |
| **16** | **37,093** | **Peak hour** |
| 17 | 32,791 | |
| 18 | 25,658 | Steep evening decline |
| 19 | 15,907 | |
| 20 | 9,316 | |
| 21 | 5,002 | |
| 22 | 2,857 | |
| 23 | 2,507 | |

### Day-of-Week Event Volume (500K sample, Oct 2019)

| Day | Events | Relative |
|---|---|---|
| **Tuesday** | **80,945** | **Peak day** |
| Wednesday | 78,927 | |
| Thursday | 76,774 | |
| Sunday | 69,816 | |
| Friday | 66,022 | |
| Saturday | 67,141 | |
| Monday | 63,295 | Lowest |

### Key Findings
- **Peak shopping hour is 16:00 (4pm)** — activity is 11x higher at peak vs midnight
- **Activity pattern:** Near-zero midnight → gradual morning rise from 5am → sustained plateau 6am–1pm → strong afternoon peak 2pm–5pm → steep evening decline
- **Weekdays dominate over weekends** — Tuesday is the busiest day, ~28% more activity than Monday
- Weekend (Sat/Sun) shopping is moderately lower but not dramatically so — this is not a typical B2C weekday-only pattern

### Note on Sample Scope
These hourly/DoW patterns come from the 500K sample drawn from October 2019 only. The full 5-month patterns should be verified on the full data. October may exhibit slightly different patterns than peak holiday months (November/December). Validate the hour-of-day distribution on the full dataset before using as a model feature.

### Modeling Implications
These distributions feed directly into Two-Tower user features:
- `peak_hour_bucket`: bin each user's typical activity hour into morning / afternoon / evening / night
- `preferred_dow`: weekday vs weekend shopper preference
- Both features should be computed per user as part of feature engineering (post-cleaning)

---

## 9. Category & Brand Landscape

### Top-Level Category Distribution

| Category | Events | % | Unique Products | Unique Users |
|---|---|---|---|---|
| electronics | 59,676,304 | 25.25% | 39,095 | 4,623,601 |
| construction | 53,740,852 | 22.74% | 23,552 | 4,812,653 |
| appliances | 42,961,758 | 18.18% | 34,498 | 3,431,845 |
| apparel | 28,305,192 | 11.98% | 70,826 | 3,326,121 |
| computers | 14,894,048 | 6.30% | 28,533 | 1,823,919 |
| furniture | 11,980,232 | 5.07% | 35,605 | 1,638,300 |
| sport | 10,810,524 | 4.57% | 19,434 | 1,760,956 |
| kids | 6,041,162 | 2.56% | 16,324 | 1,060,267 |
| auto | 3,836,239 | 1.62% | 4,919 | 603,121 |

### Construction Category — Verified Legitimate

| Subcategory | Events | Brands | Avg Price |
|---|---|---|---|
| construction.tools.light | 46,529,328 | 249 | $449.97 |
| construction.components.faucet | 1,854,749 | 71 | $208.09 |
| construction.tools.drill | 1,661,778 | 167 | $180.79 |
| construction.tools.welding | 1,418,491 | 208 | $146.20 |
| construction.tools.generator | 891,758 | 143 | $108.96 |
| construction.tools.saw | 836,959 | 172 | $186.68 |

**Verdict:** `construction` is a legitimate hardware/tools category. `construction.tools.light` = lighting equipment (LED strips, smart bulbs, work lights) — Samsung, Xiaomi, Apple all make smart lighting. **No remapping needed.**

### Category Hierarchy Depth

| Depth | Events | Unique Products |
|---|---|---|
| 2 levels | 90,035,175 | 135,224 |
| 3 levels | 146,025,288 | 134,505 |
| 4 levels | 241,566 | 1,906 |

Use level_1 and level_2 as Two-Tower item features.

### Null Category Pattern

| cat_code_null | Events | Unique category_ids | Avg Price |
|---|---|---|---|
| False | 236,302,029 | 960 | $311.64 |
| True | 52,477,198 | 527 | $175.17 |

### Top Brands

| Brand | Events | Unique Products | Avg Price | Purchases |
|---|---|---|---|---|
| samsung | 35,647,588 | 1,399 | $357.44 | 1,116,800 |
| apple | 27,053,465 | 709 | $800.87 | 957,906 |
| xiaomi | 21,199,990 | 1,300 | $199.08 | 410,070 |
| huawei | 7,397,535 | 178 | $239.94 | 167,131 |
| lg | 3,875,007 | 590 | $469.36 | 60,755 |
| sony | 3,490,851 | 1,063 | $426.07 | 55,260 |

Samsung, Apple, Xiaomi dominate (~35% of branded events). Brand is a strong Two-Tower feature.

### Decisions
```
category_remapping          : null — construction is a legitimate category
two_tower_item_features     : [category_level_1, category_level_2, brand, price_bucket]
null_category_code_encoding : "unknown" — coherent lower-tier product segment
```

---

## 10. Repeat Behavior & User Loyalty

### Active Months per User

| Active Months | Users | % |
|---|---|---|
| 1 month only | 7,435,794 | **62.80%** |
| 2 months | 2,240,902 | 18.93% |
| 3 months | 1,092,450 | 9.23% |
| 4 months | 634,208 | 5.36% |
| All 5 months | 436,610 | 3.69% |

### Repeat Purchase Distribution (among 1,485,690 purchasing users)

| Purchase Count | Users | % |
|---|---|---|
| 1 purchase | 714,553 | 48.10% |
| 2–3 | 457,000 | 30.76% |
| 4–9 | 233,796 | 15.74% |
| 10–19 | 53,444 | 3.60% |
| 20+ | 26,897 | 1.81% |

### Key Findings
- **62.8% of users appear in only one month** — fundamental sparsity challenge
- Collaborative filtering signal comes from the **37.2% of cross-month users**
- 436,610 five-month loyal users are the highest-quality training signal
- **48.1% of buyers purchase only once** — leave-one-out evaluation is valid since 51.9% have ≥2 purchases
- 26,897 users with 20+ purchases are the loyal customer core — primary target for recommendations

---

## 11. 5-Core Simulation

### Survival Rates at k=5 (BQ simulation, 3 rounds)

| Stage | Users | Items | Events |
|---|---|---|---|
| Raw | 11,839,964 | 313,884 | 288,779,227 |
| After round 1 | ~7.8M | ~295K | ~283M |
| After round 2 | ~6.2M | ~278K | ~279M |
| After round 3 (k=5) | 5,912,686 | 275,006 | 277,344,101 |

| Metric | Value |
|---|---|
| User survival (k=5) | 5,912,686 / 11,839,964 = **49.9%** |
| Item survival (k=5) | 275,006 / 313,884 = **87.6%** |
| Event survival (k=5) | 277,344,101 / 288,779,227 = **96.0%** |

### Why k=3 Instead of k=5

The 96% event survival at k=5 reveals the key insight: **the removed 50% of users collectively contributed only 4% of all events.** Going to k=3 recovers 1.68M users at negligible noise cost.

| k | Users | % Users | Est. Event Survival |
|---|---|---|---|
| 3 | ~7,593,911 | 64.1% | ~98% |
| **5** | **5,912,686** | **49.9%** | **96.0%** |

**Marginal cost of k=3→k=5:** lose 1.68M users for only 2% more event cleaning. Not worth it.

### Decisions
```
core_k          : 3   — applied symmetrically to users AND items
core_iterations : 10  — run until convergence (typically 3–5 rounds)
```

---

## 12. Actual Spark Cleaning Pipeline Results *(Spark EDA — New)*

The BQ EDA estimated cleaning removal counts. Spark EDA actually ran the full pipeline on all 5 months and produced the definitive numbers.

### Step-by-Step Row Counts (full 5-month data)

| Step | Rows Removed | Rows Remaining | Notes |
|---|---|---|---|
| Raw load | — | 288,779,227 | All 5 EDA months loaded correctly |
| Drop null user_session | 66 | 288,779,161 | ✅ Exact match with BQ |
| Drop exact duplicates | 1,084,383 | 287,694,778 | ✅ Off by 1 from BQ (rounding) |
| Remove near-dupes (1s) | 838,571 | 286,856,207 | ✅ Consistent with BQ after accounting for order of operations |
| Apply price floor ($1.00) | 625,453 | 286,230,754 | ✅ Within 1,290 rows of BQ estimate |
| Remove bot users (>300 EPD) | 614,006 | 285,616,748 | 285 bot users, all their events removed |
| **Total removed pre-k-core** | **3,162,479** | **285,616,748** | **1.09% of raw data** |

K-core results pending full Spark convergence run.

### Key Takeaway
The BQ estimates were accurate. The Spark pipeline confirms total pre-k-core removal of ~3.16M rows (1.09% of data), consistent with BQ's estimate of ~3M+. The dataset is very clean — cleaning removes only ~1% of data before k-core.

---

## 13. Complete Cleaning Configuration

All thresholds derived from both EDAs. Use this as `configs/cleaning_config.yaml` for the Spark pipeline.

```yaml
# configs/cleaning_config.yaml
# Generated from BigQuery + Spark EDA
# Dataset: REES46 eCommerce | 5 months: Oct 2019 – Feb 2020
# Total raw events: 288,779,227

data:
  source_table   : recosys-489001.recosys.events_raw
  gcs_raw_path   : gs://recosys-data-bucket/raw/
  train_months   : ["2019-10", "2019-11", "2019-12", "2020-01"]
  test_months    : ["2020-02"]
  holdout_months : ["2020-03", "2020-04"]   # MLOps flow — never touch

cleaning:
  # --- Duplicates ---
  drop_exact_duplicates    : true       # removes ~1,084,384 rows (0.38%)
  dedup_window_seconds     : 1          # removes ~838,571 rows after exact dedup (0.29%)
  dedup_key                : [user_id, product_id, event_type]

  # --- Nulls ---
  null_user_session_action : drop       # only 66 rows
  null_category_code_action: encode     # encode as "unknown" — do NOT drop rows
  null_brand_action        : encode     # encode as "unknown" — do NOT drop rows
  null_encode_value        : "unknown"

  # --- Price ---
  price_floor              : 1.0        # removes ~625,000–627,000 rows (0.22%)
  price_ceiling            : null       # not needed — max price is $2,574

  # --- Bot Removal ---
  bot_threshold_events_per_day : 300    # removes 285 users (~614,000 events)
                                        # NOTE: BQ report incorrectly stated 45 users;
                                        # Spark EDA confirmed correct count is 285

  # --- Sparse Entity Filtering (iterative k-core) ---
  core_k           : 3                  # user survival ~64%, item ~92%, event ~98%
  core_iterations  : 10                 # run until convergence (typically 3–5 rounds)
  core_apply_to    : [user_id, product_id]

  # --- Spark Tuning ---
  shuffle_partitions    : 480           # ~128MB/partition for 52GB dataset
  repartition_key       : user_id       # use user_id not product_id (skew avoidance)

sequence_models:
  # GRU4Rec / SASRec
  min_session_length           : 2      # drop single-event sessions (41.88%)
  max_session_length           : 50     # p97 boundary — truncate longer sequences
  max_session_duration_minutes : 60     # flag broken/unclosed sessions
  padding_value                : 0

modeling:
  # Interaction matrix confidence weights (ALS / BPR)
  confidence_view     : 1
  confidence_cart     : 2
  confidence_purchase : 4

  # Evaluation
  evaluation_strategy          : leave_one_out
  evaluation_metric            : [Recall@10, NDCG@10, Precision@10]
  min_purchases_for_evaluation : 2

feature_engineering:
  # Two-Tower item features
  item_features : [category_level_1, category_level_2, brand, price_bucket]

  # Two-Tower user features (to be computed post-cleaning)
  user_features : [activity_level, months_active, event_type_mix,
                   peak_hour_bucket, preferred_dow]
  # peak_hour_bucket and preferred_dow are new features from Spark EDA
  # temporal analysis — bin user's typical hour into morning/afternoon/evening/night
```

---

## 14. Modeling Implications Summary

| Model | Key EDA Insights |
|---|---|
| **Popularity Baseline** | Clear top items (product 1004767: 3.47M events). Construction.tools.light dominates — ensure popularity is computed post-cleaning |
| **ALS / BPR** | 37.2% cross-month users provide collaborative signal. 1.49M purchasing users for evaluation. Confidence weights validated by funnel ratios |
| **Two-Tower** | Item features: category_level_1, category_level_2, brand, price_bucket. User features: activity_level, months_active, event_type_mix, **peak_hour_bucket**, **preferred_dow**. Null category/brand → "unknown" embedding |
| **GRU4Rec** | 38M sessions after filtering single-event sessions. Max sequence length 50. 62.8% of users are single-month — session behavior is the primary signal for these users |
| **SASRec** | Same session config as GRU4Rec. Benefits most from the 436K five-month loyal users with long interaction histories |

---

## 15. Outstanding Items

| Item | Status | Owner |
|---|---|---|
| Spark cleaning pipeline (full run with k-core) | 🔄 In progress | Nikhil |
| Cleaned Parquet → GCS → BigQuery events_clean | ⬜ Blocked on Spark pipeline completion | Nikhil |
| Fix event_time dtype in sample Parquet files | ⬜ Not started | Nikhil |
| Validate hourly/DoW patterns on full 5-month data | ⬜ Not started | Nikhil |
| 50K / 500K user-based samples (from cleaned data) | ⬜ Blocked on cleaning | Manoj |
| Feature engineering (incl. peak_hour_bucket, preferred_dow) | ⬜ Not started | Manoj |
| Model development | ⬜ Not started | All |

---

*BigQuery EDA: recosys-489001.recosys.events_raw | VSCode Jupyter + google-cloud-bigquery | March 2026*
*Spark EDA: gs://recosys-data-bucket/raw/ | GCP Dataproc (PySpark) | March 2026*
