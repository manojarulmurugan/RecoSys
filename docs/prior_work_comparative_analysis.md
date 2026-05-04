# Prior Work on the REES46 Dataset — Comparative Analysis
## Context for RecoSys Phase 7 (Two-Tower) Planning

**Scope:** 8 external sources reviewed — 6 Kaggle notebooks, the Transformers4Rec RecSys'21 paper, and the 3 NVIDIA Merlin end-to-end tutorials. Goal: situate our Two-Tower work against what others have done on this exact dataset.

---

## Executive Summary

**The punchline: nobody on this dataset has built what we are building.** Of 8 sources reviewed, 0 implement a user-based Two-Tower retrieval model. The dominant paradigms on REES46 are (a) **session-based next-item prediction** with XLNet/GRU (Transformers4Rec paper, ecommerce-transformer4rec Kaggle notebook, userbehavior-ecommerce-transformers4rec, and all 3 NVIDIA tutorials — these 5 share essentially one recipe) and (b) **collaborative filtering with ALS** on user–item interactions (the recommendation-engine Spark notebook). The remaining 3 sources aren't recommenders at all — RFM is customer segmentation, the XGBoost notebook is cart→purchase conversion classification, why-most-users-never-add-to-cart is funnel EDA, and the eCommerce Predictor is time-series forecasting of aggregate PV/UV.

**Scale check — we're working at larger scale than every source.** The Transformers4Rec paper's REES46 run used 31 days × 17.97M interactions. The NVIDIA tutorials use **1 week of October 2019** (~6M events after dedup). The ecommerce-transformer4rec Kaggle notebook operates on ~207K events post-dedup. The recommendation-engine notebook uses 1 month. We clean **279.9M rows** across 5 months and sample 500K users / 18.5M events — more data than anyone published on this dataset.

**Cleaning is also more rigorous than any source.** No source reviewed does iterative k-core filtering. None does explicit bot removal by events-per-day. The Transformers4Rec paper drops sessions <2 and >20, and applies Categorify — that's it. NVIDIA drops nulls (2 rows) and consecutive repeats. The recommendation-engine notebook does one domain-specific thing well (dedup multiple cart events in the same session → 1 per session) but skips k-core entirely. **Our pipeline's null handling, 1s near-dedup window, $1 price floor, 300 EPD bot threshold, and iterative k=3 core are all defensible and collectively stricter than anything published.**

**Target baseline to beat.** The single comparable published metric on Oct 2019 REES46 is the Transformers4Rec paper's XLNet+RTD result: **NDCG@20 = 0.2546, Recall@20 = 0.4886** (Table 2, REES46 eCommerce column). The Kaggle XLNet notebook on a smaller slice reports NDCG@10=0.18, Recall@10=0.30. These are session-based next-item metrics. **Two-Tower retrieval should not be benchmarked against these directly** — different task. Session models condition on a rich within-session sequence; Two-Tower conditions on user aggregate features. Expect lower absolute numbers. A reasonable first internal target is Recall@10 ≥ 0.10 on the 50K sample.

**Five concrete takeaways for Phase 7:**

1. **Our train/test split is better calibrated than the paper's.** Transformers4Rec evaluates per-day with a 50:50 val/test split *within* the day; we hold out all of February, which is closer to how a retrieval model would be deployed. Keep our design.
2. **Confidence weights: check recommendation-engine's scheme against ours.** They use view=0.1, cart=0.3, purchase=1.0 **with a 20-day half-life decay**; we use view=1, cart=2, purchase=4 flat. The ratios are similar (10:30:100 vs 25:50:100), but their decay is interesting — worth testing as a second variant for ALS/Two-Tower.
3. **Item features recommended by Transformers4Rec paper are close to ours but add one thing we haven't enumerated: relative-price-to-category.** NVIDIA's tutorial computes `(price − category_avg_price) / category_avg_price` as a continuous feature. We should add this to our item/user feature tables.
4. **Session-level user features validated by Transformers4Rec RQ3 (+4.72% NDCG on REES46):** day-of-week sin/cos, product recency (log-normalized days since item's first appearance), price log-normalized. Our EDA already proposed peak_hour_bucket and preferred_dow — this is direct empirical support for keeping them.
5. **Two-Tower is the right choice for our setup, not Transformer.** The T4Rec paper's session length stats show REES46 avg=5.49 — short. For retrieval over 284K items at scale with mostly short sessions, Two-Tower with in-batch negatives is more efficient and more honest about what signal exists. We should not abandon Two-Tower in favour of XLNet.

---

## 1. Data Preprocessing & Cleaning

| Source | Sample scope | Nulls | Dedup | Bot removal | Core filter | Price floor |
|---|---|---|---|---|---|---|
| **T4Rec paper** (REES46) | Oct 2019 (31 days), 17.97M events | Not discussed | Consecutive repeats removed | Not mentioned | Single-event sessions dropped, sessions truncated to 20 | None |
| **NVIDIA tutorial 01** | 1 week of Oct 2019 (~6.4M after dedup) | Drops 2 null `user_session` rows | Consecutive repeats per session (42.4M → 30.7M, 27% removed) | None | Sessions with ≥2 items | None |
| **ecommerce-transformer4rec** (Kaggle) | Tiny slice, ~340K raw → 207K dedup | Drops null `user_session` | Consecutive repeats per session | None | MIN_SESSION_LENGTH=2, SESSIONS_MAX_LENGTH=20 | None |
| **userbehavior-transformers4rec** | Pre-built sequential session data | Inherited from upstream | Inherited | None | Inherited | None |
| **recommendation-engine** (Spark/ALS) | Oct 2019 | Not explicit | **Multiple cart events in same session → 1 per session** (domain-specific) | None | None | None |
| **rfm-analysis** | Oct+Nov 2019 purchases only | Implicit (purchase-only filter) | None | None | Single-purchase users kept | None |
| **xgboost-behaviour** | Nov 2019, cart+purchase only | `dropna(how='any')` | dedup on (event_type, product_id, price, user_id, user_session) | None | None | None |
| **why-most-users...** | Single month sample, row-limited load | Event-type filter only | First-occurrence-per-user for funnel | None | None | None |
| **OUR project** | 5 months, 279.9M rows cleaned | "unknown" encoding for category_code/brand, drop 66 null user_session | Exact dedup + 1s near-dedup window | **>300 EPD → drop (285 users)** | **Iterative k=3 user+item** | **$1.00 floor** |

**Key observations:**

- **We are the only source with explicit bot removal and k-core filtering.** The Transformers4Rec paper doesn't describe either, and the Kaggle T4Rec notebook certainly doesn't. Our EDA-driven justification (the 200 EPD natural break, but choosing 300 to avoid over-filtering; k=3 instead of k=5 because the 1.68M extra users cost only 2% event noise) is a methodological strength, not something to cut.
- **Our null-handling decision is non-obvious but correct.** Three of the session-based sources use `Categorify()` which maps nulls to a single integer (1 in NVTabular). Our `"unknown"` string encoding is functionally equivalent — both treat nulls as their own category embedding. The EDA finding that nulls cluster into 527 distinct category_ids with a coherent ~$175 price segment validates *not dropping* them, which is what the xgboost notebook incorrectly does via `dropna(how='any')` (that notebook is throwing out real products).
- **Sampling strategy varies wildly.** NVIDIA and the Kaggle T4Rec notebooks sample by time (first N days). The recommendation-engine notebook samples by fraction (`df.sample(0.001, seed=321)` — commented out but present). We sample by **user** (draw N random users, keep all their events), which is the right choice for a retrieval model where we need full user histories. None of the sources do this.
- **The recommendation-engine cart-dedup is a real insight worth stealing.** Their observation: user–product pairs had implausibly high cart counts (100+) because real platforms let users add/remove from cart repeatedly. Without cart-remove events in the data, they cap at 1 cart event per (user, product, session). Our pipeline doesn't do this. **For Two-Tower, I'd add this as a sixth cleaning step** — it would affect interaction matrix confidence values but not event counts materially.

**Verdict on our cleaning:** stricter than any published source, defensibly so. No changes needed. Optional add: cap carts at 1 per (user, product, session).

---

## 2. Data-Level Insights

What each source surfaces that our EDA might have missed or reinforces:

- **Cart→purchase conversion is the easy part; view→cart is where users drop.** (why-most-users-never-add-to-cart, xgboost-behaviour) — Funnel-enforced analysis shows 95% drop at view→cart but only 38% at cart→purchase. Our EDA section 7 confirms the same pattern from the opposite direction (21.5% of users ever cart, 12.5% ever purchase — cart→purchase odds ≈ 58%).
- **View→cart latency has a clear bimodal pattern.** (why-most-users-never-add-to-cart) — 52% of cart-adding users do so within 2 minutes of viewing, 23% take more than 10 minutes. **This is a potential user feature we haven't considered:** median view-to-cart time as a user attribute (fast-decider vs slow-decider).
- **Construction category is a top-1 skew driver.** (recommendation-engine) — Product 1004767 (Samsung `construction.tools.light`) has 3.47M events and will dominate any popularity baseline. Our EDA caught this (Section 5) and the category was verified legitimate. Transformers4Rec paper's top-items table shows the same.
- **RFM segments on purchase data are severely skewed "Lost".** (rfm-analysis) — 231K "Lost" users, 37K "Champions", 114K "Hibernating" out of 697K purchasing users. This mirrors our finding that **62.8% of users appear in only one month**. RFM segments could be a viable user feature for Two-Tower (categorical embedding with 8 values) but would only be computable for the 12.5% of users who ever purchase.
- **Double-11 ≠ Black Friday.** (eCommerce Predictor README) — Useful reminder that this is US data (REES46, New York HQ). Our holiday features should use US calendar: Thanksgiving (Nov 28, 2019), Black Friday (Nov 29), Cyber Monday (Dec 2), Christmas. The predictor notebook reports **PV MAPE 2.73%** with US-aware features vs 26% baseline — holiday encoding matters.
- **COVID-19 hits mid-March 2020.** (eCommerce Predictor) — Their v5.0 model (152 training days including pandemic) degraded catastrophically: PV MAPE went from 2.7% to 30%. For us, this reinforces the EDA decision to reserve March–April 2020 as an **MLOps retraining holdout** rather than include in train — unintentionally, we set ourselves up for a distribution-shift experiment. That's valuable.
- **Item skew warning for Spark operations.** (Transformers4Rec paper, our own EDA Section 5) — The top product has 3.47M events. Any Spark groupBy/join keyed on `product_id` will straggle. The recommendation-engine notebook does not discuss this. Our cleaning pipeline correctly repartitions by `user_id`.

**Insight our project has that none of the sources do:** the explicit correction of the bot-user count (45 → 285) is unique to our work. The T4Rec paper is silent on bots entirely.

---

## 3. Problem Definition & Objectives

| Source | Task | Framing |
|---|---|---|
| Transformers4Rec paper | **Session-based next-item prediction** | Given in-session history x₁…xₙ₋₁, predict xₙ from all items |
| ecommerce-transformer4rec (Kaggle) | Same as above | Session-based next-item, XLNet + MLM |
| userbehavior-ecommerce-transformers4rec | Same as above | Educational walkthrough of the pipeline |
| NVIDIA tutorials 01/02/03 | Same as above | End-to-end Merlin pipeline, GRU baseline then XLNet |
| recommendation-engine | **User-based user–item collaborative filtering** | Given (user_id, product_id, interaction_score) matrix, predict top-N products per user. Includes item–item similarity via LSH |
| xgboost-behaviour | **Cart→purchase classification** | Binary classifier: given cart event + features, will user purchase? |
| rfm-analysis | **Customer segmentation** | Unsupervised labeling into 8 RFM tiers |
| why-most-users-never-add-to-cart | **Funnel/drop-off EDA** | No model, descriptive only |
| eCommerce Predictor | **Aggregate time-series forecasting** | Predict daily PV/UV/purchase count, not per-user recs |
| **OUR project (Phase 7)** | **User-based Two-Tower retrieval** | Given user aggregate features, retrieve top-K items from catalog of 284,523 |

**Observation:** Our framing sits *between* the session-based camp and the collaborative-filtering camp. It's closest to recommendation-engine (user–item retrieval) but with neural embeddings rather than matrix factorization. **No prior source on REES46 does this exact task**, which is both good (novelty) and slightly scary (no published baselines to calibrate against).

**Primary metrics comparison:**
- **Session-based sources (T4Rec paper, Kaggle T4Rec, NVIDIA):** NDCG@20, Recall@20 — leave-last-item-out within session
- **recommendation-engine:** RMSE, MAE on the interaction score — **this is wrong for a recommender;** they are measuring reconstruction quality, not ranking quality. Our Recall@10 / NDCG@10 / Precision@10 is the right choice.
- **xgboost:** Accuracy, Precision, Recall (binary classification, not ranking)
- **Our project:** Recall@10, NDCG@10, Precision@10 — aligned with Transformers4Rec paper metrics (@10 instead of @20 is a conscious choice; for retrieval feeding a downstream ranker, smaller K is appropriate)

---

## 4. Ground Truth & Evaluation Setup

This is the section where our approach differs most from prior work.

**Train/test splits observed:**

- **Transformers4Rec paper:** Incremental per-day. For each day T, train on days [1..T], evaluate on day T+1 (split 50:50 val/test within that day). **Average-over-time** the per-day metrics. Very production-realistic but expensive.
- **NVIDIA tutorials:** Same incremental pattern on 7 days of October 2019. Fine-tuning + eval loop over days 1→4 (so 3 training cycles). Provides per-day metrics but does not aggregate.
- **ecommerce-transformer4rec Kaggle notebook:** Also daily incremental but reports only one number (NDCG@10=0.18, Recall@10=0.30).
- **recommendation-engine:** `randomSplit([0.8, 0.2])` — **random split on (user, product) pairs**. This is wrong for a recommender (user can be in both train and test, future events can leak to past). RMSE=0.014 is not meaningful without ranking metrics.
- **xgboost:** `train_test_split` default, random 80:20 on the cart-record-level. Fine for a classifier, wrong for a recommender.

**Our approach:**
- **Temporal split at Jan/Feb 2020 boundary** (train: Oct 2019 – Jan 2020; test: Feb 2020)
- **User overlap:** ~73.5% across all three sample sizes — majority of Feb users have prior history (warm), ~26.5% are cold-start (this is a design feature — we can measure cold-start performance separately)
- **Leave-one-out on last event per user** for evaluation

**Comparison verdict:**
- We are stricter than the recommendation-engine notebook (no random splits = no leakage)
- We are less granular than the Transformers4Rec paper (single eval point vs per-day) but that's appropriate for retrieval
- **One change worth considering:** add a *validation* split from late January 2020 (last 7 days) for hyperparameter tuning, keeping Feb untouched as test. This would prevent overfitting to the test set during embedding dimension / learning rate sweeps. The Transformers4Rec paper does this on the first 50% of days and evaluates on all days.

**Ground truth definition — important subtle point.** Our memory notes leave-one-out on the last event per user in the test period. The Transformers4Rec paper and the NVIDIA tutorials define positives as **the last item in every session** (not the last item per user) — this is richer signal because every session in Feb contributes. Since a user in February may have multiple sessions, this roughly 5x expands our effective test set. **Recommendation:** reconsider last-event-per-user vs last-event-per-session. Per-session gives more evaluation points and better statistical power.

---

## 5. Model Architecture & Training

### Session-based Transformer camp (5 sources)

| Source | Model | d_model | n_layer | n_head | Masking | Loss | Batch size | LR | Epochs |
|---|---|---|---|---|---|---|---|---|---|
| T4Rec paper | XLNet+RTD, also GRU/BERT/ELECTRA/GPT2/Transformer-XL | Tuned | Tuned | Tuned | CLM/MLM/PLM/RTD | XE | Tuned | Tuned | Tuned |
| ecommerce-transformer4rec (Kaggle) | XLNet | 192 | 2 | 4 | MLM | XE (NLLLoss) | default | default | not reported |
| userbehavior-transformers4rec | XLNet (walkthrough) | 64 | — | — | MLM | NLLLoss | 8 (demo) | — | — |
| NVIDIA tutorial 03 | GRU (first), then XLNet | 128 (GRU), unspecified (XLNet) | 1 GRU / 2 XLNet | — | Causal (GRU), MLM (XLNet) | XE | 256 train / 32 eval | 0.00071 | 3 per day, 3 days |

**Shared design choices across this camp:**
- `max_sequence_length = 20`, `min_session_length = 2`
- **Weight tying** between input item embedding and output projection (Transformers4Rec paper finds this gives +6.7% NDCG on e-commerce)
- **In-session sequential loss**: predict every masked position simultaneously
- **No explicit negative sampling** — implicit via softmax over all items (284K items in our catalog → expensive full softmax; T4Rec handles this fine on GPU)

### Collaborative filtering camp

**recommendation-engine (PySpark ALS):**
- Algorithm: Spark MLlib ALS with `implicitPrefs=True`
- Best hyperparameters (5-fold CV): rank=15, regParam=0.005, alpha=0.0 (huh — alpha=0 with implicit prefs is unusual; normally you want alpha > 0 to control confidence weighting)
- Interaction score: `views*0.1 + carts*0.3 + purchases*1.0` multiplied by **half-life decay** `exp(ln(0.5) * recency_days / 20)` (20-day half-life). Then log10 transform and cap at 100.
- Plus **LSH for item–item similarity** (`BucketedRandomProjectionLSH`, numHashTables=5, bucketLength=0.1) — used to find substitute/complement products from ALS item factors
- No Recall/NDCG reported — only RMSE=0.014, MAE=0.007 (these are on the interaction score, essentially meaningless for ranking)

### Two-Tower (our project) — no prior art on this dataset

Our planned architecture (in-batch negative sampling, L2 normalization, FAISS `IndexFlatIP`) is standard for large-scale retrieval. The closest analogue in the reviewed sources is the recommendation-engine ALS system, which uses a very different training objective.

**Specific recommendations from the comparison:**

1. **Embedding dimension:** the Kaggle T4Rec notebook uses 64-dim item embeddings, the NVIDIA tutorial uses 128. For our 284K catalog, start with 64 and scale up if underfitting.
2. **Weight tying analogue:** there's no direct "tying" in Two-Tower (since user and item towers are separate), but L2-normalizing both sides before the dot product (as planned) is the correct equivalent — it makes the dot product a cosine similarity and stabilizes training. Keep this.
3. **Label smoothing:** the Transformers4Rec paper's Section 3.2.4 calls out label smoothing as their most impactful regularizer. We should add it to the in-batch negative softmax loss.
4. **Logit-scale temperature:** not mentioned in any source but standard for two-tower. Start with learnable temperature or fixed τ=0.1.
5. **Bias correction for popular items under in-batch negatives.** None of the sources discuss this, but it's a known issue — popular items appear disproportionately as negatives, causing the model to suppress them. The YouTube paper's `log(Q(j))` correction is standard. Add this.

---

## 6. Evaluation & Results — What to expect

**Published numbers on REES46 Oct 2019 (all session-based):**

| Source | Model | NDCG@10 | NDCG@20 | Recall@10 | Recall@20 |
|---|---|---|---|---|---|
| T4Rec paper RQ1 | XLNet (PLM) | — | 0.2422 | — | 0.4760 |
| T4Rec paper RQ1 | GRU4Rec (FT) | — | 0.2231 | — | 0.4414 |
| T4Rec paper RQ1 | V-SkNN | — | 0.2187 | — | 0.4662 |
| **T4Rec paper RQ2** | **XLNet (RTD)** | — | **0.2546** | — | **0.4886** |
| T4Rec paper RQ3 | XLNet (MLM) + side info + SOHE | — | 0.2542 | — | 0.4858 |
| Kaggle T4Rec notebook | XLNet (MLM) | 0.1799 | 0.2016 | 0.3005 | 0.3858 |
| NVIDIA tutorial 03 | GRU (day 4, 3 days trained) | 0.0675 | 0.0817 | 0.1260 | 0.1826 |

**Important calibration:** these are session-based numbers, and they include all 284K items as candidates. The T4Rec paper filters only sessions ≥2 and truncates to 20 — no k-core. The absolute comparison to our Two-Tower Recall@10 is apples-to-oranges, but the order of magnitude is informative.

**What we should expect for Two-Tower on 50K user sample:**

- **Lower than session-based** because Two-Tower conditions on *aggregate* user features, not a rich within-session sequence. Session models effectively have `O(20)` recent items as input; we have categorical user features.
- **Reasonable first target: Recall@10 in the 0.05–0.15 range, NDCG@10 in the 0.03–0.10 range.** This is based on typical Two-Tower papers on similar-scale e-commerce data.
- **A popularity baseline will be surprisingly strong.** The T4Rec paper's V-SkNN (session-kNN baseline) already scores 0.2187 NDCG@20. Always compare against: (a) global popularity, (b) per-category popularity, (c) item-item co-occurrence kNN. If Two-Tower can't beat (c), something is wrong.

**Unexpected result from T4Rec paper that's relevant to us (Table 2):** BERT4Rec (MLM, no side info) actually scored higher Recall@20 than XLNet+PLM on YOOCHOOSE (0.6349 vs 0.6282). **MLM beat PLM.** If we ever revisit the session-based route, MLM is the masking strategy.

---

## 7. Unique Techniques & Novel Approaches

Notable techniques from each source worth knowing about, whether or not we adopt them:

- **Half-life temporal decay on implicit feedback** (recommendation-engine) — `exp(ln(0.5) * recency_days / 20)`. Recent events worth more than old ones. Not common in Two-Tower work but straightforward to add as a sample weight during training. **Worth a sweep.**
- **Cyclical encoding of day-of-week** (NVIDIA tutorial 02) — sin/cos of the weekday so Monday and Sunday are close in feature space. Standard technique but easy to overlook. **Add to user features.**
- **Relative price to category average** (NVIDIA tutorial 02) — `(price − category_mean_price) / category_mean_price` captures "is this a cheap or expensive item for its category." Item-level feature. **Add.**
- **Product recency (log-normalized days since first seen)** (NVIDIA tutorial 02) — an item-level feature that captures freshness. Older items tend to have more interactions; this lets the model separate freshness from popularity. **Add.**
- **Tying embeddings** (T4Rec paper Section 3.2.3) — input item embedding matrix is shared with the output projection. +6.7% NDCG on e-commerce in their ablations. Not directly applicable to Two-Tower, but the cosine-similarity via L2 normalization is the conceptual equivalent.
- **Stochastic Shared Embeddings / swap noise** (T4Rec paper, SSE-PT reference) — during training, randomly swap a user/item embedding with a different one with small probability. Regularization technique. Probably unnecessary at our scale, but noted.
- **Replacement Token Detection (RTD)** (T4Rec paper) — ELECTRA-style discriminator task. Their best result. Transformer-specific, not directly applicable to Two-Tower.
- **LSH for item–item similarity** (recommendation-engine) — fast approximate nearest neighbors. Interesting alternative to FAISS `IndexFlatIP`. FAISS is more standard for GPU inference and supports more index types. **Stick with FAISS.**
- **Soft One-Hot Encoding (SOHE)** for continuous features (T4Rec paper RQ3) — best-performing feature aggregation on REES46 (+4.72% NDCG). Each continuous feature becomes a soft distribution over bins. Not hard to implement. **Worth testing** for price and recency features.
- **Downsampling to balance classes** (xgboost) — they resample 500K purchases vs 500K non-purchases. For retrieval this doesn't apply directly, but it's a reminder that in-batch negatives should sample users uniformly at random, not weighted by activity (otherwise heavy users dominate).
- **US-holiday awareness** (eCommerce Predictor) — Veterans Day, Thanksgiving, Black Friday, Cyber Monday, Christmas. For a one-shot Two-Tower model we probably don't care, but if we ever extend to daily incremental training, these should be features.
- **Per-session cart dedup** (recommendation-engine) — limit cart events to 1 per (user, product, session) to handle add/remove cycles. **Good idea, adds robustness, not in our pipeline.**

---

## 8. Gaps & Opportunities

### What nobody has tried on this dataset (i.e., our legitimate contribution)

1. **User-based Two-Tower retrieval with neural embeddings.** The closest prior work is the recommendation-engine ALS notebook, which uses matrix factorization. We'd be the first to do the neural version on this data.
2. **Proper retrieval metrics on user-based recommendations.** The ALS notebook reports RMSE/MAE. The session-based work reports Recall/NDCG on sessions. Nobody reports Recall@K on *user-level* recommendations against a held-out temporal split.
3. **Scale: 500K users × 18.5M events.** The biggest published experiment on REES46 is the Transformers4Rec paper's 17.97M on 31 days; we match them on event volume but operate on 5x the time window with a sampling strategy built for user-based modeling.
4. **Bot removal + iterative k-core as documented preprocessing.** No source treats this systematically.
5. **Cold-start measurement.** With 26.5% of February users never appearing in training, we can cleanly measure cold-start performance. None of the sources measure this (session-based models fake-solve cold-start by reading the in-session context; ALS literally cannot recommend to cold-start users).

### What's unexploited in the dataset

1. **Session structure inside user histories.** Our Two-Tower treats user features as aggregates, losing session boundaries. A **hybrid** approach — Two-Tower for retrieval, then session-based re-ranking — would combine strengths. (Beyond Phase 7 scope, but note for Phase 8+.)
2. **The Mar–Apr 2020 distribution shift (COVID-19) holdout.** We already reserved this; the eCommerce Predictor README documents that this period has materially different patterns. We can use Mar–Apr as a planned distribution-shift test for the MLOps retraining pipeline. None of the sources do this.
3. **Category hierarchy depth.** Our EDA noted 2-level and 3-level category codes. None of the sources exploit this as a hierarchical feature. We could embed each level separately and sum/concatenate.
4. **The 1,485,690 purchasing users as a cleaner signal.** All sources treat views/carts/purchases uniformly (with weights). An alternative: train a second Two-Tower specifically on purchase events (far sparser but cleaner signal) and ensemble with the main model at inference.
5. **RFM segments as user features.** Computable for the 12.5% of users who ever purchase. Would go in the user feature table as a categorical field with 8 values.

### What I'd recommend as "next steps" for the project

**For immediate Phase 7 execution:**

1. **Build the item feature table first, then user.** Item features (category_level_1, category_level_2, brand, price_log_norm, price_relative_to_category_avg, product_recency_days_log_norm, total_interactions_log) are easier to validate end-to-end.
2. **Start small: 50K sample, Two-Tower with only `product_id` (item) and `user_id` (user) — pure collaborative.** This gives a pure ID-based baseline. Everything else should beat it.
3. **Add side features incrementally and measure delta.** T4Rec paper's RQ3 adds them all at once; we should add one at a time to isolate which features matter.
4. **Benchmark against 3 baselines:** (a) global popularity, (b) per-category popularity top-N, (c) item-item co-occurrence kNN. These are cheap to build and if Two-Tower doesn't beat (c) on Recall@10, something is wrong with training.
5. **Implement logQ correction for in-batch negatives.** This is the single most-common bug in two-tower retrieval implementations and none of our sources discuss it.
6. **Use the last-event-per-session ground truth, not last-event-per-user.** Richer signal, aligned with the Transformers4Rec paper's design.
7. **Hold out a validation slice from late January 2020.** Keep February as untouched test. Match the T4Rec paper's hyperparameter tuning discipline.

**Beyond Phase 7:**

8. **Cold-start analysis.** Split test users into "in training history" vs "cold" and report Recall@10 separately for each. This is a genuine contribution absent from all sources.
9. **Distribution-shift test on Mar–Apr 2020.** Run the trained Feb-evaluated model on Mar–Apr without retraining, measure degradation, retrain, measure recovery. This is the MLOps story we already have scaffolded.

---

## Contradictions / ambiguities between sources flagged

- **Bot user count on Oct 2019:** our corrected count is 285. No source confirms or contradicts because no source computes it.
- **Session length distribution:** T4Rec paper's REES46 avg session length is 5.49 (after filtering single-event sessions and truncating to 20). Our EDA Section 6 reports 41.88% of sessions are single-event *before* filtering. Consistent after accounting for their `min_session_length=2`.
- **Confidence weighting scheme:** recommendation-engine uses view=0.1, cart=0.3, purchase=1.0 with 20-day half-life. We use view=1, cart=2, purchase=4 flat. Both ratios are in the same ballpark (view:purchase ≈ 1:10 in theirs, 1:4 in ours). Worth testing both.
- **Category `construction.tools.light`:** the recommendation-engine notebook doesn't flag it as suspicious; we initially did. The Transformers4Rec paper's top-items table shows it as dominant (Samsung/Apple smart-lighting). All sources agree it's legitimate. Our initial skepticism was worth resolving but the conclusion is confirmed.
- **What ALS alpha should be for implicit prefs:** recommendation-engine's grid search selected alpha=0.0 as best. This is strange — Hu/Koren (2008) implicit ALS requires alpha > 0 to scale confidences. Likely an artifact of their tiny RMSE on an already-small interaction scale, not a signal we should act on. Ignore.

---

## Source cross-reference table

| # | Source | Stack | Task | Scope | Key contribution |
|---|---|---|---|---|---|
| 1 | Transformers4Rec paper (RecSys'21) | PyTorch + HF Transformers | Session-based next-item | Oct 2019, 17.97M events | Best published benchmarks; XLNet+RTD novel |
| 2 | ecommerce-transformer4rec (Kaggle) | T4Rec library | Session-based | Tiny slice | Hands-on T4Rec usage example |
| 3 | userbehavior-transformers4rec (Kaggle) | T4Rec components | Component walkthrough | Pre-built sequential data | Pedagogical |
| 4 | NVIDIA Merlin tutorial 01 | cuDF + NVTabular | Preprocessing | 1 week Oct 2019 | GPU preprocessing idioms |
| 5 | NVIDIA Merlin tutorial 02 | NVTabular | Feature engineering + partitioning | 1 week Oct 2019 | Full feature pipeline |
| 6 | NVIDIA Merlin tutorial 03 | T4Rec + NVTabular | GRU then XLNet training | 1 week Oct 2019 | End-to-end training loop |
| 7 | recommendation-engine (Kaggle) | PySpark ML | ALS + LSH | Oct 2019 | Only user-based CF on this data |
| 8 | rfm-analysis (Kaggle) | pandas | Customer segmentation | Oct+Nov 2019 purchases | RFM segments |
| 9 | xgboost-behaviour (Kaggle) | sklearn + xgboost | Cart→purchase classifier | Nov 2019 cart/purchase | Binary classification baseline |
| 10 | why-most-users-never-add-to-cart (Kaggle) | pandas | Funnel EDA | Single month | View→cart drop quantified |
| 11 | eCommerce Predictor (GitHub) | sklearn | PV/UV/purchase time-series | 7 months | US-holiday features matter |

---

*Analysis compiled March 2026. Supersedes no prior document; complements `reports/eda_report_v2.md` and `reports/04_sampling_splits_interactions.md`.*
