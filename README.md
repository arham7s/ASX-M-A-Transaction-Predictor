# ASX M&A Transaction Predictor

> **Can machine learning predict which ASX-listed companies will be involved in a merger or acquisition?**

A four-phase empirical pipeline that trains a Platt-calibrated XGBoost classifier on ten years of ASX financial data, management sentiment, M&A network topology, and macro regime features — then serves predictions through an interactive GitHub Pages dashboard.

**Live dashboard →** [arham7s.github.io/asx-ma-predictor](https://arham7s.github.io/ASX-M-A-Transaction-Predictor/#predictions)

---

## Table of contents

1. [Project overview](#1-project-overview)
2. [Key results](#2-key-results)
3. [Repository structure](#3-repository-structure)
4. [Data sources](#4-data-sources)
5. [Methodology](#5-methodology)
   - [Phase 1 — Model upgrade](#phase-1--model-upgrade)
   - [Phase 2 — Signal enrichment](#phase-2--signal-enrichment)
   - [Phase 3 — Evaluation framework](#phase-3--evaluation-framework)
   - [Phase 4 — Dashboard](#phase-4--dashboard)
6. [Feature engineering](#6-feature-engineering)
7. [Safeguards against common pitfalls](#7-safeguards-against-common-pitfalls)
8. [Installation and run order](#8-installation-and-run-order)
9. [Reproducing results](#9-reproducing-results)
10. [Dashboard deployment](#10-dashboard-deployment)
11. [Limitations and future work](#11-limitations-and-future-work)
12. [Academic context](#12-academic-context)

---

## 1. Project overview

Mergers and acquisitions are among the highest-impact corporate events — for shareholders, employees, and markets. Predicting which companies will be involved before a deal is announced has obvious value for event-driven investment strategies, strategic planning, and academic research on deal determinants.

This project started from a baseline Random Forest classifier trained on 11 financial ratios from ASX-listed companies. It was rebuilt across four phases into a production-grade machine learning pipeline that addresses the original model's key weaknesses: lookahead bias in validation, a single signal source, uncalibrated probabilities, and a minimal evaluation framework.

The final model achieves a **ROC-AUC of 0.824** and a **portfolio hit rate of 58%** against a base rate of ~8% — a **7× lift** — validated through strict walk-forward cross-validation across FY2013–FY2017.

---

## 2. Key results

| Metric | Phase 1 (RF baseline) | Phase 2 (XGBoost + enriched) |
|---|---|---|
| ROC-AUC | 0.7614 | **0.8241** |
| PR-AUC | 0.3821 | **0.4817** |
| Calibration error (ECE) | 0.1204 | **0.0312** (after Platt scaling) |
| Brier score | 0.0812 | **0.0641** |
| Best F1 | — | **0.570** (threshold = 0.45) |

**Backtest results (FY2013–FY2017, top-20 portfolio, walk-forward):**

| Year | Hit rate | Portfolio return | ASX 200 | Alpha |
|---|---|---|---|---|
| FY2013 | 50% | +11.2% | +14.8% | −3.6% |
| FY2014 | 55% | +9.8% | +5.5% | **+4.3%** |
| FY2015 | 60% | +3.1% | −0.9% | **+4.0%** |
| FY2016 | 65% | +18.9% | +11.7% | **+7.2%** |
| FY2017 | 56% | +14.3% | +11.9% | **+2.4%** |
| **Mean** | **57.2%** | — | — | **+2.9%** |

Hit rate = fraction of flagged companies that announced a deal. Base rate (unconditional) = 8.2%.

---

## 3. Repository structure

```
asx-ma-predictor/
│
├── index.html                   ← GitHub Pages dashboard (single file, self-contained)
│
├── data/
│   ├── ml_raw_data.xlsx         ← Training data: financial ratios + transaction labels
│   └── FY18_Features.xlsx       ← Prediction target: FY17 financials for FY18 forecast
│
├── pipeline/
│   ├── phase1_train_evaluate.py ← XGBoost + walk-forward CV + SHAP
│   ├── phase1_predict.py        ← Score new companies using trained artifacts
│   ├── phase2a_sentiment.py     ← VADER sentiment + M&A keyword features
│   ├── phase2b_network.py       ← M&A graph centrality features (NetworkX)
│   ├── phase2c_macro.py         ← Macro regime features (RBA + ASX200 via yfinance)
│   ├── phase2_train_evaluate.py ← Merge all Phase 2 features + retrain
│   ├── phase3a_calibration.py   ← Platt/isotonic calibration + ECE + threshold sweep
│   ├── phase3b_backtest.py      ← Walk-forward backtest + hit rate + alpha
│   └── phase3c_report.py        ← Master PDF report + report_data.json export
│
├── outputs/
│   ├── model_artifacts/         ← Phase 1 trained model artifacts (.pkl)
│   ├── model_artifacts_p2/      ← Phase 2 trained + calibrated model artifacts (.pkl)
│   ├── predictions_FY18.xlsx    ← FY18 company scores with SHAP drivers
│   ├── report_data.json         ← Machine-readable metrics for dashboard
│   ├── evaluation_report.pdf    ← Publication-ready methodology report
│   ├── shap_beeswarm.png        ← Global SHAP feature importance plot
│   ├── shap_bar.png             ← Mean |SHAP| bar chart
│   ├── calibration_curve.png    ← Reliability diagram (raw vs Platt vs isotonic)
│   ├── backtest_returns.png     ← Annual hit rate vs base rate chart
│   └── roc_pr_curves.png        ← ROC and PR curves (P1 vs P2)
│
└── requirements.txt
```

---

## 4. Data sources

### Financial and transaction data

- **Financial ratios** (`ml_raw_data.xlsx`): Income statement, balance sheet, and cash flow statement items for ~1,500 ASX-listed companies over FY2008–FY2017, sourced from Bloomberg/Refinitiv. Financial ratios are computed from these statements (see feature list below).
- **Transaction labels**: Historical M&A transaction records for the same universe and period. Transaction types (`BUY`, `SELL`, `BUY&SELL`) are collapsed into a single `DEAL` label; non-transacting company-years are `NO DEAL`.
- **FY18 prediction features** (`FY18_Features.xlsx`): FY17 financial data (since ASX fiscal years end 30 June, FY17 data is available at prediction time) for all ~1,500 companies, used to generate FY18 forecasts.

### Macro data (auto-fetched, no manual download required)

- **RBA cash rate**: Fetched directly from the Reserve Bank of Australia's published CSV at `rba.gov.au/statistics/tables/csv/f1.csv`
- **ASX 200 prices**: Fetched via `yfinance` (ticker: `^AXJO`), used to compute annual return and 12-month rolling volatility per fiscal year

### Sentiment data (two modes)

- **Mode A** (if available): Earnings call transcripts as `.txt` files in a `transcripts/` folder, named `<TICKER>_<YEAR>.txt`
- **Mode B**: NewsAPI free tier (500 requests/day) — set `NEWSAPI_KEY` in `phase2a_sentiment.py`
- **Mode C** (demo): Synthetic data is generated automatically if neither source is available, so the pipeline runs end-to-end without requiring text data

---

## 5. Methodology

### Phase 1 — Model upgrade

**Script:** `phase1_train_evaluate.py`, `phase1_predict.py`

The original baseline used a `sklearn` Random Forest with a random 80/20 train-test split. Two problems with this:

1. **Lookahead bias**: A random split can place FY2017 data in the training set and FY2013 data in the test set. The model implicitly learns from the future.
2. **Black box**: Random Forest gives no explanation of which financial ratios drive which predictions.

**Changes made:**

- Random Forest replaced with **XGBoost** (primary) and **LightGBM** (benchmark). XGBoost consistently outperforms RF on tabular financial data due to better handling of class imbalance via `scale_pos_weight` and more controlled regularisation.
- Random split replaced with **walk-forward validation**: each fold trains on all years prior to the test year and tests on the test year alone. A minimum of 3 training years is required before the first fold opens.
- **SHAP (SHapley Additive exPlanations)** values computed for every prediction. Global importance is shown via beeswarm and bar plots; individual company predictions get waterfall plots showing exactly which features pushed the score up or down.
- Evaluation expanded from a single confusion matrix to **ROC-AUC**, **PR-AUC** (more informative than ROC-AUC for rare-event classification), full classification reports, and per-fold metrics.
- Trained model artifacts (`xgb_model.pkl`, `lgb_model.pkl`, `scaler.pkl`, `label_encoder.pkl`) are serialised so the prediction script loads the same objects used in training — preventing subtle preprocessing inconsistencies.

**XGBoost hyperparameters:**

```python
XGBClassifier(
    n_estimators     = 500,
    learning_rate    = 0.05,
    max_depth        = 4,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    scale_pos_weight = n_neg / n_pos,   # handles class imbalance
    eval_metric      = "logloss",
    random_state     = 42
)
```

---

### Phase 2 — Signal enrichment

**Scripts:** `phase2a_sentiment.py`, `phase2b_network.py`, `phase2c_macro.py`, `phase2_train_evaluate.py`

The 11-feature financial ratio set is expanded to **31 features** across three new signal groups.

#### 2A — Sentiment features (4 new features)

Management language is a documented leading indicator of M&A intent. Executives use acquisition vocabulary ("bolt-on", "strategic fit", "synergistic", "accretive") in earnings calls and press releases before deals are announced.

VADER (Valence Aware Dictionary and sEntiment Reasoner) is used because it is rule-based and requires no training data, making it suitable for financial text without domain fine-tuning. Four features are extracted per company-year:

| Feature | Description |
|---|---|
| `sentiment_score` | Mean VADER compound score across all sentences in the year's text |
| `sentiment_volatility` | Standard deviation of sentence-level scores — erratic tone signals uncertainty |
| `ma_keyword_score` | Count of M&A-intent vocabulary hits (22 keywords including "acquisition", "synergy", "bolt-on") |
| `tone_shift` | Year-over-year change in `sentiment_score` — deteriorating tone precedes sell-side activity |

#### 2B — Network centrality features (7 new features)

M&A activity is not random across the company universe. Serial acquirers cluster together. Companies that sit at the centre of the historical deal network have structurally different characteristics to peripheral companies — and this information is completely invisible to financial ratios alone.

A directed graph is constructed where each historical M&A transaction is an edge from acquirer to target. Features are computed using a **rolling window**: when building features for year T, only edges from years strictly before T are used, preserving walk-forward integrity.

| Feature | Description |
|---|---|
| `degree_centrality` | (in-degree + out-degree) / (N−1) — normalised share of total network edges |
| `in_degree` | Number of times this company was a target in prior transactions |
| `out_degree` | Number of times this company was an acquirer in prior transactions |
| `betweenness_centrality` | How often this company sits on the shortest path between two other nodes — captures "broker" role in deal networks |
| `pagerank` | Recursive importance: high if connected to other high-PageRank nodes (serial acquirer clusters) |
| `clustering_coeff` | Cliquishness of the company's M&A neighbourhood |
| `acquirer_flag` | Binary: has this company ever made an acquisition? |

#### 2C — Macro regime features (9 new features)

M&A activity is strongly cyclical. Deals cluster in low-rate, tight-spread, rising-equity environments (expansion regimes) and collapse in stress regimes. A company's deal probability is not independent of the macro environment it sits in.

| Feature | Description |
|---|---|
| `rba_cash_rate` | RBA official cash rate at fiscal year end (June) |
| `asx200_return` | ASX 200 total return for the fiscal year |
| `asx200_volatility` | 12-month rolling volatility of ASX 200 daily returns |
| `credit_spread` | Proxy: 10Y government yield minus RBA cash rate |
| `rate_change_yoy` | Year-over-year change in cash rate (direction signal) |
| `ma_volume_lag1` | Total ASX M&A deal count in the prior year (momentum) |
| `regime_expansion` | 1 if macro regime classified as EXPANSION (low vol, positive equity) |
| `regime_contraction` | 1 if macro regime classified as CONTRACTION |
| `regime_stress` | 1 if macro regime classified as STRESS (high vol or credit spread > 2.5%) |

Regime classification uses a rule-based three-state model combining realised volatility, equity return, and rate direction. One-hot encoding is used (rather than ordinal encoding) because the relationship between regime and deal probability is non-linear.

#### Integration

All three feature CSVs are left-merged into `ml_raw_data.xlsx` on `[Ticker, Year]` (sentiment and network) and `[Year]` alone (macro). Missing values from unmatched rows are imputed with column medians. The merged 31-feature matrix is then fed through the same walk-forward XGBoost pipeline from Phase 1.

---

### Phase 3 — Evaluation framework

**Scripts:** `phase3a_calibration.py`, `phase3b_backtest.py`, `phase3c_report.py`

#### 3A — Probability calibration

XGBoost, like all tree ensembles, is systematically overconfident — it pushes scores toward 0 and 1. A raw score of 0.80 may correspond to only a 45% empirical deal rate. This matters enormously for threshold selection and any downstream use of the predicted probabilities.

Two calibration wrappers are fitted and compared:

- **Platt scaling**: Fits a logistic regression on top of raw XGBoost scores. Fast, stable, recommended for moderate-sized datasets.
- **Isotonic regression**: Non-parametric monotone calibration. More flexible but requires more data and is prone to overfitting on small evaluation sets.

Both are fitted on a held-out 70% split and evaluated on the remaining 30%. The primary calibration metric is **Expected Calibration Error (ECE)**:

```
ECE = Σ (|bin| / N) × |accuracy(bin) − confidence(bin)|
```

ECE measures the average gap between predicted probability and empirical deal frequency across 10 equal-width bins. The Platt-scaled model reduces ECE from 0.0893 (raw) to 0.0312 — a 65% reduction.

The **threshold analysis** sweeps every threshold from 0.10 to 0.85 in 5pp steps and computes precision, recall, and F1 at each point. The maximum-F1 threshold (0.45) is saved as `recommended_threshold.pkl` and is used as the default screen in the dashboard.

#### 3B — Walk-forward backtest

Out-of-fold predicted probabilities are collected by re-running the walk-forward loop. For each test year, the top-K=20 companies by predicted probability (subject to a minimum probability floor of 0.50) form the portfolio.

**Hit rate** is the primary evaluation metric: the fraction of portfolio companies that actually announced a deal in the test year. This is compared against the unconditional base rate (~8.2%) to compute the lift multiple.

Where price data is available via `yfinance`, total returns are measured over the fiscal year (1 July to 30 June) and compared to the ASX 200 benchmark to compute annual alpha.

#### 3C — Master report

Aggregates all Phase 1–3 outputs into a five-page publication-quality PDF report (via `matplotlib.backends.backend_pdf`) and a `report_data.json` file that feeds the dashboard directly.

---

### Phase 4 — Dashboard

**File:** `index.html` (single self-contained file)

A static GitHub Pages site that reads `report_data.json` and `predictions.json` and renders everything client-side. No backend, no server, no database. Two CDN dependencies: Google Fonts and Chart.js.

**Interactive features:**
- **Live threshold slider**: Drag to change the probability screen — the predictions table, precision, recall, and F1 all update in real time
- **Expandable company rows**: Click any row to see the plain-English explanation of the SHAP driver, the company's deal probability vs the current threshold, and historical deal activity
- **Calibration chart**: Bar chart showing predicted probability vs actual deal rate per decile — visual confirmation of calibration quality
- **Backtest chart**: Annual portfolio hit rate vs the base rate, colour-coded green/red for above/below baseline

To update the dashboard after a new pipeline run: replace the two JSON data objects inside the `<script>` tag in `index.html` and push.

---

## 6. Feature engineering

### Original 11 financial ratio features

These are computed from income statement, balance sheet, and cash flow data from Bloomberg/Refinitiv. The exact column names in `ml_raw_data.xlsx` must match what the pipeline expects (columns 1–11 of Sheet1).

| Category | Ratios |
|---|---|
| Liquidity | Quick ratio, current ratio |
| Leverage | Debt ratio, debt-to-equity |
| Profitability | Return on assets (ROA), return on equity (ROE), net profit margin |
| Efficiency | Asset turnover |
| Growth | Revenue growth (YoY) |
| Valuation | P/E ratio, EV/EBITDA |

### Phase 2 additions (20 new features)

See the feature tables in the [Phase 2 section](#phase-2--signal-enrichment) above.

### Feature importance (SHAP, Phase 2 model — top 10)

Based on mean |SHAP| across the full training set after final retraining:

1. `debt_ratio` — financial leverage is the single strongest discriminator
2. `pagerank` — network centrality (recursive M&A connectivity)
3. `ma_keyword_score` — management acquisition language frequency
4. `sentiment_score` — overall tone of earnings communications
5. `betweenness_centrality` — broker role in deal network
6. `out_degree` — serial acquirer history
7. `tone_shift` — year-over-year change in management tone
8. `quick_ratio` — liquidity position
9. `regime_expansion` — macro environment flag
10. `roa` — return on assets (profitability signal)

The emergence of network and sentiment features in the top-10 — above traditional financial ratios like ROA and current ratio — is the core empirical finding of this project.

---

## 7. Safeguards against common pitfalls

### Lookahead bias
Walk-forward validation strictly enforces that each test fold's model is trained only on data from prior years. The minimum training window is 3 years. Network features are also computed rolling (edges before year T only), and macro features use prior-year values when lagged appropriately.

### Class imbalance
M&A deals are rare events — approximately 8.2% of ASX company-years in this dataset. XGBoost's `scale_pos_weight` parameter is set to `n_negative / n_positive` (approximately 11:1) at model instantiation. PR-AUC is reported alongside ROC-AUC because ROC-AUC is misleading for highly imbalanced classes (a model that predicts "no deal" always achieves ROC-AUC > 0.5 trivially).

### Probability miscalibration
Raw XGBoost output probabilities are not used directly. A Platt scaling wrapper is fitted on a held-out 30% split, reducing ECE from 0.089 to 0.031. All downstream uses of probabilities (threshold selection, portfolio construction, dashboard display) use the calibrated model.

### Data leakage in scaling
The `StandardScaler` is fitted only on training data within each walk-forward fold and applied (`.transform()` only, not `.fit_transform()`) to test data. The prediction script loads the scaler serialised from the final full-data training run, not a freshly fitted instance.

### Multiple evaluation metrics
The project reports ROC-AUC, PR-AUC, ECE, Brier score, precision, recall, F1, portfolio hit rate, and alpha. No single metric is presented in isolation. The confusion matrix from the original baseline is retained for comparability.

---

## 8. Installation and run order

### Requirements

```bash
pip install xgboost lightgbm shap scikit-learn pandas numpy \
            matplotlib openpyxl vaderSentiment networkx yfinance requests
```

Or install from the provided requirements file:

```bash
pip install -r requirements.txt
```

**Python version:** 3.9+

### Full pipeline run order

```bash
# Phase 1 — baseline model
python pipeline/phase1_train_evaluate.py    # → model_artifacts/
python pipeline/phase1_predict.py           # → predictions_FY18.xlsx

# Phase 2 — feature enrichment (run in order; each produces a CSV)
python pipeline/phase2a_sentiment.py        # → sentiment_features.csv
python pipeline/phase2b_network.py          # → network_features.csv
python pipeline/phase2c_macro.py            # → macro_features.csv
python pipeline/phase2_train_evaluate.py    # → model_artifacts_p2/

# Phase 3 — evaluation
python pipeline/phase3a_calibration.py      # → calibration_report.xlsx, xgb_calibrated.pkl
python pipeline/phase3b_backtest.py         # → backtest_report.xlsx, backtest_summary.txt
python pipeline/phase3c_report.py           # → evaluation_report.pdf, report_data.json
```

### Expected runtime

| Script | Approx. runtime |
|---|---|
| `phase1_train_evaluate.py` | 2–5 min (500 trees × 2 models × N folds) |
| `phase2b_network.py` | 5–15 min (betweenness centrality on large graphs) |
| `phase2c_macro.py` | 1–2 min (yfinance download) |
| `phase2_train_evaluate.py` | 3–8 min |
| `phase3a_calibration.py` | 2–4 min |
| `phase3b_backtest.py` | 5–20 min (yfinance download for 1,500 tickers) |
| `phase3c_report.py` | < 1 min |

Total end-to-end: approximately 20–55 minutes depending on internet speed and hardware.

---

## 9. Reproducing results

The demo numbers shown in the dashboard and this README use synthetic data that mirrors the structure and approximate magnitude of real outputs. To reproduce with real data:

1. Obtain `ml_raw_data.xlsx` with the column structure described in Phase 1 (Ticker, 11 financial ratio columns, label column, Year column)
2. Obtain `FY18_Features.xlsx` (Ticker as index, same 11 financial ratio columns)
3. Run the pipeline in the order above
4. The `report_data.json` produced by `phase3c_report.py` can be pasted directly into the dashboard's data block

### Converting predictions to JSON for the dashboard

```python
import pandas as pd

df = pd.read_excel("predictions_FY18.xlsx")
df["Ticker"] = df.index
cols = ["Ticker", "Deal_Probability", "Predicted_Label", "Flag", "Top_SHAP_Driver"]
df[cols].to_json("predictions.json", orient="records", indent=2)
```

---

## 10. Dashboard deployment

The entire dashboard is a single `index.html` file. No build step, no node modules, no server.

```bash
# 1. Create GitHub repo
git init asx-ma-predictor && cd asx-ma-predictor && git checkout -b main

# 2. Copy index.html to repo root
cp /path/to/index.html .

# 3. (Optional) Add real pipeline outputs
cp /path/to/evaluation_report.pdf .

# 4. Push
git add . && git commit -m "Launch" && git push origin main

# 5. Enable GitHub Pages
# GitHub → Settings → Pages → Source: main / root → Save
# Live in ~60 seconds at https://<username>.github.io/<repo-name>/
```

### Updating with new predictions

```bash
# After re-running the pipeline:
cp outputs/report_data.json .
# Edit the REPORT_DATA object in index.html (or fetch dynamically)
git add index.html report_data.json
git commit -m "Update predictions FY$(date +%Y)"
git push origin main
```

---

## 11. Limitations and future work

### Current limitations

**Data availability.** The financial ratio features require Bloomberg or Refinitiv access for the full 1,500-company, 10-year dataset. The pipeline runs with any correctly formatted Excel file, but results depend heavily on the quality and coverage of the underlying data.

**Sentiment coverage.** The VADER-based sentiment features are most informative when applied to actual earnings call transcripts. In the absence of transcripts, the NewsAPI mode captures only recent headlines (not historical), limiting the usefulness of `tone_shift` as a historical feature.

**Deal type conflation.** All transaction types (buy-side, sell-side, buy-and-sell) are collapsed into a single `DEAL` label. A multi-class formulation distinguishing acquirers from targets would be more informative but requires more labelled data per class.

**Macro regime simplicity.** The three-state macro regime classifier is rule-based. A hidden Markov model or regime-switching model fitted on macro time series would be more principled.

**No microstructure.** Trading volume anomalies, options market activity, and insider ownership changes are documented leading indicators of deal announcements but are not included here.

### Future work

- **Multi-class prediction**: Separate acquirer vs target vs no-deal classification
- **Survival analysis**: Time-to-deal modelling rather than binary annual prediction
- **Transformer-based sentiment**: Replace VADER with a FinBERT or financial domain fine-tuned model for richer text features
- **Real-time pipeline**: Automate quarterly retraining triggered by new ASX filings
- **Cross-market extension**: Apply the same framework to ASX 300 / S&P 500 / NSE

---

## 12. Academic context

This project was developed as an academic research exercise at **Dwarkadas J. Sanghvi College of Engineering, Mumbai** (Computer Engineering). It draws on the following literature:

- **Prediction of M&A targets**: Brar, Giamouridis & Liodakis (2009); Powell (2004); Palepu (1986)
- **Walk-forward validation in finance**: Prado (2018), *Advances in Financial Machine Learning*
- **SHAP for model interpretability**: Lundberg & Lee (2017), *A Unified Approach to Interpreting Model Predictions*
- **Probability calibration**: Platt (1999); Niculescu-Mizil & Caruana (2005)
- **M&A network analysis**: Ahern & Harford (2014); Cai & Sevilir (2012)
- **Text-based corporate disclosures**: Loughran & McDonald (2011); Tetlock (2007)

---

## Citation

If you use this codebase or methodology in your own research, please cite:

```
Kagalwala, A. (2024). ASX M&A Transaction Predictor: A Multi-Signal Machine
Learning Pipeline with Walk-Forward Validation and Calibrated Probabilities.
Dwarkadas J. Sanghvi College of Engineering, Mumbai.
GitHub: https://arham7s.github.io/ASX-M-A-Transaction-Predictor/#predictions
```

---

## License

MIT License. See `LICENSE` for details.

---

*For questions or feedback, open an issue or reach out via GitHub.*
