# Stock Market Regime Detection Using Unsupervised Machine Learning

Identifying recurring market "regimes" (calm uptrends, volatile sell-offs, choppy
transitions) in the S&P 500 using K-Means clustering on rolling risk and trend
features, with an interactive Dash dashboard.

> **What this project is not:** it does not predict prices, and it does not claim
> a trading edge. It groups historical days by how the market was *behaving*, then
> checks whether those groups make economic sense.

---

## 1. Problem

Markets do not behave the same way all the time. Volatility clusters, drawdowns
come in episodes, and trends persist for weeks. Risk teams and portfolio managers
often talk about "regimes". This project asks:

*Can an unsupervised model, given only trailing price information, separate
S&P 500 history into a small number of interpretable, recurring regimes?*

## 2. Data

* **Source:** Yahoo Finance via the free `yfinance` library (no API key needed).
* **Ticker:** `^GSPC` (S&P 500 index), daily, about 10 years.
* **Cleaning:** handles yfinance's MultiIndex columns, timezone-aware dates,
  duplicate rows, and non-numeric, infinite or non-positive prices. Exchange holidays
  are **not** filled in, because filling them would create fake zero-return days.
* **Reliability:** downloads are retried with increasing pauses. Raw prices are cached
  to `data/sp500_raw.csv`. If a later download fails, the cached copy is used.

## 3. Features

All features at date *t* use only data up to *t* (trailing windows), so no future
information leaks into them.

| Feature | Formula | Used for |
|---|---|---|
| `ret_1d` | P<sub>t</sub> / P<sub>t-1</sub> − 1 | describing regimes |
| `log_ret` | ln(P<sub>t</sub> / P<sub>t-1</sub>) | input to volatility |
| `vol_20` | 20-day std of log returns × √252 | **clustering** |
| `vol_60` | 60-day std of log returns × √252 | **clustering** |
| `mom_20` | P<sub>t</sub> / P<sub>t-20</sub> − 1 | **clustering** |
| `mdd_60` | worst peak-to-trough fall inside the trailing 60 days | **clustering** |
| `drawdown` | P<sub>t</sub> / running max − 1 | dashboard only |

The single-day return is deliberately **excluded** from the clustering inputs. It is
mostly noise and would make regime labels flip from day to day. It is still used to
describe each regime afterwards.

## 4. Methodology

1. **Chronological split.** The first 80% of dates form the *training period*; the
   last 20% are *out-of-sample*.
2. **Standardise** the four features with `StandardScaler`, **fitted on the training
   period only**. This prevents future volatility levels from influencing how earlier
   days are scaled.
3. **K-Means** with `k = 3`, `n_init = 20`, `random_state = 42`, fitted on the
   training period. Later days are assigned to the nearest learned centroid, the same
   way a new day would be classified in practice.
4. **Interpret, then label.** Cluster numbers have no meaning. Each cluster's
   training-period averages are computed first, and then labels follow fixed rules:
   * volatility level = rank of average `vol_20` (Low / Moderate / High)
   * direction = *positive* if average daily return **and** momentum > 0,
     *negative* if both < 0, otherwise *mixed / transitional*
   * *High-volatility stressed* only if the highest-volatility cluster is negative
     **and** has the deepest average drawdown

   If the data doesn't support a clean label, the label says so.
5. **Evaluate.** Report silhouette score (training and out-of-sample), cluster sizes,
   a k = 2–6 silhouette/inertia sweep, the number of regime switches, and the average
   length of each regime episode.

## 5. Results

> Fill this section in from **your own run**. The numbers come from
> `outputs/metrics.json` and `outputs/regime_profile.csv`, and are also printed by
> `run_analysis.py`. Do not copy numbers from anywhere else.

Data period: `[START]` to `[END]` (`[N]` trading days classified)

| Regime | Days (share) | Avg daily return | Vol 20d | Momentum 20d | Max DD 60d | Avg episode length |
|---|---|---|---|---|---|---|
| `[label]` | `[ ]` | `[ ]` | `[ ]` | `[ ]` | `[ ]` | `[ ]` days |
| `[label]` | `[ ]` | `[ ]` | `[ ]` | `[ ]` | `[ ]` | `[ ]` days |
| `[label]` | `[ ]` | `[ ]` | `[ ]` | `[ ]` | `[ ]` | `[ ]` days |

* Silhouette score: training `[ ]`, out-of-sample `[ ]`
* Regime switches over the period: `[ ]`
* Latest detected regime (`[date]`): `[label]`

**Interpretation (write after looking at the table):** Do the high-volatility days
line up with known stress episodes, for example the 2020 COVID crash or the 2022
rate-hike sell-off? Are regimes persistent (episodes of weeks or months) or noisy?
Does the out-of-sample silhouette hold up against the training score?

Screenshot: `docs/dashboard.png` *(add after running the app)*

## 6. Limitations

* **Descriptive, not predictive.** A regime label summarises the last 20–60 days.
  Nothing here shows that regimes forecast future returns, and no backtest is included.
* **Lag.** Rolling windows react after the fact. A crash is usually classified as
  "high volatility" some days into it.
* **The high-volatility cluster may not be "stressed".** Volatility is also high in
  sharp *rebounds*. The labelling rule checks for this instead of assuming it.
* **A day's own values can differ from its regime's averages.** A day is assigned to
  the *nearest* centroid; the dashboard shows the day's actual values next to the label.
* **K-Means assumptions.** It favours roughly spherical, similar-sized clusters and
  has no notion of time order. Rare crisis regimes are small, so they can be absorbed
  into larger clusters.
* **Choice of k is a judgement call.** Silhouette helps but does not settle it; the
  k-sweep is shown for transparency.
* **One index, one data source.** Yahoo Finance data is unofficial and can contain
  occasional errors or gaps.
* **Static model.** The model is fitted once on the training period. Market structure
  can change after that.

Possible extensions (deliberately not in v1): Gaussian Mixture Models, Hidden Markov
Models, more assets (bonds, VIX), walk-forward refitting.

## 7. Setup (Windows)

Requires Python 3.10 or newer.

```bat
git clone https://github.com/<your-username>/sp500-regime-detection.git
cd sp500-regime-detection
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

*macOS/Linux:* use `python3 -m venv .venv` and `source .venv/bin/activate`.

## 8. Usage

```bat
:: 1) Run the analysis (downloads data, fits the model, writes outputs\)
python run_analysis.py

:: Optional: reuse the cached download, or try other settings
python run_analysis.py --offline
python run_analysis.py --k 4 --seed 7 --train-frac 0.75

:: 2) Start the dashboard, then open http://127.0.0.1:8050
python app.py
```

Exploratory and model charts are saved as interactive HTML files in
`outputs\figures\` (price, returns, return histogram, volatility, regime scatter,
k-sweep). Open them in any browser.

**Troubleshooting:** if the download fails (no internet, or a Yahoo rate limit),
wait a few minutes and retry, or upgrade yfinance: `pip install -U yfinance`. Once
one download has succeeded, `--offline` works without internet.

## 9. Project structure

```
sp500-regime-detection/
├── README.md
├── requirements.txt
├── .gitignore
├── run_analysis.py        # reproducible workflow: data -> features -> model -> outputs
├── app.py                 # Dash dashboard (reads outputs/)
├── src/
│   ├── __init__.py
│   ├── data_loader.py     # download, retries, cleaning, cache
│   ├── features.py        # trailing-window features
│   └── regimes.py         # scaling, K-Means, evaluation, labelling
├── data/                  # created at runtime (git-ignored)
└── outputs/               # created at runtime (git-ignored)
    ├── regimes.csv
    ├── regime_profile.csv
    ├── k_sweep.csv
    ├── metrics.json
    └── figures/*.html
```

## 10. Tech stack

Python · pandas · NumPy · yfinance · scikit-learn · Plotly · Dash

---

*Educational project. Not investment advice.*
