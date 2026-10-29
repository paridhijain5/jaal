# JAAL

JAAL is an offline-first Streamlit research prototype for surfacing unusual price/volume activity, disclosed entity overlap, and filing red flags in Indian small-cap stocks. Its score is an early-warning signal only; it is not evidence of manipulation, an accusation, a legal conclusion, or investment advice.

The checked-in demo uses deterministic synthetic records, including a few planted connected-entity and abnormal-trading patterns. No internet access is needed to run the app. The synthetic company names, orders, and disclosures are fictional and must not be attributed to real issuers or people.

## Setup

Python 3.10+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

Streamlit's sidebar links navigate among Watchlist, Stock Detail, Ring Radar, Time Machine, and Backtest. The sample dataset can be regenerated with `python -m jaal.sample_data`. Run checks with `python -m pytest -q`.

## Architecture

- `app.py` is the ranked Watchlist entry point; `pages/` holds the other Streamlit views.
- `jaal/ingest.py` loads the offline CSVs and provides opt-in raw CSV/JSON fetchers for public exchange and regulator exports.
- `jaal/engines/anomaly.py` computes trailing price, traded-volume, delivery, and range signals.
- `jaal/engines/graph.py` builds date-bounded stock/entity networks, repeat-entity features, and communities.
- `jaal/engines/filings.py` weights a small, explicit set of disclosure flags.
- `jaal/engines/case_twin.py` compares a current feature vector with prior synthetic penalised cases using cosine similarity.
- `jaal/features.py` assembles daily features and forward SEBI-order labels. Outcomes are evaluation-only, not model inputs.
- `jaal/model.py` trains LightGBM on a chronological training period, keeps a 20-calendar-day label embargo before the holdout, generates the 0-100 blended Jaal Score, and uses SHAP for local reasons where supported.
- `jaal/backtest.py` reports holdout precision@k, recall@k, false-positive rate, and lead time. The Backtest page compares the model against a price-only anomaly ranking.
- `data/sample/` contains the self-contained fixture tables; `tests/` covers feature behavior and the time split.

The app combines a transparent anomaly score, entity and filing signals, and a LightGBM probability. The resulting score is a prioritization aid, not a calibrated probability of misconduct. Per-row reasons use SHAP when compatible with the installed tree model and a feature-importance fallback otherwise.

## Data Sources

The fetchers accept operator-supplied export URLs because exchange archive paths, schemas, throttling, and access controls can change. They retrieve raw data; they do not automatically normalize source-specific columns into the sample schema.

- NSE historical reports and bhavcopy archives: <https://www.nseindia.com/market-data/archives>
- NSE bulk and block deals: <https://www.nseindia.com/market-data/bulk-deal> and <https://www.nseindia.com/market-data/block-deal>
- NSE shareholding patterns: <https://www.nseindia.com/companies-listing/corporate-filings-shareholding-pattern>
- NSE corporate announcements and insider disclosures: <https://www.nseindia.com/companies-listing/corporate-filings-announcements>
- BSE corporate announcements: <https://www.bseindia.com/corporates/ann.html>
- SEBI orders and proceedings (potential historical labels after manual review): <https://www.sebi.gov.in/sebiweb/home/HomeAction.do?doListing=yes&sid=1&smid=0&ssid=7>

`fetch_nse_bhavcopy`, `fetch_nse_bulk_block_deals`, `fetch_nse_shareholding`, `fetch_nse_insider_disclosures`, `fetch_nse_announcements`, `fetch_bse_csv`, and `fetch_sebi_orders` are the fetch entry points. Several endpoints require a browser session or expose downloads rather than stable machine-readable APIs; check provider terms and normalize/validate any downloaded data before use.

## Limitations

- The demo is synthetic and deliberately plants a handful of patterns; displayed backtest values do not establish real-world predictive power.
- Real SEBI orders are sparse, delayed, selected outcomes and do not provide complete ground truth. Treat labels as noisy and potentially biased.
- Exchange data may be revised, incomplete, rate-limited, or subject to access and redistribution restrictions. Verify provenance and corporate-action adjustments.
- Entity matching is difficult: name collisions, transliteration, corporate structures, and disclosure gaps can create false links or miss real ones.
- Rules and model weights are illustrative, not calibrated for decisions. The score is not a probability, trading recommendation, finding of intent, or substitute for analyst, legal, or regulatory review.
- Use for research only. Never publish or act on a risk signal as an allegation without independent verification and appropriate safeguards.
