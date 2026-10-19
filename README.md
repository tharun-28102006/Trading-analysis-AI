# Nivesh - Local Financial Intelligence Research

Nivesh is a local FastAPI, SQLite, pandas, NumPy, and browser application. It has no brokerage connection, paid provider requirement, or hosted AI inference. No orders are routed.

## Automatic free EOD data

**Repository data:** CSV and Parquet files are stored using Git LFS. Install Git LFS before cloning, then run `git lfs pull` to download the full datasets.

Nivesh bootstraps from the existing **local Tej Bazaar NSE and BSE daily Parquet files**, defaulting to `D:\Trading AI\data\out`. Set `NIVESH_TEJ_DATA_DIR` to point at another installed Tej Bazaar output directory. Daily files are read under `nse/year=YYYY/month=MM/date=YYYY-MM-DD.parquet` and the equivalent `bse/` tree. On refresh Nivesh checks the local file manifest, hashes new or changed files, validates them, and stages them inside one SQLite transaction. A fully validated import creates an immutable release; failed parsing, validation or publication preserves the active release. Synthetic demo rows remain marked `DEMO_SYNTHETIC` and are excluded from the genuine data fingerprint and model inputs. The project uses one canonical generated Parquet root (`data/out`); the older duplicate `tej-bazaar/data/out` export was removed after its files matched the canonical copies. Original Tej Bazaar raw inputs and package code remain under `tej-bazaar/`.

For incremental sessions and recent corrections, the adapter calls TejHQ's documented keyless `/v1/status` and `/v1/snapshot/{nse|bse}?date=...` API. It does not run the Tej Bazaar web scraper or collect directly from NSE/BSE sites. TejHQ publishes this dataset with an MIT license on its [Hugging Face page](https://huggingface.co/datasets/tejhq/indian-markets) and describes the API as open/free in its [documentation](https://tejhq.dev/docs); that is the publisher's licensing representation, not independent legal confirmation of exchange rights. This integration is for local research and does not redistribute market data. API status, exchange coverage and any provider-supplied publication timestamp are displayed separately from local refresh completion time.

Both exchanges are preserved independently. NSE records enter the existing `prices` research table; BSE records are stored in `exchange_market_prices` and its immutable release snapshot so overlapping symbols do not collide with NSE prices. Prices are raw and unadjusted. The source does not supply NIFTY 50 index levels, historical constituent membership or verified corporate actions through this workflow. It cannot support benchmark-relative or survivorship-safe pooled training until those separate genuine inputs are available. No automatic training occurs on refresh.

The actual local coverage, import counts, latest observation dates, last successful refresh, last attempt/errors and source attribution are available at `/api/data/status` and in Data & model health. Local import success and live snapshot fetch status are reported separately. A local bootstrap may publish validated historical files while showing that remote incremental fetching failed; that is not a claim that a live fetch succeeded.

## Start and refresh

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --reload
```

Open <http://127.0.0.1:8000> and press **Refresh data** on the overview or **Refresh data now** on Data & model health. The dashboard polls server-side progress, then reports final status, source, acquired rows, latest observed date, warnings, and errors. The server prevents overlapping jobs. The first request is a historical bootstrap; later clicks are incremental. â€œReconcile full historyâ€ is available on the health screen.

Daily refresh is enabled in `.env.example` by default. While Nivesh is running, the scheduler checks every 15 minutes after 21:30 Asia/Kolkata, skips weekends and exchange holidays using Tej Bazaar's session calendar, and retries failed attempts no more than every 30 minutes. Set `NIVESH_DAILY_REFRESH_ENABLED=0` to turn it off; restart the app after changing environment variables. No credentials or paid subscription are needed. Configure `NIVESH_DAILY_REFRESH_HOUR`, `NIVESH_DAILY_REFRESH_MINUTE`, `NIVESH_TEJ_DATA_DIR`, and `NIVESH_TEJ_OVERLAP_DAYS` as needed.

Run the offline suite with `python -m pytest -q`. `python -m app.maintenance refresh-data` starts a real refresh and waits for its recorded result. It does not use manual CSV input and does not fall back to demo data.

The local bootstrap published 8,573,249 genuine provider-attributed observations from 4,704 validated NSE/BSE Parquet files (NSE: 7,260,145 rows through 2026-10-16; BSE: 1,313,104 rows through 2026-10-19). This is local-file import evidence, not a successful live TejHQ API fetch. The live incremental API fetch is attempted on later refreshes. Models are not ready for benchmark-relative or survivorship-safe market-data training: aligned NIFTY 50 history, historical constituent membership and other point-in-time prerequisites are still absent.

## Additional local research tools

The remaining README sections document local imports, immutable release history, feature building, research releases, and model gates. These capabilities do not make manually supplied or synthetic data independently verified.

## Maintenance commands

```powershell
python -m app.maintenance discover-sources
python -m app.maintenance refresh-data
python -m app.maintenance refresh-data --reconcile-history
python -m app.maintenance run-research
python -m app.maintenance run-all
python -m app.maintenance run-all --reconcile-history
```

`run-all` checks database integrity, source eligibility, refresh state, existing model evidence, and research gates; it writes JSON and text reports under `.nivesh-data/reports/`. Existing eligible models are not retrained unless `NIVESH_AUTO_RETRAIN=1`. Optional daily refresh is enabled by default in `.env.example`; while the app is running it uses the same TejHQ EOD refresh. A report marked `blocked` means prerequisites are absent; it is not a successful download.

The dashboard refresh button invokes the TejHQ workflow and presents the actual result. Status distinguishes the last refresh from the latest observation and source quality. Do not treat daily historical observations as live quotes.

## Run on Windows

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --reload
```

Copy `.env.example` for reference, then set supported variables in PowerShell (the application does not load `.env` files automatically). For example: `$env:NIVESH_DATA_DIR='D:\Trading AI\.nivesh-data'` and `$env:NIVESH_PORT='8000'`. `NIVESH_API_TOKEN` is optional and should be set before any deliberate non-local exposure. The optional MoSPI CPI adapter requires no account credentials.

Open <http://127.0.0.1:8000>. Set `NIVESH_DATA_DIR` to move the SQLite DB, retained original imports, and local model artifacts. `NIVESH_HOST` and `NIVESH_PORT` control the local server. No secrets belong in the app configuration.

## Local data and provenance

Use the Health / Data screen or the documented `/api/data/import-*` endpoints. The app retains source files under `.nivesh-data/imports/`, hashes each file, and records run state, source name/description, license/terms reference, user-use confirmation, accepted/rejected counts, dates, basis, warnings and errors. `USER_IMPORTED_UNVERIFIED` is the default. A user confirmation plus a supplied terms reference changes quality to `USER_CONFIRMED_UNVERIFIED`; it is only an attestation, never independent license verification.

### Daily prices and benchmark CSV

Required headers:

```csv
symbol,exchange,date,open,high,low,close,volume,price_basis,source
```

Optional: `adjusted_close`, `dividends`, `stock_splits`. `exchange` is `NSE` or `NSE_INDEX`; benchmark symbol is `NIFTY50` on `NSE_INDEX`. Dates use ISO `YYYY-MM-DD`. `price_basis` must be `raw_unadjusted` or `adjusted_close`. Adjusted basis requires a positive adjusted-close value for each row. Raw close is never described as total return. Corporate actions are preserved when present; the importer warns about suspicious gaps and large unaccounted close moves instead of correcting them silently.

ZIP upload accepts CSV members (up to 25 MB per file and 100 MB expanded total), with a 25 MB compressed upload cap, a 1,000-entry/500-CSV limit, and an extreme-compression-ratio check; members are read in memory and paths are never extracted. Each member commits as its own import run. Malformed files do not roll back previously imported bundle members. Repeated SHA-256 imports are idempotent. Conflicting existing non-demo observations are retained and reported rather than overwritten.

Price imports are staged and validated before canonical writes. Exact file repeats are idempotent; different files with unchanged observations are recorded as metadata-only imports and do not start a rebuild. New sessions and backfills create a versioned refresh; corrections to existing observations are blocked until the explicit historical-correction checkbox is selected. Confirmed corrections preserve old/new values in the revision ledger. Refresh runs record affected symbols, change counts, dependency statuses, timings, and errors. Eligible existing local models may produce staged forecasts; imports never start training. A research release and its dependency state become active together after publication succeeds. Failed publication preserves the prior active release and its dependency statuses.

The health API/UI distinguishes the last refresh attempt from the latest imported market observation. `GET /api/data/refreshes` and `GET /api/data/releases` expose run and release history. `GET /api/data/status` includes the active release, latest price version, and component dependency states. Feature rows are calculated from canonical data on demand, so affected cached features/targets and historical evaluation artifacts are marked stale for explicit rebuild or evaluation rather than rewritten silently.

### Current constituents and membership history

Current constituents require an explicit `as_of_date`, `symbol`, and `name` (optional `sector`) and exactly 50 unique members. They create a dated snapshot only; they are not projected backward. The imported snapshot, not the old ten-company prototype flags, defines the NIFTY 50 screener. Historical membership uses `symbol,valid_from,valid_to` with an empty end for an open interval. Overlaps with the file or existing intervals are rejected. Pooled training requires user-confirmed effective membership intervals and does not infer historic membership from today's index.

Broader NSE reference lists use `symbol,name,exchange` (optional `sector`) and an explicit as-of date. They extend the local instrument inventory but do not imply index membership. An optional exchange session file uses one row per open `exchange,date` (`NSE` or `NSE_INDEX`). Imported session dates are stored with provenance and used only to diagnose missing or unexpected observations over the covered range; Nivesh does not generate a holiday calendar or certify the uploaded dates.

Nivesh does not bundle a current official 50-member list. Import an authoritative file and record its source, terms, and as-of date. The existing instrument file is a prototype, not a complete index universe.

### Fundamentals and events

Fundamentals require `symbol,period` and supported numeric fields `revenue_growth,profit_growth,debt_equity,roe`; `published_at` is required for point-in-time feature use. Events require `symbol,date,kind,title`; `published_at` is required for event features. Only user-confirmed rows with both event and publication dates available by the prediction close contribute to event counts and days-since-event features. Date-only publication entries are conservatively treated as available after that UTC day. Macroeconomic/geopolitical events can be represented with a suitable symbol such as `NIFTY50`, but no source or event is assumed. Feature coverage is reported per dataset.

All CSV endpoints preserve original files and source metadata. Do not upload credentials. Illustrative templates are downloadable at `/api/data/templates/{prices,constituents,membership,instruments,calendar}` and stored under `data_templates/`. Their filenames and marker rows are rejected by the corresponding importers.

## Dataset, models, and evaluation

`app/features.py` is the shared dataset builder. The imported NIFTY 50 dates define the session grid: the target uses stock and benchmark prices exactly five subsequent benchmark sessions after the origin. Missing stock sessions stay missing rather than shifting the label to a later common date. The feature set includes returns, relative returns, volatility, moving-average distances, drawdown, benchmark features, and volume/liquidity when available, plus point-in-time fundamentals and dated event features when imported. Mixed raw/adjusted bases are refused. Missing stock dates and excluded feature/target rows are reported in dataset metadata.

Training currently requires at least 750 complete aligned rows per company, user-confirmed provenance, benchmark coverage, and a consistent basis. Eligibility is not evidence that a source's terms have been checked. Walk-forward evaluation is chronological, purges five sessions, uses an initial selection fold and later evaluation folds, and reports zero and historical-mean baselines alongside Ridge and a local nonlinear RBF Ridge model. The selected model is saved as a checksummed JSON artifact (no pickle) under `.nivesh-data/models/`, with registry metadata, feature schema, training period, metrics, and data version. Current forecasts are stored separately from OOS predictions.

The explicit operations are `POST /api/models/train-company/{symbol}`, `POST /api/models/train-all-companies`, `POST /api/models/train-universe` (pooled), and `POST /api/models/train-aggregator`. Batch training discovers companies only from the latest imported NSE instrument inventory, then evaluates each independently; legacy prototype symbols are not batch candidates. Missing, unconfirmed, synthetic-only, or insufficient history is recorded as skipped. It records succeeded/skipped/failed outcomes, continues after a company failure, and persists progress and per-company results in the batch run. `GET /api/models/training/{run_id}` retrieves the persisted record. Pooled modeling requires historical membership. The aggregator consumes only stored walk-forward base-model predictions and requires aligned streams; it is kept as eligible only if it beats the zero baseline on a purged chronological holdout. `GET /api/predictions/latest` returns the latest forecast per symbol with its model/data versions and stale-origin warning.

Calibration is available through `GET /api/models/calibration?model_key=...`. It estimates the distinct event probability `P(five-session excess return > 0)` with Platt logistic calibration. The earliest dated portion is used to fit, five sessions are purged, and the latest dates are held out for reliability metrics. At least 200 fit rows, 100 evaluation rows, and 20 positive and negative outcomes in each period are required. Otherwise the result is `not_calibrated`; no confidence percentage is shown. Expected excess return remains a separate forecast field. Directional accuracy is not a probability of success, and low error is not evidence of profitability.

## Backtesting, risk, and portfolio

Backtests consume historical OOS predictions only. The present execution rule uses a signal at the origin close, enters next session open, exits at the fifth subsequent session close, and requires raw OHLC. Non-overlapping five-session cohorts, position caps, transaction-cost and slippage assumptions, NIFTY comparison, annualized and cumulative returns, drawdown, volatility, risk-adjusted ratio, trade count, exposure, and a combined-cost sensitivity table are recorded. Adjusted-close-only inputs return unavailable because they do not provide a defensible adjusted open. Results are historical simulations, not live performance.

Portfolio holdings and paper trades are local manual records. Valuations use the latest imported non-synthetic observation and can be stale or unverified. No brokerage account is connected and no real order is placed. Risk/portfolio outputs are limited by imported coverage.

## Useful API routes

- `GET /api/status`, `/api/data/status`, `/api/data/imports`: app/data health and import history.
- `GET /api/data/sources`, `/api/data/acquisition/runs`, `POST /api/data/acquisition/refresh`: source permission registry and safe acquisition attempts.
- `POST /api/data/import-csv`, `/api/data/import-zip`, `/api/data/import-fundamentals-csv`, `/api/data/import-events-csv`, `/api/data/import-constituents`, `/api/data/import-membership-history`, `/api/data/import-instruments`, `/api/data/import-exchange-calendar`: local file imports.
- `GET /api/universe/instruments`, `/api/universe/constituents`, `/api/data/templates/{template}`: dated universe inventory and templates.
- `GET /api/models/training`, `GET /api/models/compare`, `POST /api/models/train-company/{symbol}`, `/api/models/train-all-companies`, `/api/models/train-universe`, `/api/models/train-aggregator`: distinct training and registry operations.
- `GET /api/features/{symbol}`, `/api/predictions/oos`, `/api/predictions/latest`, `GET /api/models/calibration`: dataset preview, historical predictions, forecasts, and event calibration.
- `POST /api/backtests`, `GET /api/backtests`: simulation and history.
- `GET/PUT/DELETE /api/portfolio/holdings`, `POST/GET /api/portfolio/paper-trades`: local tracking only.

The compatibility refresh endpoint is local-only and does not contact any provider. Yahoo Finance and Kite collection code has been disabled; credentials or environment flags cannot re-enable it.

## Database and migration

SQLite schema changes are additive and versioned in `schema_migrations`. Before a needed migration, `init_db()` uses SQLite's online backup API. Synthetic records are preserved in place and archived. Watchlists, alerts, imports, model files, and legacy tables remain. Schema v6 adds dataset versions, correction revisions, dependency states, staged forecasts, and research release lineage. Schema v7 adds the source registry and durable acquisition-attempt history. Schema v9 stores immutable release snapshots for research reads; APIs, screeners, feature construction, and backtests read the active snapshot while refresh inference reads candidate rows. Schema v8 distinguishes documentation review, access permission, adapter, connectivity, cost, and last-attempt state. If a price publication fails, the importer restores prior canonical price/action rows and retains candidate raw payloads and revisions with failed-publication status. Schema v10 adds an isolated revision-preserving macro observation store; it is not yet a model feature. The project database currently contains 11,550 preserved synthetic legacy rows and no genuine equity observations or genuine-data model training.

## Tests and troubleshooting

```powershell
python -m compileall -q app tests
python -m pytest -q
```

Tests run offline with temporary databases and mocked/isolated data. If the app reports no eligible data, acquire a dataset under terms that permit the intended use, import stock and NIFTY50 histories with compatible basis, and record its terms reference. Company forecasts require sufficient aligned history; pooled training additionally requires historical constituent intervals. Nivesh does not download or fabricate unavailable data.

To run the app from the project directory in PowerShell:

```powershell
Set-Location 'D:\Trading AI'
.\.venv\Scripts\Activate.ps1
python -m uvicorn app.main:app --reload
```

## Current limits

- No genuine dataset was downloaded or imported; single-company, batch, pooled and aggregator training are currently blocked by data eligibility.
- Source provenance and use permission are user-declared, not independently verified.
- No official session calendar is bundled, so exchange holidays and missing-session counts cannot be independently certified. A user may import a calendar reference for diagnostics.
- Calibration has not run on genuine predictions; it remains unavailable until sample and outcome-diversity thresholds are met. The local nonlinear candidate is the NumPy RBF Ridge model; native HistGradientBoosting/LightGBM and a fee-aware editable paper-trade ledger remain future work.
- Current dashboard universe membership is not a substitute for importing the authoritative as-of constituent list.

Nivesh is a research tool, not investment advice or an execution system.
