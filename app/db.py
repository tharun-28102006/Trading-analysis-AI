import json
import sqlite3
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone

from .config import DATA_DIR, DB_PATH

_UNPINNED_RELEASE = object()
_PINNED_RELEASE_ID = ContextVar("nivesh_pinned_release_id", default=_UNPINNED_RELEASE)

SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS securities(symbol TEXT PRIMARY KEY, name TEXT NOT NULL, sector TEXT, exchange TEXT DEFAULT 'NSE', index_member INTEGER DEFAULT 1, is_demo INTEGER DEFAULT 0);
CREATE TABLE IF NOT EXISTS prices(symbol TEXT NOT NULL, date TEXT NOT NULL, open REAL, high REAL, low REAL, close REAL NOT NULL, adjusted_close REAL NOT NULL, volume INTEGER, source TEXT NOT NULL, retrieved_at TEXT NOT NULL, quality TEXT NOT NULL DEFAULT 'unverified', PRIMARY KEY(symbol,date), FOREIGN KEY(symbol) REFERENCES securities(symbol));
CREATE TABLE IF NOT EXISTS exchange_market_prices(exchange TEXT NOT NULL,symbol TEXT NOT NULL,date TEXT NOT NULL,series TEXT,isin TEXT,name TEXT,open REAL NOT NULL,high REAL NOT NULL,low REAL NOT NULL,close REAL NOT NULL,volume INTEGER NOT NULL,turnover REAL,trades INTEGER,source TEXT NOT NULL,retrieved_at TEXT NOT NULL,quality TEXT NOT NULL,price_basis TEXT NOT NULL DEFAULT 'raw_unadjusted',PRIMARY KEY(exchange,symbol,date));
CREATE INDEX IF NOT EXISTS price_date_idx ON prices(date);
CREATE TABLE IF NOT EXISTS fundamentals(symbol TEXT NOT NULL, period TEXT NOT NULL, revenue_growth REAL, profit_growth REAL, debt_equity REAL, roe REAL, source TEXT NOT NULL, retrieved_at TEXT NOT NULL, quality TEXT NOT NULL DEFAULT 'unverified', published_at TEXT, PRIMARY KEY(symbol,period));
CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY AUTOINCREMENT, symbol TEXT NOT NULL, date TEXT NOT NULL, kind TEXT NOT NULL, title TEXT NOT NULL, source TEXT NOT NULL, quality TEXT NOT NULL, published_at TEXT);
CREATE TABLE IF NOT EXISTS fundamental_observations(id INTEGER PRIMARY KEY AUTOINCREMENT,symbol TEXT NOT NULL,period TEXT NOT NULL,published_at TEXT NOT NULL,revenue_growth REAL,profit_growth REAL,debt_equity REAL,roe REAL,source TEXT NOT NULL,quality TEXT NOT NULL,import_run_id INTEGER,UNIQUE(symbol,period,published_at,source));
CREATE INDEX IF NOT EXISTS fundamental_asof_idx ON fundamental_observations(symbol,published_at);
CREATE TABLE IF NOT EXISTS models(id INTEGER PRIMARY KEY AUTOINCREMENT, symbol TEXT NOT NULL, trained_at TEXT NOT NULL, model_type TEXT NOT NULL, metrics TEXT NOT NULL, latest_prediction REAL, prediction_date TEXT, provenance TEXT NOT NULL, feature_names TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS watchlist(symbol TEXT PRIMARY KEY, added_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS alerts(id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, symbol TEXT NOT NULL, condition TEXT NOT NULL, threshold REAL NOT NULL, period TEXT NOT NULL, enabled INTEGER NOT NULL DEFAULT 1, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS alert_history(id INTEGER PRIMARY KEY AUTOINCREMENT, alert_id INTEGER NOT NULL, triggered_at TEXT NOT NULL, message TEXT NOT NULL, value REAL NOT NULL, UNIQUE(alert_id,triggered_at));
CREATE TABLE IF NOT EXISTS audit(id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT NOT NULL, action TEXT NOT NULL, details TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS raw_price_observations(id INTEGER PRIMARY KEY AUTOINCREMENT, instrument_id TEXT NOT NULL, symbol TEXT NOT NULL, provider_symbol TEXT NOT NULL, date TEXT NOT NULL, source TEXT NOT NULL, retrieved_at TEXT NOT NULL, payload_hash TEXT NOT NULL, payload_json TEXT NOT NULL, validation_status TEXT NOT NULL, UNIQUE(instrument_id,date,source,payload_hash));
CREATE INDEX IF NOT EXISTS raw_price_lookup ON raw_price_observations(symbol,date,retrieved_at);
CREATE TABLE IF NOT EXISTS corporate_actions(id INTEGER PRIMARY KEY AUTOINCREMENT, instrument_id TEXT NOT NULL, symbol TEXT NOT NULL, date TEXT NOT NULL, dividends REAL, stock_splits REAL, source TEXT NOT NULL, retrieved_at TEXT NOT NULL, quality TEXT NOT NULL);
CREATE UNIQUE INDEX IF NOT EXISTS corporate_action_unique ON corporate_actions(symbol,date,source);
CREATE TABLE IF NOT EXISTS refresh_runs(id INTEGER PRIMARY KEY AUTOINCREMENT, started_at TEXT NOT NULL, finished_at TEXT, status TEXT NOT NULL, triggered_by TEXT NOT NULL, source TEXT NOT NULL, instruments_requested INTEGER NOT NULL DEFAULT 0, instruments_updated INTEGER NOT NULL DEFAULT 0, rows_written INTEGER NOT NULL DEFAULT 0, latest_observation_date TEXT, errors_json TEXT NOT NULL DEFAULT '[]', warnings_json TEXT NOT NULL DEFAULT '[]');
CREATE TABLE IF NOT EXISTS data_import_runs(id INTEGER PRIMARY KEY AUTOINCREMENT, started_at TEXT NOT NULL, finished_at TEXT, filename TEXT NOT NULL, sha256 TEXT NOT NULL, stored_path TEXT, status TEXT NOT NULL, source TEXT, rows_read INTEGER NOT NULL DEFAULT 0, rows_imported INTEGER NOT NULL DEFAULT 0, duplicates INTEGER NOT NULL DEFAULT 0, symbols_json TEXT NOT NULL DEFAULT '[]', warnings_json TEXT NOT NULL DEFAULT '[]', errors_json TEXT NOT NULL DEFAULT '[]', earliest_date TEXT, latest_date TEXT, UNIQUE(sha256));
CREATE TABLE IF NOT EXISTS archived_demo_prices(symbol TEXT NOT NULL,date TEXT NOT NULL,open REAL,high REAL,low REAL,close REAL NOT NULL,adjusted_close REAL NOT NULL,volume INTEGER,source TEXT NOT NULL,retrieved_at TEXT NOT NULL,quality TEXT NOT NULL,price_basis TEXT NOT NULL DEFAULT 'synthetic',PRIMARY KEY(symbol,date));
CREATE TABLE IF NOT EXISTS data_sources(id INTEGER PRIMARY KEY AUTOINCREMENT,source_name TEXT NOT NULL,dataset_description TEXT NOT NULL DEFAULT '',license_reference TEXT NOT NULL DEFAULT '',user_confirmed_use INTEGER NOT NULL DEFAULT 0,created_at TEXT NOT NULL,UNIQUE(source_name,dataset_description,license_reference));
CREATE TABLE IF NOT EXISTS constituent_membership(symbol TEXT NOT NULL,valid_from TEXT NOT NULL,valid_to TEXT,source_id INTEGER,import_run_id INTEGER,quality TEXT NOT NULL,PRIMARY KEY(symbol,valid_from));
CREATE INDEX IF NOT EXISTS membership_date_idx ON constituent_membership(valid_from,valid_to,symbol);
CREATE TABLE IF NOT EXISTS constituent_snapshots(as_of_date TEXT NOT NULL,symbol TEXT NOT NULL,company_name TEXT NOT NULL,sector TEXT,source_id INTEGER,import_run_id INTEGER,quality TEXT NOT NULL,PRIMARY KEY(as_of_date,symbol));
CREATE TABLE IF NOT EXISTS training_runs(id INTEGER PRIMARY KEY AUTOINCREMENT,mode TEXT NOT NULL,status TEXT NOT NULL,started_at TEXT NOT NULL,finished_at TEXT,progress INTEGER NOT NULL DEFAULT 0,total INTEGER NOT NULL DEFAULT 0,errors_json TEXT NOT NULL DEFAULT '[]',summary_json TEXT NOT NULL DEFAULT '{}');
CREATE TABLE IF NOT EXISTS model_versions(id INTEGER PRIMARY KEY AUTOINCREMENT,training_run_id INTEGER NOT NULL,model_key TEXT NOT NULL,model_type TEXT NOT NULL,artifact_path TEXT NOT NULL,artifact_sha256 TEXT NOT NULL,data_version TEXT NOT NULL,feature_schema_json TEXT NOT NULL,config_json TEXT NOT NULL,metrics_json TEXT NOT NULL,training_period_json TEXT NOT NULL,status TEXT NOT NULL,created_at TEXT NOT NULL,UNIQUE(model_key,artifact_sha256));
CREATE TABLE IF NOT EXISTS oos_predictions(id INTEGER PRIMARY KEY AUTOINCREMENT,training_run_id INTEGER NOT NULL,model_key TEXT NOT NULL,symbol TEXT NOT NULL,origin_date TEXT NOT NULL,target_end_date TEXT NOT NULL,prediction REAL NOT NULL,actual_excess REAL NOT NULL,stock_return REAL NOT NULL,benchmark_return REAL NOT NULL,data_version TEXT NOT NULL,price_basis TEXT NOT NULL DEFAULT 'unknown',UNIQUE(training_run_id,model_key,symbol,origin_date));
CREATE INDEX IF NOT EXISTS oos_date_idx ON oos_predictions(origin_date,symbol,model_key);
CREATE TABLE IF NOT EXISTS current_forecasts(id INTEGER PRIMARY KEY AUTOINCREMENT,model_version_id INTEGER NOT NULL,symbol TEXT NOT NULL,origin_date TEXT NOT NULL,horizon_sessions INTEGER NOT NULL,prediction REAL NOT NULL,price_basis TEXT NOT NULL,created_at TEXT NOT NULL,UNIQUE(model_version_id,symbol,origin_date));
CREATE TABLE IF NOT EXISTS backtest_runs(id INTEGER PRIMARY KEY AUTOINCREMENT,created_at TEXT NOT NULL,status TEXT NOT NULL,config_json TEXT NOT NULL,results_json TEXT NOT NULL,errors_json TEXT NOT NULL DEFAULT '[]');
CREATE TABLE IF NOT EXISTS portfolio_positions(symbol TEXT PRIMARY KEY,quantity REAL NOT NULL,cost_basis REAL NOT NULL,updated_at TEXT NOT NULL,notes TEXT NOT NULL DEFAULT '');
CREATE TABLE IF NOT EXISTS paper_trades(id INTEGER PRIMARY KEY AUTOINCREMENT,symbol TEXT NOT NULL,side TEXT NOT NULL,quantity REAL NOT NULL,reference_price REAL NOT NULL,trade_date TEXT NOT NULL,created_at TEXT NOT NULL,notes TEXT NOT NULL DEFAULT '');
CREATE TABLE IF NOT EXISTS schema_migrations(version INTEGER PRIMARY KEY,applied_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS training_run_items(run_id INTEGER NOT NULL,symbol TEXT NOT NULL,status TEXT NOT NULL,reason TEXT,model_version_id INTEGER,metrics_json TEXT NOT NULL DEFAULT '{}',updated_at TEXT NOT NULL,PRIMARY KEY(run_id,symbol));
CREATE TABLE IF NOT EXISTS instrument_universe(symbol TEXT NOT NULL,as_of_date TEXT NOT NULL,name TEXT NOT NULL,sector TEXT,exchange TEXT NOT NULL,source_id INTEGER,import_run_id INTEGER,quality TEXT NOT NULL,PRIMARY KEY(symbol,as_of_date));
CREATE INDEX IF NOT EXISTS instrument_universe_date_idx ON instrument_universe(as_of_date,symbol);
CREATE TABLE IF NOT EXISTS exchange_sessions(exchange TEXT NOT NULL,date TEXT NOT NULL,source_id INTEGER,import_run_id INTEGER,quality TEXT NOT NULL,PRIMARY KEY(exchange,date));
CREATE TABLE IF NOT EXISTS macro_observations(id INTEGER PRIMARY KEY AUTOINCREMENT,source TEXT NOT NULL,dataset TEXT NOT NULL,year INTEGER NOT NULL,month INTEGER NOT NULL,geography TEXT NOT NULL,sector TEXT NOT NULL,group_name TEXT NOT NULL,subgroup TEXT NOT NULL,index_value REAL,inflation REAL,status TEXT,published_at TEXT,retrieved_at TEXT NOT NULL,payload_hash TEXT NOT NULL,record_hash TEXT NOT NULL UNIQUE,record_json TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS macro_period_idx ON macro_observations(dataset,year,month,geography,sector);
"""

@contextmanager
def connect():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    _install_pinned_views(conn)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


@contextmanager
def pin_published_release(release_id):
    """Pin published views to one immutable release across request connections."""
    token=_PINNED_RELEASE_ID.set(None if release_id is None else int(release_id))
    try: yield
    finally: _PINNED_RELEASE_ID.reset(token)

def active_release_id():
    with connect() as conn:
        row=conn.execute("SELECT value FROM app_state WHERE key='active_release_id'").fetchone()
        return int(row['value']) if row else None

def _install_pinned_views(conn):
    release_id=_PINNED_RELEASE_ID.get()
    if release_id is _UNPINNED_RELEASE: return
    for table in _SNAPSHOT_TABLES:
        view=f'published_{table}'
        conn.execute(f'DROP VIEW IF EXISTS temp.{view}')
        columns=[r['name'] for r in conn.execute(f'PRAGMA main.table_info({table})')]
        if not columns: continue
        if release_id is not None and conn.execute('SELECT 1 FROM main.research_snapshot_releases WHERE release_id=?',(release_id,)).fetchone():
            select=', '.join(f's."{col}"' for col in columns)
            conn.execute(f'CREATE TEMP VIEW {view} AS SELECT {select} FROM main.release_snapshot_{table} s WHERE s.release_id={release_id}')
        else:
            select=', '.join(f'c."{col}"' for col in columns)
            conn.execute(f'CREATE TEMP VIEW {view} AS SELECT {select} FROM main.{table} c')

def init_db():
    # Back up the user's existing database before applying any additive schema changes.
    database_existed=DB_PATH.exists()
    if database_existed:
        probe=sqlite3.connect(DB_PATH)
        try:
            tables={r[0] for r in probe.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            versioned='schema_migrations' in tables and probe.execute('SELECT COALESCE(MAX(version),0) FROM schema_migrations').fetchone()[0]>=11
            needs_migration=bool(tables) and (not versioned or 'model_versions' not in tables or ('prices' in tables and 'price_basis' not in {r[1] for r in probe.execute('PRAGMA table_info(prices)')}))
        finally: probe.close()
        if needs_migration: backup_database()
    with connect() as c:
        c.executescript(SCHEMA)
        price_cols={r['name'] for r in c.execute('PRAGMA table_info(prices)')}
        if 'price_basis' not in price_cols:
            c.execute("ALTER TABLE prices ADD COLUMN price_basis TEXT NOT NULL DEFAULT 'unknown'")
            c.execute("UPDATE prices SET price_basis=CASE WHEN quality='DEMO_SYNTHETIC' THEN 'synthetic' WHEN quality='KITE_CONNECT_RAW_UNADJUSTED' THEN 'raw_unadjusted' WHEN quality='YAHOO_FINANCE_UNVERIFIED' THEN 'adjusted_close' ELSE 'unknown' END")
        c.execute("INSERT OR IGNORE INTO archived_demo_prices(symbol,date,open,high,low,close,adjusted_close,volume,source,retrieved_at,quality,price_basis) SELECT symbol,date,open,high,low,close,adjusted_close,volume,source,retrieved_at,quality,CASE WHEN price_basis='unknown' THEN 'synthetic' ELSE price_basis END FROM prices WHERE quality='DEMO_SYNTHETIC'")
        c.execute('CREATE INDEX IF NOT EXISTS import_run_started_idx ON data_import_runs(started_at)')
        import_cols={r['name'] for r in c.execute('PRAGMA table_info(data_import_runs)')}
        for name,decl in {'dataset_description':"TEXT NOT NULL DEFAULT ''",'license_reference':"TEXT NOT NULL DEFAULT ''",'user_confirmed_use':'INTEGER NOT NULL DEFAULT 0','accepted_rows':'INTEGER NOT NULL DEFAULT 0','rejected_rows':'INTEGER NOT NULL DEFAULT 0','price_basis':'TEXT'}.items():
            if name not in import_cols: c.execute(f'ALTER TABLE data_import_runs ADD COLUMN {name} {decl}')
        for table in ('fundamentals','events'):
            cols={r['name'] for r in c.execute(f'PRAGMA table_info({table})')}
            if 'published_at' not in cols: c.execute(f'ALTER TABLE {table} ADD COLUMN published_at TEXT')
        oos_cols={r['name'] for r in c.execute('PRAGMA table_info(oos_predictions)')}
        if 'price_basis' not in oos_cols: c.execute("ALTER TABLE oos_predictions ADD COLUMN price_basis TEXT NOT NULL DEFAULT 'unknown'")
        cols={r['name'] for r in c.execute('PRAGMA table_info(securities)')}
        additions={'instrument_id':'TEXT','provider_symbol':'TEXT'}
        missing={name:decl for name,decl in additions.items() if name not in cols}
        if missing:
            if database_existed: backup_database()
            for name,decl in missing.items(): c.execute(f'ALTER TABLE securities ADD COLUMN {name} {decl}')
        c.execute('CREATE UNIQUE INDEX IF NOT EXISTS security_instrument_id_unique ON securities(instrument_id) WHERE instrument_id IS NOT NULL')
        c.execute("UPDATE refresh_runs SET status='interrupted',finished_at=? WHERE status='running'",(datetime.now(timezone.utc).isoformat(),))
        c.execute('INSERT OR IGNORE INTO schema_migrations(version,applied_at) VALUES(4,?)',(datetime.now(timezone.utc).isoformat(),))
        c.execute('INSERT OR IGNORE INTO schema_migrations(version,applied_at) VALUES(5,?)',(datetime.now(timezone.utc).isoformat(),))
    _apply_v6_migration()
    _apply_v7_migration()
    _apply_v8_migration()
    _apply_v9_migration()
    _apply_v10_migration()
    _apply_v11_migration()
    _ensure_published_views()


def _apply_v6_migration():
    """Add versioned import, revision, dependency, and release lineage atomically."""
    with connect() as c:
        version=c.execute('SELECT COALESCE(MAX(version),0) FROM schema_migrations').fetchone()[0]
        if version>=6: return
        c.execute('BEGIN IMMEDIATE')
        try:
            additions={
                'data_import_runs': {'content_sha256':'TEXT','version_id':'INTEGER','refresh_run_id':'INTEGER','change_summary_json':"TEXT NOT NULL DEFAULT '{}'",'correction_confirmed':'INTEGER NOT NULL DEFAULT 0'},
                'refresh_runs': {'import_run_id':'INTEGER','version_id':'INTEGER','change_summary_json':"TEXT NOT NULL DEFAULT '{}'",'dependencies_json':"TEXT NOT NULL DEFAULT '[]'",'step_timings_json':"TEXT NOT NULL DEFAULT '{}'"},
                'current_forecasts': {'version_id':'INTEGER','input_fingerprint':'TEXT','status':"TEXT NOT NULL DEFAULT 'current'",'stale_reason':'TEXT'},
            }
            for table, fields in additions.items():
                present={r['name'] for r in c.execute(f'PRAGMA table_info({table})')}
                for name,decl in fields.items():
                    if name not in present: c.execute(f'ALTER TABLE {table} ADD COLUMN {name} {decl}')
            c.execute("CREATE TABLE IF NOT EXISTS dataset_versions(id INTEGER PRIMARY KEY AUTOINCREMENT,dataset_type TEXT NOT NULL,content_fingerprint TEXT NOT NULL,metadata_fingerprint TEXT NOT NULL,parent_version_id INTEGER,import_run_id INTEGER,created_at TEXT NOT NULL,change_kind TEXT NOT NULL,summary_json TEXT NOT NULL DEFAULT '{}',UNIQUE(dataset_type,content_fingerprint,metadata_fingerprint))")
            c.execute("CREATE TABLE IF NOT EXISTS data_revisions(id INTEGER PRIMARY KEY AUTOINCREMENT,dataset_type TEXT NOT NULL,symbol TEXT NOT NULL,observation_date TEXT NOT NULL,change_kind TEXT NOT NULL,old_value_json TEXT,new_value_json TEXT NOT NULL,import_run_id INTEGER NOT NULL,created_at TEXT NOT NULL,UNIQUE(import_run_id,symbol,observation_date,change_kind))")
            c.execute("CREATE INDEX IF NOT EXISTS data_revision_lookup ON data_revisions(dataset_type,symbol,observation_date)")
            c.execute("CREATE TABLE IF NOT EXISTS dependency_states(component TEXT NOT NULL,scope_key TEXT NOT NULL,status TEXT NOT NULL,input_fingerprint TEXT,version_id INTEGER,updated_at TEXT NOT NULL,detail_json TEXT NOT NULL DEFAULT '{}',PRIMARY KEY(component,scope_key))")
            c.execute("CREATE TABLE IF NOT EXISTS research_releases(id INTEGER PRIMARY KEY AUTOINCREMENT,version_id INTEGER NOT NULL,refresh_run_id INTEGER NOT NULL,status TEXT NOT NULL,created_at TEXT NOT NULL,published_at TEXT,fingerprint TEXT NOT NULL,validation_json TEXT NOT NULL DEFAULT '{}',errors_json TEXT NOT NULL DEFAULT '[]')")
            c.execute("CREATE TABLE IF NOT EXISTS candidate_forecasts(id INTEGER PRIMARY KEY AUTOINCREMENT,refresh_run_id INTEGER NOT NULL,version_id INTEGER NOT NULL,model_version_id INTEGER NOT NULL,symbol TEXT NOT NULL,origin_date TEXT NOT NULL,horizon_sessions INTEGER NOT NULL,prediction REAL NOT NULL,price_basis TEXT NOT NULL,input_fingerprint TEXT NOT NULL,created_at TEXT NOT NULL,UNIQUE(refresh_run_id,model_version_id,symbol,origin_date))")
            c.execute("CREATE TABLE IF NOT EXISTS app_state(key TEXT PRIMARY KEY,value TEXT NOT NULL,updated_at TEXT NOT NULL)")
            c.execute("INSERT OR IGNORE INTO schema_migrations(version,applied_at) VALUES(6,?)",(datetime.now(timezone.utc).isoformat(),))
            c.commit()
        except Exception:
            c.rollback()
            raise

def _apply_v7_migration():
    """Add the source registry and durable acquisition-attempt ledger."""
    with connect() as c:
        version=c.execute('SELECT COALESCE(MAX(version),0) FROM schema_migrations').fetchone()[0]
        if version>=7: return
        c.execute('BEGIN IMMEDIATE')
        try:
            c.execute("CREATE TABLE IF NOT EXISTS source_registry(source_key TEXT PRIMARY KEY,name TEXT NOT NULL,owner TEXT NOT NULL,data_types_json TEXT NOT NULL,coverage_json TEXT NOT NULL,access_method TEXT NOT NULL,permission_status TEXT NOT NULL,authentication TEXT NOT NULL,pricing_summary TEXT NOT NULL,terms_url TEXT NOT NULL,docs_url TEXT NOT NULL,request_limits TEXT NOT NULL,attribution TEXT NOT NULL,notes TEXT NOT NULL,checked_at TEXT NOT NULL,last_success_at TEXT,last_error TEXT)")
            c.execute("CREATE TABLE IF NOT EXISTS acquisition_runs(id INTEGER PRIMARY KEY AUTOINCREMENT,started_at TEXT NOT NULL,finished_at TEXT,status TEXT NOT NULL,attempted_sources_json TEXT NOT NULL DEFAULT '[]',errors_json TEXT NOT NULL DEFAULT '[]',summary_json TEXT NOT NULL DEFAULT '{}')")
            c.execute('CREATE INDEX IF NOT EXISTS acquisition_runs_started_idx ON acquisition_runs(started_at)')
            c.execute("INSERT OR IGNORE INTO schema_migrations(version,applied_at) VALUES(7,?)",(datetime.now(timezone.utc).isoformat(),))
            c.commit()
        except Exception:
            c.rollback()
            raise

def _apply_v8_migration():
    """Add explicit source review, adapter, connectivity and attempt status."""
    with connect() as c:
        version=c.execute('SELECT COALESCE(MAX(version),0) FROM schema_migrations').fetchone()[0]
        if version>=8: return
        c.execute('BEGIN IMMEDIATE')
        try:
            columns={r['name'] for r in c.execute('PRAGMA table_info(source_registry)')}
            additions={
                'documentation_status':"TEXT NOT NULL DEFAULT 'not_reviewed'",
                'access_status':"TEXT NOT NULL DEFAULT 'uncertain'",
                'adapter_status':"TEXT NOT NULL DEFAULT 'not_implemented'",
                'connectivity_status':"TEXT NOT NULL DEFAULT 'not_tested'",
                'access_cost_type':"TEXT NOT NULL DEFAULT 'unknown'",
                'last_attempt_at':'TEXT',
                'last_attempt_status':'TEXT',
            }
            for name,decl in additions.items():
                if name not in columns: c.execute(f'ALTER TABLE source_registry ADD COLUMN {name} {decl}')
            # v7 stored explanatory permission notes in last_error; those were
            # never failed provider attempts and must not appear as such.
            c.execute('UPDATE source_registry SET last_error=NULL WHERE last_attempt_at IS NULL AND last_error=notes')
            c.execute("INSERT OR IGNORE INTO schema_migrations(version,applied_at) VALUES(8,?)",(datetime.now(timezone.utc).isoformat(),))
            c.commit()
        except Exception:
            c.rollback()
            raise

_SNAPSHOT_TABLES = ('prices','exchange_market_prices','securities','fundamentals','fundamental_observations','events','corporate_actions',
                    'constituent_snapshots','constituent_membership','instrument_universe','exchange_sessions')

def _apply_v9_migration():
    """Create immutable per-release copies for all research input tables."""
    with connect() as c:
        version=c.execute('SELECT COALESCE(MAX(version),0) FROM schema_migrations').fetchone()[0]
        if version>=9: return
        c.execute('BEGIN IMMEDIATE')
        try:
            c.execute('CREATE TABLE IF NOT EXISTS research_snapshot_releases(release_id INTEGER PRIMARY KEY,captured_at TEXT NOT NULL,table_counts_json TEXT NOT NULL DEFAULT \'{}\')')
            for table in _SNAPSHOT_TABLES:
                c.execute(f'CREATE TABLE IF NOT EXISTS release_snapshot_{table} AS SELECT 0 AS release_id,* FROM {table} WHERE 0')
                c.execute(f'CREATE INDEX IF NOT EXISTS release_snapshot_{table}_release_idx ON release_snapshot_{table}(release_id)')
            c.execute("INSERT OR IGNORE INTO schema_migrations(version,applied_at) VALUES(9,?)",(datetime.now(timezone.utc).isoformat(),))
            active=c.execute("SELECT CAST(value AS INTEGER) FROM app_state WHERE key='active_release_id'").fetchone()
            if active:
                capture_release_snapshot(c,int(active[0]))
            c.commit()
        except Exception:
            c.rollback()
            raise

def _apply_v10_migration():
    """Add an isolated, revision-preserving store for permitted macro observations."""
    with connect() as c:
        version=c.execute('SELECT COALESCE(MAX(version),0) FROM schema_migrations').fetchone()[0]
        if version>=10: return
        c.execute('BEGIN IMMEDIATE')
        try:
            c.execute('''CREATE TABLE IF NOT EXISTS macro_observations(
                id INTEGER PRIMARY KEY AUTOINCREMENT,source TEXT NOT NULL,dataset TEXT NOT NULL,
                year INTEGER NOT NULL,month INTEGER NOT NULL,geography TEXT NOT NULL,sector TEXT NOT NULL,
                group_name TEXT NOT NULL,subgroup TEXT NOT NULL,index_value REAL,inflation REAL,status TEXT,
                published_at TEXT,retrieved_at TEXT NOT NULL,payload_hash TEXT NOT NULL,
                record_hash TEXT NOT NULL UNIQUE,record_json TEXT NOT NULL)''')
            c.execute('CREATE INDEX IF NOT EXISTS macro_period_idx ON macro_observations(dataset,year,month,geography,sector)')
            c.execute("INSERT OR IGNORE INTO schema_migrations(version,applied_at) VALUES(10,?)",(datetime.now(timezone.utc).isoformat(),))
            c.commit()
        except Exception:
            c.rollback()
            raise

def _apply_v11_migration():
    """Track per-exchange Tej Bazaar refreshes and file-level provenance."""
    with connect() as c:
        version=c.execute('SELECT COALESCE(MAX(version),0) FROM schema_migrations').fetchone()[0]
        if version>=11: return
        c.execute('BEGIN IMMEDIATE')
        try:
            c.execute('''CREATE TABLE IF NOT EXISTS market_data_files(
                exchange TEXT NOT NULL,relative_path TEXT NOT NULL,observation_date TEXT NOT NULL,
                sha256 TEXT NOT NULL,size_bytes INTEGER NOT NULL,modified_ns INTEGER NOT NULL DEFAULT 0,row_count INTEGER NOT NULL,
                imported_at TEXT NOT NULL,source TEXT NOT NULL,PRIMARY KEY(exchange,relative_path))''')
            c.execute('CREATE INDEX IF NOT EXISTS market_data_files_date_idx ON market_data_files(exchange,observation_date)')
            c.execute('''CREATE TABLE IF NOT EXISTS market_exchange_refresh(
                exchange TEXT PRIMARY KEY,last_attempt_at TEXT,last_successful_at TEXT,
                last_source_fetch_at TEXT,last_source_fetch_status TEXT,latest_market_date TEXT,
                latest_source_date TEXT,rows_in_source INTEGER NOT NULL DEFAULT 0,
                rows_imported INTEGER NOT NULL DEFAULT 0,rows_updated INTEGER NOT NULL DEFAULT 0,
                rows_unchanged INTEGER NOT NULL DEFAULT 0,status TEXT,error TEXT,
                source_published_at TEXT,updated_at TEXT)''')
            c.execute('CREATE TABLE IF NOT EXISTS release_snapshot_exchange_market_prices AS SELECT 0 AS release_id,* FROM exchange_market_prices WHERE 0')
            c.execute('CREATE INDEX IF NOT EXISTS release_snapshot_exchange_market_prices_release_idx ON release_snapshot_exchange_market_prices(release_id)')
            c.execute("INSERT OR IGNORE INTO schema_migrations(version,applied_at) VALUES(11,?)",(datetime.now(timezone.utc).isoformat(),))
            c.commit()
        except Exception:
            c.rollback()
            raise

def capture_release_snapshot(conn, release_id:int):
    """Copy the canonical input universe into one immutable release atomically."""
    if conn.execute('SELECT 1 FROM research_snapshot_releases WHERE release_id=?',(release_id,)).fetchone(): return
    counts={}
    for table in _SNAPSHOT_TABLES:
        snapshot=f'release_snapshot_{table}'
        conn.execute(f'INSERT INTO {snapshot} SELECT ?,* FROM {table}',(release_id,))
        counts[table]=conn.execute(f'SELECT changes()').fetchone()[0]
    conn.execute('INSERT INTO research_snapshot_releases(release_id,captured_at,table_counts_json) VALUES(?,?,?)',(release_id,datetime.now(timezone.utc).isoformat(),json.dumps(counts,sort_keys=True)))

def _ensure_published_views():
    with connect() as c:
        for table in _SNAPSHOT_TABLES:
            view=f'published_{table}'
            cols=[r['name'] for r in c.execute(f'PRAGMA table_info({table})')]
            snapshot=f'release_snapshot_{table}'
            canonical=', '.join(f'c."{col}"' for col in cols)
            snap=', '.join(f's."{col}"' for col in cols)
            c.execute(f'DROP VIEW IF EXISTS {view}')
            c.execute(f'''CREATE VIEW {view} AS
                SELECT {snap} FROM {snapshot} s
                WHERE s.release_id=(SELECT CAST(value AS INTEGER) FROM app_state WHERE key='active_release_id')
                UNION ALL
                SELECT {canonical} FROM {table} c
                WHERE NOT EXISTS (SELECT 1 FROM research_snapshot_releases x WHERE x.release_id=(SELECT CAST(value AS INTEGER) FROM app_state WHERE key='active_release_id'))''')

def backup_database():
    """Create a consistent SQLite backup using the online backup API."""
    if not DB_PATH.exists(): return None
    stamp=datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')
    target=DB_PATH.with_name(f'{DB_PATH.stem}.backup-{stamp}{DB_PATH.suffix}')
    source_conn=sqlite3.connect(DB_PATH,timeout=30); backup_conn=sqlite3.connect(target)
    try: source_conn.backup(backup_conn)
    finally: backup_conn.close(); source_conn.close()
    return target

def audit(action: str, details: dict):
    with connect() as c:
        c.execute("INSERT INTO audit(timestamp,action,details) VALUES(?,?,?)", (datetime.now(timezone.utc).isoformat(), action, json.dumps(details, default=str)))
