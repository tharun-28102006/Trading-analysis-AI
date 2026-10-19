from datetime import date, timedelta
import json
import time
from pathlib import Path

import polars as pl
import pytest

from app import acquisition, config, db, market
from app.providers import tej
from app.providers import tej_bazaar_local
from app import tej_bazaar_lifecycle


@pytest.fixture
def local_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db, 'DATA_DIR', tmp_path)
    monkeypatch.setattr(db, 'DB_PATH', tmp_path / 'nivesh.sqlite3')
    monkeypatch.setattr(config, 'DATA_DIR', tmp_path)
    monkeypatch.setattr(market, 'DATA_DIR', tmp_path)
    root = tmp_path / 'tej-out'
    monkeypatch.setenv('NIVESH_TEJ_DATA_DIR', str(root))
    monkeypatch.setenv('NIVESH_TEJ_OVERLAP_DAYS', '7')
    db.init_db()
    acquisition.sync_source_registry()
    for exchange in ('nse','bse'):
        day = date.today() - timedelta(days=3)
        folder = root / exchange / f'year={day.year}' / f'month={day.month:02d}'
        folder.mkdir(parents=True, exist_ok=True)
        _write_file(folder / f'date={day.isoformat()}.parquet', exchange.upper(), day, 100.0)
    return tmp_path


def _write_file(path: Path, exchange: str, day: date, close: float):
    series='EQ' if exchange=='NSE' else 'A'
    pl.DataFrame([{'date':day,'symbol':'RELIANCE' if exchange=='NSE' else '500325','series':series,
        'isin':'INE002A01018','name':'Reliance Industries','open':100.0,'high':max(103.0,close),
        'low':99.0,'close':close,'last':close,'prev_close':99.0,'volume':1200,
        'turnover':120000.0,'trades':25}]).write_parquet(path)


class FakeTejClient:
    def __init__(self): pass
    def status(self): return {'trading_date':'2010-01-01'}
    def snapshot(self, exchange, day): raise AssertionError('snapshot is outside the configured overlap')


def run(monkeypatch, *, client=FakeTejClient, reconcile=False):
    monkeypatch.setattr(tej, 'TejClient', client)
    monkeypatch.setattr(tej_bazaar_lifecycle, 'TejClient', client)
    started = market.start_refresh('test', reconcile)
    assert started['status'] == 'running'
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        result = market.refresh_job(started['run_id'])
        if result['status'] != 'running': return result
        time.sleep(.02)
    pytest.fail('Refresh job did not finish within twenty seconds')


def test_local_nse_bse_bootstrap_and_no_new_data_are_idempotent(local_db, monkeypatch):
    first = run(monkeypatch)
    assert first['status'] == 'partial_success'
    assert first['rows_written'] == 2
    with db.connect() as conn:
        active=conn.execute("SELECT value FROM app_state WHERE key='active_release_id'").fetchone()[0]
        assert conn.execute("SELECT status FROM research_releases WHERE id=?",(active,)).fetchone()[0] == 'published'
        assert conn.execute("SELECT COUNT(*) FROM prices WHERE quality='TEJ_BAZAAR_LOCAL_LICENSED'").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM exchange_market_prices WHERE quality='TEJ_BAZAAR_LOCAL_LICENSED'").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM raw_price_observations WHERE source LIKE 'tej_bazaar_local:%'").fetchone()[0] == 2
    second = run(monkeypatch)
    assert second['status'] == 'unchanged'
    assert second['rows_written'] == 0
    status=market.get_data_status()
    assert status['exchange_refresh_status']['NSE']['rows'] == 1
    assert status['exchange_refresh_status']['BSE']['rows'] == 1
    assert status['last_successful_refresh_ist']


def test_corrected_local_file_publishes_revision_and_release(local_db, monkeypatch):
    first = run(monkeypatch)
    with db.connect() as conn:
        before=int(conn.execute("SELECT value FROM app_state WHERE key='active_release_id'").fetchone()[0])
    path=Path(local_db/'tej-out'/'nse')
    file=next(path.glob('year=*/month=*/date=*.parquet'))
    day=date.fromisoformat(file.stem.split('=',1)[1])
    _write_file(file,'NSE',day,101.5)
    result=run(monkeypatch,reconcile=True)
    assert result['status']=='succeeded'
    with db.connect() as conn:
        assert conn.execute("SELECT close FROM published_prices WHERE symbol='RELIANCE' AND quality='TEJ_BAZAAR_LOCAL_LICENSED'").fetchone()[0] == 101.5
        assert conn.execute('SELECT COUNT(*) FROM data_revisions').fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM research_releases WHERE status IN ('published','superseded')").fetchone()[0] == 2
        assert int(conn.execute("SELECT value FROM app_state WHERE key='active_release_id'").fetchone()[0]) != before


def test_provider_failure_after_bootstrap_preserves_active_release(local_db, monkeypatch):
    first=run(monkeypatch)
    with db.connect() as conn:
        before=int(conn.execute("SELECT value FROM app_state WHERE key='active_release_id'").fetchone()[0])
    class Offline:
        def status(self): raise tej.TejError('mock provider unavailable')
    failed=run(monkeypatch,client=Offline)
    assert failed['status']=='failed'
    with db.connect() as conn:
        assert int(conn.execute("SELECT value FROM app_state WHERE key='active_release_id'").fetchone()[0])==before
        assert conn.execute("SELECT COUNT(*) FROM published_prices WHERE quality='TEJ_BAZAAR_LOCAL_LICENSED'").fetchone()[0]==1


def test_invalid_local_file_fails_validation_without_publication(local_db, monkeypatch):
    first=run(monkeypatch)
    with db.connect() as conn:
        before=int(conn.execute("SELECT value FROM app_state WHERE key='active_release_id'").fetchone()[0])
    path=next((local_db/'tej-out'/'nse').glob('year=*/month=*/date=*.parquet'))
    day=date.fromisoformat(path.stem.split('=',1)[1])
    _write_file(path,'NSE',day,98.0) # close below the source low of 99
    result=run(monkeypatch,reconcile=True)
    assert result['status']=='failed'
    with db.connect() as conn:
        assert int(conn.execute("SELECT value FROM app_state WHERE key='active_release_id'").fetchone()[0])==before
        assert conn.execute("SELECT close FROM published_prices WHERE symbol='RELIANCE'").fetchone()[0]==100.0


def test_valid_provider_symbol_with_bse_hash_marker_is_preserved(local_db):
    root=Path(local_db/'tej-out')
    path=next((root/'bse').glob('year=*/month=*/date=*.parquet'))
    frame=pl.read_parquet(path).with_columns(pl.lit('SBIN#').alias('symbol'))
    frame.write_parquet(path)
    file=tej_bazaar_local.discover_files(root,'BSE')[0]
    parsed=tej_bazaar_local.read_and_validate_file(file)
    assert parsed['symbol'].item()=='SBIN#'


def test_refresh_job_rejects_overlapping_jobs(local_db, monkeypatch):
    class Slow(FakeTejClient):
        def status(self): time.sleep(.3); return super().status()
    monkeypatch.setattr(tej,'TejClient',Slow)
    monkeypatch.setattr(tej_bazaar_lifecycle,'TejClient',Slow)
    first=market.start_refresh('test')
    second=market.start_refresh('test')
    assert first['status']=='running'
    assert second['status']=='already_running'
    deadline=time.monotonic()+10
    while time.monotonic()<deadline and market.refresh_job(first['run_id'])['status']=='running':
        time.sleep(.02)
