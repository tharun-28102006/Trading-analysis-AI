import json
import sqlite3
from pathlib import Path

from app import db, market


def prepare(tmp_path, monkeypatch):
    monkeypatch.setattr(db,'DATA_DIR',tmp_path)
    monkeypatch.setattr(db,'DB_PATH',tmp_path/'lifecycle.sqlite3')
    monkeypatch.setattr(market,'DATA_DIR',tmp_path)
    db.init_db()


def price_csv(rows, source='LicensedLocalFile'):
    return ('symbol,exchange,date,open,high,low,close,volume,price_basis,source,adjusted_close,dividends,stock_splits\n'+
            ''.join(f'{symbol},{exchange},{date},{op},{hi},{lo},{close},{volume},raw_unadjusted,{source},,{dividend},{split}\n'
                    for symbol,exchange,date,op,hi,lo,close,volume,dividend,split in rows)).encode()


def test_identical_observations_from_different_file_do_not_rebuild(tmp_path,monkeypatch):
    prepare(tmp_path,monkeypatch)
    body=price_csv([('AAA','NSE','2026-10-15',100,102,99,101,1000,0,0)])
    first=market.import_csv(body,'first.csv')
    before=len(market.get_data_status()['dependency_statuses'])
    duplicate=market.import_csv(body,'first.csv')
    assert duplicate.get('idempotent') is True
    with db.connect() as c:
        assert c.execute('SELECT COUNT(*) FROM refresh_runs').fetchone()[0]==1
    second=market.import_csv(b'# another permitted local copy\n'+body,'second.csv')
    assert first['lifecycle']['release_status']=='published'
    assert second['change_summary']['change_kind']=='metadata_only'
    assert second['rows_imported']==0
    assert len(market.get_data_status()['dependency_statuses'])==before
    with db.connect() as c:
        assert c.execute('SELECT COUNT(*) FROM prices WHERE quality<>\'DEMO_SYNTHETIC\'').fetchone()[0]==1
        assert c.execute('SELECT COUNT(*) FROM research_releases WHERE status IN (\'published\',\'superseded\')').fetchone()[0]==2


def test_new_benchmark_session_propagates_dependencies_without_retraining(tmp_path,monkeypatch):
    prepare(tmp_path,monkeypatch)
    rows=[('AAA','NSE','2026-10-15',100,102,99,101,1000,0,0),('BBB','NSE','2026-10-15',50,52,49,51,800,0,0),('NIFTY50','NSE_INDEX','2026-10-15',200,202,199,201,2000,0,0)]
    first=market.import_csv(price_csv(rows),'initial.csv')
    rows2=[('NIFTY50','NSE_INDEX','2026-10-16',201,203,200,202,2100,0,0)]
    second=market.import_csv(price_csv(rows2),'benchmark-session.csv')
    assert first['lifecycle']['release_status']=='published'
    with db.connect() as c:
        c.execute("INSERT INTO prices(symbol,date,open,high,low,close,adjusted_close,volume,source,retrieved_at,quality,price_basis) VALUES('AAA','2026-10-16',101,103,100,102,102,1100,'unpublished-test','now','USER_CONFIRMED_UNVERIFIED','raw_unadjusted')")
        assert c.execute("SELECT COUNT(*) FROM published_prices WHERE symbol='AAA'").fetchone()[0]==1
        c.execute("DELETE FROM prices WHERE symbol='AAA' AND date='2026-10-16'")
    assert second['change_summary']['change_kind']=='new_session'
    assert set(second['change_summary']['affected_symbols'])=={'AAA','BBB','NIFTY50'}
    statuses={(r['component'],r['scope_key']):r['status'] for r in market.get_data_status()['dependency_statuses']}
    assert statuses[('feature_datasets','AAA')]=='stale'
    assert statuses[('feature_datasets','BBB')]=='stale'
    assert statuses[('forecasts','AAA')]=='insufficient_data'
    assert statuses[('company_models','AAA')]=='stale'
    with db.connect() as c:
        assert c.execute("SELECT COUNT(*) FROM published_prices WHERE symbol='NIFTY50'").fetchone()[0]==2
        active=int(c.execute("SELECT value FROM app_state WHERE key='active_release_id'").fetchone()[0])
        assert c.execute('SELECT COUNT(*) FROM release_snapshot_prices WHERE release_id=?',(active,)).fetchone()[0] >= 3
    with db.connect() as c:
        assert c.execute("SELECT COUNT(*) FROM published_prices WHERE symbol='NIFTY50'").fetchone()[0]==2
        active=int(c.execute("SELECT value FROM app_state WHERE key='active_release_id'").fetchone()[0])
        assert c.execute('SELECT COUNT(*) FROM release_snapshot_prices WHERE release_id=?',(active,)).fetchone()[0] >= 3
    with db.connect() as c:
        assert c.execute("SELECT COUNT(*) FROM training_runs").fetchone()[0]==0


def test_historical_correction_requires_confirmation_then_records_revision_and_invalidates(tmp_path,monkeypatch):
    prepare(tmp_path,monkeypatch)
    base=[('AAA','NSE','2026-10-15',100,102,99,101,1000,0,0),('NIFTY50','NSE_INDEX','2026-10-15',200,202,199,201,2000,0,0)]
    market.import_csv(price_csv(base),'base.csv')
    correction=price_csv([('NIFTY50','NSE_INDEX','2026-10-15',200,203,199,202,2000,0,0)])
    blocked=market.import_csv(correction,'corrected.csv')
    assert blocked['status']=='blocked_correction_confirmation'
    with db.connect() as c:
        assert c.execute("SELECT close FROM prices WHERE symbol='NIFTY50'").fetchone()[0]==201
        assert c.execute('SELECT COUNT(*) FROM data_revisions').fetchone()[0]==0
    revised=market.import_csv(correction,'corrected.csv',allow_historical_corrections=True)
    assert revised['change_summary']['change_kind']=='historical_correction'
    with db.connect() as c:
        revision=c.execute('SELECT * FROM data_revisions').fetchone()
        assert revision['change_kind']=='historical_correction'
        assert json.loads(revision['old_value_json'])['close']==201
        assert json.loads(revision['new_value_json'])['close']==202
        assert c.execute("SELECT close FROM prices WHERE symbol='NIFTY50'").fetchone()[0]==202
    assert set(revised['change_summary']['affected_symbols'])=={'AAA','NIFTY50'}
    assert any(r['component']=='targets' and r['scope_key']=='AAA' and r['status']=='stale' for r in market.get_data_status()['dependency_statuses'])


def test_failed_release_publication_keeps_previous_active_release(tmp_path,monkeypatch):
    prepare(tmp_path,monkeypatch)
    first=market.import_csv(price_csv([('AAA','NSE','2026-10-15',100,102,99,101,1000,0,0)]),'base.csv')
    with db.connect() as c:
        active_before=c.execute("SELECT value FROM app_state WHERE key='active_release_id'").fetchone()[0]
        c.execute("CREATE TRIGGER fail_candidate BEFORE INSERT ON research_releases BEGIN SELECT RAISE(ABORT,'publication unavailable'); END")
    second=market.import_csv(price_csv([('AAA','NSE','2026-10-16',101,103,100,102,1100,0,0)]),'next.csv')
    assert second['lifecycle']['release_status']=='failed'
    with db.connect() as c:
        assert c.execute("SELECT value FROM app_state WHERE key='active_release_id'").fetchone()[0]==active_before
        assert c.execute("SELECT status FROM research_releases WHERE id=?",(int(active_before),)).fetchone()[0]=='published'
        assert c.execute("SELECT status FROM refresh_runs ORDER BY id DESC LIMIT 1").fetchone()[0]=='failed'
        assert c.execute("SELECT COUNT(*) FROM prices WHERE symbol='AAA'").fetchone()[0]==1
        assert c.execute("SELECT COUNT(*) FROM published_prices WHERE symbol='AAA'").fetchone()[0]==1
        assert c.execute("SELECT date FROM prices WHERE symbol='AAA'").fetchone()[0]=='2026-10-15'
        assert c.execute("SELECT status FROM data_import_runs ORDER BY id DESC LIMIT 1").fetchone()[0]=='failed_publication'
        assert c.execute("SELECT COUNT(*) FROM raw_price_observations WHERE validation_status='CANDIDATE_PUBLICATION_FAILED'").fetchone()[0]==1
        assert c.execute("SELECT COUNT(*) FROM dependency_states WHERE version_id=(SELECT version_id FROM refresh_runs ORDER BY id DESC LIMIT 1)").fetchone()[0]==0


def test_schema_v6_migration_backs_up_and_preserves_legacy_rows(tmp_path,monkeypatch):
    monkeypatch.setattr(db,'DATA_DIR',tmp_path)
    monkeypatch.setattr(db,'DB_PATH',tmp_path/'upgrade.sqlite3')
    con=sqlite3.connect(db.DB_PATH)
    con.executescript(db.SCHEMA)
    con.execute("INSERT INTO securities(symbol,name) VALUES('AAA','Alpha')")
    con.execute("INSERT INTO prices(symbol,date,close,adjusted_close,source,retrieved_at,quality) VALUES('AAA','2026-10-15',1,1,'legacy','now','DEMO_SYNTHETIC')")
    con.execute("INSERT INTO schema_migrations(version,applied_at) VALUES(5,'prior')")
    con.commit();con.close()
    db.init_db()
    backups=list(tmp_path.glob('upgrade.backup-*.sqlite3'))
    assert len(backups)==1
    backup=sqlite3.connect(backups[0])
    assert backup.execute('SELECT COUNT(*) FROM prices').fetchone()[0]==1
    assert backup.execute('SELECT MAX(version) FROM schema_migrations').fetchone()[0]==5
    backup.close()
    with db.connect() as c:
        assert c.execute('SELECT MAX(version) FROM schema_migrations').fetchone()[0]==11
        assert c.execute('SELECT COUNT(*) FROM prices').fetchone()[0]==1
        assert c.execute('SELECT COUNT(*) FROM archived_demo_prices').fetchone()[0]==1


def test_published_reads_stay_pinned_across_release_switch(tmp_path,monkeypatch):
    prepare(tmp_path,monkeypatch)
    with db.connect() as c:
        c.execute("INSERT INTO securities(symbol,name) VALUES('AAA','Alpha')")
        c.execute("INSERT INTO prices(symbol,date,close,adjusted_close,source,retrieved_at,quality,price_basis) VALUES('AAA','2026-10-15',10,10,'test','now','USER_CONFIRMED_UNVERIFIED','raw_unadjusted')")
        first=c.execute("INSERT INTO research_releases(version_id,refresh_run_id,status,created_at,fingerprint) VALUES(1,1,'published','now','one')").lastrowid
        db.capture_release_snapshot(c,first)
        c.execute("INSERT INTO app_state(key,value,updated_at) VALUES('active_release_id',?,'now') ON CONFLICT(key) DO UPDATE SET value=excluded.value",(str(first),))
    with db.pin_published_release(first):
        with db.connect() as c:
            assert c.execute("SELECT COUNT(*) FROM published_prices WHERE symbol='AAA'").fetchone()[0]==1
        with db.connect() as c:
            second=c.execute("INSERT INTO research_releases(version_id,refresh_run_id,status,created_at,fingerprint) VALUES(2,2,'published','now','two')").lastrowid
            c.execute("INSERT INTO prices(symbol,date,close,adjusted_close,source,retrieved_at,quality,price_basis) VALUES('AAA','2026-10-16',11,11,'test','now','USER_CONFIRMED_UNVERIFIED','raw_unadjusted')")
            db.capture_release_snapshot(c,second)
            c.execute("UPDATE app_state SET value=? WHERE key='active_release_id'",(str(second),))
        with db.connect() as c:
            assert c.execute("SELECT COUNT(*) FROM published_prices WHERE symbol='AAA'").fetchone()[0]==1
            assert c.execute("SELECT MAX(date) FROM published_prices WHERE symbol='AAA'").fetchone()[0]=='2026-10-15'
    with db.connect() as c:
        assert c.execute("SELECT MAX(date) FROM published_prices WHERE symbol='AAA'").fetchone()[0]=='2026-10-16'
