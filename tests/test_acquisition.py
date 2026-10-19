import json
import sqlite3

from fastapi.testclient import TestClient

from app import acquisition, db, main


def test_free_source_registry_explains_permission_and_coverage_gates(tmp_path, monkeypatch):
    monkeypatch.setattr(db, 'DATA_DIR', tmp_path)
    monkeypatch.setattr(db, 'DB_PATH', tmp_path / 'sources.sqlite3')
    db.init_db()
    sources = {row['source_key']: row for row in acquisition.list_sources()}
    assert sources['nse_research_channel']['access_status'] == 'blocked_accreditation_and_approval'
    assert sources['nse_research_channel']['access_cost_type'] == 'free_restricted'
    assert sources['nse_public_website']['permission_status'] == 'not_permitted_for_automated_research_ingestion'
    assert sources['mospi_cpi_api']['access_status'] == 'free_open_api'
    assert sources['mospi_cpi_api']['adapter_status'] == 'implemented'
    assert sources['mospi_cpi_api']['connectivity_status'] == 'not_tested'
    assert all('dhan' not in row['source_key'] for row in sources.values())


def test_acquisition_gate_records_blocked_attempt_without_network(tmp_path, monkeypatch):
    monkeypatch.setattr(db, 'DATA_DIR', tmp_path)
    monkeypatch.setattr(db, 'DB_PATH', tmp_path / 'sources.sqlite3')
    monkeypatch.setattr(acquisition, 'SOURCE_CATALOG', tuple(
        {**source, 'adapter_status': 'not_implemented'} for source in acquisition.SOURCE_CATALOG))
    db.init_db()
    result = acquisition.run_acquisition('test')
    assert result['status'] == 'blocked'
    assert result['network_requests_made'] == 0
    with db.connect() as conn:
        run = conn.execute('SELECT * FROM acquisition_runs WHERE id=?', (result['run_id'],)).fetchone()
        summary = json.loads(run['summary_json'])
        assert run['status'] == 'blocked'
        assert summary['data_acquired'] is False


def test_source_and_refresh_api_report_mospi_failure_without_fabricating_data(tmp_path, monkeypatch):
    monkeypatch.setattr(db, 'DATA_DIR', tmp_path)
    monkeypatch.setattr(db, 'DB_PATH', tmp_path / 'api.sqlite3')
    from app.providers import mospi
    class OfflineClient:
        def fetch_cpi(self):
            raise mospi.MoSPIError('MoSPI request failed (URLError).')
    monkeypatch.setattr(mospi, 'MoSPIClient', OfflineClient)
    with TestClient(main.app) as client:
        sources = client.get('/api/data/sources').json()['sources']
        assert any(row['source_key'] == 'nse_research_channel' for row in sources)
        response = client.post('/api/data/acquisition/refresh')
        assert response.status_code == 200
        assert response.json()['status'] == 'failed'
        assert response.json()['data_acquired'] is False


def test_v6_to_v8_migrations_back_up_and_preserve_data(tmp_path, monkeypatch):
    monkeypatch.setattr(db, 'DATA_DIR', tmp_path)
    monkeypatch.setattr(db, 'DB_PATH', tmp_path / 'upgrade.sqlite3')
    db.init_db()
    with db.connect() as conn:
        conn.execute("INSERT INTO securities(symbol,name) VALUES('ABC','Alpha Beta')")
        conn.execute("INSERT INTO prices(symbol,date,close,adjusted_close,source,retrieved_at,quality,price_basis) VALUES('ABC','2026-10-15',100,100,'local','now','USER_IMPORTED_UNVERIFIED','raw_unadjusted')")
        conn.execute('DROP TABLE acquisition_runs')
        conn.execute('DROP TABLE source_registry')
        conn.execute('DELETE FROM schema_migrations WHERE version>6')
    db.init_db()
    backups = list(tmp_path.glob('upgrade.backup-*.sqlite3'))
    assert len(backups) == 1
    backup = sqlite3.connect(backups[0])
    assert backup.execute('SELECT MAX(version) FROM schema_migrations').fetchone()[0] == 6
    assert backup.execute('SELECT COUNT(*) FROM prices WHERE quality<>\'DEMO_SYNTHETIC\'').fetchone()[0] == 1
    assert backup.execute("PRAGMA integrity_check").fetchone()[0] == 'ok'
    backup.close()
    with db.connect() as conn:
        assert conn.execute('SELECT MAX(version) FROM schema_migrations').fetchone()[0] == 11
        assert conn.execute("SELECT close FROM prices WHERE symbol='ABC'").fetchone()[0] == 100
        assert conn.execute("SELECT COUNT(*) FROM source_registry").fetchone()[0] == 0


def test_v7_to_v8_registry_migration_preserves_sources_and_adds_statuses(tmp_path, monkeypatch):
    monkeypatch.setattr(db, 'DATA_DIR', tmp_path)
    monkeypatch.setattr(db, 'DB_PATH', tmp_path / 'upgrade-v7.sqlite3')
    db.init_db()
    with db.connect() as conn:
        conn.execute('DELETE FROM schema_migrations WHERE version=8')
        for column in ('documentation_status','access_status','adapter_status','connectivity_status','access_cost_type','last_attempt_at','last_attempt_status'):
            conn.execute(f'ALTER TABLE source_registry DROP COLUMN {column}')
        conn.execute("INSERT INTO source_registry(source_key,name,owner,data_types_json,coverage_json,access_method,permission_status,authentication,pricing_summary,terms_url,docs_url,request_limits,attribution,notes,checked_at) VALUES('keep','Keep','Owner','[]','[]','local','uncertain','none','free','terms','docs','none','none','retained','today')")
        conn.execute('DELETE FROM schema_migrations WHERE version>7')
    db.init_db()
    backups=list(tmp_path.glob('upgrade-v7.backup-*.sqlite3'))
    assert len(backups)==1
    old=sqlite3.connect(backups[0])
    assert old.execute('SELECT MAX(version) FROM schema_migrations').fetchone()[0]==7
    old.close()
    with db.connect() as conn:
        row=conn.execute("SELECT * FROM source_registry WHERE source_key='keep'").fetchone()
        assert row['name']=='Keep'
        assert row['documentation_status']=='not_reviewed'
        assert row['adapter_status']=='not_implemented'
        assert conn.execute('SELECT MAX(version) FROM schema_migrations').fetchone()[0]==11
