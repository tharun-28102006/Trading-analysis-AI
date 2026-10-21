import json
import urllib.error

import pytest

from app.providers.mospi import MoSPIClient, MoSPIError, parse_cpi, store_cpi


def test_cpi_parser_normalizes_documented_fields_and_keeps_publication_unknown():
    payload=json.dumps({'data':[{'year':2024,'month':'January','state':'All India','sector':'Combined',
        'group':'Food and beverages','subgroup':'Cereals','index':180.2,'inflation':8.4,'status':'F'}]}).encode()
    rows=parse_cpi(payload)
    assert rows[0]['year']==2024 and rows[0]['month']==1
    assert rows[0]['index_value']==180.2 and rows[0]['inflation']==8.4
    assert rows[0]['published_at'] is None


@pytest.mark.parametrize('row',[{}, {'year':2024,'month':'Bogus','index':1},
    {'year':2024,'month':1,'index':'NaN'}, {'year':2024,'month':1,'index':-1},
    {'year':2024,'month':1,'inflation':'bad'}])
def test_cpi_parser_rejects_invalid_rows(row):
    with pytest.raises(MoSPIError):
        parse_cpi(json.dumps({'data':[row]}).encode())


def test_mospi_client_uses_current_2024_endpoint_and_sanitizes_http_errors(tmp_path,monkeypatch):
    from app import config
    monkeypatch.setattr(config,'DATA_DIR',tmp_path)
    calls=[]
    payload=json.dumps({'data':[{'year':2026,'month':'January','index':150.2}]}).encode()
    def mock_request(url,**kwargs):
        calls.append((url,kwargs))
        return payload
    client=MoSPIClient(request=mock_request,sleeper=lambda _:None)
    result,rows=client.fetch_cpi()
    assert result==payload and rows[0]['index_value']==150.2
    assert '/api/cpi/getCPIData' in calls[0][0]
    assert 'base_year=2024' in calls[0][0] and 'series=Current' in calls[0][0]
    def unavailable(*args,**kwargs):
        raise urllib.error.HTTPError('https://api.mospi.gov.in/api/cpi/getCPIData',503,'Unavailable',{},None)
    client=MoSPIClient(request=unavailable,sleeper=lambda _:None)
    with pytest.raises(MoSPIError) as error:
        client.fetch_cpi()
    assert 'https://' not in str(error.value)


def test_cpi_storage_retains_raw_bytes_and_is_idempotent(tmp_path,monkeypatch):
    from app import config, db
    monkeypatch.setattr(config,'DATA_DIR',tmp_path)
    monkeypatch.setattr(db,'DATA_DIR',tmp_path)
    monkeypatch.setattr(db,'DB_PATH',tmp_path/'macro.sqlite3')
    db.init_db()
    payload=json.dumps({'data':[{'year':2024,'month':'January','state':'All India','sector':'Combined',
        'group':'Food','subgroup':'Cereals','index':180.2,'inflation':8.4,'status':'F'}]}).encode()
    rows=parse_cpi(payload)
    first=store_cpi(payload,rows); second=store_cpi(payload,rows)
    assert first['new_rows']==1 and second['new_rows']==0 and second['duplicate_rows']==1
    assert first['payload_sha256'] in first['raw_payload_path']
    assert (tmp_path/'provider_payloads'/'mospi'/f"{first['payload_sha256']}.json").read_bytes()==payload
    with db.connect() as conn:
        assert conn.execute('SELECT COUNT(*) FROM macro_observations').fetchone()[0]==1
        assert conn.execute('SELECT published_at FROM macro_observations').fetchone()[0] is None


def test_macro_refresh_uses_mocked_provider_without_claiming_equity_data(tmp_path,monkeypatch):
    from app import acquisition, config, db
    from app.providers import mospi
    monkeypatch.setattr(config,'DATA_DIR',tmp_path)
    monkeypatch.setattr(db,'DATA_DIR',tmp_path)
    monkeypatch.setattr(db,'DB_PATH',tmp_path/'refresh.sqlite3')
    db.init_db()
    payload=json.dumps({'data':[{'year':2024,'month':'January','state':'All India','sector':'Combined',
        'group':'Food','subgroup':'Cereals','index':180.2,'inflation':8.4,'status':'F'}]}).encode()
    class MockClient:
        def __init__(self): pass
        def fetch_cpi(self): return payload,mospi.parse_cpi(payload)
    monkeypatch.setattr(mospi,'MoSPIClient',MockClient)
    result=acquisition.run_acquisition('test')
    assert result['status']=='succeeded'
    assert result['equity_market_data_acquired'] is False
    assert result['categories'][0]['new_rows']==1
    repeat=acquisition.run_acquisition('test')
    assert repeat['status']=='succeeded' and repeat['categories'][0]['new_rows']==0


def test_schema_v9_migration_backs_up_before_adding_macro_store(tmp_path,monkeypatch):
    import sqlite3
    from app import db
    monkeypatch.setattr(db,'DATA_DIR',tmp_path)
    monkeypatch.setattr(db,'DB_PATH',tmp_path/'upgrade.sqlite3')
    db.init_db()
    with db.connect() as conn:
        conn.execute('DROP TABLE macro_observations')
        conn.execute('DELETE FROM schema_migrations WHERE version>=10')
    db.init_db()
    backups=list(tmp_path.glob('upgrade.backup-*.sqlite3'))
    assert len(backups)==1
    backup=sqlite3.connect(backups[0])
    assert backup.execute('SELECT MAX(version) FROM schema_migrations').fetchone()[0]==9
    assert backup.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
    assert not backup.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='macro_observations'").fetchone()
    backup.close()
    with db.connect() as conn:
        assert conn.execute('SELECT MAX(version) FROM schema_migrations').fetchone()[0]==11
        assert conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='macro_observations'").fetchone()
