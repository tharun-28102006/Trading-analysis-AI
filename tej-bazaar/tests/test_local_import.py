import sqlite3
from pathlib import Path
import io
import zipfile

import pytest
from fastapi.testclient import TestClient

from app import db, main, market


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(db, 'DATA_DIR', tmp_path)
    monkeypatch.setattr(db, 'DB_PATH', tmp_path / 'local.sqlite3')
    monkeypatch.setattr(market, 'DATA_DIR', tmp_path)
    monkeypatch.setenv('NIVESH_AUTO_REFRESH', 'false')
    with TestClient(main.app) as c:
        yield c


def csv(rows=None):
    return ("symbol,exchange,date,open,high,low,close,volume,price_basis,source,adjusted_close,dividends,stock_splits\n" +
            (rows or "RELIANCE,NSE,2026-01-05,100,103,99,102,1000,adjusted_close,LicensedPersonalDataset,101,0,0\n"))


def upload(client, body):
    return client.post('/api/data/import-csv', files={'file': ('prices.csv', body.encode(), 'text/csv')})


def test_valid_import_preserves_basis_provenance_and_original(client, tmp_path):
    response = upload(client, csv())
    assert response.status_code == 200, response.text
    data = response.json()
    assert data['rows_imported'] == 1 and data['quality'] == 'USER_IMPORTED_UNVERIFIED'
    assert data['original_file']
    with db.connect() as conn:
        row = conn.execute("SELECT source,quality,price_basis FROM prices WHERE symbol='RELIANCE'").fetchone()
        assert tuple(row) == ('user_csv:LicensedPersonalDataset', 'USER_IMPORTED_UNVERIFIED', 'adjusted_close')
        assert conn.execute('SELECT COUNT(*) FROM raw_price_observations').fetchone()[0] == 1


def test_user_confirmation_is_recorded_as_attestation_not_independent_verification(client):
    response=client.post('/api/data/import-csv',files={'file':('prices.csv',csv().encode(),'text/csv')},data={'dataset_description':'Personal research sample','license_reference':'https://provider.example/terms','user_confirmed_use':'true'})
    assert response.status_code==200,response.text
    assert response.json()['quality']=='USER_CONFIRMED_UNVERIFIED'
    run=client.get('/api/data/imports').json()[0]
    assert run['user_confirmed_use']==1
    assert run['license_reference']=='https://provider.example/terms'
    assert client.get('/api/data/status').json()['has_verified_market_data'] is False


def test_repeated_file_is_idempotent(client):
    body = csv()
    assert upload(client, body).json()['rows_imported'] == 1
    again = upload(client, body).json()
    assert again['idempotent'] is True
    with db.connect() as conn:
        assert conn.execute('SELECT COUNT(*) FROM prices').fetchone()[0] == 1
        assert conn.execute('SELECT COUNT(*) FROM data_import_runs').fetchone()[0] == 1


@pytest.mark.parametrize('body,detail', [
    ('symbol,date,close\nRELIANCE,2026-01-05,100\n', 'Missing required columns'),
    (csv('RELIANCE,NSE,not-a-date,100,103,99,102,1000,raw_unadjusted,LicensedDataset,,0,0\n'), 'Invalid observation date'),
    (csv('RELIANCE,NSE,2026-01-05,100,99,99,102,1000,raw_unadjusted,LicensedDataset,,0,0\n'), 'Invalid OHLC'),
    (csv('RELIANCE,NSE,2026-01-05,100,103,99,102,1000,raw_unadjusted,ILLUSTRATIVE_SYNTHETIC_EXAMPLE,,0,0\n'), 'not importable'),
])
def test_invalid_files_are_rejected_and_recorded(client, body, detail):
    response = upload(client, body)
    assert response.status_code == 422
    assert detail.lower() in response.json()['detail'].lower()
    assert client.get('/api/data/imports').json()[-1]['errors_json'] != '[]'


def test_illustrative_template_is_clearly_marked_and_rejected(client):
    template = Path('data_templates/prices_illustrative_NOT_FOR_IMPORT.csv').read_bytes()
    assert b'ILLUSTRATIVE' in template and b'ILLUSTRATIVE_SYNTHETIC_EXAMPLE' in template
    response = client.post('/api/data/import-csv', files={'file': ('example.csv', template, 'text/csv')})
    assert response.status_code == 422
    assert 'not importable' in response.json()['detail']


def test_duplicates_are_rejected_without_silent_dedup(client):
    row = 'RELIANCE,NSE,2026-01-05,100,103,99,102,1000,raw_unadjusted,LicensedDataset,,0,0\n'
    assert 'Duplicate symbol/date' in upload(client, csv(row + row)).json()['detail']


def test_gap_warning_and_raw_basis_are_reported(client):
    rows = ('RELIANCE,NSE,2026-01-05,100,103,99,102,1000,raw_unadjusted,LicensedDataset,,0,0\n'
            'RELIANCE,NSE,2026-01-12,102,104,101,103,1000,raw_unadjusted,LicensedDataset,,0,0\n')
    result = upload(client, csv(rows)).json()
    assert result['rows_imported'] == 2
    assert any('gap' in w['warning'] for w in result['warnings'])
    with db.connect() as conn:
        assert conn.execute("SELECT DISTINCT price_basis FROM prices").fetchone()[0] == 'raw_unadjusted'


def test_demo_overlap_is_archived_before_real_import_replaces_active_row(client):
    with db.connect() as conn:
        conn.execute("INSERT INTO securities(symbol,name) VALUES('RELIANCE','Reliance')")
        conn.execute("INSERT INTO prices(symbol,date,open,high,low,close,adjusted_close,volume,source,retrieved_at,quality,price_basis) VALUES('RELIANCE','2026-01-05',99,103,98,100,100,1000,'synthetic_demo','test','DEMO_SYNTHETIC','synthetic')")
    result = upload(client, csv()).json()
    assert result['rows_imported'] == 1
    with db.connect() as conn:
        assert conn.execute("SELECT quality FROM prices WHERE symbol='RELIANCE' AND date='2026-01-05'").fetchone()[0] == 'USER_IMPORTED_UNVERIFIED'
        assert conn.execute("SELECT COUNT(*) FROM archived_demo_prices WHERE symbol='RELIANCE' AND date='2026-01-05'").fetchone()[0] == 1


def test_failed_storage_transaction_preserves_rows_and_marks_run_failed(client):
    with db.connect() as conn:
        conn.execute("CREATE TRIGGER reject_test BEFORE INSERT ON prices BEGIN SELECT RAISE(ABORT,'forced failure'); END")
    response = upload(client, csv())
    assert response.status_code == 422
    with db.connect() as conn:
        assert conn.execute('SELECT COUNT(*) FROM prices').fetchone()[0] == 0
        run = conn.execute('SELECT status,errors_json FROM data_import_runs').fetchone()
        assert run['status'] == 'failed' and 'forced failure' in run['errors_json']


def test_health_reports_coverage_source_quality_and_training_block(client):
    assert upload(client, csv()).status_code == 200
    health = client.get('/api/data/health').json()
    reliance = next(x for x in health['instrument_coverage'] if x['symbol'] == 'RELIANCE')
    assert reliance['rows'] == 1
    assert reliance['training_eligible'] is False
    assert reliance['earliest_date'] == '2026-01-05'
    assert health['has_non_synthetic_observations'] is True
    assert health['has_verified_market_data'] is False


def test_zip_bundle_imports_csv_members_and_retains_zip(client):
    archive=io.BytesIO()
    with zipfile.ZipFile(archive,'w') as z: z.writestr('prices.csv',csv())
    response=client.post('/api/data/import-zip',files={'file':('bundle.zip',archive.getvalue(),'application/zip')})
    assert response.status_code==200,response.text
    assert response.json()['files_imported']==1 and response.json()['rows_imported']==1
    assert Path(response.json()['bundle_file']).exists()


def test_zip_rejects_excessive_entry_count(tmp_path, monkeypatch):
    monkeypatch.setattr(db,'DATA_DIR',tmp_path)
    monkeypatch.setattr(db,'DB_PATH',tmp_path/'zip.sqlite3')
    db.init_db()
    archive=io.BytesIO()
    with zipfile.ZipFile(archive,'w') as z:
        for i in range(1001): z.writestr(f'{i}.txt','')
    with pytest.raises(ValueError,match='1,000 entries'):
        market.import_price_zip(archive.getvalue(),'many.zip')


def test_dated_constituent_snapshot_does_not_create_historical_membership(client):
    names='symbol,name,sector\n'+''.join(f'SYM{i:02d},Company {i},Sector\n' for i in range(50))
    response=client.post('/api/data/import-constituents',files={'file':('nifty.csv',names.encode(),'text/csv')},data={'as_of_date':'2026-09-30','source_name':'User supplied NSE Indices constituent file','license_reference':'terms reference','user_confirmed_use':'true'})
    assert response.status_code==200,response.text
    data=response.json()
    assert data['rows_imported']==50 and data['historical_membership_created'] is False
    assert client.get('/api/universe/constituents').json()['as_of_date']=='2026-09-30'
    with db.connect() as conn: assert conn.execute('SELECT COUNT(*) FROM constituent_membership').fetchone()[0]==0


def test_broader_instrument_and_calendar_imports_are_dated_and_diagnostic(client):
    instruments='symbol,name,exchange,sector\nABC,Alpha Beta Corp,NSE,Technology\n'
    response=client.post('/api/data/import-instruments',files={'file':('nse_list.csv',instruments.encode(),'text/csv')},data={'as_of_date':'2026-01-07','source_name':'Permitted local instrument list','license_reference':'terms','user_confirmed_use':'true'})
    assert response.status_code==200,response.text
    inventory=client.get('/api/universe/instruments').json()
    assert inventory['imported_instruments']==1 and any(row['symbol']=='ABC' for row in inventory['instruments'])
    # Imported inventory entries are available in securities/universe even with no prices.
    assert any(row['symbol']=='ABC' for row in client.get('/api/universe').json())
    prices='symbol,exchange,date,open,high,low,close,volume,price_basis,source\nABC,NSE,2026-01-05,10,11,9,10,100,raw_unadjusted,PermittedData\nABC,NSE,2026-01-07,10,11,9,10,100,raw_unadjusted,PermittedData\n'
    assert upload(client,prices).status_code==200
    calendar='exchange,date\nNSE,2026-01-05\nNSE,2026-01-06\nNSE,2026-01-07\n'
    response=client.post('/api/data/import-exchange-calendar',files={'file':('sessions.csv',calendar.encode(),'text/csv')},data={'source_name':'Permitted local session reference','license_reference':'terms','user_confirmed_use':'true'})
    assert response.status_code==200,response.text
    data=client.get('/api/data/status').json()
    coverage=next(row for row in data['instrument_coverage'] if row['symbol']=='ABC')
    assert coverage['missing_session_count']==1


def test_downloadable_illustrative_universe_templates_are_rejected(client):
    response=client.get('/api/data/templates/constituents')
    assert response.status_code==200 and 'ILLUSTRATIVE_NOT_IMPORTABLE' in response.text
    result=client.post('/api/data/import-constituents',files={'file':('constituents_illustrative_NOT_FOR_IMPORT.csv',response.content,'text/csv')},data={'as_of_date':'2026-01-01','source_name':'example','license_reference':'none'})
    assert result.status_code==422 and 'illustrative' in result.text.lower()


def test_historical_membership_intervals_are_effective_dated_and_non_overlapping(client):
    body='symbol,valid_from,valid_to\nABC,2020-01-01,2021-12-31\nABC,2022-01-01,\n'
    response=client.post('/api/data/import-membership-history',files={'file':('membership.csv',body.encode(),'text/csv')},data={'source_name':'Permitted historical constituent source','license_reference':'terms','user_confirmed_use':'true'})
    assert response.status_code==200,response.text
    with db.connect() as c:
        rows=c.execute("SELECT valid_from,valid_to,quality FROM constituent_membership WHERE symbol='ABC' ORDER BY valid_from").fetchall()
    assert [(r['valid_from'],r['valid_to']) for r in rows]==[('2020-01-01','2021-12-31'),('2022-01-01',None)]
    assert all(r['quality']=='USER_CONFIRMED_UNVERIFIED' for r in rows)
    overlap='symbol,valid_from,valid_to\nABC,2021-06-01,2022-02-01\n'
    rejected=client.post('/api/data/import-membership-history',files={'file':('overlap.csv',overlap.encode(),'text/csv')},data={'source_name':'Second source','license_reference':'terms','user_confirmed_use':'true'})
    assert rejected.status_code==422 and 'overlap' in rejected.text.lower()


def test_batch_training_run_id_is_persisted_before_request_returns(client):
    response=client.post('/api/models/train-all-companies')
    assert response.status_code==200,response.text
    run_id=response.json()['run_id']
    assert run_id is not None
    detail=client.get(f'/api/models/training/{run_id}')
    assert detail.status_code==200 and detail.json()['run']['mode']=='company_batch'


def test_paper_trade_never_creates_order_without_reference_observation(client):
    assert client.post('/api/portfolio/paper-trades',json={'symbol':'RELIANCE','side':'BUY','quantity':1}).status_code==409
    response=client.get('/api/portfolio').json()
    assert response['brokerage_connected'] is False and response['trading_enabled'] is False


def test_later_row_failure_rolls_back_new_rows_but_keeps_previous_valid_data(client):
    assert upload(client, csv()).status_code == 200
    with db.connect() as conn:
        conn.execute("CREATE TRIGGER reject_hdfc BEFORE INSERT ON prices WHEN NEW.symbol='HDFCBANK' BEGIN SELECT RAISE(ABORT,'forced failure'); END")
    rows = ('TCS,NSE,2026-01-06,100,103,99,102,1000,raw_unadjusted,LicensedDataset,,0,0\n'
            'HDFCBANK,NSE,2026-01-06,100,103,99,102,1000,raw_unadjusted,LicensedDataset,,0,0\n')
    assert upload(client, csv(rows)).status_code == 422
    with db.connect() as conn:
        assert conn.execute('SELECT COUNT(*) FROM prices').fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM prices WHERE symbol='TCS'").fetchone()[0] == 0
