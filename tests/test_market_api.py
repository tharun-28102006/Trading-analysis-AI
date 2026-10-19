import time

from fastapi.testclient import TestClient

from app import db, main, market


def setup_client(tmp_path, monkeypatch):
    monkeypatch.setattr(db,'DATA_DIR',tmp_path)
    monkeypatch.setattr(db,'DB_PATH',tmp_path/'api.sqlite3')
    monkeypatch.setattr(market,'DATA_DIR',tmp_path)
    monkeypatch.setenv('NIVESH_TEJ_DATA_DIR',str(tmp_path/'tej-empty'))
    monkeypatch.setenv('NIVESH_AUTO_REFRESH','false')
    monkeypatch.delenv('YAHOO_FINANCE_PERMISSION_CONFIRMED',raising=False)
    monkeypatch.delenv('KITE_API_KEY',raising=False)
    monkeypatch.delenv('KITE_ACCESS_TOKEN',raising=False)
    return TestClient(main.app)


def test_data_status_and_dashboard_show_empty_market_state(tmp_path,monkeypatch):
    with setup_client(tmp_path,monkeypatch) as client:
        status=client.get('/api/data/status')
        assert status.status_code==200
        assert status.json()['latest_market_observation_date'] is None
        home=client.get('/')
        assert home.status_code==200
        assert 'Import local data' in home.text
        assert 'No real market observations imported' in home.text
        for view in ('overview','watch','charts','portfolio','holdings','positions','orders','screener','predictions','risk','health','settings'):
            assert f'id="view-{view}"' in home.text
        assert 'No brokerage portfolio connected' in home.text
        assert 'No eligible prediction' in home.text


def test_refresh_endpoint_runs_free_provider_and_never_uses_kite(tmp_path,monkeypatch):
    from app.providers import tej
    from app import tej_bazaar_lifecycle
    class OfflineClient:
        def status(self): raise tej.TejError('TejHQ request failed (offline).')
    monkeypatch.setattr(tej,'TejClient',OfflineClient)
    monkeypatch.setattr(tej_bazaar_lifecycle,'TejClient',OfflineClient)
    with setup_client(tmp_path,monkeypatch) as client:
        response=client.post('/api/data/refresh')
        assert response.status_code==200
        assert response.json()['status']=='running'
        run_id=response.json()['run_id']
        deadline=time.time()+3
        while time.time()<deadline:
            result=client.get(f'/api/data/refresh/jobs/{run_id}').json()
            if result['status']!='running': break
            time.sleep(.01)
        assert result['status']=='failed'
        assert 'No NSE/BSE Tej Bazaar Parquet files' in result['errors'][0]
        assert client.get('/api/data/status').json()['has_real_market_data'] is False
        dashboard=client.get('/').text
        assert 'completed_files' in dashboard
        assert 'Last successful refresh' in dashboard
        assert 'Asia/Kolkata' in dashboard
        train=client.post('/api/models/reliance/train')
        assert train.status_code==200 and train.json()['status']=='queued'
        deadline=time.time()+2
        while time.time()<deadline:
            state=client.get('/api/models/training').json()
            if state['job']['status']=='failed': break
            time.sleep(.01)
        assert state['job']['status']=='failed'
        assert not state['model_versions']


def test_demo_prices_remain_stored_but_are_hidden_by_default(tmp_path,monkeypatch):
    with setup_client(tmp_path,monkeypatch) as client:
        market.seed_demo_data()
        assert client.get('/api/status').json()['demo_observations']>0
        assert client.get('/api/company/RELIANCE/prices').json()==[]
        result=client.post('/api/screen',json={'period':'1m','universe':'nifty50'}).json()
        assert result['results']==[]
