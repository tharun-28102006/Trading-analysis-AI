import json
import numpy as np
import pandas as pd

from app import db, models
from app.calibration import calibration_status


def prepare(tmp_path,monkeypatch):
    monkeypatch.setattr(db,'DATA_DIR',tmp_path); monkeypatch.setattr(db,'DB_PATH',tmp_path/'orchestration.sqlite3'); db.init_db()


def test_batch_training_skips_ineligible_and_continues_after_company_failure(tmp_path,monkeypatch):
    prepare(tmp_path,monkeypatch)
    with db.connect() as c:
        c.executemany("INSERT INTO securities(symbol,name) VALUES(?,?)",[(s,s) for s in ('AAA','BBB','CCC','NIFTY50')])
        c.executemany("INSERT INTO instrument_universe(symbol,as_of_date,name,sector,exchange,quality) VALUES(?,'2026-10-01',?,NULL,'NSE','USER_CONFIRMED_UNVERIFIED')",[(s,s) for s in ('AAA','BBB','CCC')])
        c.executemany("INSERT INTO prices(symbol,date,close,adjusted_close,source,retrieved_at,quality,price_basis) VALUES(?,?,100,100,'fixture','now','USER_CONFIRMED_UNVERIFIED','raw_unadjusted')",[(s,'2026-01-01') for s in ('AAA','BBB','CCC')])
    monkeypatch.setattr(models,'training_eligibility',lambda s:{'eligible':s in {'AAA','BBB'},'reason':'insufficient test history'})
    def fake_train(symbol):
        if symbol=='BBB': raise RuntimeError('controlled per-company failure')
        return {'run_id':17,'model_version_id':31,'metrics':{'ridge':{'mae':0.1}}}
    monkeypatch.setattr(models,'train_company',fake_train)
    result=models.train_all_eligible_companies()
    assert result['status']=='partial_success'
    assert result['succeeded']==1 and result['failed']==1 and result['skipped']==1
    assert [x['symbol'] for x in result['results']]==['AAA','BBB']
    with db.connect() as c:
        run=c.execute('SELECT * FROM training_runs WHERE id=?',(result['run_id'],)).fetchone()
        items={r['symbol']:dict(r) for r in c.execute('SELECT * FROM training_run_items WHERE run_id=?',(result['run_id'],))}
    assert run['progress']==3 and run['total']==3
    assert json.loads(run['summary_json'])['completed']==2
    assert items['AAA']['status']=='succeeded' and items['AAA']['model_version_id']==31
    assert items['BBB']['status']=='failed' and 'controlled' in items['BBB']['reason']
    assert items['CCC']['status']=='skipped'


def test_batch_discovers_only_the_latest_imported_inventory(tmp_path,monkeypatch):
    prepare(tmp_path,monkeypatch)
    with db.connect() as c:
        c.executemany("INSERT INTO securities(symbol,name,exchange) VALUES(?,?,'NSE')",[('AAA','AAA'),('LEGACY','Legacy')])
        c.execute("INSERT INTO instrument_universe(symbol,as_of_date,name,exchange,quality) VALUES('AAA','2026-10-01','AAA','NSE','USER_IMPORTED_UNVERIFIED')")
        c.execute("INSERT INTO instrument_universe(symbol,as_of_date,name,exchange,quality) VALUES('BBB','2026-10-15','BBB','NSE','USER_IMPORTED_UNVERIFIED')")
    monkeypatch.setattr(models,'training_eligibility',lambda s:{'eligible':False,'reason':'no eligible price history'})
    result=models.train_all_eligible_companies()
    assert result['discovered']==1
    with db.connect() as c:
        symbols=[r[0] for r in c.execute('SELECT symbol FROM training_run_items WHERE run_id=?',(result['run_id'],))]
    assert symbols==['BBB']


def test_batch_reports_when_no_company_is_eligible(tmp_path,monkeypatch):
    prepare(tmp_path,monkeypatch)
    with db.connect() as c:
        c.execute("INSERT INTO securities(symbol,name) VALUES('AAA','AAA')")
        c.execute("INSERT INTO prices(symbol,date,close,adjusted_close,source,retrieved_at,quality,price_basis) VALUES('AAA','2026-01-01',100,100,'fixture','now','USER_CONFIRMED_UNVERIFIED','raw_unadjusted')")
    monkeypatch.setattr(models,'training_eligibility',lambda s:{'eligible':False,'reason':'not enough real history'})
    result=models.train_all_eligible_companies()
    assert result['status']=='no_eligible_companies' and result['succeeded']==0
    with db.connect() as c:
        assert c.execute('SELECT status FROM training_runs WHERE id=?',(result['run_id'],)).fetchone()[0]=='no_eligible_companies'


def test_calibration_uses_earlier_fit_and_later_purged_evaluation_dates(tmp_path,monkeypatch):
    prepare(tmp_path,monkeypatch)
    dates=pd.bdate_range('2020-01-01',periods=500)
    rng=np.random.default_rng(22)
    with db.connect() as c:
        run_id=c.execute("INSERT INTO training_runs(mode,status,started_at) VALUES('test','succeeded','now')").lastrowid
        for i,d in enumerate(dates):
            positive=i%2==0; actual=(0.01 if positive else -0.01)
            score=(0.01 if positive else -0.01)+float(rng.normal(0,.02))
            c.execute('INSERT INTO oos_predictions(training_run_id,model_key,symbol,origin_date,target_end_date,prediction,actual_excess,stock_return,benchmark_return,data_version,price_basis) VALUES(?,?,?,?,?,?,?,?,?,?,?)',
                      (run_id,'pooled:ridge','AAA',d.strftime('%Y-%m-%d'),(d+pd.offsets.BDay(5)).strftime('%Y-%m-%d'),score,actual,actual,0,'fixture-version','raw_unadjusted'))
    result=calibration_status('pooled:ridge')
    assert result['status']=='calibrated'
    assert result['fit_date_range'][1] < result['evaluation_date_range'][0]
    assert result['evaluation_reliability']['brier_score']>=0
    assert result['counts']['fit']>=200 and result['counts']['evaluation']>=100


def test_calibration_is_unavailable_without_prediction_stream(tmp_path,monkeypatch):
    prepare(tmp_path,monkeypatch)
    assert calibration_status('pooled:ridge')['status']=='not_calibrated'


def test_registered_model_and_current_forecast_keep_version_metadata(tmp_path,monkeypatch):
    prepare(tmp_path,monkeypatch)
    monkeypatch.setattr(models,'MODEL_DIR',tmp_path/'models')
    run_id=models._new_training_run('company:RELIANCE',800)
    metrics={'ridge':{'mae':0.01,'observations':120,'evaluation_date_range':['2024-01-01','2025-01-01']}}
    period={'start':'2020-01-01','end':'2025-01-01','price_basis':'adjusted_close'}
    version_id,digest,path=models._save_artifact(run_id,'company:RELIANCE','ridge',{'kind':'constant','value':0.0},'dataset-v1',['ret_5'],metrics,period)
    with db.connect() as c:
        c.execute('INSERT INTO current_forecasts(model_version_id,symbol,origin_date,horizon_sessions,prediction,price_basis,created_at) VALUES(?,?,?,?,?,?,?)',(version_id,'RELIANCE','2025-01-02',5,0.02,'adjusted_close','now'))
    artifact=models.load_model_version(version_id)
    forecast=models.latest_model()
    assert artifact['data_version']=='dataset-v1' and artifact['training_period']==period and artifact['price_basis']=='adjusted_close'
    assert forecast['model_version_id']==version_id and forecast['data_version']=='dataset-v1'
    assert forecast['feature_schema']==['ret_5'] and forecast['price_basis']=='adjusted_close'
    assert forecast['calibration']['status']=='not_calibrated'
    assert forecast['symbol']=='RELIANCE' and forecast['stale'] is True and forecast['stale_warning']


def test_latest_forecast_keeps_lineage_stale_flag_even_when_origin_is_recent(tmp_path,monkeypatch):
    from datetime import date
    from app import main
    prepare(tmp_path,monkeypatch)
    with db.connect() as c:
        run=c.execute("INSERT INTO training_runs(mode,status,started_at) VALUES('company','succeeded','now')").lastrowid
        version=c.execute("INSERT INTO dataset_versions(dataset_type,content_fingerprint,metadata_fingerprint,created_at,change_kind) VALUES('prices','current-fingerprint','meta','now','new_session')").lastrowid
        model=c.execute("INSERT INTO model_versions(training_run_id,model_key,model_type,artifact_path,artifact_sha256,data_version,feature_schema_json,config_json,metrics_json,training_period_json,status,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",(run,'company:AAA','ridge','unused','digest','old-fingerprint','[]','{}','{}','{}','eligible','now')).lastrowid
        c.execute('INSERT INTO current_forecasts(model_version_id,symbol,origin_date,horizon_sessions,prediction,price_basis,created_at,version_id,input_fingerprint,status) VALUES(?,?,?,?,?,?,?,?,?,?)',(model,'AAA',date.today().isoformat(),5,0.01,'raw_unadjusted','now',version,'old-fingerprint','current'))
    result=main.current_forecasts()
    forecast=result['forecasts'][0]
    assert forecast['stale'] is True
    assert forecast['status']=='stale'
    assert 'fingerprint' in forecast['stale_warning'].lower()
