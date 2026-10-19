from contextlib import asynccontextmanager
import asyncio
import hmac
import json
import logging
import os
from pathlib import Path
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from . import alerts
from . import acquisition
from . import market
from .analytics import company_research, compare_benchmark, screen
from .assistant import execute_command
from .config import APP_NAME
from .db import active_release_id, audit, connect, init_db, pin_published_release
from .market import get_data_status, import_csv, import_events_csv, import_fundamentals_csv, import_price_zip, import_constituents_csv, import_membership_history_csv, import_instruments_csv, import_exchange_calendar_csv
from .models import latest_model, start_training, training_status
from .ranking import rank
from .features import build_dataset, training_eligibility
from .backtest import run_backtest, backtest_history

logger=logging.getLogger('nivesh.alerts')

async def alert_scheduler():
    while True:
        await asyncio.sleep(900)
        try: await asyncio.to_thread(alerts.evaluate_alerts)
        except Exception: logger.exception('Scheduled alert evaluation failed')
        if os.getenv('NIVESH_DAILY_REFRESH_ENABLED','1').strip().lower() in {'1','true','yes'}:
            try:
                hour=min(max(int(os.getenv('NIVESH_DAILY_REFRESH_HOUR','21')),0),23)
                minute=min(max(int(os.getenv('NIVESH_DAILY_REFRESH_MINUTE','30')),0),59)
                local_now=datetime.now(ZoneInfo('Asia/Kolkata'))
                if (local_now.hour,local_now.minute)>=(hour,minute):
                    from pipeline.holidays import is_trading_day
                    if not is_trading_day(local_now.date(),'NSE'):
                        continue
                    with connect() as conn:
                        row=conn.execute('SELECT started_at,finished_at,status FROM refresh_runs WHERE source=? ORDER BY id DESC LIMIT 1',(market.TEJ_BAZAAR_SOURCE,)).fetchone()
                    last_day=datetime.fromisoformat(row['started_at']).astimezone(ZoneInfo('Asia/Kolkata')).date() if row else None
                    terminal_success=bool(row and row['status'] in {'succeeded','unchanged'} and last_day==local_now.date())
                    last_finish=datetime.fromisoformat(row['finished_at']).astimezone(ZoneInfo('Asia/Kolkata')) if row and row['finished_at'] else None
                    retry_due=last_day!=local_now.date() or last_finish is None or (local_now-last_finish).total_seconds()>=1800
                    if not terminal_success and retry_due:
                        market.start_refresh('scheduled')
            except Exception:
                logger.exception('Scheduled daily market-data refresh failed')

@asynccontextmanager
async def lifespan(app):
    init_db()
    acquisition.sync_source_registry()
    task=asyncio.create_task(alert_scheduler())
    try: yield
    finally:
        task.cancel()
        try: await task
        except asyncio.CancelledError: pass

app=FastAPI(title=APP_NAME,version='0.1.0',description='Local-first, evidence-labeled Indian equity research assistant',lifespan=lifespan)
app.add_middleware(CORSMiddleware,allow_origins=['http://127.0.0.1:8000','http://localhost:8000'],allow_methods=['GET','POST','PUT','DELETE'],allow_headers=['*'])
@app.middleware('http')
async def optional_api_auth(request, call_next):
    # Local-only use needs no token. Set NIVESH_API_TOKEN before any network exposure.
    token=os.getenv('NIVESH_API_TOKEN')
    if token and request.url.path.startswith('/api/'):
        supplied=request.headers.get('authorization','')
        if not hmac.compare_digest(supplied, 'Bearer '+token):
            from fastapi.responses import JSONResponse
            return JSONResponse({'detail':'Authentication required. Provide the configured local API token.'},status_code=401)
    return await call_next(request)
STATIC=Path(__file__).parent/'static'

class ScreenRequest(BaseModel):
    period:str='1m'; min_return:float|None=None; min_excess:float|None=None; max_volatility:float|None=None; max_drawdown:float|None=None; universe:str='nifty50'
class CommandRequest(BaseModel): command:str=Field(max_length=1000)
class WatchRequest(BaseModel): symbol:str
class AlertRequest(BaseModel):
    symbol:str; condition:str; threshold:float; period:str='1m'; name:str|None=None
class PortfolioPositionRequest(BaseModel): symbol:str; quantity:float=Field(ge=0); cost_basis:float=Field(gt=0); notes:str=''
class PaperTradeRequest(BaseModel): symbol:str; side:str; quantity:float=Field(gt=0); notes:str=''
class BacktestRequest(BaseModel): model_key:str='pooled:ridge'; top_n:int=Field(default=5,ge=1,le=50); bottom_n:int=Field(default=0,ge=0,le=50); max_weight:float=Field(default=.2,gt=0,le=1); transaction_cost_bps:float=Field(default=5,ge=0,le=500); slippage_bps:float=Field(default=5,ge=0,le=500)

@app.get('/')
def home(): return FileResponse(STATIC/'index.html')
@app.get('/manifest.webmanifest')
def manifest(): return FileResponse(STATIC/'manifest.webmanifest',media_type='application/manifest+json')
@app.get('/sw.js')
def service_worker(): return FileResponse(STATIC/'sw.js',media_type='application/javascript')
@app.get('/icon.svg')
def icon(): return FileResponse(STATIC/'icon.svg',media_type='image/svg+xml')
@app.get('/api/status')
def status():
    with connect() as c:
        provenance_filter=" WHERE quality<>'DEMO_SYNTHETIC'"
        # Use the canonical tables and persisted acquisition counts. Snapshot views
        # intentionally have only release_id indexes and full scans here make the
        # dashboard status endpoint unusably slow after a large first import.
        nse_market_rows=c.execute("SELECT rows_imported FROM market_exchange_refresh WHERE exchange='NSE'").fetchone()
        counts={'securities':c.execute('SELECT COUNT(*) FROM securities').fetchone()[0],
                'prices':nse_market_rows[0] if nse_market_rows else c.execute("SELECT COUNT(*) FROM prices WHERE quality<>'DEMO_SYNTHETIC'").fetchone()[0],
                'fundamentals':c.execute("SELECT COUNT(*) FROM fundamentals WHERE quality<>'DEMO_SYNTHETIC'").fetchone()[0],
                'events':c.execute("SELECT COUNT(*) FROM events WHERE quality<>'DEMO_SYNTHETIC'").fetchone()[0],
                'macro_observations':c.execute('SELECT COUNT(*) FROM macro_observations').fetchone()[0],
                'models':c.execute("SELECT COUNT(*) FROM model_versions WHERE status='eligible'").fetchone()[0],
                'legacy_model_records':c.execute('SELECT COUNT(*) FROM models').fetchone()[0]}
        demo=c.execute("SELECT COUNT(*) FROM prices WHERE quality='DEMO_SYNTHETIC'").fetchone()[0]
        stored_counts={t:c.execute(f'SELECT COUNT(*) FROM {t}').fetchone()[0] for t in ['prices','fundamentals','events']}
    return {'name':APP_NAME,'status':'ok','database':'sqlite','counts':counts,'stored_counts':stored_counts,'demo_observations':demo,'synthetic_demo_mode':False,'isolated_demo_observations':demo,'trading_enabled':False,'message':'Local research only. Synthetic observations are excluded from analytics and training.'}
@app.get('/api/data/status')
def market_data_status(): return get_data_status()
@app.get('/api/data/refreshes')
def refresh_history():
    with connect() as c: rows=[dict(r) for r in c.execute('SELECT * FROM refresh_runs ORDER BY id DESC LIMIT 50')]
    return {'runs':rows}
@app.get('/api/data/releases')
def release_history():
    with connect() as c:
        active=c.execute("SELECT value FROM app_state WHERE key='active_release_id'").fetchone()
        rows=[dict(r) for r in c.execute('SELECT * FROM research_releases ORDER BY id DESC LIMIT 50')]
    return {'active_release_id':int(active['value']) if active else None,'releases':rows}
@app.get('/api/data/sources')
def data_sources(): return {'checked_at':acquisition.CHECKED_AT,'sources':acquisition.list_sources()}
@app.get('/api/data/macro/cpi')
def macro_cpi(limit:int=500):
    limit=min(max(limit,1),2000)
    with connect() as conn:
        rows=[dict(row) for row in conn.execute('''SELECT source,dataset,year,month,geography,sector,group_name,subgroup,
            index_value,inflation,status,published_at,retrieved_at,payload_hash FROM macro_observations
            ORDER BY year DESC,month DESC,geography,sector,group_name,subgroup LIMIT ?''',(limit,))]
        summary=conn.execute("SELECT COUNT(*) rows,MIN(year*100+month) first_period,MAX(year*100+month) last_period,COUNT(*)-COUNT(published_at) missing_publication_time FROM macro_observations WHERE source='MoSPI CPI API'").fetchone()
    return {'dataset':'MoSPI CPI current series, 2024 base','rows':rows,'coverage':dict(summary),
            'model_eligible':False,
            'note':'CPI is stored separately and is not yet integrated into the feature pipeline. Rows without source-supplied publication timestamps are not point-in-time model inputs.'}
@app.get('/api/data/acquisition/runs')
def acquisition_runs(limit:int=50): return {'runs':acquisition.acquisition_history(limit)}
@app.post('/api/data/acquisition/refresh')
def acquisition_refresh(reconcile_history:bool=False): return acquisition.run_acquisition('dashboard',reconcile_history)
@app.get('/api/data/health')
def market_data_health(): return get_data_status()
@app.get('/api/data/imports')
def import_history(): return get_data_status()['import_history']
@app.get('/api/data/templates/{template_name}')
def data_template(template_name:str):
    allowed={'prices':'prices_illustrative_NOT_FOR_IMPORT.csv','constituents':'constituents_illustrative_NOT_FOR_IMPORT.csv','membership':'membership_illustrative_NOT_FOR_IMPORT.csv','instruments':'instruments_illustrative_NOT_FOR_IMPORT.csv','calendar':'exchange_sessions_illustrative_NOT_FOR_IMPORT.csv'}
    if template_name not in allowed: raise HTTPException(404,'Template not found.')
    return FileResponse(Path(__file__).parent.parent/'data_templates'/allowed[template_name],filename=allowed[template_name],media_type='text/csv')
@app.post('/api/data/refresh')
def market_data_refresh(reconcile_history:bool=False): return market.start_refresh('manual',reconcile_history)
@app.get('/api/data/refresh/jobs/{run_id}')
def market_data_refresh_job(run_id:int):
    try: return market.refresh_job(run_id)
    except ValueError as exc: raise HTTPException(404,str(exc))
@app.get('/api/universe')
def universe():
    sql="SELECT s.symbol,s.name,s.sector,s.exchange,CASE WHEN cs.symbol IS NOT NULL THEN 1 ELSE 0 END AS index_member,s.is_demo,s.instrument_id,s.provider_symbol,COUNT(p.date) observations,MAX(p.date) latest_date,GROUP_CONCAT(DISTINCT p.source) sources FROM published_securities s LEFT JOIN published_prices p ON p.symbol=s.symbol AND p.quality<>'DEMO_SYNTHETIC' LEFT JOIN published_constituent_snapshots cs ON cs.symbol=s.symbol AND cs.as_of_date=(SELECT MAX(as_of_date) FROM published_constituent_snapshots) AND cs.quality<>'DEMO_SYNTHETIC' GROUP BY s.symbol ORDER BY index_member DESC,s.symbol"
    with connect() as c: return [dict(r) for r in c.execute(sql)]
@app.post('/api/screen')
def run_screen(body:ScreenRequest):
    try:
        result=screen(**body.model_dump()); audit('screen',body.model_dump()); return result
    except ValueError as e: raise HTTPException(422,str(e))
@app.post('/api/query')
def query(body:CommandRequest):
    result=execute_command(body.command); audit('nl_query',{'command':body.command,'status':result['status']}); return result
@app.get('/api/company/{symbol}')
def company(symbol:str,period:str='3m'): return company_research(symbol,period)
@app.get('/api/company/{symbol}/prices')
def prices(symbol:str,limit:int=260):
    with connect() as c: rows=c.execute("SELECT date,close,adjusted_close,volume,source,quality,price_basis FROM published_prices WHERE symbol=? AND quality<>'DEMO_SYNTHETIC' ORDER BY date DESC LIMIT ?",(symbol.upper(),min(max(limit,1),1500))).fetchall()
    return list(reversed([dict(r) for r in rows]))
@app.post('/api/models/reliance/train')
def train():
    try: return start_training('company','RELIANCE')
    except ValueError as e: raise HTTPException(409,str(e))
@app.get('/api/models/reliance/latest')
def model_latest(): return latest_model()
@app.get('/api/models/latest')
def model_latest_for_symbol(symbol:str|None=None): return latest_model(symbol)
@app.post('/api/models/train-universe')
def model_universe_train():
    try: return start_training('pooled')
    except ValueError as e: raise HTTPException(409,str(e))
@app.get('/api/models/training')
def models_training_status(): return training_status()
@app.get('/api/models/calibration')
def model_calibration(model_key:str='company:ridge'):
    from .calibration import calibration_status
    return calibration_status(model_key)
@app.post('/api/models/train-company/{symbol}')
def company_train(symbol:str):
    try: return start_training('company',symbol.upper())
    except ValueError as e: raise HTTPException(409,str(e))
@app.post('/api/models/train-aggregator')
def aggregator_train():
    try: return start_training('aggregator')
    except ValueError as e: raise HTTPException(409,str(e))
@app.post('/api/models/train-all-companies')
def company_batch_train():
    try: return start_training('company_batch')
    except ValueError as e: raise HTTPException(409,str(e))
@app.get('/api/models/training/{run_id}')
def training_run_detail(run_id:int):
    with connect() as c:
        run=c.execute('SELECT * FROM training_runs WHERE id=?',(run_id,)).fetchone()
        if not run: raise HTTPException(404,'Training run not found.')
        items=[dict(r) for r in c.execute('SELECT i.*,v.model_type AS selected_model,v.metrics_json AS registered_metrics_json FROM training_run_items i LEFT JOIN model_versions v ON v.id=i.model_version_id WHERE i.run_id=? ORDER BY i.symbol',(run_id,))]
    return {'run':dict(run),'items':items}
@app.get('/api/models/compare')
def model_compare():
    with connect() as c: rows=c.execute('SELECT id,model_key,model_type,data_version,feature_schema_json,metrics_json,training_period_json,status,created_at FROM model_versions ORDER BY id DESC').fetchall()
    return [{'id':r['id'],'model_key':r['model_key'],'model_type':r['model_type'],'data_version':r['data_version'],'feature_schema':json.loads(r['feature_schema_json']),'metrics':json.loads(r['metrics_json']),'training_period':json.loads(r['training_period_json']),'price_basis':json.loads(r['training_period_json']).get('price_basis','unknown'),'status':r['status'],'created_at':r['created_at']} for r in rows]
@app.get('/api/predictions/latest')
def current_forecasts():
    with connect() as c:
        rows=c.execute("SELECT f.*,v.model_key,v.model_type,v.data_version,v.feature_schema_json,v.training_period_json,v.metrics_json FROM current_forecasts f JOIN model_versions v ON v.id=f.model_version_id WHERE v.status='eligible' ORDER BY f.symbol,f.origin_date DESC,f.id DESC").fetchall()
        current_version=c.execute("SELECT content_fingerprint FROM dataset_versions WHERE dataset_type='prices' ORDER BY id DESC LIMIT 1").fetchone()
    current_fingerprint=current_version['content_fingerprint'] if current_version else None
    latest={}
    for row in rows: latest.setdefault(row['symbol'],dict(row))
    from .calibration import calibrated_probability
    for forecast in latest.values():
        fingerprint_matches=bool(current_fingerprint and forecast.get('input_fingerprint')==current_fingerprint)
        try: origin_stale=(datetime.now().date()-datetime.strptime(forecast['origin_date'],'%Y-%m-%d').date()).days>5
        except ValueError: origin_stale=True
        forecast['stale']=forecast.get('status')!='current' or not fingerprint_matches or origin_stale
        if forecast['stale']:
            forecast['status']='stale'
            forecast['stale_warning']=forecast.get('stale_reason') or ('Forecast input fingerprint does not match the latest canonical price version.' if not fingerprint_matches else 'Forecast origin is more than five calendar days old.')
        else: forecast['stale_warning']=None
        oos_key='master_aggregator' if forecast['model_key']=='master_aggregator' else ('company:' if forecast['model_key'].startswith('company:') else 'pooled:')+forecast['model_type']
        forecast['calibration']=calibrated_probability(oos_key,forecast['prediction'])
        # Preserve lineage and status failures calculated above; age is an
        # additional stale condition, never a replacement for them.
        try: origin_old=(datetime.now().date()-datetime.strptime(forecast['origin_date'],'%Y-%m-%d').date()).days>5
        except ValueError: origin_old=True
        forecast['stale']=bool(forecast['stale'] or origin_old)
        if forecast['stale']:
            forecast['status']='stale'
            forecast['stale_warning']=forecast.get('stale_warning') or 'Forecast origin is more than five calendar days old; do not treat as current.'
        for source,target in [('feature_schema_json','feature_schema'),('training_period_json','training_period'),('metrics_json','metrics')]: forecast[target]=json.loads(forecast.pop(source))
    return {'available':bool(latest),'forecasts':list(latest.values()),'note':'Expected excess return and calibrated probability of positive excess are separate measures. Calibration remains unavailable until chronological OOS evidence is sufficient.'}
@app.get('/api/predictions/oos')
def historical_predictions(model_key:str|None=None,limit:int=1000):
    with connect() as c:
        if model_key: rows=c.execute('SELECT * FROM oos_predictions WHERE model_key=? ORDER BY origin_date DESC,symbol LIMIT ?',(model_key,min(max(limit,1),5000))).fetchall()
        else: rows=c.execute('SELECT * FROM oos_predictions ORDER BY origin_date DESC,symbol LIMIT ?',(min(max(limit,1),5000),)).fetchall()
    return {'predictions':[dict(r) for r in rows],'count':len(rows)}
@app.get('/api/features/{symbol}')
def feature_preview(symbol:str,limit:int=50):
    try:
        frame,meta=build_dataset(symbol.upper()); rows=frame.tail(min(max(limit,1),250)).reset_index(drop=True).replace({float('nan'):None}).to_dict(orient='records')
        return {'available':True,'metadata':meta,'rows':rows,'eligibility':training_eligibility(symbol)}
    except ValueError as e: return {'available':False,'symbol':symbol.upper(),'reason':str(e),'eligibility':training_eligibility(symbol)}
@app.post('/api/backtests')
def backtest_create(body:BacktestRequest):
    try:
        result=run_backtest(**body.model_dump()); audit('backtest',{'status':result['status'],'model_key':body.model_key}); return result
    except ValueError as e: raise HTTPException(422,str(e))
@app.get('/api/backtests')
def backtests_list(): return backtest_history()
@app.get('/api/risk/{symbol}')
def instrument_risk(symbol:str,period:str='1y'):
    result=compare_benchmark(symbol.upper(),period)
    return {'available':result.get('available',False),'symbol':symbol.upper(),'period':period,'source':result.get('source'),'price_basis':result.get('price_basis'),'return':result.get('return'),'annualized_volatility':result.get('annualized_volatility'),'maximum_drawdown':result.get('max_drawdown'),'benchmark_return':result.get('benchmark_return'),'excess_return':result.get('simple_excess_return'),'reason':result.get('reason')}
@app.post('/api/rank')
def aggregate(body:ScreenRequest):
    try: return rank(period=body.period,universe=body.universe)
    except ValueError as e: raise HTTPException(422,str(e))
@app.get('/api/watchlist')
def watchlist(): return alerts.get_watchlist()
@app.post('/api/watchlist')
def watch_add(body:WatchRequest):
    try: result=alerts.add_watch(body.symbol); audit('watch_add',result); return result
    except ValueError as e: raise HTTPException(422,str(e))
@app.delete('/api/watchlist/{symbol}')
def watch_remove(symbol:str): return alerts.delete_watch(symbol)
@app.get('/api/alerts')
def alert_list(): return {'rules':alerts.list_alerts(),'history':alerts.alert_history()}
@app.post('/api/alerts')
def alert_add(body:AlertRequest):
    try: result=alerts.add_alert(body.model_dump()); audit('alert_add',{'id':result['id'],'symbol':result['symbol']}); return result
    except ValueError as e: raise HTTPException(422,str(e))
@app.delete('/api/alerts/{alert_id}')
def alert_remove(alert_id:int): return alerts.delete_alert(alert_id)
@app.post('/api/alerts/evaluate')
def alert_run(): return alerts.evaluate_alerts()
@app.post('/api/data/import-csv')
async def data_import(file:UploadFile=File(...),dataset_description:str=Form(''),license_reference:str=Form(''),user_confirmed_use:bool=Form(False),allow_historical_corrections:bool=Form(False)):
    if not file.filename.lower().endswith('.csv'): raise HTTPException(415,'Upload a .csv file.')
    raw=await file.read(25_000_001)
    if len(raw)>25_000_000: raise HTTPException(413,'CSV must be at most 25 MB.')
    try: result=import_csv(raw,file.filename,dataset_description=dataset_description,license_reference=license_reference,user_confirmed_use=user_confirmed_use,allow_historical_corrections=allow_historical_corrections); audit('csv_import',result); return result
    except ValueError as e: raise HTTPException(422,str(e))
@app.post('/api/data/import-fundamentals-csv')
async def fundamentals_import(file:UploadFile=File(...),dataset_description:str=Form(''),license_reference:str=Form(''),user_confirmed_use:bool=Form(False),source_name:str=Form('')):
    if not file.filename.lower().endswith('.csv'): raise HTTPException(415,'Upload a .csv file.')
    raw=await file.read(25_000_001)
    if len(raw)>25_000_000: raise HTTPException(413,'CSV must be at most 25 MB.')
    try: result=import_fundamentals_csv(raw,file.filename,dataset_description=dataset_description,license_reference=license_reference,user_confirmed_use=user_confirmed_use,source_name=source_name or None); audit('fundamentals_import',result); return result
    except ValueError as e: raise HTTPException(422,str(e))
@app.post('/api/data/import-events-csv')
async def events_import(file:UploadFile=File(...),dataset_description:str=Form(''),license_reference:str=Form(''),user_confirmed_use:bool=Form(False),source_name:str=Form('')):
    if not file.filename.lower().endswith('.csv'): raise HTTPException(415,'Upload a .csv file.')
    raw=await file.read(25_000_001)
    if len(raw)>25_000_000: raise HTTPException(413,'CSV must be at most 25 MB.')
    try: result=import_events_csv(raw,file.filename,dataset_description=dataset_description,license_reference=license_reference,user_confirmed_use=user_confirmed_use,source_name=source_name or None); audit('events_import',result); return result
    except ValueError as e: raise HTTPException(422,str(e))

@app.post('/api/data/import-zip')
async def import_zip(file:UploadFile=File(...),dataset_description:str=Form(''),license_reference:str=Form(''),user_confirmed_use:bool=Form(False),allow_historical_corrections:bool=Form(False)):
    if not file.filename.lower().endswith('.zip'): raise HTTPException(415,'Upload a .zip bundle.')
    raw=await file.read(25_000_001)
    if len(raw)>25_000_000: raise HTTPException(413,'ZIP must be at most 25 MB compressed.')
    try: result=import_price_zip(raw,file.filename,dataset_description=dataset_description,license_reference=license_reference,user_confirmed_use=user_confirmed_use,allow_historical_corrections=allow_historical_corrections); audit('zip_import',{'files_imported':result['files_imported'],'rows_imported':result['rows_imported'],'status':result['status']}); return result
    except ValueError as e: raise HTTPException(422,str(e))

@app.post('/api/data/import-constituents')
async def import_constituents(file:UploadFile=File(...),as_of_date:str=Form(...),source_name:str=Form(...),dataset_description:str=Form(''),license_reference:str=Form(''),user_confirmed_use:bool=Form(False)):
    if not file.filename.lower().endswith('.csv'): raise HTTPException(415,'Upload a .csv constituent list.')
    raw=await file.read(25_000_001)
    if len(raw)>25_000_000: raise HTTPException(413,'CSV must be at most 25 MB.')
    try: result=import_constituents_csv(raw,file.filename,as_of_date=as_of_date,source_name=source_name,dataset_description=dataset_description,license_reference=license_reference,user_confirmed_use=user_confirmed_use); audit('constituents_import',{'as_of_date':result.get('as_of_date'),'rows':result.get('rows_imported')}); return result
    except ValueError as e: raise HTTPException(422,str(e))

@app.post('/api/data/import-membership-history')
async def import_membership_history(file:UploadFile=File(...),source_name:str=Form(...),dataset_description:str=Form(''),license_reference:str=Form(''),user_confirmed_use:bool=Form(False)):
    if not file.filename.lower().endswith('.csv'): raise HTTPException(415,'Upload a .csv membership interval file.')
    raw=await file.read(25_000_001)
    if len(raw)>25_000_000: raise HTTPException(413,'CSV must be at most 25 MB.')
    try: result=import_membership_history_csv(raw,file.filename,source_name=source_name,dataset_description=dataset_description,license_reference=license_reference,user_confirmed_use=user_confirmed_use); audit('membership_history_import',{'rows':result.get('rows_imported')}); return result
    except ValueError as e: raise HTTPException(422,str(e))

@app.post('/api/data/import-instruments')
async def import_instruments(file:UploadFile=File(...),as_of_date:str=Form(...),source_name:str=Form(...),dataset_description:str=Form(''),license_reference:str=Form(''),user_confirmed_use:bool=Form(False)):
    if not file.filename.lower().endswith('.csv'): raise HTTPException(415,'Upload a .csv instrument list.')
    raw=await file.read(25_000_001)
    if len(raw)>25_000_000: raise HTTPException(413,'CSV must be at most 25 MB.')
    try: result=import_instruments_csv(raw,file.filename,as_of_date=as_of_date,source_name=source_name,dataset_description=dataset_description,license_reference=license_reference,user_confirmed_use=user_confirmed_use); audit('instrument_list_import',result); return result
    except ValueError as e: raise HTTPException(422,str(e))

@app.post('/api/data/import-exchange-calendar')
async def import_exchange_calendar(file:UploadFile=File(...),source_name:str=Form(...),dataset_description:str=Form(''),license_reference:str=Form(''),user_confirmed_use:bool=Form(False)):
    if not file.filename.lower().endswith('.csv'): raise HTTPException(415,'Upload a .csv exchange session list.')
    raw=await file.read(25_000_001)
    if len(raw)>25_000_000: raise HTTPException(413,'CSV must be at most 25 MB.')
    try: result=import_exchange_calendar_csv(raw,file.filename,source_name=source_name,dataset_description=dataset_description,license_reference=license_reference,user_confirmed_use=user_confirmed_use); audit('exchange_calendar_import',result); return result
    except ValueError as e: raise HTTPException(422,str(e))

@app.get('/api/universe/constituents')
def current_constituents():
    with connect() as c:
        latest=c.execute('SELECT MAX(as_of_date) FROM published_constituent_snapshots').fetchone()[0]
        if not latest: return {'available':False,'as_of_date':None,'constituents':[],'reason':'No dated NIFTY 50 constituent snapshot has been imported.'}
        rows=c.execute('SELECT * FROM published_constituent_snapshots WHERE as_of_date=? ORDER BY symbol',(latest,)).fetchall()
    return {'available':True,'as_of_date':latest,'constituents':[dict(r) for r in rows],'quality':rows[0]['quality'] if rows else None}

@app.get('/api/universe/instruments')
def instrument_inventory():
    data=get_data_status()
    return {'as_of_date':data['universe']['instrument_as_of_date'],'imported_instruments':data['universe']['imported_instruments'],'instruments':data['universe']['imported_list'],'coverage':data['instrument_coverage'],'note':'The built-in 10-company list is a prototype. User-imported instrument records are shown with dated provenance; membership is separate.'}

@app.get('/api/portfolio')
def portfolio():
    with connect() as c:
        positions=c.execute("SELECT h.*,s.name FROM portfolio_positions h LEFT JOIN published_securities s USING(symbol) ORDER BY h.symbol").fetchall(); out=[]
        for p in positions:
            price=c.execute("SELECT date,close,source,quality,price_basis FROM published_prices WHERE symbol=? AND quality<>'DEMO_SYNTHETIC' ORDER BY date DESC LIMIT 1",(p['symbol'],)).fetchone()
            value=float(p['quantity']*price['close']) if price else None
            out.append({**dict(p),'latest_observation':dict(price) if price else None,'market_value':value,'unrealized_pnl':None if value is None else value-float(p['quantity']*p['cost_basis']),'valuation_note':'Historical local observation; not a live quote.' if price else 'No non-synthetic local price available.'})
    return {'available':bool(out),'portfolio_type':'local_manual_tracking','positions':out,'brokerage_connected':False,'trading_enabled':False}

@app.put('/api/portfolio/holdings')
def portfolio_position(body:PortfolioPositionRequest):
    symbol=body.symbol.strip().upper()
    with connect() as c:
        if not c.execute('SELECT 1 FROM published_securities WHERE symbol=?',(symbol,)).fetchone(): raise HTTPException(404,'Unknown local instrument. Import its instrument data first.')
        c.execute('INSERT INTO portfolio_positions(symbol,quantity,cost_basis,updated_at,notes) VALUES(?,?,?,?,?) ON CONFLICT(symbol) DO UPDATE SET quantity=excluded.quantity,cost_basis=excluded.cost_basis,updated_at=excluded.updated_at,notes=excluded.notes',(symbol,body.quantity,body.cost_basis,datetime.now(timezone.utc).isoformat(),body.notes[:500]))
    audit('portfolio_position_set',{'symbol':symbol,'quantity':body.quantity})
    return {'status':'saved','symbol':symbol,'portfolio_type':'local_manual_tracking','trading_enabled':False}

@app.delete('/api/portfolio/holdings/{symbol}')
def portfolio_position_delete(symbol:str):
    with connect() as c: c.execute('DELETE FROM portfolio_positions WHERE symbol=?',(symbol.upper(),))
    return {'deleted':symbol.upper()}

@app.post('/api/portfolio/paper-trades')
def paper_trade(body:PaperTradeRequest):
    symbol=body.symbol.strip().upper(); side=body.side.strip().upper()
    if side not in {'BUY','SELL'}: raise HTTPException(422,'side must be BUY or SELL.')
    now=datetime.now(timezone.utc).isoformat()
    with connect() as c:
        quote=c.execute("SELECT date,close,source,quality FROM published_prices WHERE symbol=? AND quality<>'DEMO_SYNTHETIC' ORDER BY date DESC LIMIT 1",(symbol,)).fetchone()
        if not quote: raise HTTPException(409,'Paper trading needs a non-synthetic local reference close; no trade was recorded.')
        held=c.execute('SELECT quantity,cost_basis FROM portfolio_positions WHERE symbol=?',(symbol,)).fetchone(); old_qty=float(held['quantity']) if held else 0.0
        if side=='SELL' and body.quantity>old_qty: raise HTTPException(409,'Paper sell exceeds locally tracked quantity.')
        price=float(quote['close']); new_qty=old_qty+(body.quantity if side=='BUY' else -body.quantity)
        old_cost=float(held['cost_basis']) if held else 0.0
        new_cost=((old_qty*old_cost+body.quantity*price)/new_qty if side=='BUY' and new_qty>0 else old_cost if new_qty>0 else 0.0)
        c.execute('INSERT INTO paper_trades(symbol,side,quantity,reference_price,trade_date,created_at,notes) VALUES(?,?,?,?,?,?,?)',(symbol,side,body.quantity,price,quote['date'],now,(body.notes+' · reference '+quote['source']+' '+quote['quality'])[:500]))
        c.execute('INSERT INTO portfolio_positions(symbol,quantity,cost_basis,updated_at,notes) VALUES(?,?,?,?,?) ON CONFLICT(symbol) DO UPDATE SET quantity=excluded.quantity,cost_basis=excluded.cost_basis,updated_at=excluded.updated_at',(symbol,new_qty,new_cost,now,'Paper-trade simulation only'))
    audit('paper_trade',{'symbol':symbol,'side':side,'quantity':body.quantity,'reference_date':quote['date']})
    return {'status':'simulated','symbol':symbol,'side':side,'quantity':body.quantity,'reference_price':price,'reference_date':quote['date'],'source':quote['source'],'quality':quote['quality'],'real_order_placed':False}

@app.get('/api/portfolio/paper-trades')
def paper_trade_history():
    with connect() as c: rows=c.execute('SELECT * FROM paper_trades ORDER BY id DESC LIMIT 500').fetchall()
    return {'trades':[dict(r) for r in rows],'real_orders':False}

