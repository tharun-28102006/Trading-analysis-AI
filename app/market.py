"""Market data providers, deterministic local demo data, validation, and ingestion."""
import io
import json
import hashlib
import csv
import math
import re
import threading
import time
import zipfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Protocol
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .db import connect, capture_release_snapshot
from .config import DATA_DIR

UNIVERSE = [
    ("RELIANCE", "Reliance Industries", "Energy"), ("TCS", "Tata Consultancy Services", "Information Technology"),
    ("HDFCBANK", "HDFC Bank", "Financial Services"), ("ICICIBANK", "ICICI Bank", "Financial Services"),
    ("INFY", "Infosys", "Information Technology"), ("BHARTIARTL", "Bharti Airtel", "Telecommunication"),
    ("ITC", "ITC", "Consumer Goods"), ("SBIN", "State Bank of India", "Financial Services"),
    ("LT", "Larsen & Toubro", "Construction"), ("HINDUNILVR", "Hindustan Unilever", "Consumer Goods"),
    ("NIFTY50", "NIFTY 50 (demo benchmark)", "Index"),
]

class MarketDataProvider(Protocol):
    name: str
    def history(self, symbol: str, start: str, end: str) -> pd.DataFrame: ...

YAHOO_SOURCE='yahoo_finance_yfinance'
YAHOO_QUALITY='YAHOO_FINANCE_UNVERIFIED'
KITE_SOURCE='zerodha_kite_connect'
KITE_QUALITY='KITE_CONNECT_RAW_UNADJUSTED'
UNIVERSE_PATH=Path(__file__).with_name('instruments.json')
_refresh_lock=threading.Lock()
_refresh_state={'status':'idle','started_at':None,'run_id':None}

def load_instruments() -> list[dict]:
    data=json.loads(UNIVERSE_PATH.read_text(encoding='utf-8'))
    items=data.get('instruments',[])
    ids=[x.get('instrument_id') for x in items]; symbols=[x.get('symbol') for x in items]; tickers=[x.get('provider_symbol') for x in items]
    if not items or any(not x for x in ids+symbols+tickers) or len(set(ids))!=len(ids) or len(set(symbols))!=len(symbols):
        raise ValueError('Instrument configuration has missing or duplicate identifiers, symbols, or provider mappings.')
    if any(x.get('provider') not in {'kite_connect','yahoo_finance','tejhq'} for x in items): raise ValueError('Unsupported provider in instrument configuration.')
    return items

class YahooFinanceProvider:
    """Retired adapter; an environment flag is not proof of collection rights."""
    name=YAHOO_SOURCE
    def assert_allowed(self):
        raise RuntimeError('Yahoo Finance collection is disabled: automated collection and research/simulation use are not verified as permitted. An environment flag cannot authorize it.')
    def history(self,symbol:str,start:str,end:str)->pd.DataFrame:
        self.assert_allowed()

class KiteConnectProvider:
    """Retired broker adapter; Nivesh does not require or use broker credentials."""
    name=KITE_SOURCE
    quality=KITE_QUALITY
    def assert_allowed(self):
        raise RuntimeError('Kite Connect collection is disabled: this application does not require a broker account, and account/API access does not establish historical-data or research-use rights.')
    def history(self,symbol,start,end):
        self.assert_allowed()

def normalize_yahoo_frame(frame:pd.DataFrame,symbol:str) -> tuple[pd.DataFrame,list[str]]:
    if frame is None or frame.empty: raise ValueError('Provider returned no daily observations.')
    df=frame.copy()
    if isinstance(df.columns,pd.MultiIndex): df.columns=df.columns.get_level_values(0)
    df.columns=[str(c).strip().lower().replace(' ','_') for c in df.columns]
    if 'date' not in df:
        if isinstance(df.index,pd.DatetimeIndex): df=df.reset_index(); df.columns=[str(c).strip().lower().replace(' ','_') for c in df.columns]
        if 'date' not in df and 'datetime' in df: df=df.rename(columns={'datetime':'date'})
    aliases={'adj_close':'adjusted_close','stock_splits':'stock_splits','dividends':'dividends'}
    df=df.rename(columns=aliases)
    required={'date','open','high','low','close','adjusted_close'}
    missing=required-set(df.columns)
    if missing: raise ValueError('Provider response missing required fields: '+', '.join(sorted(missing)))
    df['symbol']=symbol
    df['date']=pd.to_datetime(df.date,errors='coerce').dt.strftime('%Y-%m-%d')
    for col in ['open','high','low','close','adjusted_close','volume','dividends','stock_splits']:
        if col in df: df[col]=pd.to_numeric(df[col],errors='coerce')
    df=df.dropna(subset=['date','open','high','low','close','adjusted_close'])
    if df.empty: raise ValueError('Provider response contains no valid price rows.')
    if df.duplicated(['symbol','date']).any():
        df=df.drop_duplicates(['symbol','date'],keep='last')
    prices=df[['open','high','low','close','adjusted_close']]
    if (prices<=0).any().any(): raise ValueError('Provider returned non-positive OHLC or adjusted prices.')
    bad=(df.high < df[['open','close','low']].max(axis=1)) | (df.low > df[['open','close','high']].min(axis=1))
    if bad.any(): raise ValueError(f'Provider returned invalid OHLC relationships on {int(bad.sum())} row(s).')
    if 'volume' in df and (df.volume.dropna()<0).any(): raise ValueError('Provider returned negative volume.')
    warnings=[]
    if 'volume' not in df or df.volume.isna().any(): warnings.append('Volume is unavailable for one or more observations and was kept missing.')
    for action_col in ['dividends','stock_splits']:
        if action_col not in df: df[action_col]=0.0
        else: df[action_col]=df[action_col].fillna(0.0)
    if 'volume' not in df: df['volume']=np.nan
    return df.sort_values('date').reset_index(drop=True),warnings

def _ten_year_start(today:date) -> date:
    try: return today.replace(year=today.year-10)
    except ValueError: return today.replace(year=today.year-10,day=28)

def refresh_market_data(provider:MarketDataProvider|None=None,triggered_by='manual',reconcile=False) -> dict:
    if provider is None: raise RuntimeError('Remote refresh is disabled. Supply a specifically authorized provider adapter explicitly, or import a permitted local CSV.')
    instruments=load_instruments(); started=datetime.now(timezone.utc)
    source=getattr(provider,'name','custom_provider'); quality=getattr(provider,'quality',YAHOO_QUALITY if source==YAHOO_SOURCE else KITE_QUALITY)
    with connect() as c:
        cur=c.execute("INSERT INTO refresh_runs(started_at,status,triggered_by,source,instruments_requested) VALUES(?,?,?,?,?)",(started.isoformat(),'running',triggered_by,getattr(provider,'name','custom_provider'),len(instruments)))
        run_id=cur.lastrowid
        for item in instruments:
            c.execute("INSERT INTO securities(symbol,name,sector,exchange,index_member,is_demo,instrument_id,provider_symbol) VALUES(?,?,?,?,?,?,?,?) ON CONFLICT(symbol) DO UPDATE SET name=excluded.name,sector=excluded.sector,exchange=excluded.exchange,index_member=excluded.index_member,instrument_id=excluded.instrument_id,provider_symbol=excluded.provider_symbol",(item['symbol'],item['name'],item.get('sector'),item['exchange'],int(item.get('index_member',False)),0,item['instrument_id'],item['provider_symbol']))
    errors=[]; warnings=[]; updated=0; rows_written=0; latest=None
    try:
        if hasattr(provider,'assert_allowed'): provider.assert_allowed()
        today=date.today(); initial=_ten_year_start(today); end=(today.isoformat() if source==KITE_SOURCE else (today+timedelta(days=1)).isoformat())
        for item in instruments:
            symbol=item['symbol']; ticker=item['provider_symbol']
            with connect() as c:
                last=c.execute('SELECT MAX(date) FROM prices WHERE symbol=? AND source=?',(symbol,source)).fetchone()[0]
            start=initial if reconcile or not last else max(initial,date.fromisoformat(last)-timedelta(days=10))
            try:
                raw=provider.history(ticker,start.isoformat(),end)
                clean,row_warnings=normalize_yahoo_frame(raw,symbol)
                if source==KITE_SOURCE and ('volume' not in clean or clean.volume.isna().any()):
                    raise ValueError('Kite Connect response is missing required daily volume values.')
                outside_range=(clean.date<start.isoformat()) | ((clean.date>end) if source==KITE_SOURCE else (clean.date>=end))
                if outside_range.any():
                    raise ValueError('Provider returned observations outside the requested date range.')
                retrieved=datetime.now(timezone.utc).isoformat(); symbol_rows=0; saved_latest=clean.date.max()
                with connect() as c:
                    for _,r in clean.iterrows():
                        payload={k:(None if pd.isna(r[k]) else float(r[k])) for k in ['open','high','low','close','adjusted_close','volume','dividends','stock_splits']}
                        payload_json=json.dumps(payload,sort_keys=True,allow_nan=False); payload_hash=hashlib.sha256(payload_json.encode('utf-8')).hexdigest()
                        c.execute('INSERT OR IGNORE INTO raw_price_observations(instrument_id,symbol,provider_symbol,date,source,retrieved_at,payload_hash,payload_json,validation_status) VALUES(?,?,?,?,?,?,?,?,?)',(item['instrument_id'],symbol,ticker,r.date,source,retrieved,payload_hash,payload_json,'STRUCTURALLY_VALID_UNVERIFIED_SOURCE'))
                        c.execute('INSERT INTO prices(symbol,date,open,high,low,close,adjusted_close,volume,source,retrieved_at,quality) VALUES(?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(symbol,date) DO UPDATE SET open=excluded.open,high=excluded.high,low=excluded.low,close=excluded.close,adjusted_close=excluded.adjusted_close,volume=excluded.volume,source=excluded.source,retrieved_at=excluded.retrieved_at,quality=excluded.quality WHERE prices.source IN (?,?)',(symbol,r.date,float(r.open),float(r.high),float(r.low),float(r.close),float(r.adjusted_close),None if pd.isna(r.volume) else int(r.volume),source,retrieved,quality,'synthetic_demo',source))
                        if c.execute('SELECT changes()').fetchone()[0]: symbol_rows+=1
                        if r.dividends or r.stock_splits:
                            c.execute('INSERT INTO corporate_actions(instrument_id,symbol,date,dividends,stock_splits,source,retrieved_at,quality) VALUES(?,?,?,?,?,?,?,?) ON CONFLICT(symbol,date,source) DO UPDATE SET dividends=excluded.dividends,stock_splits=excluded.stock_splits,retrieved_at=excluded.retrieved_at,quality=excluded.quality',(item['instrument_id'],symbol,r.date,float(r.dividends),float(r.stock_splits),source,retrieved,quality))
                updated+=int(symbol_rows>0); rows_written+=symbol_rows; latest=max(latest,saved_latest) if latest else saved_latest
                if source==KITE_SOURCE: row_warnings.append('Kite supplies raw close only; adjusted_close stores an explicitly unadjusted close proxy and no corporate actions were supplied.')
                warnings.extend({'symbol':symbol,'warning':w} for w in row_warnings)
            except Exception as e:
                errors.append({'symbol':symbol,'provider_symbol':ticker,'error':f'{type(e).__name__}: {e}'[:500]})
        status='succeeded' if not errors else 'partial_success' if updated else 'failed'
    except Exception as e:
        errors.append({'symbol':None,'error':f'{type(e).__name__}: {e}'[:500]}); status='failed'
    finished=datetime.now(timezone.utc).isoformat()
    with connect() as c:
        c.execute('UPDATE refresh_runs SET finished_at=?,status=?,instruments_updated=?,rows_written=?,latest_observation_date=?,errors_json=?,warnings_json=? WHERE id=?',(finished,status,updated,rows_written,latest,json.dumps(errors),json.dumps(warnings),run_id))
    result={'run_id':run_id,'status':status,'started_at':started.isoformat(),'finished_at':finished,'source':source,'quality':quality,'instruments_requested':len(instruments),'instruments_updated':updated,'rows_written':rows_written,'latest_observation_date':latest,'errors':errors,'warnings':warnings,'history_target':'up to 10 years daily, subject to provider coverage','reconciliation':'full-history' if reconcile else 'incremental with 10-calendar-day overlap'}
    return result

TEJ_SOURCE='TejHQ Indian Markets (NSE EOD)'
TEJ_QUALITY='TEJ_PROVIDER_LICENSED'
TEJ_LICENSE='https://huggingface.co/datasets/tejhq/indian-markets (MIT dataset license as published by TejHQ); https://tejhq.dev/docs'
TEJ_BAZAAR_SOURCE='Tej Bazaar local NSE/BSE Parquet + TejHQ keyless snapshots'
TEJ_BAZAAR_QUALITY='TEJ_BAZAAR_LOCAL_LICENSED'

def _validate_tej_rows(rows, symbol, start, end):
    if not isinstance(rows,list): raise ValueError('Provider returned a malformed data list.')
    parsed=[]; seen=set()
    for ix,row in enumerate(rows):
        if not isinstance(row,dict): raise ValueError(f'Provider returned a non-object observation for {symbol}.')
        try:
            d=date.fromisoformat(str(row.get('date'))).isoformat()
        except ValueError as exc: raise ValueError(f'Provider returned an invalid date for {symbol}.') from exc
        if d<start or d>end: raise ValueError(f'Provider returned {symbol} observation outside the requested date range.')
        if d in seen: raise ValueError(f'Provider returned duplicate {symbol} observations for {d}.')
        seen.add(d)
        values={}
        for key in ('open','high','low','close','volume'):
            try: value=float(row[key])
            except (KeyError,TypeError,ValueError) as exc: raise ValueError(f'Provider returned invalid {key} for {symbol} on {d}.') from exc
            if not math.isfinite(value): raise ValueError(f'Provider returned non-finite {key} for {symbol} on {d}.')
            values[key]=value
        if any(values[k]<=0 for k in ('open','high','low','close')) or values['volume']<0 or values['volume']%1:
            raise ValueError(f'Provider returned invalid positive OHLC or whole-share volume for {symbol} on {d}.')
        if values['high']<max(values['open'],values['close'],values['low']) or values['low']>min(values['open'],values['close'],values['high']):
            raise ValueError(f'Provider returned invalid OHLC relationships for {symbol} on {d}.')
        parsed.append({'symbol':symbol,'exchange':'NSE','date':d,'open':values['open'],'high':values['high'],'low':values['low'],
            'close':values['close'],'adjusted_close':values['close'],'volume':int(values['volume']),
            'price_basis':'raw_unadjusted','source':TEJ_SOURCE})
    return sorted(parsed,key=lambda x:x['date'])

def _set_refresh_progress(run_id, *, completed, total, current_symbol=None, rows=0, provider_trading_date=None, errors=None):
    detail={'progress':{'completed_symbols':completed,'total_symbols':total,'current_symbol':current_symbol,
        'rows_received':rows,'provider_trading_date':provider_trading_date}}
    warnings=json.dumps(detail,sort_keys=True)
    with connect() as c:
        c.execute('UPDATE refresh_runs SET instruments_updated=?,rows_written=?,warnings_json=? WHERE id=?',
            (completed,rows,warnings,run_id))

def _run_tej_refresh(run_id, reconcile=False):
    from .providers.tej import TejClient, retain_payload
    started=time.monotonic(); errors=[]; warnings=[]; all_rows=[]; payloads=[]
    try:
        instruments=[x for x in load_instruments() if x.get('exchange')=='NSE' and x.get('symbol')!='NIFTY50']
        if not instruments: raise ValueError('No NSE equity symbols are configured for TejHQ refresh.')
        client=TejClient(); status=client.status(); provider_date=status['trading_date']
        if date.fromisoformat(provider_date)>date.today()+timedelta(days=1): raise ValueError('TejHQ status reports a future trading date.')
        if (date.today()-date.fromisoformat(provider_date)).days>7:
            raise ValueError(f'TejHQ data is stale: provider trading date is {provider_date}, more than seven days behind today.')
        end=provider_date; total=len(instruments)
        for index,item in enumerate(instruments,1):
            symbol=item['symbol']; _set_refresh_progress(run_id,completed=index-1,total=total,current_symbol=symbol,rows=len(all_rows),provider_trading_date=provider_date)
            with connect() as c:
                last=c.execute("SELECT MAX(date) FROM prices WHERE symbol=? AND source LIKE 'tejhq:%' AND quality<>'DEMO_SYNTHETIC'",(symbol,)).fetchone()[0]
            start='2010-01-04' if reconcile or not last else max('2010-01-04',(date.fromisoformat(last)-timedelta(days=10)).isoformat())
            payload,rows=client.history(item['provider_symbol'],start,end)
            digest,path=retain_payload(payload)
            payloads.append({'symbol':symbol,'sha256':digest,'path':path,'retrieved_at':datetime.now(timezone.utc).isoformat()})
            parsed=_validate_tej_rows(rows,symbol,start,end)
            if not parsed and not last: raise ValueError(f'TejHQ returned no historical observations for newly configured {symbol}.')
            if parsed and date.fromisoformat(max(row['date'] for row in parsed))>date.fromisoformat(provider_date):
                raise ValueError(f'TejHQ returned {symbol} observations newer than its published trading date.')
            if parsed:
                ordered=[r['date'] for r in parsed]
                gaps=[]
                for a,b in zip(ordered,ordered[1:]):
                    cursor=date.fromisoformat(a)+timedelta(days=1); missing=[]
                    while cursor<date.fromisoformat(b):
                        if cursor.weekday()<5: missing.append(cursor.isoformat())
                        cursor+=timedelta(days=1)
                    if missing: gaps.append({'after':a,'before':b,'possible_missing_weekdays':missing})
                if gaps:
                    warnings.append({'symbol':symbol,'warning':'Potential missing weekday sessions require review; the provider endpoint does not supply an authoritative exchange-session calendar.','gaps':gaps[:100]})
            all_rows.extend(parsed)
            _set_refresh_progress(run_id,completed=index,total=total,current_symbol=None,rows=len(all_rows),provider_trading_date=provider_date)
        if not all_rows:
            with connect() as c:
                latest=c.execute("SELECT MAX(date) FROM prices WHERE quality<>'DEMO_SYNTHETIC'").fetchone()[0]
            status_label='unchanged'
            result={'status':status_label,'source':TEJ_SOURCE,'rows_received':0,'rows_imported':0,'latest_observation_date':latest,
                'provider_trading_date':provider_date,'warnings':warnings,'errors':[],'no_new_data':True,'payloads':payloads}
        else:
            import_buffer=io.StringIO(newline='')
            fieldnames=['symbol','exchange','date','open','high','low','close','adjusted_close','volume','price_basis','source']
            writer=csv.DictWriter(import_buffer,fieldnames=fieldnames,lineterminator='\n'); writer.writeheader(); writer.writerows(all_rows)
            imported=import_csv(import_buffer.getvalue().encode('utf-8'),f'tejhq-refresh-{provider_date}.csv',
                dataset_description='TejHQ NSE EOD bhavcopy OHLCV; raw unadjusted daily observations for configured Nivesh equities.',
                license_reference=TEJ_LICENSE,user_confirmed_use=False,allow_historical_corrections=True,
                provider_namespace='tejhq',provider_quality=TEJ_QUALITY)
            if imported.get('status') in {'failed','failed_publication','blocked_correction_confirmation'}:
                raise ValueError(imported.get('message') or 'Provider observation publication failed; previous release remains active.')
            if imported.get('lifecycle') and imported['lifecycle'].get('release_status')!='published':
                raise ValueError(imported['lifecycle'].get('error') or 'The candidate data release could not be published.')
            with connect() as c:
                latest=c.execute("SELECT MAX(date) FROM published_prices WHERE quality=?",(TEJ_QUALITY,)).fetchone()[0]
            no_new=bool(imported.get('idempotent')) or imported.get('rows_imported',0)==0
            imported_rows=0 if imported.get('idempotent') else imported.get('rows_imported',0)
            status_label='unchanged' if no_new else 'succeeded'
            result={'status':status_label,'source':TEJ_SOURCE,'rows_received':len(all_rows),'rows_imported':imported_rows,'rows_written':imported_rows,
                'duplicates':imported.get('duplicates',0),'latest_observation_date':latest,'provider_trading_date':provider_date,
                'import':imported,'warnings':warnings+imported.get('warnings',[]),'errors':[],'no_new_data':no_new,'payloads':payloads}
        if result['latest_observation_date'] and result['latest_observation_date']<provider_date:
            result['warnings'].append({'warning':f'Configured instruments have no observation on provider trading date {provider_date}; latest acquired observation is {result["latest_observation_date"]}.'})
        finished=datetime.now(timezone.utc).isoformat()
        with connect() as c:
            c.execute('UPDATE refresh_runs SET finished_at=?,status=?,source=?,instruments_updated=?,rows_written=?,latest_observation_date=?,errors_json=?,warnings_json=?,step_timings_json=? WHERE id=?',
                (finished,result['status'],TEJ_SOURCE,len(instruments),result.get('rows_written',0),result['latest_observation_date'],json.dumps([],sort_keys=True),
                 json.dumps({'warnings':result['warnings'],'provider_payloads':payloads,'provider_trading_date':provider_date,'no_new_data':result['no_new_data']},sort_keys=True),
                 json.dumps({'elapsed_seconds':round(time.monotonic()-started,3),'provider_requests':1+len(instruments)},sort_keys=True),run_id))
            c.execute("UPDATE source_registry SET connectivity_status='reachable',last_attempt_at=?,last_attempt_status=?,last_success_at=?,last_error=NULL WHERE source_key='tejhq_indian_markets'",
                (finished,result['status'],finished))
        _refresh_state.update({'status':result['status'],'run_id':run_id,'finished_at':finished,'result':result})
        return result
    except Exception as exc:
        message=f'{type(exc).__name__}: {str(exc)[:500]}'
        errors=[message]
        finished=datetime.now(timezone.utc).isoformat()
        with connect() as c:
            c.execute('UPDATE refresh_runs SET finished_at=?,status=?,source=?,errors_json=?,step_timings_json=? WHERE id=?',
                (finished,'failed',TEJ_SOURCE,json.dumps(errors),json.dumps({'elapsed_seconds':round(time.monotonic()-started,3)}),run_id))
            c.execute("UPDATE source_registry SET connectivity_status='unreachable',last_attempt_at=?,last_attempt_status='failed',last_error=? WHERE source_key='tejhq_indian_markets'",
                (finished,message))
        _refresh_state.update({'status':'failed','run_id':run_id,'finished_at':finished,'error':message})
        return {'run_id':run_id,'status':'failed','source':TEJ_SOURCE,'errors':errors,'error':message,'latest_observation_date':None}
    finally:
        _refresh_lock.release()

def start_refresh(triggered_by='manual',reconcile=False) -> dict:
    if not _refresh_lock.acquire(blocking=False):
        with connect() as c: row=c.execute("SELECT id,started_at,status FROM refresh_runs WHERE status='running' ORDER BY id DESC LIMIT 1").fetchone()
        return {'status':'already_running','run_id':row['id'] if row else _refresh_state.get('run_id'),'message':'A market-data refresh is already in progress.'}
    try:
        instruments=['NSE','BSE']
        started=datetime.now(timezone.utc).isoformat()
        with connect() as c:
            c.execute('BEGIN IMMEDIATE')
            pending=c.execute("SELECT id,started_at FROM refresh_runs WHERE source=? AND status='running' ORDER BY id DESC LIMIT 1",(TEJ_BAZAAR_SOURCE,)).fetchone()
            if pending:
                try: age=datetime.now(timezone.utc)-datetime.fromisoformat(pending['started_at'])
                except ValueError: age=timedelta(days=1)
                if age<timedelta(hours=6):
                    _refresh_lock.release()
                    return {'run_id':pending['id'],'status':'already_running','message':'A TejHQ market-data refresh is already in progress.'}
                c.execute("UPDATE refresh_runs SET finished_at=?,status='interrupted',errors_json=? WHERE id=?",
                    (started,json.dumps(['Previous refresh exceeded the six-hour recovery window and was marked interrupted.']),pending['id']))
            run_id=c.execute("INSERT INTO refresh_runs(started_at,status,triggered_by,source,instruments_requested,errors_json,warnings_json) VALUES(?,'running',?,?,?,'[]','{}')",
                (started,triggered_by,TEJ_BAZAAR_SOURCE,len(instruments))).lastrowid
            c.execute("UPDATE source_registry SET last_attempt_at=?,last_attempt_status='running',last_error=NULL WHERE source_key='tejhq_indian_markets'",(started,))
        _refresh_state.update({'status':'running','run_id':run_id,'started_at':started,'finished_at':None})
        threading.Thread(target=_run_tej_bazaar_refresh,args=(run_id,reconcile),name=f'nivesh-tej-refresh-{run_id}',daemon=True).start()
        return {'run_id':run_id,'status':'running','source':TEJ_BAZAAR_SOURCE,'started_at':started,'progress':{'completed_exchanges':0,'total_exchanges':2}}
    except Exception:
        _refresh_lock.release()
        raise

def refresh_job(run_id:int) -> dict:
    with connect() as c: row=c.execute('SELECT * FROM refresh_runs WHERE id=?',(run_id,)).fetchone()
    if not row: raise ValueError('Refresh job not found.')
    result=dict(row)
    try: result['warnings']=json.loads(result.get('warnings_json') or '{}')
    except ValueError: result['warnings']={}
    try: result['errors']=json.loads(result.get('errors_json') or '[]')
    except ValueError: result['errors']=[]
    result['progress']=(result['warnings'].get('progress') if isinstance(result['warnings'],dict) else None)
    return result

def _refresh_background(triggered_by,reconcile):
    return start_refresh(triggered_by,reconcile)

def _run_tej_bazaar_refresh(run_id,reconcile=False):
    lock_released=False
    try:
        from .tej_bazaar_lifecycle import refresh
        result=refresh(run_id,reconcile)
        _refresh_lock.release()
        lock_released=True
        fetch_status=result.get('provider_fetch_status')
        connectivity='reachable' if fetch_status=='succeeded' else ('not_tested' if fetch_status=='deferred_bootstrap' else 'unreachable')
        finished=datetime.now(timezone.utc).isoformat()
        error_text='; '.join((x.get('error') if isinstance(x,dict) else str(x)) for x in result.get('errors',[])) or None
        with connect() as c:
            c.execute("UPDATE source_registry SET connectivity_status=?,last_attempt_at=?,last_attempt_status=?,last_success_at=CASE WHEN ?='reachable' THEN ? ELSE last_success_at END,last_error=? WHERE source_key='tejhq_indian_markets'",
                (connectivity,finished,'deferred' if fetch_status=='deferred_bootstrap' else result.get('status'),connectivity,finished,error_text))
        _refresh_state.update({'status':result.get('status'),'run_id':run_id,'finished_at':datetime.now(timezone.utc).isoformat(),'result':result})
        return result
    finally:
        if not lock_released:
            _refresh_lock.release()

def get_data_status() -> dict:
    with connect() as c:
        last=c.execute('SELECT * FROM refresh_runs ORDER BY id DESC LIMIT 1').fetchone()
        success=c.execute("SELECT MAX(finished_at) FROM refresh_runs WHERE status IN ('succeeded','partial_success','unchanged')").fetchone()[0]
        # The active immutable release and the canonical tables change in the same
        # SQLite transaction. Use canonical indexes for dashboard summaries; scanning
        # unindexed release snapshot copies here makes health unusable on large imports.
        latest_row=c.execute("SELECT date,source,quality,price_basis FROM prices WHERE quality<>'DEMO_SYNTHETIC' ORDER BY date DESC LIMIT 1").fetchone()
        latest=latest_row['date'] if latest_row else None
        exchange_refresh={r['exchange']:dict(r) for r in c.execute('SELECT * FROM market_exchange_refresh ORDER BY exchange')}
        quality=[]
        for exchange,source_name in (('NSE','tej_bazaar_local:NSE'),('BSE','tej_bazaar_local:BSE')):
            summary=exchange_refresh.get(exchange,{})
            if summary.get('rows_imported'):
                earliest=c.execute('SELECT MIN(observation_date) FROM market_data_files WHERE exchange=?',(exchange,)).fetchone()[0]
                quality.append({'quality':'TEJ_BAZAAR_LOCAL_LICENSED','source':source_name,'price_basis':'raw_unadjusted',
                    'rows':summary['rows_imported'],'earliest_date':earliest,'latest_date':summary.get('latest_market_date')})
        # Health displays a useful sample instead of recomputing coverage diagnostics
        # for every instrument on every page load. Per-symbol history endpoints remain exact.
        coverage=[dict(r) for r in c.execute("""SELECT s.symbol,s.name,s.sector,s.exchange,COUNT(p.date) rows,
            MIN(p.date) earliest_date,MAX(p.date) latest_date,GROUP_CONCAT(DISTINCT p.source) source,
            GROUP_CONCAT(DISTINCT p.price_basis) price_basis,GROUP_CONCAT(DISTINCT p.quality) provenance_status,
            SUM(CASE WHEN p.quality='DEMO_SYNTHETIC' THEN 1 ELSE 0 END) synthetic_rows,
            SUM(CASE WHEN p.quality='USER_IMPORTED_UNVERIFIED' THEN 1 ELSE 0 END) unverified_rows
            FROM securities s LEFT JOIN prices p ON p.symbol=s.symbol AND p.quality<>'DEMO_SYNTHETIC'
            WHERE EXISTS(SELECT 1 FROM prices p2 WHERE p2.symbol=s.symbol AND p2.quality<>'DEMO_SYNTHETIC')
            GROUP BY s.symbol ORDER BY s.index_member DESC,s.symbol LIMIT 100""")]
        imports=[dict(r) for r in c.execute('SELECT id,started_at,finished_at,filename,status,source,rows_read,rows_imported,accepted_rows,rejected_rows,duplicates,dataset_description,license_reference,user_confirmed_use,price_basis,warnings_json,errors_json,earliest_date,latest_date FROM data_import_runs ORDER BY id DESC LIMIT 25')]
        benchmark=c.execute("SELECT COUNT(*) rows,MIN(date) earliest_date,MAX(date) latest_date,GROUP_CONCAT(DISTINCT source) source,GROUP_CONCAT(DISTINCT price_basis) price_basis FROM prices WHERE symbol='NIFTY50' AND quality<>'DEMO_SYNTHETIC'").fetchone()
        calendar={exchange:[dict(r) for r in c.execute('SELECT date,quality FROM published_exchange_sessions WHERE exchange=? ORDER BY date',(exchange,))] for exchange in ('NSE','NSE_INDEX')}
        latest_universe=c.execute('SELECT MAX(as_of_date) FROM published_instrument_universe').fetchone()[0]
        imported_instruments=c.execute('SELECT COUNT(*) FROM published_instrument_universe WHERE as_of_date=?',(latest_universe,)).fetchone()[0] if latest_universe else 0
        has_constituent_snapshot=bool(c.execute('SELECT COUNT(*) FROM published_constituent_snapshots').fetchone()[0])
        imported_instrument_rows=[dict(r) for r in c.execute('SELECT * FROM published_instrument_universe WHERE as_of_date=? ORDER BY symbol',(latest_universe,))] if latest_universe else []
        lifecycle_run=c.execute("SELECT id,started_at,finished_at,status,source,instruments_updated,rows_written,latest_observation_date,errors_json,warnings_json,change_summary_json,dependencies_json FROM refresh_runs ORDER BY id DESC LIMIT 1").fetchone()
        active_release_id=c.execute("SELECT value FROM app_state WHERE key='active_release_id'").fetchone()
        active_release=c.execute('SELECT id,version_id,refresh_run_id,status,created_at,published_at,fingerprint,validation_json FROM research_releases WHERE id=?',(int(active_release_id['value']),)).fetchone() if active_release_id else None
        latest_data_version=c.execute("SELECT id,content_fingerprint,change_kind,created_at,summary_json FROM dataset_versions WHERE dataset_type='prices' ORDER BY id DESC LIMIT 1").fetchone()
        dependency_statuses=[dict(r) for r in c.execute('SELECT component,scope_key,status,version_id,updated_at,detail_json FROM dependency_states ORDER BY component,scope_key')]
        exchange_coverage={
            'NSE':{'rows':exchange_refresh.get('NSE',{}).get('rows_imported',0),
                'earliest_date':c.execute("SELECT MIN(observation_date) FROM market_data_files WHERE exchange='NSE'").fetchone()[0],
                'latest_date':exchange_refresh.get('NSE',{}).get('latest_market_date'),
                'symbols':c.execute("SELECT COUNT(*) FROM securities WHERE exchange='NSE' AND is_demo=0").fetchone()[0]},
            'BSE':{'rows':exchange_refresh.get('BSE',{}).get('rows_imported',0),
                'earliest_date':c.execute("SELECT MIN(observation_date) FROM market_data_files WHERE exchange='BSE'").fetchone()[0],
                'latest_date':exchange_refresh.get('BSE',{}).get('latest_market_date'),
                'symbols':c.execute("SELECT COUNT(DISTINCT symbol) FROM exchange_market_prices WHERE exchange='BSE' AND quality<>'DEMO_SYNTHETIC'").fetchone()[0]},
        }
        for exchange in ('NSE','BSE'):
            exchange_coverage[exchange].update(exchange_refresh.get(exchange,{}))
            exchange_coverage[exchange]['data_source']='Tej Bazaar local Parquet / TejHQ snapshot API'
            latest_exchange=exchange_coverage[exchange].get('latest_market_date') or exchange_coverage[exchange].get('latest_date')
            try: exchange_coverage[exchange]['is_stale']=(datetime.now(ZoneInfo('Asia/Kolkata')).date()-date.fromisoformat(latest_exchange)).days>5 if latest_exchange else True
            except ValueError: exchange_coverage[exchange]['is_stale']=True
    stale=True
    if latest:
        try: stale=(datetime.now(ZoneInfo('Asia/Kolkata')).date()-date.fromisoformat(latest)).days>5
        except ValueError: pass
    eligible_any=False
    for row in coverage:
        row['duplicate_count']=0 # symbol/date is unique in the canonical prices table
        exchange='NSE_INDEX' if row['symbol']=='NIFTY50' else row['exchange']
        calendar_rows=calendar.get(exchange,[]); sessions=[x['date'] for x in calendar_rows]
        if row['earliest_date'] and sessions:
            overlap_start=max(row['earliest_date'],min(sessions)); overlap_end=min(row['latest_date'],max(sessions))
            reference=[d for d in sessions if overlap_start<=d<=overlap_end] if overlap_start<=overlap_end else []
            with connect() as c: observed={r[0] for r in c.execute("SELECT date FROM prices WHERE symbol=? AND quality<>'DEMO_SYNTHETIC' AND date>=? AND date<=?",(row['symbol'],overlap_start,overlap_end))} if reference else set()
            row['calendar_diagnostic']={'available':bool(reference),'exchange':exchange,'calendar_start':min(sessions),'calendar_end':max(sessions),'coverage_start':overlap_start,'coverage_end':overlap_end,'sessions_in_coverage':len(reference),'missing_expected_sessions':len(set(reference)-observed),'unexpected_observation_dates':len(observed-set(reference)),'calendar_quality':sorted({x['quality'] for x in calendar_rows})}
            row['missing_session_count']=row['calendar_diagnostic']['missing_expected_sessions']
        else:
            row['calendar_diagnostic']={'available':False,'exchange':exchange,'reason':'No imported exchange-session calendar covers this instrument.'}; row['missing_session_count']=None
        with connect() as c:
            membership=c.execute("SELECT COUNT(*) n,GROUP_CONCAT(DISTINCT quality) quality FROM constituent_membership WHERE symbol=?",(row['symbol'],)).fetchone()
        row['historical_membership_intervals']=membership['n']; row['membership_provenance']=membership['quality']
        row['membership_available']=row['historical_membership_intervals']>0
        from .features import training_eligibility
        eligibility=({'eligible':False,'reason':'Aligned NIFTY 50 benchmark history is unavailable.'}
            if not benchmark['rows'] else training_eligibility(row['symbol']))
        row['training_eligible']=eligibility['eligible']; row['training_eligibility_reason']=eligibility['reason']
        eligible_any=eligible_any or eligibility['eligible']
    def ist(value):
        if not value: return None
        try:
            stamp=datetime.fromisoformat(value.replace('Z','+00:00'))
            if stamp.tzinfo is None: stamp=stamp.replace(tzinfo=timezone.utc)
            return stamp.astimezone(ZoneInfo('Asia/Kolkata')).isoformat()
        except (ValueError,AttributeError): return None
    return {'refresh_in_progress':bool(lifecycle_run and lifecycle_run['status']=='running'),'refresh_state':{'status':lifecycle_run['status'] if lifecycle_run else 'idle'},'last_attempt':dict(last) if last else None,'last_successful_refresh':success,'last_successful_refresh_ist':ist(success),'latest_market_observation_date':latest,'source':latest_row['source'] if latest_row else None,'quality':latest_row['quality'] if latest_row else 'NO_MARKET_DATA','price_basis':latest_row['price_basis'] if latest_row else None,'is_stale':stale,'delayed_or_incomplete':True,'demo_mode':False,'freshness_note':'Historical end-of-day observations from TejHQ; not live quotes. No NIFTY 50 index history or historical membership is supplied by this refresh path.','exchange_refresh_status':exchange_coverage,'quality_counts':quality,'instrument_coverage':coverage,'import_history':imports,'latest_lifecycle_run':dict(lifecycle_run) if lifecycle_run else None,'active_release':dict(active_release) if active_release else None,'latest_data_version':dict(latest_data_version) if latest_data_version else None,'dependency_statuses':dependency_statuses,'benchmark':dict(benchmark),'universe':{'imported_instruments':imported_instruments,'instrument_as_of_date':latest_universe,'imported_list':imported_instrument_rows,'historical_membership_instruments':sum(bool(r['membership_available']) for r in coverage),'current_constituents_available':has_constituent_snapshot},'has_real_market_data':bool(latest_row),'has_non_synthetic_observations':bool(latest_row),'has_verified_market_data':False,'training_enabled':eligible_any,'training_note':'Provider-license provenance is separate from the existing user-confirmed model eligibility gate. Training remains blocked until the NIFTY 50 benchmark and other alignment prerequisites are met.'}

class LocalDatabaseProvider:
    name = "local_database"
    def history(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        with connect() as c:
            return pd.read_sql_query("SELECT * FROM published_prices WHERE symbol=? AND date>=? AND date<=? AND quality<>'DEMO_SYNTHETIC' ORDER BY date", c, params=(symbol, start, end))

def seed_demo_data():
    with connect() as c:
        if c.execute("SELECT COUNT(*) FROM prices").fetchone()[0]:
            return
        now = datetime.now(timezone.utc).isoformat()
        c.executemany("INSERT OR IGNORE INTO securities(symbol,name,sector,index_member,is_demo) VALUES(?,?,?,?,1)", [(s,n,sec,0 if s=='NIFTY50' else 1) for s,n,sec in UNIVERSE])
        dates = pd.bdate_range(end=pd.Timestamp.now().normalize(), periods=1050)
        rng = np.random.default_rng(91827)
        benchmark = 22000 * np.exp(np.cumsum(rng.normal(0.00038, 0.0095, len(dates))))
        for i,(symbol,_,_) in enumerate(UNIVERSE):
            base = benchmark if symbol == "NIFTY50" else (1000 + i*700) * np.exp(np.cumsum(rng.normal(0.00035 + (i-4)*0.000015, 0.016 + (i%3)*0.003, len(dates))))
            vol = rng.integers(500_000, 15_000_000, len(dates))
            rows=[]
            for d, close, volume in zip(dates, base, vol):
                rows.append((symbol,d.strftime('%Y-%m-%d'),float(close*.995),float(close*1.012),float(close*.988),float(close),float(close),int(volume),'synthetic_demo',now,'DEMO_SYNTHETIC'))
            c.executemany("INSERT OR IGNORE INTO prices(symbol,date,open,high,low,close,adjusted_close,volume,source,retrieved_at,quality) VALUES(?,?,?,?,?,?,?,?,?,?,?)",rows)
        fund_rows=[]
        for i,(s,_,_) in enumerate(UNIVERSE[:-1]):
            fund_rows.append((s,'FY2025',float(4+i%5),float(5+(i*3)%14),float(.1+(i%7)*.32),float(9+(i*7)%20),'synthetic_demo',now,'DEMO_SYNTHETIC'))
        c.executemany("INSERT OR IGNORE INTO fundamentals(symbol,period,revenue_growth,profit_growth,debt_equity,roe,source,retrieved_at,quality) VALUES(?,?,?,?,?,?,?,?,?)",fund_rows)
        events=[('RELIANCE','2025-10-20','earnings','Illustrative quarterly results entry'),('TCS','2025-10-15','earnings','Illustrative quarterly results entry'),('INFY','2025-10-10','corporate','Illustrative company update')]
        c.executemany("INSERT INTO events(symbol,date,kind,title,source,quality) VALUES(?,?,?,?,?,?)",[(s,d,k,t,'synthetic_demo','DEMO_SYNTHETIC') for s,d,k,t in events])

def validate_price_frame(df: pd.DataFrame) -> tuple[pd.DataFrame,list[str]]:
    required={'symbol','date','close','adjusted_close'}
    missing=required-set(df.columns)
    if missing: raise ValueError('Missing required columns: '+', '.join(sorted(missing)))
    out=df.copy(); warnings=[]
    out['symbol']=out.symbol.astype(str).str.upper().str.strip()
    out['date']=pd.to_datetime(out.date,errors='coerce').dt.strftime('%Y-%m-%d')
    for col in ['open','high','low','close','adjusted_close','volume']:
        if col in out: out[col]=pd.to_numeric(out[col],errors='coerce')
    out=out.dropna(subset=['symbol','date','close','adjusted_close'])
    if (out[['close','adjusted_close']]<=0).any().any(): raise ValueError('Prices must be positive.')
    if out.duplicated(['symbol','date']).any():
        warnings.append('Duplicate symbol/date rows found; the last row was retained.')
        out=out.drop_duplicates(['symbol','date'],keep='last')
    if out.empty: raise ValueError('No valid observations in uploaded CSV.')
    return out,warnings


def _price_observation(row):
    return {k:(None if k not in row or pd.isna(row.get(k)) else (str(row[k]) if k=='price_basis' else float(row[k]))) for k in ('open','high','low','close','adjusted_close','volume','dividends','stock_splits','price_basis')}


def _canonical_price_fingerprint(conn):
    rows=conn.execute("SELECT p.symbol,p.date,p.open,p.high,p.low,p.close,p.adjusted_close,p.volume,p.price_basis,COALESCE(a.dividends,0) dividends,COALESCE(a.stock_splits,0) stock_splits FROM prices p LEFT JOIN (SELECT symbol,date,SUM(COALESCE(dividends,0)) dividends,SUM(COALESCE(stock_splits,0)) stock_splits FROM corporate_actions WHERE quality<>'DEMO_SYNTHETIC' GROUP BY symbol,date) a ON a.symbol=p.symbol AND a.date=p.date WHERE p.quality<>'DEMO_SYNTHETIC' ORDER BY p.symbol,p.date").fetchall()
    digest=hashlib.sha256()
    for row in rows:
        digest.update(json.dumps([row[k] for k in ('symbol','date','open','high','low','close','adjusted_close','volume','price_basis','dividends','stock_splits')],separators=(',',':'),allow_nan=False).encode())
        digest.update(b'\n')
    return digest.hexdigest()


def _refresh_current_forecasts(symbols, version_id, fingerprint, refresh_id):
    """Refresh inference from eligible local model artifacts; never retrains."""
    from .models import _current_feature_row, _predict, load_model_version
    outcomes={}
    for symbol in symbols:
        with connect() as c:
            versions=list(c.execute("SELECT * FROM model_versions WHERE model_key=? AND status='eligible' ORDER BY id DESC LIMIT 1",(f'company:{symbol}',)))
            pooled=c.execute("SELECT * FROM model_versions WHERE model_key='pooled' AND status='eligible' ORDER BY id DESC LIMIT 1").fetchone()
            in_current_index=bool(c.execute("SELECT 1 FROM constituent_snapshots WHERE symbol=? AND as_of_date=(SELECT MAX(as_of_date) FROM constituent_snapshots) AND quality<>'DEMO_SYNTHETIC' LIMIT 1",(symbol,)).fetchone())
            if pooled and in_current_index: versions.append(pooled)
        if not versions:
            outcomes[symbol]={'status':'insufficient_data','reason':'No eligible company model is registered; training is never automatic.'}
            continue
        model_outcomes=[]
        try:
            for version in versions:
                artifact=load_model_version(version['id']); columns=json.loads(version['feature_schema_json'])
                try: features,origin=_current_feature_row(symbol,columns,published=False)
                except ValueError as exc:
                    model_outcomes.append({'model_version_id':version['id'],'status':'insufficient_data','reason':str(exc)[:300]})
                    continue
                prediction=float(_predict(artifact['model'],features)[0]); created=datetime.now(timezone.utc).isoformat()
                with connect() as c:
                    c.execute('INSERT OR REPLACE INTO candidate_forecasts(refresh_run_id,version_id,model_version_id,symbol,origin_date,horizon_sessions,prediction,price_basis,input_fingerprint,created_at) VALUES(?,?,?,?,?,?,?,?,?,?)',(refresh_id,version_id,version['id'],symbol,origin,5,prediction,artifact.get('price_basis','unknown'),fingerprint,created))
                model_outcomes.append({'model_version_id':version['id'],'model_key':version['model_key'],'status':'current','origin_date':origin})
            outcomes[symbol]={'status':'current' if any(x['status']=='current' for x in model_outcomes) else 'insufficient_data','models':model_outcomes}
        except Exception as exc:
            raise RuntimeError(f'Forecast candidate failed for {symbol}: {type(exc).__name__}: {str(exc)[:300]}') from None
    return outcomes


def _coordinate_price_import(import_run_id, change_summary, metadata, affected_symbols):
    """Create a versioned refresh run and publish only after deterministic state checks."""
    now=datetime.now(timezone.utc).isoformat(); started=time.monotonic()
    kind=change_summary.get('change_kind','unchanged')
    affected_symbols=sorted(set(affected_symbols))
    if 'NIFTY50' in affected_symbols:
        with connect() as c:
            dependents=[r[0] for r in c.execute("SELECT DISTINCT symbol FROM prices WHERE quality<>'DEMO_SYNTHETIC'")]
        affected_symbols=sorted(set(affected_symbols)|set(dependents))
        change_summary['benchmark_propagation']={'affected_symbols':affected_symbols,'reason':'NIFTY50 observations are benchmark inputs to every instrument with imported price history.'}
        change_summary['affected_symbols']=affected_symbols
    with connect() as c:
        fingerprint=_canonical_price_fingerprint(c)
        latest=c.execute("SELECT id FROM dataset_versions WHERE dataset_type='prices' ORDER BY id DESC LIMIT 1").fetchone()
        meta_fingerprint=hashlib.sha256(json.dumps(metadata,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        content_fingerprint=fingerprint
        prior=c.execute("SELECT id FROM dataset_versions WHERE dataset_type='prices' AND content_fingerprint=? AND metadata_fingerprint=?",(content_fingerprint,meta_fingerprint)).fetchone()
        if prior: version_id=prior['id']
        else:
            cur=c.execute('INSERT INTO dataset_versions(dataset_type,content_fingerprint,metadata_fingerprint,parent_version_id,import_run_id,created_at,change_kind,summary_json) VALUES(?,?,?,?,?,?,?,?)',( 'prices',content_fingerprint,meta_fingerprint,latest['id'] if latest else None,import_run_id,now,kind,json.dumps(change_summary,sort_keys=True)))
            version_id=cur.lastrowid
        cur=c.execute("INSERT INTO refresh_runs(started_at,status,triggered_by,source,instruments_requested,import_run_id,version_id,change_summary_json) VALUES(?,'running','local_import','local_csv',?,?,?,?)",(now,len(affected_symbols),import_run_id,version_id,json.dumps(change_summary,sort_keys=True)))
        refresh_id=cur.lastrowid
        c.execute('UPDATE data_import_runs SET version_id=?,refresh_run_id=?,content_sha256=?,change_summary_json=? WHERE id=?',(version_id,refresh_id,content_fingerprint,json.dumps(change_summary,sort_keys=True),import_run_id))
    step_start=time.monotonic(); prediction_outcomes={}
    if kind in {'new_session','backfill'}:
        try: prediction_outcomes=_refresh_current_forecasts(affected_symbols,version_id,fingerprint,refresh_id)
        except Exception as exc:
            with connect() as c: c.execute("UPDATE refresh_runs SET finished_at=?,status='failed',errors_json=? WHERE id=?",(datetime.now(timezone.utc).isoformat(),json.dumps([f'{type(exc).__name__}: {str(exc)[:400]}']),refresh_id))
            return {'refresh_run_id':refresh_id,'version_id':version_id,'release_id':None,'release_status':'failed','error':str(exc)[:400]}
    steps={'canonical_fingerprint_seconds':round(time.monotonic()-started,6),'inference_seconds':round(time.monotonic()-step_start,6)}
    historical=kind=='historical_correction'
    dependency_rows=[]
    components={
        'price_charts':('current','Imported canonical observations are read directly.'),
        'indicators':('current','Computed from canonical price observations on request.'),
        'feature_datasets':('stale','Feature rows are rebuilt on demand; cached datasets require rebuild.'),
        'targets':('stale','Five-session labels depend on benchmark-session price versions.'),
        'company_models':('stale','The imported price version differs from model training lineage; inference may use the last good eligible model, but retraining is explicit.'),
        'pooled_model_inputs':('stale','Pooled inputs depend on the changed local price version.'),
        'aggregator_inputs':('stale','Aggregator eligibility must be re-evaluated against the current aligned base-model inputs.'),
        'oos_predictions':('stale','Historical OOS records are preserved; changed price versions require re-evaluation.'),
        'calibration':('stale','Calibration is retained as historical evidence and must be refreshed after price changes.'),
        'backtests':('stale','Stored simulations are not silently rewritten after price changes.'),
        'rankings':('current','Rankings are computed from the current canonical dataset on request.'),
        'portfolio_valuations':('current','Valuations resolve latest local prices on request.'),
        'paper_trade_analytics':('current','Trade records are independent; valuation uses current local prices.'),
    }
    for symbol in affected_symbols:
        forecast=prediction_outcomes.get(symbol,{'status':'stale' if historical else 'insufficient_data','reason':'No eligible local model inference was available.'})
        components['forecasts']=(forecast['status'],forecast.get('reason','Current local-model inference completed.'))
        for component,(status,reason) in components.items():
            dependency_rows.append((component,symbol,status,fingerprint,version_id,now,json.dumps({'reason':reason,'change_kind':kind,'origin_date':forecast.get('origin_date'),'model_version_id':forecast.get('model_version_id')})))
    candidate_id=None
    try:
        with connect() as c:
            # Publish dependency lineage atomically with the active release and forecasts.
            c.executemany('INSERT INTO dependency_states(component,scope_key,status,input_fingerprint,version_id,updated_at,detail_json) VALUES(?,?,?,?,?,?,?) ON CONFLICT(component,scope_key) DO UPDATE SET status=excluded.status,input_fingerprint=excluded.input_fingerprint,version_id=excluded.version_id,updated_at=excluded.updated_at,detail_json=excluded.detail_json',dependency_rows)
            cur=c.execute("INSERT INTO research_releases(version_id,refresh_run_id,status,created_at,published_at,fingerprint,validation_json,errors_json) VALUES(?,?,'candidate',?,NULL,?,?, '[]')",(version_id,refresh_id,now,fingerprint,json.dumps({'canonical_validation':'passed','synthetic_rows_excluded':True,'inference':prediction_outcomes},sort_keys=True)))
            candidate_id=cur.lastrowid
            capture_release_snapshot(c,candidate_id)
            stale_reason='Historical price version changed; dependent outputs require explicit rebuild or retraining.' if historical else 'A newer local price version was imported; refreshing the forecast requires eligible history and an eligible model.'
            for symbol in affected_symbols:
                c.execute("UPDATE current_forecasts SET status='stale',stale_reason=? WHERE symbol=?",(stale_reason,symbol))
            candidates=c.execute('SELECT * FROM candidate_forecasts WHERE refresh_run_id=? ORDER BY symbol,model_version_id',(refresh_id,)).fetchall()
            for forecast in candidates:
                c.execute("INSERT INTO current_forecasts(model_version_id,symbol,origin_date,horizon_sessions,prediction,price_basis,created_at,version_id,input_fingerprint,status,stale_reason) VALUES(?,?,?,?,?,?,?,?,?,'current',NULL) ON CONFLICT(model_version_id,symbol,origin_date) DO UPDATE SET prediction=excluded.prediction,price_basis=excluded.price_basis,created_at=excluded.created_at,version_id=excluded.version_id,input_fingerprint=excluded.input_fingerprint,status='current',stale_reason=NULL",(forecast['model_version_id'],forecast['symbol'],forecast['origin_date'],forecast['horizon_sessions'],forecast['prediction'],forecast['price_basis'],forecast['created_at'],forecast['version_id'],forecast['input_fingerprint']))
            # Candidate becomes active only after all version/dependency writes succeed.
            c.execute("UPDATE research_releases SET status='superseded' WHERE status='published'")
            c.execute("UPDATE research_releases SET status='published',published_at=? WHERE id=?",(datetime.now(timezone.utc).isoformat(),candidate_id))
            c.execute("INSERT INTO app_state(key,value,updated_at) VALUES('active_release_id',?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value,updated_at=excluded.updated_at",(str(candidate_id),datetime.now(timezone.utc).isoformat()))
            c.execute("UPDATE refresh_runs SET finished_at=?,status='succeeded',instruments_updated=?,rows_written=?,latest_observation_date=(SELECT MAX(date) FROM prices WHERE quality<>'DEMO_SYNTHETIC'),errors_json='[]',warnings_json=?,dependencies_json=?,step_timings_json=? WHERE id=?",(datetime.now(timezone.utc).isoformat(),len(affected_symbols),change_summary.get('new_observations',0)+change_summary.get('corrected_observations',0),json.dumps(change_summary.get('warnings',[])),json.dumps([{'symbol':s,**prediction_outcomes.get(s,{'status':'insufficient_data'})} for s in affected_symbols]),json.dumps(steps),refresh_id))
        return {'refresh_run_id':refresh_id,'version_id':version_id,'release_id':candidate_id,'release_status':'published','dependencies':prediction_outcomes}
    except Exception as exc:
        with connect() as c:
            c.execute("UPDATE refresh_runs SET finished_at=?,status='failed',errors_json=?,step_timings_json=? WHERE id=?",(datetime.now(timezone.utc).isoformat(),json.dumps([f'{type(exc).__name__}: {str(exc)[:400]}']),json.dumps(steps),refresh_id))
            if candidate_id: c.execute("UPDATE research_releases SET status='failed',errors_json=? WHERE id=?",(json.dumps([str(exc)[:400]]),candidate_id))
        return {'refresh_run_id':refresh_id,'version_id':version_id,'release_id':None,'release_status':'failed','error':str(exc)[:400]}


def _coordinate_reference_import(import_run_id, dataset_type, affected_symbols, row_count):
    """Version non-price local inputs and stale only their declared dependents."""
    queries={
        'events':"SELECT symbol,date,kind,title,source,quality,published_at FROM events WHERE quality<>'DEMO_SYNTHETIC' ORDER BY symbol,date,id",
        'constituents':"SELECT as_of_date,symbol,company_name,sector,source_id,quality FROM constituent_snapshots WHERE as_of_date=(SELECT MAX(as_of_date) FROM constituent_snapshots) ORDER BY symbol",
        'membership':"SELECT symbol,valid_from,valid_to,source_id,quality FROM constituent_membership ORDER BY symbol,valid_from",
        'instruments':"SELECT symbol,as_of_date,name,sector,exchange,source_id,quality FROM instrument_universe WHERE as_of_date=(SELECT MAX(as_of_date) FROM instrument_universe) ORDER BY symbol",
        'calendar':"SELECT exchange,date,source_id,quality FROM exchange_sessions ORDER BY exchange,date",
    }
    now=datetime.now(timezone.utc).isoformat(); started=time.monotonic(); symbols=sorted(set(affected_symbols)); candidate_id=None
    try:
        with connect() as c:
            if dataset_type=='fundamentals':
                rows={'facts':[dict(r) for r in c.execute("SELECT symbol,period,revenue_growth,profit_growth,debt_equity,roe,source,quality,published_at FROM fundamentals WHERE quality<>'DEMO_SYNTHETIC' ORDER BY symbol,period")],
                      'point_in_time_observations':[dict(r) for r in c.execute("SELECT symbol,period,published_at,revenue_growth,profit_growth,debt_equity,roe,source,quality FROM fundamental_observations WHERE quality<>'DEMO_SYNTHETIC' ORDER BY symbol,period,published_at,source")]}
            else: rows=[dict(r) for r in c.execute(queries[dataset_type])]
            fingerprint=hashlib.sha256(json.dumps(rows,sort_keys=True,separators=(',',':'),default=str).encode()).hexdigest()
            run=c.execute('SELECT * FROM data_import_runs WHERE id=?',(import_run_id,)).fetchone()
            metadata={k:run[k] for k in ('source','dataset_description','license_reference','user_confirmed_use')}
            metadata_fp=hashlib.sha256(json.dumps(metadata,sort_keys=True,separators=(',',':')).encode()).hexdigest()
            parent=c.execute('SELECT id FROM dataset_versions WHERE dataset_type=? ORDER BY id DESC LIMIT 1',(dataset_type,)).fetchone()
            old=c.execute('SELECT id FROM dataset_versions WHERE dataset_type=? AND content_fingerprint=? AND metadata_fingerprint=?',(dataset_type,fingerprint,metadata_fp)).fetchone()
            if old: version_id=old['id']
            else:
                version_id=c.execute('INSERT INTO dataset_versions(dataset_type,content_fingerprint,metadata_fingerprint,parent_version_id,import_run_id,created_at,change_kind,summary_json) VALUES(?,?,?,?,?,?,?,?)',(dataset_type,fingerprint,metadata_fp,parent['id'] if parent else None,import_run_id,now,f'{dataset_type}_update',json.dumps({'rows':row_count,'affected_symbols':symbols},sort_keys=True))).lastrowid
            refresh_id=c.execute("INSERT INTO refresh_runs(started_at,status,triggered_by,source,instruments_requested,import_run_id,version_id,change_summary_json) VALUES(?,'running','local_import','local_csv',?,?,?,?)",(now,len(symbols),import_run_id,version_id,json.dumps({'dataset_type':dataset_type,'rows':row_count,'affected_symbols':symbols}))).lastrowid
            c.execute('UPDATE data_import_runs SET version_id=?,refresh_run_id=?,content_sha256=?,change_summary_json=? WHERE id=?',(version_id,refresh_id,fingerprint,json.dumps({'dataset_type':dataset_type,'rows':row_count,'affected_symbols':symbols}),import_run_id))
        dependent_map={
            'fundamentals':['feature_datasets','company_models','pooled_model_inputs','forecasts'],
            'events':['feature_datasets','company_models','pooled_model_inputs','forecasts'],
            'constituents':['instrument_universe','rankings','pooled_model_inputs','aggregator_inputs'],
            'membership':['historical_membership','pooled_model_inputs','aggregator_inputs','rankings'],
            'instruments':['instrument_universe','batch_training_candidates','rankings'],
            'calendar':['session_calendar_diagnostics'],
        }
        state='stale' if dataset_type not in {'calendar','instruments'} else 'current'
        dep_symbols=symbols or ['*']
        dependencies=[(component,symbol,state,fingerprint,version_id,now,json.dumps({'reason':f'{dataset_type} dataset version updated; retraining is not automatic.','row_count':row_count})) for component in dependent_map[dataset_type] for symbol in (dep_symbols if component not in {'instrument_universe','historical_membership','session_calendar_diagnostics','batch_training_candidates'} else ['*'])]
        with connect() as c:
            c.executemany('INSERT INTO dependency_states(component,scope_key,status,input_fingerprint,version_id,updated_at,detail_json) VALUES(?,?,?,?,?,?,?) ON CONFLICT(component,scope_key) DO UPDATE SET status=excluded.status,input_fingerprint=excluded.input_fingerprint,version_id=excluded.version_id,updated_at=excluded.updated_at,detail_json=excluded.detail_json',dependencies)
            if dataset_type in {'fundamentals','events','constituents','membership'}:
                for symbol in dep_symbols:
                    c.execute("UPDATE current_forecasts SET status='stale',stale_reason=? WHERE symbol=?",(f'{dataset_type} inputs changed; eligible model inference or retraining is required.',symbol))
            candidate_id=c.execute("INSERT INTO research_releases(version_id,refresh_run_id,status,created_at,published_at,fingerprint,validation_json,errors_json) VALUES(?,?,'candidate',?,NULL,?,?, '[]')",(version_id,refresh_id,now,fingerprint,json.dumps({'canonical_validation':'passed','dataset_type':dataset_type,'rows':row_count,'affected_symbols':symbols}))).lastrowid
            capture_release_snapshot(c,candidate_id)
            c.execute("UPDATE research_releases SET status='superseded' WHERE status='published'")
            c.execute("UPDATE research_releases SET status='published',published_at=? WHERE id=?",(datetime.now(timezone.utc).isoformat(),candidate_id))
            c.execute("INSERT INTO app_state(key,value,updated_at) VALUES('active_release_id',?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value,updated_at=excluded.updated_at",(str(candidate_id),datetime.now(timezone.utc).isoformat()))
            c.execute("UPDATE refresh_runs SET finished_at=?,status='succeeded',instruments_updated=?,rows_written=?,latest_observation_date=(SELECT MAX(date) FROM prices WHERE quality<>'DEMO_SYNTHETIC'),dependencies_json=?,step_timings_json=? WHERE id=?",(datetime.now(timezone.utc).isoformat(),len(symbols),row_count,json.dumps([{'component':x[0],'scope':x[1],'status':x[2]} for x in dependencies]),json.dumps({'version_and_publish_seconds':round(time.monotonic()-started,6)}),refresh_id))
        return {'refresh_run_id':refresh_id,'version_id':version_id,'release_id':candidate_id,'release_status':'published','dataset_type':dataset_type,'affected_symbols':symbols}
    except Exception as exc:
        with connect() as c:
            row=c.execute('SELECT MAX(id) FROM refresh_runs WHERE import_run_id=?',(import_run_id,)).fetchone()
            refresh_id=row[0]
            if refresh_id: c.execute("UPDATE refresh_runs SET finished_at=?,status='failed',errors_json=? WHERE id=?",(datetime.now(timezone.utc).isoformat(),json.dumps([f'{type(exc).__name__}: {str(exc)[:400]}']),refresh_id))
            if candidate_id: c.execute("UPDATE research_releases SET status='failed',errors_json=? WHERE id=?",(json.dumps([str(exc)[:400]]),candidate_id))
        return {'refresh_run_id':refresh_id,'release_id':None,'release_status':'failed','error':str(exc)[:400]}

def import_csv(contents: bytes, filename: str, *, dataset_description='', license_reference='', user_confirmed_use=False, allow_historical_corrections=False, max_bytes=25_000_000, provider_namespace=None, provider_quality=None) -> dict:
    """Strict, idempotent local daily OHLCV importer. Original bytes are retained."""
    if len(contents)>max_bytes: raise ValueError(f'CSV must be at most {max_bytes // 1_000_000} MB.')
    digest=hashlib.sha256(contents).hexdigest(); now=datetime.now(timezone.utc).isoformat()
    dataset_description=(dataset_description or '').strip()[:500]
    license_reference=(license_reference or '').strip()[:1000]
    user_confirmed_use=bool(user_confirmed_use)
    if user_confirmed_use and not license_reference: raise ValueError('A license or terms reference is required when confirming intended research use.')
    if provider_namespace not in {None,'tejhq'}: raise ValueError('Unsupported provider namespace.')
    if provider_quality not in {None,'TEJ_PROVIDER_LICENSED'}: raise ValueError('Unsupported provider quality label.')
    source_prefix=f'{provider_namespace}:' if provider_namespace else 'user_csv:'
    safe_name=Path(filename or 'upload.csv').name.replace('..','_')
    import_dir=DATA_DIR/'imports'; import_dir.mkdir(parents=True,exist_ok=True)
    saved=import_dir/f'{digest[:16]}-{safe_name}'
    if not saved.exists(): saved.write_bytes(contents)
    with connect() as c:
        prior=c.execute('SELECT * FROM data_import_runs WHERE sha256=?',(digest,)).fetchone()
        if prior and not (prior['status']=='blocked_correction_confirmation' and allow_historical_corrections):
            result=dict(prior); result['idempotent']=True
            for k in ('symbols_json','warnings_json','errors_json'):
                result[k.replace('_json','')]=json.loads(result.pop(k))
            return result
        if prior:
            run_id=prior['id']
            c.execute("UPDATE data_import_runs SET status='validating',started_at=?,finished_at=NULL,correction_confirmed=1 WHERE id=?",(now,run_id))
        else:
            cur=c.execute("INSERT INTO data_import_runs(started_at,filename,sha256,stored_path,status,dataset_description,license_reference,user_confirmed_use,correction_confirmed) VALUES(?,?,?,?,?,?,?,?,?)",(now,safe_name,digest,str(saved),'validating',dataset_description,license_reference,int(user_confirmed_use),int(allow_historical_corrections)))
            run_id=cur.lastrowid
    try:
        raw=pd.read_csv(io.BytesIO(contents), comment='#')
        raw.columns=[str(x).strip().lower() for x in raw.columns]
        required={'symbol','exchange','date','open','high','low','close','volume','price_basis','source'}
        missing=required-set(raw.columns)
        if missing: raise ValueError('Missing required columns: '+', '.join(sorted(missing)))
        if raw.empty: raise ValueError('CSV contains no data rows.')
        raw['symbol']=raw.symbol.astype(str).str.strip().str.upper()
        raw['exchange']=raw.exchange.astype(str).str.strip().str.upper()
        raw['source']=raw.source.astype(str).str.strip()
        raw['price_basis']=raw.price_basis.astype(str).str.strip().str.lower()
        if raw[['symbol','exchange','source']].isna().any().any() or raw[['symbol','exchange','source']].astype(str).apply(lambda s:s.str.strip().str.lower().isin(['','nan','none'])).any().any(): raise ValueError('Symbol, exchange, and source must be non-empty for every row.')
        if not set(raw.exchange).issubset({'NSE','NSE_INDEX'}): raise ValueError('exchange must be NSE or NSE_INDEX.')
        if ((raw.symbol=='NIFTY50')!=(raw.exchange=='NSE_INDEX')).any(): raise ValueError('Use NIFTY50 with NSE_INDEX for the benchmark; use NSE for equity symbols.')
        if not set(raw.price_basis).issubset({'raw_unadjusted','adjusted_close'}): raise ValueError('price_basis must be raw_unadjusted or adjusted_close.')
        if raw.source.str.contains(r'synthetic|illustrative|demo',case=False,regex=True).any(): raise ValueError('Illustrative/synthetic template rows are not importable. Replace them with real permitted observations.')
        parsed=pd.to_datetime(raw.date,errors='coerce')
        if parsed.isna().any(): raise ValueError(f'Invalid observation date in row(s): {list((parsed[parsed.isna()].index+2)[:10])}.')
        raw['date']=parsed.dt.strftime('%Y-%m-%d')
        numeric=['open','high','low','close','volume']
        for col in ['adjusted_close','dividends','stock_splits']:
            if col in raw:
                raw[col]=pd.to_numeric(raw[col],errors='coerce')
                if col in {'dividends','stock_splits'}: raw[col]=raw[col].fillna(0)
        for col in numeric: raw[col]=pd.to_numeric(raw[col],errors='coerce')
        if raw[numeric].isna().any().any(): raise ValueError('OHLCV and supplied adjustment/action values must be numeric and non-empty.')
        if (raw.volume%1!=0).any(): raise ValueError('Volume must be a whole-number share count.')
        for col in ('dividends','stock_splits'):
            if col in raw and (raw[col]<0).any(): raise ValueError(f'{col} cannot be negative.')
        if raw.duplicated(['symbol','date']).any(): raise ValueError('Duplicate symbol/date observations in upload; correct the source file and retry.')
        prices=raw[['open','high','low','close']]
        if (prices<=0).any().any() or (raw.volume<0).any(): raise ValueError('OHLC prices must be positive and volume cannot be negative.')
        bad=(raw.high<raw[['open','close','low']].max(axis=1))|(raw.low>raw[['open','close','high']].min(axis=1))
        if bad.any(): raise ValueError(f'Invalid OHLC relationships in row(s): {list((raw.index[bad]+2)[:10])}.')
        if 'adjusted_close' not in raw: raw['adjusted_close']=np.nan
        adjusted_missing=raw.adjusted_close.isna()
        if ((raw.price_basis=='adjusted_close')&adjusted_missing).any(): raise ValueError('adjusted_close basis requires an adjusted_close value in every row.')
        raw.loc[raw.price_basis=='raw_unadjusted','adjusted_close']=raw.loc[raw.price_basis=='raw_unadjusted','adjusted_close'].fillna(raw.loc[raw.price_basis=='raw_unadjusted','close'])
        if (raw.adjusted_close<=0).any(): raise ValueError('adjusted_close values must be positive when supplied.')
        warnings=[]
        for symbol,group in raw.sort_values('date').groupby('symbol'):
            dates=pd.to_datetime(group.date)
            if len(dates)>1:
                gaps=dates.diff().dt.days
                if (gaps>4).any(): warnings.append({'symbol':symbol,'warning':f'{int((gaps>4).sum())} date gap(s) exceed four calendar days; holidays and suspensions require review.'})
                if len(group)>1:
                    prior=group.close.shift(1); jumps=(group.close/prior-1).abs()>.35
                    for ix in group.index[jumps.fillna(False)]:
                        action_cols=[x for x in ('dividends','stock_splits') if x in raw]
                        has_action=any(float(raw.at[ix,x] or 0)>0 for x in action_cols)
                        if not has_action: warnings.append({'symbol':symbol,'date':raw.at[ix,'date'],'warning':'Close moved over 35% from prior supplied observation without a supplied corporate action; review basis/actions.'})
        symbols=sorted(raw.symbol.unique().tolist()); count=0; duplicates=0; new_count=0; corrected_count=0; unchanged_count=0; backfill_count=0
        staged=[]; correction_rows=[]
        with connect() as c:
            last_dates={s:r[0] for s,r in ((s,c.execute("SELECT MAX(date) FROM prices WHERE symbol=? AND quality<>'DEMO_SYNTHETIC'",(s,)).fetchone()) for s in symbols)}
            for _,r in raw.iterrows():
                existing=c.execute('SELECT * FROM prices WHERE symbol=? AND date=?',(r.symbol,r.date)).fetchone()
                incoming={'open':float(r.open),'high':float(r.high),'low':float(r.low),'close':float(r.close),'adjusted_close':float(r.adjusted_close),'volume':int(r.volume),'price_basis':str(r.price_basis)}
                for action_col in ('dividends','stock_splits'):
                    if action_col in raw: incoming[action_col]=float(r[action_col] or 0)
                changed=[]
                if existing and existing['quality']!='DEMO_SYNTHETIC':
                    for key,value in incoming.items():
                        if key in {'dividends','stock_splits'}:
                            old_row=c.execute(f"SELECT COALESCE(SUM({key}),0) FROM corporate_actions WHERE symbol=? AND date=? AND quality<>'DEMO_SYNTHETIC'",(r.symbol,r.date)).fetchone(); old=float(old_row[0] or 0)
                        else: old=existing[key]
                        if (old!=value) if key=='price_basis' else (old is None or abs(float(old)-float(value))>1e-9): changed.append(key)
                kind='correction' if existing and existing['quality']!='DEMO_SYNTHETIC' and changed else ('unchanged' if existing and existing['quality']!='DEMO_SYNTHETIC' else ('backfill' if last_dates.get(r.symbol) and r.date<last_dates[r.symbol] else 'new_session'))
                existing_actions=[dict(x) for x in c.execute("SELECT * FROM corporate_actions WHERE symbol=? AND date=? AND source=?",(r.symbol,r.date,f'{source_prefix}{r.source}'))]
                entry={'row':r,'existing':dict(existing) if existing else None,'existing_actions':existing_actions,'incoming':incoming,'changed':changed,'kind':kind}
                staged.append(entry)
                if kind=='correction': correction_rows.append(entry)
        if correction_rows and not allow_historical_corrections:
            warnings.extend({'symbol':x['row'].symbol,'date':x['row'].date,'warning':'Historical values differ in '+', '.join(x['changed'])+'; canonical row was not changed. Re-import with explicit historical-correction confirmation to revise it.'} for x in correction_rows)
            summary={'change_kind':'correction_confirmation_required','new_observations':0,'corrected_observations':len(correction_rows),'unchanged_observations':len(staged)-len(correction_rows),'backfills':0,'affected_symbols':[],'warnings':warnings}
            with connect() as c: c.execute("UPDATE data_import_runs SET finished_at=?,status='blocked_correction_confirmation',source=?,rows_read=?,rows_imported=0,accepted_rows=0,rejected_rows=?,duplicates=?,symbols_json=?,warnings_json=?,errors_json=?,change_summary_json=? WHERE id=?",(datetime.now(timezone.utc).isoformat(),','.join(sorted(raw.source.unique())),len(raw),len(correction_rows),len(correction_rows),json.dumps(symbols),json.dumps(warnings),json.dumps(['Explicit historical correction confirmation is required.']),json.dumps(summary,sort_keys=True),run_id))
            return {'run_id':run_id,'status':'blocked_correction_confirmation','rows_read':len(raw),'rows_imported':0,'correction_count':len(correction_rows),'symbols':symbols,'warnings':warnings,'message':'No canonical rows changed. Review the listed revisions, then explicitly confirm and re-import.'}
        source_fingerprint=hashlib.sha256(json.dumps({'source_names':sorted(raw.source.unique()),'description':dataset_description,'license_reference':license_reference,'user_confirmed_use':user_confirmed_use},sort_keys=True,separators=(',',':')).encode()).hexdigest()
        for entry in staged:
            if entry['kind']=='new_session': new_count+=1
            elif entry['kind']=='backfill': new_count+=1; backfill_count+=1
            elif entry['kind']=='correction': corrected_count+=1
            else: unchanged_count+=1
        change_kind='historical_correction' if corrected_count else ('new_session' if new_count and not backfill_count else ('backfill' if new_count else ('metadata_only' if unchanged_count else 'unchanged')))
        affected_symbols=sorted({x['row'].symbol for x in staged if x['kind']!='unchanged'})
        change_summary={'change_kind':change_kind,'new_observations':new_count,'corrected_observations':corrected_count,'unchanged_observations':unchanged_count,'backfills':backfill_count,'affected_symbols':affected_symbols,'warnings':warnings,'source_metadata_fingerprint':source_fingerprint}
        with connect() as c:
            for source_name in sorted(raw.source.unique()):
                c.execute('INSERT OR IGNORE INTO data_sources(source_name,dataset_description,license_reference,user_confirmed_use,created_at) VALUES(?,?,?,?,?)',(source_name,dataset_description,license_reference,int(user_confirmed_use),now))
            quality=provider_quality or ('USER_CONFIRMED_UNVERIFIED' if user_confirmed_use else 'USER_IMPORTED_UNVERIFIED')
            for symbol,exchange in raw[['symbol','exchange']].drop_duplicates().itertuples(index=False,name=None):
                c.execute("INSERT INTO securities(symbol,name,sector,exchange,index_member,is_demo) VALUES(?,?,?, ?,0,0) ON CONFLICT(symbol) DO UPDATE SET exchange=excluded.exchange,is_demo=0",(symbol,symbol,'Unclassified',exchange))
            for entry in staged:
                r=entry['row']; existing=entry['existing']; values=entry['incoming']
                if entry['kind']=='unchanged':
                    duplicates+=1
                    continue
                if entry['kind']=='correction':
                    old={k:existing[k] for k in values if k not in {'dividends','stock_splits'}}
                    for action_col in ('dividends','stock_splits'):
                        old[action_col]=float(c.execute(f"SELECT COALESCE(SUM({action_col}),0) FROM corporate_actions WHERE symbol=? AND date=? AND quality<>'DEMO_SYNTHETIC'",(r.symbol,r.date)).fetchone()[0] or 0)
                    c.execute('INSERT INTO data_revisions(dataset_type,symbol,observation_date,change_kind,old_value_json,new_value_json,import_run_id,created_at) VALUES(?,?,?,?,?,?,?,?)',('prices',r.symbol,r.date,'historical_correction',json.dumps(old,sort_keys=True),json.dumps(values,sort_keys=True),run_id,now))
                if existing:
                    c.execute('INSERT OR IGNORE INTO archived_demo_prices(symbol,date,open,high,low,close,adjusted_close,volume,source,retrieved_at,quality,price_basis) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',(existing['symbol'],existing['date'],existing['open'],existing['high'],existing['low'],existing['close'],existing['adjusted_close'],existing['volume'],existing['source'],existing['retrieved_at'],existing['quality'],'synthetic'))
                source=f'{source_prefix}{r.source}'
                basis=r.price_basis
                c.execute("INSERT INTO prices(symbol,date,open,high,low,close,adjusted_close,volume,source,retrieved_at,quality,price_basis) VALUES(?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(symbol,date) DO UPDATE SET open=excluded.open,high=excluded.high,low=excluded.low,close=excluded.close,adjusted_close=excluded.adjusted_close,volume=excluded.volume,source=excluded.source,retrieved_at=excluded.retrieved_at,quality=excluded.quality,price_basis=excluded.price_basis WHERE prices.quality='DEMO_SYNTHETIC' OR (prices.quality<>'DEMO_SYNTHETIC' AND ?=1)",(r.symbol,r.date,float(r.open),float(r.high),float(r.low),float(r.close),float(r.adjusted_close),int(r.volume),source,now,quality,basis,int(allow_historical_corrections)))
                instrument_id=f'{r.exchange}_{r.symbol}'
                payload={k:(None if k not in raw or pd.isna(r.get(k)) else float(r[k])) for k in ['open','high','low','close','adjusted_close','volume','dividends','stock_splits']}
                pj=json.dumps(payload,sort_keys=True,allow_nan=False); ph=hashlib.sha256(pj.encode()).hexdigest()
                validation='PROVIDER_LICENSED_VALIDATED' if provider_namespace else ('USER_ATTESTED_UNVERIFIED' if user_confirmed_use else 'STRUCTURALLY_VALID_UNVERIFIED_PROVENANCE')
                c.execute('INSERT OR IGNORE INTO raw_price_observations(instrument_id,symbol,provider_symbol,date,source,retrieved_at,payload_hash,payload_json,validation_status) VALUES(?,?,?,?,?,?,?,?,?)',(instrument_id,r.symbol,r.symbol,r.date,source,now,ph,pj,validation))
                if 'dividends' in raw or 'stock_splits' in raw:
                    prior_action=c.execute('SELECT dividends,stock_splits FROM corporate_actions WHERE symbol=? AND date=? AND source=?',(r.symbol,r.date,source)).fetchone()
                    dividends=float(r['dividends'] or 0) if 'dividends' in raw else (float(prior_action['dividends'] or 0) if prior_action else 0.0)
                    splits=float(r['stock_splits'] or 0) if 'stock_splits' in raw else (float(prior_action['stock_splits'] or 0) if prior_action else 0.0)
                    if dividends or splits:
                        c.execute('INSERT INTO corporate_actions(instrument_id,symbol,date,dividends,stock_splits,source,retrieved_at,quality) VALUES(?,?,?,?,?,?,?,?) ON CONFLICT(symbol,date,source) DO UPDATE SET dividends=excluded.dividends,stock_splits=excluded.stock_splits,retrieved_at=excluded.retrieved_at,quality=excluded.quality',(instrument_id,r.symbol,r.date,dividends,splits,source,now,quality))
                    elif prior_action: c.execute('DELETE FROM corporate_actions WHERE symbol=? AND date=? AND source=?',(r.symbol,r.date,source))
                count+=1
            c.execute("UPDATE data_import_runs SET finished_at=?,status=?,source=?,rows_read=?,rows_imported=?,accepted_rows=?,rejected_rows=0,duplicates=?,symbols_json=?,warnings_json=?,earliest_date=?,latest_date=?,price_basis=?,change_summary_json=? WHERE id=?",(datetime.now(timezone.utc).isoformat(),'succeeded' if not duplicates else ('unchanged' if not count else 'partial_success'),','.join(sorted(raw.source.unique())),len(raw),count,count,duplicates,json.dumps(symbols),json.dumps(warnings),raw.date.min(),raw.date.max(),','.join(sorted(raw.price_basis.unique())),json.dumps(change_summary,sort_keys=True),run_id))
        lifecycle=_coordinate_price_import(run_id,change_summary,{'source_names':sorted(raw.source.unique()),'description':dataset_description,'license_reference':license_reference,'user_confirmed_use':user_confirmed_use},affected_symbols) if change_kind!='unchanged' else None
        if lifecycle and lifecycle.get('release_status')!='published':
            # Canonical rows are the legacy read surface. Roll them back when
            # publication fails so those readers continue to see the last good
            # state. Preserve raw payloads and revision records as failed-candidate
            # lineage; they are never promoted to the active research state.
            with connect() as c:
                for entry in staged:
                    if entry['kind']=='unchanged': continue
                    row=entry['row']; old=entry['existing']
                    if old:
                        c.execute('INSERT OR REPLACE INTO prices(symbol,date,open,high,low,close,adjusted_close,volume,source,retrieved_at,quality,price_basis) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',tuple(old.get(k) for k in ('symbol','date','open','high','low','close','adjusted_close','volume','source','retrieved_at','quality','price_basis')))
                    else:
                        c.execute('DELETE FROM prices WHERE symbol=? AND date=?',(row.symbol,row.date))
                    action_source=f'{source_prefix}{row.source}'
                    c.execute('DELETE FROM corporate_actions WHERE symbol=? AND date=? AND source=?',(row.symbol,row.date,action_source))
                    for action in entry['existing_actions']:
                        c.execute('INSERT OR REPLACE INTO corporate_actions(instrument_id,symbol,date,dividends,stock_splits,source,retrieved_at,quality) VALUES(?,?,?,?,?,?,?,?)',tuple(action.get(k) for k in ('instrument_id','symbol','date','dividends','stock_splits','source','retrieved_at','quality')))
                    c.execute("UPDATE raw_price_observations SET validation_status='CANDIDATE_PUBLICATION_FAILED' WHERE source LIKE ? AND retrieved_at=? AND symbol IN (%s)" % ','.join('?' for _ in symbols),(source_prefix+'%',now,*symbols))
                c.execute("UPDATE data_import_runs SET status='failed_publication',errors_json=? WHERE id=?",(json.dumps([lifecycle.get('error','Research release publication failed; canonical rows were restored.')]),run_id))
            lifecycle['canonical_rows_restored']=True
        result_status='failed_publication' if lifecycle and lifecycle.get('release_status')!='published' else ('succeeded' if not duplicates else ('unchanged' if not count else 'partial_success'))
        return {'run_id':run_id,'rows_read':len(raw),'rows_imported':count,'accepted_rows':count,'rejected_rows':0,'duplicates':duplicates,'symbols':symbols,'warnings':warnings,'quality':quality,'idempotent':False,'original_file':str(saved),'status':result_status,'change_summary':change_summary,'lifecycle':lifecycle,'user_confirmed_use':user_confirmed_use}
    except Exception as e:
        with connect() as c: c.execute("UPDATE data_import_runs SET finished_at=?,status='failed',errors_json=? WHERE id=?",(datetime.now(timezone.utc).isoformat(),json.dumps([str(e)[:500]]),run_id))
        raise ValueError(str(e)) from e


def import_price_zip(contents:bytes,filename:str,*,dataset_description='',license_reference='',user_confirmed_use=False,allow_historical_corrections=False):
    """Import each daily-price CSV member atomically; never extract paths to disk."""
    if len(contents)>25_000_000: raise ValueError('ZIP must be at most 25 MB compressed.')
    digest=hashlib.sha256(contents).hexdigest(); safe=Path(filename or 'bundle.zip').name
    target=DATA_DIR/'imports'/f'{digest[:16]}-{safe}'; target.parent.mkdir(parents=True,exist_ok=True)
    if not target.exists(): target.write_bytes(contents)
    try: archive=zipfile.ZipFile(io.BytesIO(contents))
    except zipfile.BadZipFile as exc: raise ValueError('Upload is not a valid ZIP file.') from exc
    with archive:
        if len(archive.infolist())>1000: raise ValueError('ZIP contains more than 1,000 entries.')
        files=[item for item in archive.infolist() if not item.is_dir() and item.filename.lower().endswith('.csv')]
        if not files: raise ValueError('ZIP must contain at least one CSV file.')
        if len(files)>500: raise ValueError('ZIP contains more than 500 CSV files.')
        if sum(f.file_size for f in files)>100_000_000 or any(f.file_size>25_000_000 for f in files): raise ValueError('ZIP expanded CSV content exceeds the 100 MB bundle or 25 MB per-file limit.')
        if any(f.flag_bits & 0x1 for f in files): raise ValueError('Encrypted ZIP members are not supported.')
        if any(f.file_size>1_000_000 and f.file_size/max(1,f.compress_size)>1000 for f in files): raise ValueError('ZIP member compression ratio exceeds the safe limit.')
        results=[]; errors=[]
        for info in files:
            if info.filename.startswith('/') or '..' in Path(info.filename).parts:
                errors.append({'file':info.filename,'error':'Unsafe member path rejected.'}); continue
            try:
                results.append(import_csv(archive.read(info),Path(info.filename).name,dataset_description=dataset_description,license_reference=license_reference,user_confirmed_use=user_confirmed_use,allow_historical_corrections=allow_historical_corrections))
            except Exception as exc: errors.append({'file':Path(info.filename).name,'error':str(exc)[:500]})
    return {'status':'succeeded' if not errors else 'partial_success' if results else 'failed','bundle_file':str(target),'files_imported':len(results),'files_failed':len(errors),'rows_imported':sum(r.get('rows_imported',0) for r in results),'results':results,'errors':errors,'atomicity':'Each CSV member is transactionally imported as its own run; a bad member cannot roll back an earlier committed member.'}

def _normalise_publication_time(value):
    raw=str(value).strip()
    if re.fullmatch(r'\d{4}-\d{2}-\d{2}',raw):
        # A date-only disclosure has unknown intraday availability; conservatively
        # make it available only after that UTC day, hence no same-session use.
        return raw+'T23:59:59+00:00'
    parsed=pd.to_datetime(raw,errors='coerce',utc=True)
    return None if pd.isna(parsed) else parsed.isoformat()


def import_fundamentals_csv(contents:bytes,filename:str,*,dataset_description='',license_reference='',user_confirmed_use=False,source_name=None) -> dict:
    if user_confirmed_use and not license_reference: raise ValueError('A license or terms reference is required when confirming intended research use.')
    try: df=pd.read_csv(io.BytesIO(contents),comment='#'); df.columns=[str(x).strip().lower() for x in df.columns]
    except Exception as e: raise ValueError(f'Could not read fundamentals CSV: {e}')
    required={'symbol','period','revenue_growth','profit_growth','debt_equity','roe'}; missing=required-set(df.columns)
    if missing: raise ValueError('Missing required columns: '+', '.join(sorted(missing)))
    df['symbol']=df.symbol.astype(str).str.strip().str.upper(); df['period']=df.period.astype(str).str.strip()
    if df.symbol.isin(['','NAN']).any() or df.period.isin(['','NAN']).any(): raise ValueError('Every fundamentals row requires symbol and period.')
    for col in required-{'symbol','period'}: df[col]=pd.to_numeric(df[col],errors='coerce')
    if df[list(required-{'symbol','period'})].isna().any().any(): raise ValueError('Fundamentals metrics must be numeric and non-empty.')
    if 'published_at' in df:
        df['published_at']=df.published_at.map(_normalise_publication_time)
        df['published_at']=pd.to_datetime(df.published_at,errors='coerce',utc=True)
        if df.published_at.isna().any(): raise ValueError('published_at must be a valid point-in-time publication date for every supplied row.')
    now=datetime.now(timezone.utc).isoformat(); source_name=(source_name or f'user_csv:{Path(filename).name}').strip()
    run_id,prior=_start_metadata_import(contents,filename,'fundamentals',dataset_description,license_reference,user_confirmed_use)
    if prior: return {'run_id':prior['id'],'status':prior['status'],'rows_imported':prior['rows_imported'],'idempotent':True}
    quality='USER_CONFIRMED_UNVERIFIED' if user_confirmed_use else 'USER_IMPORTED_UNVERIFIED'; warnings=[]
    if 'published_at' not in df: warnings.append({'warning':'No published_at column: these rows are not eligible for point-in-time model features.'})
    try:
        with connect() as c:
            c.execute('INSERT OR IGNORE INTO data_sources(source_name,dataset_description,license_reference,user_confirmed_use,created_at) VALUES(?,?,?,?,?)',(source_name,dataset_description,license_reference,int(user_confirmed_use),now))
            for symbol in df.symbol.unique(): c.execute("INSERT OR IGNORE INTO securities(symbol,name,sector,index_member,is_demo) VALUES(?,?,?,0,0)",(symbol,symbol,'Unclassified'))
            for r in df.itertuples(index=False):
                published=pd.to_datetime(getattr(r,'published_at'),utc=True).isoformat() if hasattr(r,'published_at') else None
                c.execute("INSERT INTO fundamentals(symbol,period,revenue_growth,profit_growth,debt_equity,roe,source,retrieved_at,quality,published_at) VALUES(?,?,?,?,?,?,?,?,?,?) ON CONFLICT(symbol,period) DO UPDATE SET revenue_growth=excluded.revenue_growth,profit_growth=excluded.profit_growth,debt_equity=excluded.debt_equity,roe=excluded.roe,source=excluded.source,retrieved_at=excluded.retrieved_at,quality=excluded.quality,published_at=excluded.published_at",(r.symbol,r.period,float(r.revenue_growth),float(r.profit_growth),float(r.debt_equity),float(r.roe),source_name,now,quality,published))
                if published: c.execute('INSERT OR IGNORE INTO fundamental_observations(symbol,period,published_at,revenue_growth,profit_growth,debt_equity,roe,source,quality,import_run_id) VALUES(?,?,?,?,?,?,?,?,?,?)',(r.symbol,r.period,published,float(r.revenue_growth),float(r.profit_growth),float(r.debt_equity),float(r.roe),source_name,quality,run_id))
            c.execute("UPDATE data_import_runs SET finished_at=?,status='succeeded',source=?,rows_read=?,rows_imported=?,accepted_rows=?,symbols_json=?,warnings_json=?,earliest_date=?,latest_date=? WHERE id=?",(datetime.now(timezone.utc).isoformat(),source_name,len(df),len(df),len(df),json.dumps(sorted(df.symbol.unique())),json.dumps(warnings),str(df.published_at.min().date()) if 'published_at' in df else None,str(df.published_at.max().date()) if 'published_at' in df else None,run_id))
        lifecycle=_coordinate_reference_import(run_id,'fundamentals',df.symbol.unique().tolist(),len(df))
        return {'run_id':run_id,'rows_imported':len(df),'symbols':sorted(df.symbol.unique().tolist()),'quality':quality,'warnings':warnings,'point_in_time_available':'published_at' in df,'lifecycle':lifecycle,'user_confirmed_use':user_confirmed_use}
    except Exception as e:
        with connect() as c: c.execute("UPDATE data_import_runs SET finished_at=?,status='failed',errors_json=? WHERE id=?",(datetime.now(timezone.utc).isoformat(),json.dumps([str(e)[:500]]),run_id))
        raise ValueError(str(e)) from e


def import_events_csv(contents:bytes,filename:str,*,dataset_description='',license_reference='',user_confirmed_use=False,source_name=None) -> dict:
    if user_confirmed_use and not license_reference: raise ValueError('A license or terms reference is required when confirming intended research use.')
    try: df=pd.read_csv(io.BytesIO(contents),comment='#'); df.columns=[str(x).strip().lower() for x in df.columns]
    except Exception as e: raise ValueError(f'Could not read events CSV: {e}')
    required={'symbol','date','kind','title'}; missing=required-set(df.columns)
    if missing: raise ValueError('Missing required columns: '+', '.join(sorted(missing)))
    df['symbol']=df.symbol.astype(str).str.strip().str.upper(); df['date']=pd.to_datetime(df.date,errors='coerce')
    if df.date.isna().any() or df[['symbol','kind','title']].astype(str).apply(lambda col:col.str.strip().isin(['','nan'])).any().any(): raise ValueError('Every event requires valid symbol, date, kind, and title.')
    if 'published_at' in df:
        df['published_at']=df.published_at.map(_normalise_publication_time)
        df['published_at']=pd.to_datetime(df.published_at,errors='coerce',utc=True)
        if df.published_at.isna().any(): raise ValueError('published_at must be valid for every supplied event row.')
    now=datetime.now(timezone.utc).isoformat(); source_name=(source_name or f'user_csv:{Path(filename).name}').strip()
    run_id,prior=_start_metadata_import(contents,filename,'events',dataset_description,license_reference,user_confirmed_use)
    if prior: return {'run_id':prior['id'],'status':prior['status'],'rows_imported':prior['rows_imported'],'idempotent':True}
    quality='USER_CONFIRMED_UNVERIFIED' if user_confirmed_use else 'USER_IMPORTED_UNVERIFIED'; warnings=[]
    if 'published_at' not in df: warnings.append({'warning':'No published_at column: event availability cannot be established point-in-time.'})
    try:
        with connect() as c:
            c.execute('INSERT OR IGNORE INTO data_sources(source_name,dataset_description,license_reference,user_confirmed_use,created_at) VALUES(?,?,?,?,?)',(source_name,dataset_description,license_reference,int(user_confirmed_use),now))
            for symbol in df.symbol.unique(): c.execute("INSERT OR IGNORE INTO securities(symbol,name,sector,index_member,is_demo) VALUES(?,?,?,0,0)",(symbol,symbol,'Unclassified'))
            for r in df.itertuples(index=False):
                published=pd.to_datetime(getattr(r,'published_at'),utc=True).isoformat() if hasattr(r,'published_at') else None
                c.execute('INSERT INTO events(symbol,date,kind,title,source,quality,published_at) VALUES(?,?,?,?,?,?,?)',(r.symbol,r.date.strftime('%Y-%m-%d'),str(r.kind),str(r.title),source_name,quality,published))
            c.execute("UPDATE data_import_runs SET finished_at=?,status='succeeded',source=?,rows_read=?,rows_imported=?,accepted_rows=?,symbols_json=?,warnings_json=?,earliest_date=?,latest_date=? WHERE id=?",(datetime.now(timezone.utc).isoformat(),source_name,len(df),len(df),len(df),json.dumps(sorted(df.symbol.unique())),json.dumps(warnings),str(df.date.min().date()),str(df.date.max().date()),run_id))
        lifecycle=_coordinate_reference_import(run_id,'events',df.symbol.unique().tolist(),len(df))
        return {'run_id':run_id,'rows_imported':len(df),'symbols':sorted(df.symbol.unique().tolist()),'quality':quality,'warnings':warnings,'point_in_time_available':'published_at' in df,'lifecycle':lifecycle,'user_confirmed_use':user_confirmed_use}
    except Exception as e:
        with connect() as c: c.execute("UPDATE data_import_runs SET finished_at=?,status='failed',errors_json=? WHERE id=?",(datetime.now(timezone.utc).isoformat(),json.dumps([str(e)[:500]]),run_id))
        raise ValueError(str(e)) from e


def _start_metadata_import(contents,filename,dataset_kind,description,license_reference,user_confirmed):
    digest=hashlib.sha256(contents).hexdigest(); now=datetime.now(timezone.utc).isoformat(); safe=Path(filename or 'upload.csv').name
    target=DATA_DIR/'imports'/f'{digest[:16]}-{safe}'; target.parent.mkdir(parents=True,exist_ok=True)
    if not target.exists(): target.write_bytes(contents)
    with connect() as c:
        prior=c.execute('SELECT * FROM data_import_runs WHERE sha256=?',(digest,)).fetchone()
        if prior: return None,dict(prior)
        cur=c.execute("INSERT INTO data_import_runs(started_at,filename,sha256,stored_path,status,dataset_description,license_reference,user_confirmed_use) VALUES(?,?,?,?,?,?,?,?)",(now,safe,digest,str(target),f'{dataset_kind}_validating',description,license_reference,int(user_confirmed)))
    return cur.lastrowid,None


def import_constituents_csv(contents:bytes,filename:str,*,as_of_date:str,source_name:str,dataset_description='',license_reference='',user_confirmed_use=False):
    """Import a dated 50-name snapshot; it never manufactures earlier membership."""
    if not source_name.strip(): raise ValueError('Provide the declared source name for this constituent snapshot.')
    if user_confirmed_use and not license_reference: raise ValueError('Provide a license/terms reference for the user-use confirmation.')
    as_of=pd.to_datetime(as_of_date,errors='coerce')
    if pd.isna(as_of): raise ValueError('A valid constituent as-of date is required.')
    try: df=pd.read_csv(io.BytesIO(contents),comment='#')
    except Exception as exc: raise ValueError(f'Could not read constituents CSV: {exc}') from exc
    df.columns=[str(v).strip().lower().replace(' ','_') for v in df.columns]
    if _is_illustrative(df,filename,source_name): raise ValueError('Illustrative template rows are intentionally not importable. Replace them with a genuine source file.')
    if 'company_name' in df and 'name' not in df: df=df.rename(columns={'company_name':'name'})
    if 'industry' in df and 'sector' not in df: df=df.rename(columns={'industry':'sector'})
    missing={'symbol','name'}-set(df.columns)
    if missing: raise ValueError('Missing constituent columns: '+', '.join(sorted(missing)))
    df['symbol']=df.symbol.astype(str).str.strip().str.upper(); df['name']=df.name.astype(str).str.strip()
    if df.symbol.isin(['','NAN','NIFTY50']).any() or df.name.isin(['','nan']).any(): raise ValueError('Every constituent requires a symbol and company name; omit the index row.')
    if df.symbol.duplicated().any(): raise ValueError('Duplicate constituent symbols are not allowed.')
    if len(df)!=50: raise ValueError(f'NIFTY 50 constituent snapshots must contain exactly 50 unique equity symbols; received {len(df)}.')
    with connect() as c:
        prior_date=c.execute('SELECT MAX(as_of_date) FROM constituent_snapshots').fetchone()[0]
        prior_symbols=[r[0] for r in c.execute('SELECT symbol FROM constituent_snapshots WHERE as_of_date=?',(prior_date,))] if prior_date else []
    digest=hashlib.sha256(contents).hexdigest(); now=datetime.now(timezone.utc).isoformat(); safe=Path(filename or 'constituents.csv').name
    target=DATA_DIR/'imports'/f'{digest[:16]}-{safe}'; target.parent.mkdir(parents=True,exist_ok=True)
    if not target.exists(): target.write_bytes(contents)
    with connect() as c:
        existing=c.execute('SELECT * FROM data_import_runs WHERE sha256=?',(digest,)).fetchone()
        if existing: return {'run_id':existing['id'],'status':existing['status'],'idempotent':True,'rows_imported':existing['rows_imported']}
        cur=c.execute("INSERT INTO data_import_runs(started_at,filename,sha256,stored_path,status,source,rows_read,rows_imported,accepted_rows,dataset_description,license_reference,user_confirmed_use,earliest_date,latest_date) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",(now,safe,digest,str(target),'constituents_validating',source_name,len(df),0,0,dataset_description,license_reference,int(user_confirmed_use),as_of.strftime('%Y-%m-%d'),as_of.strftime('%Y-%m-%d'))); run_id=cur.lastrowid
        c.execute('INSERT OR IGNORE INTO data_sources(source_name,dataset_description,license_reference,user_confirmed_use,created_at) VALUES(?,?,?,?,?)',(source_name,dataset_description,license_reference,int(user_confirmed_use),now))
        source_id=c.execute('SELECT id FROM data_sources WHERE source_name=? AND dataset_description=? AND license_reference=?',(source_name,dataset_description,license_reference)).fetchone()[0]
        quality='USER_CONFIRMED_UNVERIFIED' if user_confirmed_use else 'USER_IMPORTED_UNVERIFIED'
        c.execute("UPDATE securities SET index_member=0 WHERE exchange='NSE' AND symbol<>'NIFTY50'")
        for r in df.itertuples(index=False):
            sector=getattr(r,'sector',None); sector=None if pd.isna(sector) else str(sector)
            c.execute("INSERT INTO securities(symbol,name,sector,exchange,index_member,is_demo) VALUES(?,?,?,'NSE',1,0) ON CONFLICT(symbol) DO UPDATE SET name=excluded.name,sector=COALESCE(excluded.sector,securities.sector),exchange='NSE',index_member=1,is_demo=0",(r.symbol,r.name,sector))
            c.execute('INSERT INTO constituent_snapshots(as_of_date,symbol,company_name,sector,source_id,import_run_id,quality) VALUES(?,?,?,?,?,?,?)',(as_of.strftime('%Y-%m-%d'),r.symbol,r.name,sector,source_id,run_id,quality))
            c.execute('INSERT OR REPLACE INTO instrument_universe(symbol,as_of_date,name,sector,exchange,source_id,import_run_id,quality) VALUES(?,?,?,?,?,?,?,?)',(r.symbol,as_of.strftime('%Y-%m-%d'),r.name,sector,'NSE',source_id,run_id,quality))
        c.execute("UPDATE data_import_runs SET finished_at=?,status='succeeded',rows_imported=?,accepted_rows=?,symbols_json=? WHERE id=?",(datetime.now(timezone.utc).isoformat(),len(df),len(df),json.dumps(sorted(df.symbol.tolist())),run_id))
    lifecycle=_coordinate_reference_import(run_id,'constituents',sorted(set(df.symbol.tolist())|set(prior_symbols)),len(df))
    return {'run_id':run_id,'status':'succeeded','as_of_date':as_of.strftime('%Y-%m-%d'),'rows_imported':len(df),'symbols':sorted(df.symbol.tolist()),'historical_membership_created':False,'lifecycle':lifecycle,'quality':'USER_CONFIRMED_UNVERIFIED' if user_confirmed_use else 'USER_IMPORTED_UNVERIFIED','source_independently_verified':False}


def import_membership_history_csv(contents:bytes,filename:str,*,source_name:str,dataset_description='',license_reference='',user_confirmed_use=False):
    """Import effective-from/to intervals; pooled training uses only these intervals."""
    if not source_name.strip(): raise ValueError('Provide the declared source name for membership history.')
    try: df=pd.read_csv(io.BytesIO(contents),comment='#')
    except Exception as exc: raise ValueError(f'Could not read membership CSV: {exc}') from exc
    df.columns=[str(v).strip().lower() for v in df.columns]
    if _is_illustrative(df,filename,source_name): raise ValueError('Illustrative template rows are intentionally not importable. Replace them with genuine membership data.')
    required={'symbol','valid_from','valid_to'}
    if not required.issubset(df.columns): raise ValueError('Membership history requires symbol,valid_from,valid_to; use an empty valid_to for an open interval.')
    df['symbol']=df.symbol.astype(str).str.strip().str.upper(); df['valid_from']=pd.to_datetime(df.valid_from,errors='coerce')
    df['valid_to']=pd.to_datetime(df.valid_to,errors='coerce')
    if df.valid_from.isna().any(): raise ValueError('Every membership interval requires a valid valid_from date.')
    if df.duplicated(['symbol','valid_from']).any(): raise ValueError('Duplicate symbol/valid_from membership rows.')
    for r in df.itertuples():
        if pd.notna(r.valid_to) and r.valid_to<r.valid_from: raise ValueError('valid_to cannot precede valid_from.')
    for symbol,group in df.sort_values('valid_from').groupby('symbol'):
        prior_end=None; first=True
        for r in group.itertuples():
            if prior_end is None and not first: raise ValueError(f'{symbol} has an open membership interval followed by another interval.')
            if prior_end is not None and r.valid_from<=prior_end: raise ValueError(f'{symbol} has overlapping membership intervals.')
            prior_end=r.valid_to; first=False
    if user_confirmed_use and not license_reference: raise ValueError('Provide a license/terms reference for the user-use confirmation.')
    digest=hashlib.sha256(contents).hexdigest(); now=datetime.now(timezone.utc).isoformat(); safe=Path(filename or 'membership.csv').name
    target=DATA_DIR/'imports'/f'{digest[:16]}-{safe}'; target.parent.mkdir(parents=True,exist_ok=True)
    if not target.exists(): target.write_bytes(contents)
    with connect() as c:
        prior=c.execute('SELECT * FROM data_import_runs WHERE sha256=?',(digest,)).fetchone()
        if prior: return {'run_id':prior['id'],'status':prior['status'],'idempotent':True,'rows_imported':prior['rows_imported']}
        cur=c.execute("INSERT INTO data_import_runs(started_at,filename,sha256,stored_path,status,source,rows_read,dataset_description,license_reference,user_confirmed_use) VALUES(?,?,?,?,?,?,?,?,?,?)",(now,safe,digest,str(target),'membership_validating',source_name,len(df),dataset_description,license_reference,int(user_confirmed_use))); run_id=cur.lastrowid
        c.execute('INSERT OR IGNORE INTO data_sources(source_name,dataset_description,license_reference,user_confirmed_use,created_at) VALUES(?,?,?,?,?)',(source_name,dataset_description,license_reference,int(user_confirmed_use),now))
        source_id=c.execute('SELECT id FROM data_sources WHERE source_name=? AND dataset_description=? AND license_reference=?',(source_name,dataset_description,license_reference)).fetchone()[0]
        quality='USER_CONFIRMED_UNVERIFIED' if user_confirmed_use else 'USER_IMPORTED_UNVERIFIED'
        for r in df.itertuples():
            start=r.valid_from.strftime('%Y-%m-%d'); end=r.valid_to.strftime('%Y-%m-%d') if pd.notna(r.valid_to) else None
            prior=c.execute('SELECT valid_from,valid_to FROM published_constituent_membership WHERE symbol=?',(r.symbol,)).fetchall()
            for old in prior:
                if (old['valid_to'] is None or start<=old['valid_to']) and (end is None or old['valid_from']<=end):
                    raise ValueError(f'{r.symbol} membership interval {start}–{end or "open"} overlaps an existing imported interval.')
            c.execute("INSERT INTO securities(symbol,name,exchange,index_member,is_demo) VALUES(?,?,'NSE',0,0) ON CONFLICT(symbol) DO NOTHING",(r.symbol,r.symbol))
            c.execute('INSERT INTO constituent_membership(symbol,valid_from,valid_to,source_id,import_run_id,quality) VALUES(?,?,?,?,?,?)',(r.symbol,start,end,source_id,run_id,quality))
        c.execute("UPDATE data_import_runs SET finished_at=?,status='succeeded',rows_imported=?,accepted_rows=?,symbols_json=?,earliest_date=?,latest_date=? WHERE id=?",(datetime.now(timezone.utc).isoformat(),len(df),len(df),json.dumps(sorted(df.symbol.unique())),df.valid_from.min().strftime('%Y-%m-%d'),df.valid_from.max().strftime('%Y-%m-%d'),run_id))
    lifecycle=_coordinate_reference_import(run_id,'membership',df.symbol.unique().tolist(),len(df))
    return {'run_id':run_id,'status':'succeeded','rows_imported':len(df),'symbols':sorted(df.symbol.unique()),'lifecycle':lifecycle,'quality':quality,'source_independently_verified':False}


def _is_illustrative(df,filename='',source_name=''):
    text=' '.join([str(filename),str(source_name),*map(str,df.columns),*map(str,df.to_numpy().ravel().tolist())]).lower()
    return any(marker in text for marker in ('illustrative','not_for_import','not importable','sample company','sample_symbol'))


def import_instruments_csv(contents:bytes,filename:str,*,as_of_date:str,source_name:str,dataset_description='',license_reference='',user_confirmed_use=False):
    """Import an as-of NSE instrument reference list without inventing membership."""
    if not source_name.strip(): raise ValueError('Provide the declared source name for this instrument list.')
    if user_confirmed_use and not license_reference: raise ValueError('A license or terms reference is required when confirming intended research use.')
    as_of=pd.to_datetime(as_of_date,errors='coerce')
    if pd.isna(as_of): raise ValueError('A valid instrument-list as-of date is required.')
    try: df=pd.read_csv(io.BytesIO(contents),comment='#')
    except Exception as exc: raise ValueError(f'Could not read instrument list: {exc}') from exc
    df.columns=[str(x).strip().lower().replace(' ','_') for x in df.columns]
    if _is_illustrative(df,filename,source_name): raise ValueError('Illustrative template rows are intentionally not importable.')
    if not {'symbol','name','exchange'}.issubset(df.columns): raise ValueError('Instrument list requires symbol,name,exchange; optional sector.')
    df['symbol']=df.symbol.astype(str).str.strip().str.upper(); df['name']=df.name.astype(str).str.strip(); df['exchange']=df.exchange.astype(str).str.strip().str.upper()
    if df.symbol.duplicated().any() or df.symbol.str.fullmatch(r'[A-Z0-9&._-]{1,30}').eq(False).any(): raise ValueError('Instrument symbols must be unique and use supported NSE symbol characters.')
    if df.name.eq('').any() or not df.exchange.eq('NSE').all(): raise ValueError('Every instrument needs a name and exchange NSE.')
    with connect() as c:
        prior_date=c.execute('SELECT MAX(as_of_date) FROM instrument_universe').fetchone()[0]
        prior_symbols=[r[0] for r in c.execute('SELECT symbol FROM published_instrument_universe WHERE as_of_date=?',(prior_date,))] if prior_date else []
    digest=hashlib.sha256(contents).hexdigest(); now=datetime.now(timezone.utc).isoformat(); safe=Path(filename or 'instruments.csv').name
    target=DATA_DIR/'imports'/f'{digest[:16]}-{safe}'; target.parent.mkdir(parents=True,exist_ok=True)
    if not target.exists(): target.write_bytes(contents)
    with connect() as c:
        prior=c.execute('SELECT * FROM data_import_runs WHERE sha256=?',(digest,)).fetchone()
        if prior: return {'run_id':prior['id'],'status':prior['status'],'idempotent':True,'rows_imported':prior['rows_imported']}
        cur=c.execute("INSERT INTO data_import_runs(started_at,filename,sha256,stored_path,status,source,rows_read,dataset_description,license_reference,user_confirmed_use,earliest_date,latest_date) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",(now,safe,str(digest),str(target),'instrument_list_validating',source_name,len(df),dataset_description,license_reference,int(user_confirmed_use),as_of.strftime('%Y-%m-%d'),as_of.strftime('%Y-%m-%d'))); run_id=cur.lastrowid
        c.execute('INSERT OR IGNORE INTO data_sources(source_name,dataset_description,license_reference,user_confirmed_use,created_at) VALUES(?,?,?,?,?)',(source_name,dataset_description,license_reference,int(user_confirmed_use),now))
        source_id=c.execute('SELECT id FROM data_sources WHERE source_name=? AND dataset_description=? AND license_reference=?',(source_name,dataset_description,license_reference)).fetchone()[0]
        quality='USER_CONFIRMED_UNVERIFIED' if user_confirmed_use else 'USER_IMPORTED_UNVERIFIED'
        for r in df.itertuples(index=False):
            sector=getattr(r,'sector',None); sector=None if pd.isna(sector) else str(sector)
            c.execute("INSERT INTO securities(symbol,name,sector,exchange,index_member,is_demo) VALUES(?,?,?,'NSE',0,0) ON CONFLICT(symbol) DO UPDATE SET name=excluded.name,sector=COALESCE(excluded.sector,securities.sector),exchange='NSE',is_demo=0",(r.symbol,r.name,sector))
            c.execute('INSERT OR REPLACE INTO instrument_universe(symbol,as_of_date,name,sector,exchange,source_id,import_run_id,quality) VALUES(?,?,?,?,?,?,?,?)',(r.symbol,as_of.strftime('%Y-%m-%d'),r.name,sector,'NSE',source_id,run_id,quality))
        c.execute("UPDATE data_import_runs SET finished_at=?,status='succeeded',rows_imported=?,accepted_rows=?,symbols_json=? WHERE id=?",(datetime.now(timezone.utc).isoformat(),len(df),len(df),json.dumps(sorted(df.symbol)),run_id))
    lifecycle=_coordinate_reference_import(run_id,'instruments',sorted(set(df.symbol.tolist())|set(prior_symbols)),len(df))
    return {'run_id':run_id,'status':'succeeded','as_of_date':as_of.strftime('%Y-%m-%d'),'rows_imported':len(df),'lifecycle':lifecycle,'quality':quality,'historical_membership_created':False}


def import_exchange_calendar_csv(contents:bytes,filename:str,*,source_name:str,dataset_description='',license_reference='',user_confirmed_use=False):
    """Import explicit open-session dates for diagnostics; no calendar is inferred."""
    if not source_name.strip(): raise ValueError('Provide the declared source name for the exchange calendar.')
    if user_confirmed_use and not license_reference: raise ValueError('A license or terms reference is required when confirming intended research use.')
    try: df=pd.read_csv(io.BytesIO(contents),comment='#')
    except Exception as exc: raise ValueError(f'Could not read exchange calendar: {exc}') from exc
    df.columns=[str(x).strip().lower() for x in df.columns]
    if _is_illustrative(df,filename,source_name): raise ValueError('Illustrative template rows are intentionally not importable.')
    if not {'exchange','date'}.issubset(df.columns): raise ValueError('Exchange calendar requires exchange,date with one row per open session.')
    df['exchange']=df.exchange.astype(str).str.upper().str.strip(); df['date']=pd.to_datetime(df.date,errors='coerce')
    if df.date.isna().any() or not df.exchange.isin(['NSE','NSE_INDEX']).all() or df.duplicated(['exchange','date']).any(): raise ValueError('Calendar has invalid dates, unsupported exchanges, or duplicate sessions.')
    digest=hashlib.sha256(contents).hexdigest(); now=datetime.now(timezone.utc).isoformat(); safe=Path(filename or 'calendar.csv').name
    target=DATA_DIR/'imports'/f'{digest[:16]}-{safe}'; target.parent.mkdir(parents=True,exist_ok=True)
    if not target.exists(): target.write_bytes(contents)
    quality='USER_CONFIRMED_UNVERIFIED' if user_confirmed_use else 'USER_IMPORTED_UNVERIFIED'
    with connect() as c:
        prior=c.execute('SELECT * FROM data_import_runs WHERE sha256=?',(digest,)).fetchone()
        if prior: return {'run_id':prior['id'],'status':prior['status'],'idempotent':True,'rows_imported':prior['rows_imported']}
        cur=c.execute("INSERT INTO data_import_runs(started_at,filename,sha256,stored_path,status,source,rows_read,dataset_description,license_reference,user_confirmed_use,earliest_date,latest_date) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",(now,safe,digest,str(target),'calendar_validating',source_name,len(df),dataset_description,license_reference,int(user_confirmed_use),df.date.min().strftime('%Y-%m-%d'),df.date.max().strftime('%Y-%m-%d'))); run_id=cur.lastrowid
        c.execute('INSERT OR IGNORE INTO data_sources(source_name,dataset_description,license_reference,user_confirmed_use,created_at) VALUES(?,?,?,?,?)',(source_name,dataset_description,license_reference,int(user_confirmed_use),now))
        source_id=c.execute('SELECT id FROM data_sources WHERE source_name=? AND dataset_description=? AND license_reference=?',(source_name,dataset_description,license_reference)).fetchone()[0]
        c.executemany('INSERT OR IGNORE INTO exchange_sessions(exchange,date,source_id,import_run_id,quality) VALUES(?,?,?,?,?)',[(r.exchange,r.date.strftime('%Y-%m-%d'),source_id,run_id,quality) for r in df.itertuples()])
        c.execute("UPDATE data_import_runs SET finished_at=?,status='succeeded',rows_imported=?,accepted_rows=?,symbols_json=? WHERE id=?",(datetime.now(timezone.utc).isoformat(),len(df),len(df),json.dumps(sorted(df.exchange.unique())),run_id))
    lifecycle=_coordinate_reference_import(run_id,'calendar',[],len(df))
    return {'run_id':run_id,'status':'succeeded','rows_imported':len(df),'lifecycle':lifecycle,'quality':quality,'source_independently_verified':False}
