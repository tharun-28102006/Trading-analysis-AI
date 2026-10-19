"""Atomic Nivesh import and release publication for Tej Bazaar market files."""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from .db import capture_release_snapshot, connect
from .providers import tej_bazaar_local as local
from .providers.tej import TejClient, retain_payload

SOURCE = 'tej_bazaar_local'
QUALITY = 'TEJ_BAZAAR_LOCAL_LICENSED'
LICENSE = 'https://huggingface.co/datasets/tejhq/indian-markets (MIT dataset license as published by TejHQ); https://tejhq.dev/docs'
EXCHANGES = ('NSE','BSE')


def _now():
    return datetime.now(timezone.utc).isoformat()


def _ist_today():
    return datetime.now(ZoneInfo('Asia/Kolkata')).date()


def _set_progress(run_id, exchange, completed, total, current=None, rows=0, provider_date=None, stage='import', conn=None):
    detail={'progress':{'stage':stage,'exchange':exchange,'completed_files':completed,'total_files':total,
        'current_file':current,'rows_staged':rows,'provider_trading_date':provider_date}}
    if conn is not None:
        conn.execute('UPDATE refresh_runs SET warnings_json=? WHERE id=?',(json.dumps(detail,sort_keys=True),run_id))
    else:
        with connect() as c:
            c.execute('UPDATE refresh_runs SET warnings_json=? WHERE id=?',(json.dumps(detail,sort_keys=True),run_id))


def _snapshot_meta(status):
    # Only retain provider-supplied publication timestamps; retrieval time is separate.
    for key in ('published_at','source_published_at','prices_published_at','updated_at'):
        value=status.get(key)
        if isinstance(value,str) and value.strip(): return value
    return None


def _refresh_from_api(root, baseline_dates, today):
    client=TejClient()
    status=client.status()
    provider_date=status['trading_date']
    if (today-date.fromisoformat(provider_date)).days>7:
        raise RuntimeError(f'TejHQ status is stale: latest provider trading date {provider_date}.')
    payload_hash=None; payload_path=None
    # The status endpoint is itself a successful network check. It is retained raw too.
    status_payload=getattr(client,'_last_status_payload',None)
    if status_payload:
        payload_hash,payload_path=retain_payload(status_payload)
    from .providers.tej_bazaar_local import fetch_incremental_snapshots
    result=fetch_incremental_snapshots(root,latest_by_exchange=baseline_dates,today=today,status_date=provider_date,client=client)
    return {'status':'failed' if any(r['status']=='failed' for r in result.values()) else 'succeeded',
        'provider_trading_date':provider_date,'source_published_at':_snapshot_meta(status),
        'status_payload_sha256':payload_hash,'status_payload_path':payload_path,'exchanges':result}


def _normal(value):
    if value is None: return None
    if isinstance(value,float) and (math.isnan(value) or math.isinf(value)): return None
    if hasattr(value,'isoformat'): return value.isoformat()
    return value


def _stage_files(conn, files, run_id, provider_date):
    conn.execute('''CREATE TEMP TABLE tej_stage(
        exchange TEXT NOT NULL,symbol TEXT NOT NULL,date TEXT NOT NULL,series TEXT,isin TEXT,name TEXT,
        open REAL NOT NULL,high REAL NOT NULL,low REAL NOT NULL,close REAL NOT NULL,last REAL,prev_close REAL,
        volume INTEGER NOT NULL,turnover REAL,trades INTEGER,source_file TEXT NOT NULL,file_sha256 TEXT NOT NULL,
        payload_hash TEXT NOT NULL,payload_json TEXT NOT NULL,PRIMARY KEY(exchange,symbol,date))''')
    processed=[]; total_rows=0
    by_exchange={ex:[f for f in files if f.exchange==ex] for ex in EXCHANGES}
    for exchange in EXCHANGES:
        source_files=by_exchange[exchange]
        for ix,source in enumerate(source_files,1):
            frame=local.read_and_validate_file(source)
            if source.sha256 is None:
                raise ValueError(f'No content hash was computed for {source.relative_path}.')
            batch=[]
            for row in local.frame_rows(frame):
                normalized={k:_normal(v) for k,v in row.items()}
                normalized['date']=date.fromisoformat(normalized['date']).isoformat()
                price_fields={k:normalized.get(k) for k in ('open','high','low','close','volume','date','symbol','series','isin')}
                payload=json.dumps(price_fields,sort_keys=True,separators=(',',':'),allow_nan=False)
                ph=hashlib.sha256(payload.encode()).hexdigest()
                batch.append((exchange,normalized['symbol'],normalized['date'],normalized.get('series'),normalized.get('isin'),normalized.get('name'),
                    normalized['open'],normalized['high'],normalized['low'],normalized['close'],normalized.get('last'),normalized.get('prev_close'),
                    int(normalized['volume']),normalized.get('turnover'),normalized.get('trades'),source.relative_path,source.sha256,ph,payload))
                if len(batch)>=5000:
                    conn.executemany('INSERT INTO tej_stage VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',batch)
                    total_rows+=len(batch); batch.clear()
            if batch:
                conn.executemany('INSERT INTO tej_stage VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',batch)
                total_rows+=len(batch)
            processed.append((source,frame.height))
            if ix%25==0 or ix==len(source_files):
                _set_progress(run_id,exchange,ix,len(source_files),source.relative_path,total_rows,provider_date,'validate_and_stage')
    return processed,total_rows


def _change_count(conn, exchange):
    if exchange=='NSE':
        row=conn.execute('''SELECT
            SUM(CASE WHEN p.symbol IS NULL OR p.quality='DEMO_SYNTHETIC' THEN 1 ELSE 0 END) new_rows,
            SUM(CASE WHEN p.quality<>'DEMO_SYNTHETIC' AND (p.source LIKE 'tej_bazaar_local:%' OR p.source LIKE 'tejhq:%') AND
              (p.open IS NOT s.open OR p.high IS NOT s.high OR p.low IS NOT s.low OR p.close IS NOT s.close OR p.volume IS NOT s.volume OR p.price_basis<>'raw_unadjusted') THEN 1 ELSE 0 END) updated_rows,
            SUM(CASE WHEN p.symbol IS NOT NULL AND p.quality<>'DEMO_SYNTHETIC' AND NOT (p.source LIKE 'tej_bazaar_local:%' OR p.source LIKE 'tejhq:%') THEN 1 ELSE 0 END) preserved_conflicts,
            SUM(CASE WHEN p.quality<>'DEMO_SYNTHETIC' AND p.symbol IS NOT NULL AND (p.source LIKE 'tej_bazaar_local:%' OR p.source LIKE 'tejhq:%') AND
              p.open IS s.open AND p.high IS s.high AND p.low IS s.low AND p.close IS s.close AND p.volume IS s.volume AND p.price_basis='raw_unadjusted' THEN 1 ELSE 0 END) unchanged_rows
          FROM tej_stage s LEFT JOIN prices p ON p.symbol=s.symbol AND p.date=s.date WHERE s.exchange='NSE' ''').fetchone()
    else:
        row=conn.execute('''SELECT
            SUM(CASE WHEN p.symbol IS NULL THEN 1 ELSE 0 END) new_rows,
            SUM(CASE WHEN p.symbol IS NOT NULL AND (p.open IS NOT s.open OR p.high IS NOT s.high OR p.low IS NOT s.low OR p.close IS NOT s.close OR p.volume IS NOT s.volume OR p.series IS NOT s.series OR COALESCE(p.isin,'')<>COALESCE(s.isin,'') OR COALESCE(p.name,'')<>COALESCE(s.name,'')) THEN 1 ELSE 0 END) updated_rows,
            0 preserved_conflicts,
            SUM(CASE WHEN p.symbol IS NOT NULL AND p.open IS s.open AND p.high IS s.high AND p.low IS s.low AND p.close IS s.close AND p.volume IS s.volume AND p.series IS s.series AND COALESCE(p.isin,'')=COALESCE(s.isin,'') AND COALESCE(p.name,'')=COALESCE(s.name,'') THEN 1 ELSE 0 END) unchanged_rows
          FROM tej_stage s LEFT JOIN exchange_market_prices p ON p.exchange='BSE' AND p.symbol=s.symbol AND p.date=s.date WHERE s.exchange='BSE' ''').fetchone()
    return {k:int(row[k] or 0) for k in ('new_rows','updated_rows','preserved_conflicts','unchanged_rows')}


def _fingerprint(conn):
    digest=hashlib.sha256()
    queries=[("SELECT symbol,date,open,high,low,close,adjusted_close,volume,price_basis FROM prices WHERE quality<>'DEMO_SYNTHETIC' ORDER BY symbol,date",'NSE'),
             ("SELECT symbol,date,open,high,low,close,volume,series,isin,name FROM exchange_market_prices WHERE quality<>'DEMO_SYNTHETIC' ORDER BY symbol,date",'BSE')]
    for sql,exchange in queries:
        for row in conn.execute(sql):
            digest.update(exchange.encode()+b'|')
            digest.update(json.dumps(list(row),separators=(',',':'),allow_nan=False).encode())
            digest.update(b'\n')
    return digest.hexdigest()


def _apply_staged_rows(conn, run_id, retrieved_at, counts):
    # Preserve demo values only in the explicit archive; they are never promoted as data.
    conn.execute('''INSERT INTO securities(symbol,name,sector,exchange,index_member,is_demo,instrument_id,provider_symbol)
        SELECT symbol,COALESCE(name,symbol),'Unclassified','NSE',0,0,'NSE_EQ_'||symbol,symbol FROM (
          SELECT symbol,name,ROW_NUMBER() OVER(PARTITION BY symbol ORDER BY date DESC) rn FROM tej_stage WHERE exchange='NSE') WHERE rn=1
        ON CONFLICT(symbol) DO UPDATE SET name=excluded.name,exchange='NSE',is_demo=0,instrument_id=excluded.instrument_id,provider_symbol=excluded.provider_symbol''')
    conn.execute('''INSERT OR IGNORE INTO archived_demo_prices(symbol,date,open,high,low,close,adjusted_close,volume,source,retrieved_at,quality,price_basis)
        SELECT p.symbol,p.date,p.open,p.high,p.low,p.close,p.adjusted_close,p.volume,p.source,p.retrieved_at,p.quality,'synthetic'
        FROM prices p JOIN tej_stage s ON s.exchange='NSE' AND s.symbol=p.symbol AND s.date=p.date WHERE p.quality='DEMO_SYNTHETIC' ''')
    conn.execute('''INSERT INTO data_revisions(dataset_type,symbol,observation_date,change_kind,old_value_json,new_value_json,import_run_id,created_at)
        SELECT 'prices',p.symbol,p.date,'historical_correction',
          json_object('open',p.open,'high',p.high,'low',p.low,'close',p.close,'adjusted_close',p.adjusted_close,'volume',p.volume,'price_basis',p.price_basis),
          json_object('open',s.open,'high',s.high,'low',s.low,'close',s.close,'adjusted_close',s.close,'volume',s.volume,'price_basis','raw_unadjusted'),?,?
        FROM tej_stage s JOIN prices p ON p.symbol=s.symbol AND p.date=s.date
        WHERE s.exchange='NSE' AND p.quality<>'DEMO_SYNTHETIC' AND (p.source LIKE 'tej_bazaar_local:%' OR p.source LIKE 'tejhq:%')
          AND (p.open IS NOT s.open OR p.high IS NOT s.high OR p.low IS NOT s.low OR p.close IS NOT s.close OR p.volume IS NOT s.volume OR p.price_basis<>'raw_unadjusted')''',(run_id,retrieved_at))
    conn.execute('''INSERT INTO data_revisions(dataset_type,symbol,observation_date,change_kind,old_value_json,new_value_json,import_run_id,created_at)
        SELECT 'exchange_prices:BSE',p.symbol,p.date,'historical_correction',
          json_object('open',p.open,'high',p.high,'low',p.low,'close',p.close,'volume',p.volume,'series',p.series,'isin',p.isin,'name',p.name),
          json_object('open',s.open,'high',s.high,'low',s.low,'close',s.close,'volume',s.volume,'series',s.series,'isin',s.isin,'name',s.name),?,?
        FROM tej_stage s JOIN exchange_market_prices p ON p.exchange='BSE' AND p.symbol=s.symbol AND p.date=s.date
        WHERE s.exchange='BSE' AND (p.open IS NOT s.open OR p.high IS NOT s.high OR p.low IS NOT s.low OR p.close IS NOT s.close OR p.volume IS NOT s.volume OR p.series IS NOT s.series OR COALESCE(p.isin,'')<>COALESCE(s.isin,'') OR COALESCE(p.name,'')<>COALESCE(s.name,''))''',(run_id,retrieved_at))
    conn.execute('''INSERT INTO prices(symbol,date,open,high,low,close,adjusted_close,volume,source,retrieved_at,quality,price_basis)
        SELECT symbol,date,open,high,low,close,close,volume,'tej_bazaar_local:NSE',?,?,'raw_unadjusted' FROM tej_stage WHERE exchange='NSE'
        ON CONFLICT(symbol,date) DO UPDATE SET open=excluded.open,high=excluded.high,low=excluded.low,close=excluded.close,adjusted_close=excluded.adjusted_close,volume=excluded.volume,source=excluded.source,retrieved_at=excluded.retrieved_at,quality=excluded.quality,price_basis=excluded.price_basis
        WHERE prices.quality='DEMO_SYNTHETIC' OR ((prices.source LIKE 'tej_bazaar_local:%' OR prices.source LIKE 'tejhq:%') AND
        (prices.open IS NOT excluded.open OR prices.high IS NOT excluded.high OR prices.low IS NOT excluded.low OR prices.close IS NOT excluded.close OR prices.volume IS NOT excluded.volume OR prices.price_basis<>'raw_unadjusted'))''',(retrieved_at,QUALITY))
    conn.execute('''INSERT INTO exchange_market_prices(exchange,symbol,date,series,isin,name,open,high,low,close,volume,turnover,trades,source,retrieved_at,quality,price_basis)
        SELECT 'BSE',symbol,date,series,isin,name,open,high,low,close,volume,turnover,trades,'tej_bazaar_local:BSE',?,?,'raw_unadjusted' FROM tej_stage WHERE exchange='BSE'
        ON CONFLICT(exchange,symbol,date) DO UPDATE SET series=excluded.series,isin=excluded.isin,name=excluded.name,open=excluded.open,high=excluded.high,low=excluded.low,close=excluded.close,volume=excluded.volume,turnover=excluded.turnover,trades=excluded.trades,source=excluded.source,retrieved_at=excluded.retrieved_at,quality=excluded.quality,price_basis=excluded.price_basis
        WHERE exchange_market_prices.open IS NOT excluded.open OR exchange_market_prices.high IS NOT excluded.high OR exchange_market_prices.low IS NOT excluded.low OR exchange_market_prices.close IS NOT excluded.close OR exchange_market_prices.volume IS NOT excluded.volume OR exchange_market_prices.series IS NOT excluded.series OR COALESCE(exchange_market_prices.isin,'')<>COALESCE(excluded.isin,'') OR COALESCE(exchange_market_prices.name,'')<>COALESCE(excluded.name,'')''',(retrieved_at,QUALITY))
    conn.execute('''INSERT INTO raw_price_observations(instrument_id,symbol,provider_symbol,date,source,retrieved_at,payload_hash,payload_json,validation_status)
        SELECT exchange||'_'||symbol,symbol,symbol,date,'tej_bazaar_local:'||exchange,?,payload_hash,payload_json,'PROVIDER_LICENSED_VALIDATED' FROM tej_stage WHERE 1
        ON CONFLICT(instrument_id,date,source,payload_hash) DO NOTHING''',(retrieved_at,))
    return counts


def refresh(run_id: int, reconcile: bool = False) -> dict:
    """Try keyless recent snapshots, validate local partitions, and publish once atomically."""
    started=time.monotonic(); root=local.default_data_root(); retrieved_at=_now(); errors=[]; source_status=None
    try:
        root=root.resolve()
        with connect() as c:
            c.execute('INSERT INTO market_exchange_refresh(exchange,last_attempt_at,status,updated_at) VALUES(?,? ,?,?) ON CONFLICT(exchange) DO UPDATE SET last_attempt_at=excluded.last_attempt_at,status=excluded.status,updated_at=excluded.updated_at',('NSE',retrieved_at,'running',retrieved_at))
            c.execute('INSERT INTO market_exchange_refresh(exchange,last_attempt_at,status,updated_at) VALUES(?,? ,?,?) ON CONFLICT(exchange) DO UPDATE SET last_attempt_at=excluded.last_attempt_at,status=excluded.status,updated_at=excluded.updated_at',('BSE',retrieved_at,'running',retrieved_at))
            previous_release=c.execute("SELECT value FROM app_state WHERE key='active_release_id'").fetchone()
            old_market={
                'NSE':c.execute("SELECT MAX(date) FROM prices WHERE quality=? AND source='tej_bazaar_local:NSE'",(QUALITY,)).fetchone()[0],
                'BSE':c.execute("SELECT MAX(date) FROM exchange_market_prices WHERE quality=?",(QUALITY,)).fetchone()[0],
            }
            manifest={ex:{r['relative_path']:(r['sha256'],r['size_bytes'],r['modified_ns']) for r in c.execute('SELECT * FROM market_data_files WHERE exchange=?',(ex,))} for ex in EXCHANGES}
        latest_files={}
        for ex in EXCHANGES:
            _set_progress(run_id,ex,0,0,rows=0,stage='scan_local_files')
            # Listing files is cheap. Hash only untracked/changed files plus the configured correction overlap.
            all_paths=local.discover_files(root,ex,known={},hash_files=False)
            latest_files[ex]=max((f.observation_date for f in all_paths),default=None)
            base_dates=[d for d in (old_market[ex],latest_files[ex]) if d]
            if base_dates:
                baseline=min(base_dates)
                overlap_from=date.fromisoformat(baseline)-timedelta(days=local.overlap_days())
            else: overlap_from=None
            hashed=local.discover_files(root,ex,known=manifest[ex],hash_from=overlap_from)
            latest_files[ex]=max((f.observation_date for f in hashed),default=latest_files[ex])
            # Keep the file list for a second pass after API snapshot writes.
            latest_files[ex+'_files']=hashed
            _set_progress(run_id,ex,len(hashed),len(hashed),rows=0,stage='local_file_hashes_ready')
        api_errors=[]
        # If this is a later run, a failed provider fetch cannot alter the last published release.
        bootstrap=not previous_release
        if bootstrap:
            # Publish the already-downloaded source first. The next refresh performs the live incremental probe.
            source_status={'status':'deferred_bootstrap','provider_trading_date':None,'source_published_at':None,'exchanges':{},'errors':[]}
        else:
            try:
                _set_progress(run_id,'NSE+BSE',0,0,rows=0,stage='check_provider_status')
                client=TejClient(); provider_status=client.status(); provider_date=provider_status['trading_date']
                baseline={ex:min([d for d in (old_market[ex],latest_files[ex]) if d],default=None) for ex in EXCHANGES}
                from .providers.tej_bazaar_local import fetch_incremental_snapshots
                api_result=fetch_incremental_snapshots(root,latest_by_exchange=baseline,today=_ist_today(),status_date=provider_date,client=client,
                    progress=lambda exchange,day,completed,rows:_set_progress(run_id,exchange,completed,0,day,rows,provider_date,'fetch_snapshots'))
                for ex,result in api_result.items():
                    if result['status']=='failed': api_errors.extend({'exchange':ex,**e} for e in result['errors'])
                source_status={'status':'failed' if api_errors else 'succeeded','provider_trading_date':provider_date,
                    'source_published_at':next((provider_status[k] for k in ('published_at','source_published_at','prices_published_at','updated_at') if isinstance(provider_status.get(k),str)),None),
                    'exchanges':api_result}
                _set_progress(run_id,'NSE+BSE',0,0,rows=0,provider_date=provider_date,stage='provider_snapshots_checked')
            except Exception as exc:
                api_errors.append({'exchange':'both','error':f'{type(exc).__name__}: {str(exc)[:400]}'})
                source_status={'status':'failed','provider_trading_date':None,'source_published_at':None,'exchanges':{},'errors':api_errors}
        if api_errors and not bootstrap:
            raise RuntimeError('TejHQ snapshot refresh failed; previous published release kept intact. '+ '; '.join(x['error'] for x in api_errors[:4]))
        processed_files={}; source_latest={}
        for ex in EXCHANGES:
            prior=manifest[ex]
            candidates=latest_files[ex+'_files']
            # Re-scan only when the live API actually wrote snapshots; otherwise reuse the first hashed inventory.
            base=old_market[ex] or latest_files[ex]
            overlap_from=date.fromisoformat(base)-timedelta(days=local.overlap_days()) if base else None
            fetched=source_status.get('exchanges',{}).get(ex,{}).get('rows_fetched',0)>0
            current=local.discover_files(root,ex,known=prior,hash_from=overlap_from,force_hash=reconcile) if fetched or reconcile else candidates
            changed=[f for f in current if f.sha256 and (f.relative_path not in prior or prior[f.relative_path][0]!=f.sha256)]
            processed_files[ex]=changed
            source_latest[ex]=max((f.observation_date for f in current),default=None)
        files=[f for ex in EXCHANGES for f in processed_files[ex]]
        if not root.exists() or not any(local.discover_files(root,ex,known={},hash_files=False) for ex in EXCHANGES):
            raise FileNotFoundError(f'No NSE/BSE Tej Bazaar Parquet files were found under {root}.')
        if not files:
            finished=_now()
            with connect() as c:
                for ex in EXCHANGES:
                    c.execute('''INSERT INTO market_exchange_refresh(exchange,last_attempt_at,last_successful_at,last_source_fetch_at,last_source_fetch_status,latest_market_date,latest_source_date,rows_in_source,rows_imported,rows_updated,rows_unchanged,status,error,source_published_at,updated_at)
                      VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(exchange) DO UPDATE SET last_attempt_at=excluded.last_attempt_at,last_successful_at=excluded.last_successful_at,last_source_fetch_at=excluded.last_source_fetch_at,last_source_fetch_status=excluded.last_source_fetch_status,latest_source_date=excluded.latest_source_date,status=excluded.status,error=excluded.error,source_published_at=excluded.source_published_at,updated_at=excluded.updated_at''',
                      (ex,retrieved_at,finished,retrieved_at if source_status['status']!='deferred_bootstrap' else None,source_status['status'],old_market[ex],source_latest[ex],0,0,0,0,'unchanged',json.dumps(api_errors),source_status.get('source_published_at'),finished))
                run_status='failed' if api_errors else 'unchanged'
                warnings={'api':source_status,'exchanges':{ex:{'latest_market_date':old_market[ex],'latest_source_date':source_latest[ex],'rows_imported':0,'rows_updated':0,'rows_unchanged':0,'status':'unchanged'} for ex in EXCHANGES}}
                c.execute('UPDATE refresh_runs SET finished_at=?,status=?,source=?,errors_json=?,warnings_json=?,step_timings_json=? WHERE id=?',
                    (finished,run_status,SOURCE,json.dumps(api_errors),json.dumps(warnings,sort_keys=True),json.dumps({'elapsed_seconds':round(time.monotonic()-started,3)}),run_id))
                c.execute("UPDATE refresh_runs SET latest_observation_date=(SELECT MAX(date) FROM prices WHERE quality<>'DEMO_SYNTHETIC') WHERE id=?",(run_id,))
            return {'run_id':run_id,'status':run_status,'source':SOURCE,'rows_written':0,'errors':api_errors,'warnings':warnings,'exchanges':warnings['exchanges'],'provider_fetch_status':source_status['status']}
        finished=_now(); manifest_digest=hashlib.sha256('\n'.join(f'{f.relative_path}:{f.sha256}' for f in sorted(files,key=lambda x:x.relative_path)).encode()).hexdigest()
        counts={}; publication=None; run_import_id=None; changed_rows=0; total_rows=0; per_exchange={}
        with connect() as c:
            processed, total_rows=_stage_files(c,files,run_id,source_status.get('provider_trading_date'))
            c.commit()  # Staged rows are connection-local; canonical changes remain untouched.
            c.execute('BEGIN IMMEDIATE')
            counts={ex:_change_count(c,ex) for ex in EXCHANGES}
            changed_rows=sum(v['new_rows']+v['updated_rows'] for v in counts.values())
            conflicts=sum(v['preserved_conflicts'] for v in counts.values())
            # A file manifest hash makes retries resumable; this run's rows and manifest commit together.
            existing=c.execute('SELECT id FROM data_import_runs WHERE sha256=?',(manifest_digest,)).fetchone()
            if existing:
                run_import_id=existing['id']
                c.execute("UPDATE data_import_runs SET started_at=?,finished_at=NULL,status='validating',stored_path=?,source=?,rows_read=?,errors_json='[]' WHERE id=?",
                    (retrieved_at,str(root),SOURCE,total_rows,run_import_id))
            else:
                run_import_id=c.execute('''INSERT INTO data_import_runs(started_at,filename,sha256,stored_path,status,source,rows_read,dataset_description,license_reference,user_confirmed_use,correction_confirmed)
                    VALUES(?,?,?,?,?,?,?,?,?,0,1)''',(retrieved_at,'tej-bazaar-local-parquet',manifest_digest,str(root),'validating',SOURCE,total_rows,
                    'Tej Bazaar NSE/BSE daily equity Parquet; validated raw exchange OHLCV.','; '.join([LICENSE]))).lastrowid
            if changed_rows:
                _apply_staged_rows(c,run_import_id,retrieved_at,counts)
                changed_symbols={ex:[r[0] for r in c.execute('SELECT DISTINCT symbol FROM tej_stage WHERE exchange=? ORDER BY symbol',(ex,))] for ex in EXCHANGES}
                kind='historical_correction' if any(v['updated_rows'] for v in counts.values()) else ('backfill' if not previous_release else 'new_session')
                summary={'change_kind':kind,'rows_read':total_rows,'new_observations':sum(v['new_rows'] for v in counts.values()),
                    'corrected_observations':sum(v['updated_rows'] for v in counts.values()),'preserved_nonprovider_conflicts':conflicts,
                    'file_count':len(processed),'file_sha256':[{'exchange':f.exchange,'path':f.relative_path,'sha256':f.sha256,'rows':n} for f,n in processed],
                    'exchanges':{ex:{**counts[ex],'latest_source_date':source_latest[ex]} for ex in EXCHANGES},
                    'synthetic_rows_excluded_from_fingerprint':True,'provider_fetch':source_status,'source_published_at':source_status.get('source_published_at')}
                fingerprint=_fingerprint(c)
                meta={'source':SOURCE,'license_reference':LICENSE,'dataset_description':'Tej Bazaar local daily NSE/BSE equity Parquet; TejHQ keyless snapshots for incremental updates.'}
                metadata_fp=hashlib.sha256(json.dumps(meta,sort_keys=True,separators=(',',':')).encode()).hexdigest()
                latest=c.execute("SELECT id FROM dataset_versions WHERE dataset_type='prices' ORDER BY id DESC LIMIT 1").fetchone()
                prior_version=c.execute("SELECT id FROM dataset_versions WHERE dataset_type='prices' AND content_fingerprint=? AND metadata_fingerprint=?",(fingerprint,metadata_fp)).fetchone()
                version_id=prior_version['id'] if prior_version else c.execute('''INSERT INTO dataset_versions(dataset_type,content_fingerprint,metadata_fingerprint,parent_version_id,import_run_id,created_at,change_kind,summary_json)
                    VALUES('prices',?,?,?,?,?,?,?)''',(fingerprint,metadata_fp,latest['id'] if latest else None,run_import_id,finished,kind,json.dumps(summary,sort_keys=True))).lastrowid
                affected_nse=changed_symbols['NSE']
                components={'price_charts':'current','indicators':'current','feature_datasets':'stale','targets':'stale','company_models':'stale','pooled_model_inputs':'stale','aggregator_inputs':'stale','oos_predictions':'stale','calibration':'stale','backtests':'stale','rankings':'current','portfolio_valuations':'current','paper_trade_analytics':'current','forecasts':'stale'}
                dep=[]
                for symbol in affected_nse:
                    for component,state in components.items():
                        dep.append((component,symbol,state,fingerprint,version_id,finished,json.dumps({'reason':'Tej Bazaar price version changed; rebuild/retrain follows existing policy.','exchange':'NSE','change_kind':kind})))
                if counts['BSE']['new_rows']+counts['BSE']['updated_rows']:
                    dep.append(('exchange_price_charts','BSE','current',fingerprint,version_id,finished,json.dumps({'reason':'BSE data is available from the published exchange-price view.','change_kind':kind})))
                c.executemany('''INSERT INTO dependency_states(component,scope_key,status,input_fingerprint,version_id,updated_at,detail_json) VALUES(?,?,?,?,?,?,?)
                    ON CONFLICT(component,scope_key) DO UPDATE SET status=excluded.status,input_fingerprint=excluded.input_fingerprint,version_id=excluded.version_id,updated_at=excluded.updated_at,detail_json=excluded.detail_json''',dep)
                refresh_id=c.execute('''INSERT INTO refresh_runs(started_at,finished_at,status,triggered_by,source,instruments_requested,instruments_updated,rows_written,latest_observation_date,errors_json,warnings_json,import_run_id,version_id,change_summary_json,dependencies_json,step_timings_json)
                    VALUES(?,?, 'candidate','local_import',?,?,?,?,?,'[]','{}',?,?,?,'[]','{}')''',(retrieved_at,None,SOURCE,len(changed_symbols['NSE'])+len(changed_symbols['BSE']),len(affected_nse),changed_rows,None,run_import_id,version_id,json.dumps(summary,sort_keys=True))).lastrowid
                c.execute('UPDATE data_import_runs SET version_id=?,refresh_run_id=?,content_sha256=?,change_summary_json=?,source=?,rows_imported=?,accepted_rows=?,duplicates=?,status=? WHERE id=?',
                    (version_id,refresh_id,fingerprint,json.dumps(summary,sort_keys=True),SOURCE,changed_rows,changed_rows,total_rows-changed_rows,'publishing',run_import_id))
                candidate=c.execute("INSERT INTO research_releases(version_id,refresh_run_id,status,created_at,published_at,fingerprint,validation_json,errors_json) VALUES(?,?,'candidate',?,NULL,?,?, '[]')",
                    (version_id,refresh_id,finished,fingerprint,json.dumps({'canonical_validation':'passed','synthetic_rows_excluded':True,'source_file_count':len(processed),'source_file_manifest_sha256':manifest_digest,'rows':total_rows,'provider_fetch_status':source_status['status']},sort_keys=True))).lastrowid
                # Snapshot capture and pointer swap are inside this same transaction.
                capture_release_snapshot(c,candidate)
                c.execute("UPDATE current_forecasts SET status='stale',stale_reason='A newer genuine local market-data release was published.' WHERE symbol IN (SELECT DISTINCT symbol FROM tej_stage WHERE exchange='NSE')")
                c.execute("UPDATE research_releases SET status='superseded' WHERE status='published'")
                c.execute("UPDATE research_releases SET status='published',published_at=? WHERE id=?",(finished,candidate))
                c.execute("INSERT INTO app_state(key,value,updated_at) VALUES('active_release_id',?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value,updated_at=excluded.updated_at",(str(candidate),finished))
                publication={'refresh_run_id':refresh_id,'version_id':version_id,'release_id':candidate,'release_status':'published'}
                c.execute('UPDATE refresh_runs SET finished_at=?,status=?,rows_written=?,latest_observation_date=(SELECT MAX(date) FROM prices WHERE quality<>\'DEMO_SYNTHETIC\'),errors_json=?,warnings_json=?,dependencies_json=?,step_timings_json=? WHERE id=?',
                    (finished,'succeeded' if source_status['status']=='succeeded' else 'partial_success',changed_rows,json.dumps(api_errors),json.dumps({'source_file_count':len(processed),'provider_fetch':source_status},sort_keys=True),json.dumps([{'component':'market_release','scope':'NSE+BSE','status':'published'}]),json.dumps({'elapsed_seconds':round(time.monotonic()-started,3)}),refresh_id))
                c.execute('UPDATE data_import_runs SET status=?,finished_at=?,errors_json=? WHERE id=?',('succeeded' if source_status['status']=='succeeded' else 'partial_success',finished,json.dumps(api_errors),run_import_id))
                c.execute('UPDATE refresh_runs SET status=?,finished_at=?,rows_written=?,latest_observation_date=(SELECT MAX(date) FROM prices WHERE quality<>\'DEMO_SYNTHETIC\'),errors_json=?,warnings_json=?,change_summary_json=?,step_timings_json=? WHERE id=?',
                    ('succeeded' if source_status['status']=='succeeded' else 'partial_success',finished,changed_rows,json.dumps(api_errors),json.dumps({'api':source_status,'import':summary},sort_keys=True),json.dumps(summary,sort_keys=True),json.dumps({'elapsed_seconds':round(time.monotonic()-started,3)}),refresh_id))
            else:
                summary={'change_kind':'unchanged','rows_read':total_rows,'new_observations':0,'corrected_observations':0,'preserved_nonprovider_conflicts':conflicts,
                    'file_count':len(processed),'provider_fetch':source_status,'source_published_at':source_status.get('source_published_at')}
                c.execute('UPDATE data_import_runs SET status=?,finished_at=?,rows_imported=0,accepted_rows=0,duplicates=?,change_summary_json=? WHERE id=?',
                    ('unchanged' if source_status['status']=='succeeded' else 'failed',finished,total_rows,json.dumps(summary,sort_keys=True),run_import_id))
            # Persist every newly checked/updated source file. This and the price commit share the release transaction.
            for source_file,row_count in processed:
                c.execute('''INSERT INTO market_data_files(exchange,relative_path,observation_date,sha256,size_bytes,modified_ns,row_count,imported_at,source)
                    VALUES(?,?,?,?,?,?,?,?,?) ON CONFLICT(exchange,relative_path) DO UPDATE SET observation_date=excluded.observation_date,sha256=excluded.sha256,size_bytes=excluded.size_bytes,modified_ns=excluded.modified_ns,row_count=excluded.row_count,imported_at=excluded.imported_at,source=excluded.source''',
                    (source_file.exchange,source_file.relative_path,source_file.observation_date,source_file.sha256,source_file.size_bytes,source_file.modified_ns,row_count,finished,SOURCE))
            # Refresh exact per-exchange status, including local latest observation and fetched row counts.
            for ex in EXCHANGES:
                if ex=='NSE': latest_market=c.execute("SELECT MAX(date) FROM prices WHERE quality=? AND source='tej_bazaar_local:NSE'",(QUALITY,)).fetchone()[0]
                else: latest_market=c.execute("SELECT MAX(date) FROM exchange_market_prices WHERE exchange='BSE' AND quality=?",(QUALITY,)).fetchone()[0]
                api_ex=source_status.get('exchanges',{}).get(ex,{})
                per_exchange[ex]={'latest_market_date':latest_market,'latest_source_date':source_latest[ex],
                    'rows_in_source':sum(n for f,n in processed if f.exchange==ex),
                    'rows_imported':counts[ex]['new_rows'],'rows_updated':counts[ex]['updated_rows'],
                    'rows_unchanged':counts[ex]['unchanged_rows'],'preserved_conflicts':counts[ex]['preserved_conflicts'],
                    'source_fetch_status':api_ex.get('status',source_status['status']),
                    'source_fetch_rows':api_ex.get('rows_fetched',0),'source_published_at':source_status.get('source_published_at'),
                    'status':'partial_success' if api_errors else ('succeeded' if changed_rows else 'unchanged')}
                c.execute('''INSERT INTO market_exchange_refresh(exchange,last_attempt_at,last_successful_at,last_source_fetch_at,last_source_fetch_status,latest_market_date,latest_source_date,rows_in_source,rows_imported,rows_updated,rows_unchanged,status,error,source_published_at,updated_at)
                  VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(exchange) DO UPDATE SET last_attempt_at=excluded.last_attempt_at,last_successful_at=excluded.last_successful_at,last_source_fetch_at=excluded.last_source_fetch_at,last_source_fetch_status=excluded.last_source_fetch_status,latest_market_date=excluded.latest_market_date,latest_source_date=excluded.latest_source_date,rows_in_source=excluded.rows_in_source,rows_imported=excluded.rows_imported,rows_updated=excluded.rows_updated,rows_unchanged=excluded.rows_unchanged,status=excluded.status,error=excluded.error,source_published_at=excluded.source_published_at,updated_at=excluded.updated_at''',
                  (ex,retrieved_at,finished,retrieved_at if source_status['status']!='deferred_bootstrap' else None,per_exchange[ex]['source_fetch_status'],latest_market,source_latest[ex],per_exchange[ex]['rows_in_source'],counts[ex]['new_rows'],counts[ex]['updated_rows'],counts[ex]['unchanged_rows'],per_exchange[ex]['status'],json.dumps([e for e in api_errors if e.get('exchange') in (ex,'both')]),source_status.get('source_published_at'),finished))
            if not changed_rows:
                final_status='unchanged' if source_status['status']=='succeeded' else 'failed'
                c.execute('UPDATE refresh_runs SET finished_at=?,status=?,source=?,rows_written=0,errors_json=?,warnings_json=?,step_timings_json=? WHERE id=?',
                    (finished,final_status,SOURCE,json.dumps(api_errors),json.dumps({'api':source_status,'exchanges':per_exchange},sort_keys=True),json.dumps({'elapsed_seconds':round(time.monotonic()-started,3)},sort_keys=True),run_id))
                c.execute('UPDATE data_import_runs SET refresh_run_id=?,status=?,finished_at=? WHERE id=?',(run_id,final_status,finished,run_import_id))
            else:
                final_status='succeeded' if source_status['status']=='succeeded' else 'partial_success'
                c.execute('UPDATE refresh_runs SET finished_at=?,status=?,source=?,rows_written=?,errors_json=?,warnings_json=?,version_id=?,import_run_id=?,change_summary_json=?,step_timings_json=? WHERE id=?',
                    (finished,final_status,SOURCE,changed_rows,json.dumps(api_errors),json.dumps({'api':source_status,'exchanges':per_exchange,'import':summary},sort_keys=True),publication['version_id'] if publication else None,run_import_id,json.dumps(summary,sort_keys=True),json.dumps({'elapsed_seconds':round(time.monotonic()-started,3)},sort_keys=True),run_id))
                c.execute('UPDATE data_import_runs SET refresh_run_id=?,status=?,finished_at=? WHERE id=?',(run_id,final_status,finished,run_import_id))
        return {'run_id':run_id,'status':final_status,'source':SOURCE,'rows_written':changed_rows,'rows_read':total_rows,'publication':publication,
            'exchanges':per_exchange,'errors':api_errors,'warnings':[] if not conflicts else [{'warning':f'{conflicts} existing non-Tej observations were retained.'}],
            'provider_trading_date':source_status.get('provider_trading_date'),'source_published_at':source_status.get('source_published_at'),
            'provider_fetch_status':source_status.get('status')}
    except Exception as exc:
        message=f'{type(exc).__name__}: {str(exc)[:600]}'
        finished=_now()
        with connect() as c:
            c.execute('UPDATE refresh_runs SET finished_at=?,status=?,source=?,errors_json=?,warnings_json=?,step_timings_json=? WHERE id=?',
                (finished,'failed',SOURCE,json.dumps([message]),json.dumps({'provider_fetch':source_status},sort_keys=True),json.dumps({'elapsed_seconds':round(time.monotonic()-started,3)}),run_id))
            for ex in EXCHANGES:
                c.execute('''INSERT INTO market_exchange_refresh(exchange,last_attempt_at,last_source_fetch_at,last_source_fetch_status,status,error,updated_at)
                  VALUES(?,?,?,?,?,?,?) ON CONFLICT(exchange) DO UPDATE SET last_attempt_at=excluded.last_attempt_at,last_source_fetch_at=excluded.last_source_fetch_at,last_source_fetch_status=excluded.last_source_fetch_status,status='failed',error=excluded.error,updated_at=excluded.updated_at''',
                  (ex,retrieved_at,retrieved_at,source_status.get('status','failed') if source_status else 'failed','failed',message,finished))
        return {'run_id':run_id,'status':'failed','source':SOURCE,'rows_written':0,'errors':[message],'latest_observation_date':None}
