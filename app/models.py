"""Local, leakage-aware model registry. No external inference or serialization APIs."""
import hashlib
import json
import os
import threading
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .config import MODEL_DIR
from .db import connect
from .features import FEATURES, build_dataset, build_panel, training_eligibility

MIN_TRAINING_ROWS = 750
_job_lock = threading.Lock()
_job = {'status': 'idle', 'run_id': None}


def _forecast_lineage():
    with connect() as c:
        row=c.execute("SELECT id,content_fingerprint FROM dataset_versions WHERE dataset_type='prices' ORDER BY id DESC LIMIT 1").fetchone()
    return (row['id'],row['content_fingerprint']) if row else (None,None)


def _fit_ridge(x, y, alpha=10.0):
    x=np.asarray(x,dtype=float); y=np.asarray(y,dtype=float)
    mean=x.mean(axis=0); scale=x.std(axis=0); scale[scale==0]=1.0
    z=(x-mean)/scale; ymean=float(y.mean())
    coef=np.linalg.solve(z.T@z+alpha*np.eye(z.shape[1]),z.T@(y-ymean))
    return {'kind':'ridge','mean':mean.tolist(),'scale':scale.tolist(),'ymean':ymean,'coef':coef.tolist(),'alpha':alpha}


def _predict(model, x):
    x=np.asarray(x,dtype=float)
    if model['kind']=='constant': return np.full(len(x),float(model['value']))
    z=(x-np.asarray(model['mean']))/np.asarray(model['scale'])
    if model['kind']=='ridge': return z@np.asarray(model['coef'])+model['ymean']
    centers=np.asarray(model['centers']); gamma=float(model['gamma'])
    k=np.exp(-gamma*((z[:,None,:]-centers[None,:,:])**2).sum(axis=2))
    return k@np.asarray(model['coef'])+model['ymean']


def _fit_rbf(x,y,seed=29,components=48,alpha=1.0):
    """Small deterministic RBF feature model; nonlinear baseline needs only NumPy."""
    x=np.asarray(x,dtype=float); y=np.asarray(y,dtype=float)
    mean=x.mean(axis=0); scale=x.std(axis=0); scale[scale==0]=1.0; z=(x-mean)/scale
    rng=np.random.default_rng(seed); n=min(components,len(z)); centers=z[rng.choice(len(z),n,replace=False)]
    gamma=1.0/max(1,z.shape[1]); k=np.exp(-gamma*((z[:,None,:]-centers[None,:,:])**2).sum(axis=2))
    ymean=float(y.mean()); coef=np.linalg.solve(k.T@k+alpha*np.eye(n),k.T@(y-ymean))
    return {'kind':'rbf_ridge','mean':mean.tolist(),'scale':scale.tolist(),'ymean':ymean,'coef':coef.tolist(),'centers':centers.tolist(),'gamma':gamma,'alpha':alpha,'seed':seed}


def _metrics(actual,predicted):
    y=np.asarray(actual,dtype=float); p=np.asarray(predicted,dtype=float); e=y-p
    corr=float(np.corrcoef(y,p)[0,1]) if len(y)>2 and np.std(y)>0 and np.std(p)>0 else None
    return {'mae':float(np.mean(np.abs(e))),'rmse':float(np.sqrt(np.mean(e**2))),
            'directional_accuracy':float(np.mean((p>0)==(y>0))),'rank_ic':corr,'observations':int(len(y))}


def _walk_forward(frame,feature_cols,folds=4,purge=5):
    dates=sorted(frame.origin_date.unique()); n=len(dates)
    if n<MIN_TRAINING_ROWS: raise ValueError(f'Only {n} eligible rows; {MIN_TRAINING_ROWS} are required.')
    starts=np.linspace(int(n*.55),n,folds+1,dtype=int)
    predictions=[]; fold_metrics=[]; fold_date_ranges=[]; selection_date_set=[]
    for i in range(folds):
        test_start=starts[i]; test_end=starts[i+1]
        if test_end<=test_start: continue
        train_dates=dates[:max(0,test_start-purge)]
        test_dates=dates[test_start:test_end]
        if i==0: selection_date_set=list(test_dates)
        tr=frame[frame.origin_date.isin(train_dates)]; te=frame[frame.origin_date.isin(test_dates)]
        if len(tr)<100 or te.empty: continue
        fold_date_ranges.append((min(test_dates),max(test_dates)))
        xtr=tr[feature_cols].to_numpy(float); ytr=tr.target.to_numpy(float); xte=te[feature_cols].to_numpy(float)
        candidates={'zero_excess':np.zeros(len(te)),'historical_mean':np.full(len(te),float(ytr.mean()))}
        ridge=_fit_ridge(xtr,ytr); rbf=_fit_rbf(xtr,ytr,seed=29+i)
        candidates['ridge']=_predict(ridge,xte); candidates['rbf_ridge']=_predict(rbf,xte)
        fold_metrics.append({name:_metrics(te.target.to_numpy(),pred) for name,pred in candidates.items()})
        for idx,(_,row) in enumerate(te.iterrows()):
            for name,pred in candidates.items():
                predictions.append({'model_key':name,'symbol':row.get('symbol'),'origin_date':row.origin_date,
                    'target_end_date':row.target_end_date,'prediction':float(pred[idx]),'actual_excess':float(row.target),
                    'stock_return':float(row.stock_return),'benchmark_return':float(row.benchmark_return)})
    if not fold_metrics: raise ValueError('No chronological fold had sufficient training and evaluation rows.')
    if len(fold_metrics)<2: raise ValueError('At least two chronological validation folds are required: one for training-only model selection and later folds for evaluation.')
    aggregate={}
    for name in ('zero_excess','historical_mean','ridge','rbf_ridge'):
        selection=fold_metrics[0][name]
        values=[m[name] for m in fold_metrics[1:]]
        aggregate[name]={k:(float(np.mean([v[k] for v in values if v[k] is not None])) if any(v[k] is not None for v in values) else None) for k in ('mae','rmse','directional_accuracy','rank_ic')}
        aggregate[name]['observations']=sum(v['observations'] for v in values)
        aggregate[name]['folds']=[{**{k:v[k] for k in ('mae','rmse','directional_accuracy','rank_ic','observations')},'evaluation_start':fold_date_ranges[i+1][0],'evaluation_end':fold_date_ranges[i+1][1]} for i,v in enumerate(values)]
        aggregate[name]['selection_fold']={k:selection[k] for k in ('mae','rmse','directional_accuracy','rank_ic','observations')}
        aggregate[name]['selection_dates']=selection_date_set
        if len(fold_date_ranges)>1: aggregate[name]['evaluation_date_range']=[fold_date_ranges[1][0],fold_date_ranges[-1][1]]
    return predictions,aggregate


def _current_feature_row(symbol,feature_cols,*,published=True):
    # Feature construction shares the dataset builder's information set. The last
    # labeled row is five sessions stale, so calculate the latest x directly.
    from .features import eligible_price_history, _price_column, EVENT_FEATURES, _known_event_features
    stock=eligible_price_history(symbol,published=published); bench=eligible_price_history('NIFTY50',published=published)
    if stock.empty or bench.empty: raise ValueError('Current forecast requires user-confirmed stock and benchmark data.')
    sc,sb=_price_column(stock); bc,bb=_price_column(bench)
    if sb!=bb: raise ValueError('Cannot forecast with mismatched stock and benchmark price bases.')
    z=pd.DataFrame({'p':stock[sc],'v':stock.volume,'b':bench[bc]}).dropna(subset=['p','b'])
    p=z.p; b=z.b; r=p.pct_change(); rb=b.pct_change(); x={}
    for n in (1,5,20,60): x[f'ret_{n}']=p.pct_change(n).iloc[-1]
    for n in (5,20,60): x[f'relative_ret_{n}']=(p.pct_change(n)-b.pct_change(n)).iloc[-1]
    x['volatility_20']=r.rolling(20,min_periods=20).std().iloc[-1]; x['volatility_60']=r.rolling(60,min_periods=60).std().iloc[-1]
    x['ma20_distance']=p.iloc[-1]/p.rolling(20,min_periods=20).mean().iloc[-1]-1
    x['ma50_distance']=p.iloc[-1]/p.rolling(50,min_periods=50).mean().iloc[-1]-1
    x['drawdown_60']=p.iloc[-1]/p.rolling(60,min_periods=20).max().iloc[-1]-1
    x['market_ret_20']=b.pct_change(20).iloc[-1]; x['market_volatility_20']=rb.rolling(20,min_periods=20).std().iloc[-1]
    x['volume_change_5']=z.v.pct_change(5).iloc[-1]; x['average_turnover_20']=(p*z.v).rolling(20,min_periods=20).mean().iloc[-1]
    if 'volume_change_5' not in feature_cols: x.pop('volume_change_5',None); x.pop('average_turnover_20',None)
    if any(col.startswith('fundamental_') for col in feature_cols):
        from .features import FUNDAMENTAL_FEATURES
        origin_time=(z.index[-1].tz_localize('Asia/Kolkata')+pd.Timedelta(hours=15,minutes=30)).tz_convert('UTC').isoformat()
        with connect() as c:
            record=c.execute(f"SELECT revenue_growth,profit_growth,debt_equity,roe FROM {'published_fundamental_observations' if published else 'fundamental_observations'} WHERE symbol=? AND quality='USER_CONFIRMED_UNVERIFIED' AND published_at<=? ORDER BY published_at DESC LIMIT 1",(symbol.upper(),origin_time)).fetchone()
        if record:
            x.update(dict(zip(FUNDAMENTAL_FEATURES,[record[k] for k in ('revenue_growth','profit_growth','debt_equity','roe')])))
    if any(col in EVENT_FEATURES for col in feature_cols):
        event_rows,_=_known_event_features(symbol,z.index,published=published)
        if event_rows is not None: x.update(event_rows.iloc[-1].to_dict())
    row=np.asarray([[x[k] for k in feature_cols]],dtype=float)
    if not np.isfinite(row).all(): raise ValueError('Insufficient complete latest features for a current forecast.')
    return row, z.index[-1].strftime('%Y-%m-%d')


def _save_artifact(run_id,model_key,model_type,model,data_version,feature_cols,metrics,period):
    MODEL_DIR.mkdir(parents=True,exist_ok=True)
    artifact={'format':'nivesh-local-model-v1','run_id':run_id,'model_key':model_key,'model_type':model_type,
              'model':model,'feature_columns':feature_cols,'data_version':data_version,'training_period':period,
              'metrics':metrics,'price_basis':period.get('price_basis','unknown')}
    blob=json.dumps(artifact,separators=(',',':'),allow_nan=False).encode()
    digest=hashlib.sha256(blob).hexdigest(); path=MODEL_DIR/f'{model_key.replace(":","_")}-{run_id}-{digest[:12]}.json'
    temp=path.with_suffix('.tmp'); temp.write_bytes(blob); os.replace(temp,path)
    with connect() as c:
        cur=c.execute('INSERT INTO model_versions(training_run_id,model_key,model_type,artifact_path,artifact_sha256,data_version,feature_schema_json,config_json,metrics_json,training_period_json,status,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',
            (run_id,model_key,model_type,str(path),digest,data_version,json.dumps(feature_cols),json.dumps({'algorithm':model_type,'min_rows':MIN_TRAINING_ROWS,'purge_sessions':5}),json.dumps(metrics),json.dumps(period),'eligible',datetime.now(timezone.utc).isoformat()))
        return cur.lastrowid,digest,path


def load_model_version(version_id):
    """Load only a registered JSON artifact whose checksum and format match."""
    with connect() as c: row=c.execute("SELECT * FROM model_versions WHERE id=? AND status='eligible'",(version_id,)).fetchone()
    if not row: raise ValueError('No eligible registered model version.')
    path=MODEL_DIR/(__import__('pathlib').Path(row['artifact_path']).name)
    blob=path.read_bytes()
    if hashlib.sha256(blob).hexdigest()!=row['artifact_sha256']: raise ValueError('Registered model artifact checksum mismatch.')
    model=json.loads(blob)
    if model.get('format')!='nivesh-local-model-v1': raise ValueError('Unsupported registered model artifact format.')
    return model


def _new_training_run(mode,total=0):
    with connect() as c:
        cur=c.execute("INSERT INTO training_runs(mode,status,started_at,total) VALUES(?,'running',?,?)",(mode,datetime.now(timezone.utc).isoformat(),total))
        return cur.lastrowid


def train_company(symbol):
    symbol=symbol.upper(); run_id=_new_training_run('company:'+symbol,0)
    try:
        eligibility=training_eligibility(symbol)
        if not eligibility['eligible']: raise ValueError(eligibility['reason'])
        data,meta=build_dataset(symbol)
        if len(data)<MIN_TRAINING_ROWS: raise ValueError(f'{symbol} has {len(data)} eligible rows; {MIN_TRAINING_ROWS} required.')
        with connect() as c: c.execute('UPDATE training_runs SET total=? WHERE id=?',(len(data),run_id))
    except Exception as exc:
        with connect() as c: c.execute("UPDATE training_runs SET status='failed',finished_at=?,errors_json=? WHERE id=?",(datetime.now(timezone.utc).isoformat(),json.dumps([str(exc)[:500]]),run_id))
        raise
    cols=meta['feature_columns']
    version_id=None
    try:
        oos,metrics=_walk_forward(data,cols)
        tr=data; x=tr[cols].to_numpy(float); y=tr.target.to_numpy(float)
        candidates={'ridge':_fit_ridge(x,y),'rbf_ridge':_fit_rbf(x,y),'zero_excess':{'kind':'constant','value':0.0},'historical_mean':{'kind':'constant','value':float(y.mean())}}
        selected=min(candidates,key=lambda m:metrics[m]['selection_fold']['mae'])
        version_id,digest,path=_save_artifact(run_id,'company:'+symbol,selected,candidates[selected],meta['data_version'],cols,metrics,{'start':str(data.origin_date.min()),'end':str(data.origin_date.max()),'price_basis':meta['price_basis']})
        with connect() as c:
            for p in oos:
                if p['symbol'] is None: p['symbol']=symbol
                if p['model_key']!=selected or p['origin_date'] in set(metrics[selected]['selection_dates']): continue
                c.execute('INSERT OR IGNORE INTO oos_predictions(training_run_id,model_key,symbol,origin_date,target_end_date,prediction,actual_excess,stock_return,benchmark_return,data_version,price_basis) VALUES(?,?,?,?,?,?,?,?,?,?,?)',(run_id,'company:'+selected,p['symbol'],p['origin_date'],p['target_end_date'],p['prediction'],p['actual_excess'],p['stock_return'],p['benchmark_return'],meta['data_version'],meta['price_basis']))
        latest,origin=_current_feature_row(symbol,cols)
        pred=float(_predict(candidates[selected],latest)[0])
        input_version,input_fingerprint=_forecast_lineage()
        with connect() as c:
            c.execute("INSERT INTO current_forecasts(model_version_id,symbol,origin_date,horizon_sessions,prediction,price_basis,created_at,version_id,input_fingerprint,status,stale_reason) VALUES(?,?,?,?,?,?,?,?,?,'current',NULL) ON CONFLICT(model_version_id,symbol,origin_date) DO UPDATE SET prediction=excluded.prediction,created_at=excluded.created_at,version_id=excluded.version_id,input_fingerprint=excluded.input_fingerprint,status='current',stale_reason=NULL",(version_id,symbol,origin,5,pred,meta['price_basis'],datetime.now(timezone.utc).isoformat(),input_version,input_fingerprint))
            c.execute("UPDATE training_runs SET status='succeeded',finished_at=?,progress=?,summary_json=? WHERE id=?",(datetime.now(timezone.utc).isoformat(),len(data),json.dumps({'model_version_id':version_id,'selected_model':selected,'metrics':metrics,'data':meta}),run_id))
        return {'run_id':run_id,'status':'succeeded','symbol':symbol,'model_version_id':version_id,'selected_model':selected,'metrics':metrics,'current_forecast':{'origin_date':origin,'prediction':pred,'horizon_sessions':5,'price_basis':meta['price_basis']},'artifact_sha256':digest,'artifact_path':str(path),'note':'User-confirmed dataset provenance is not independently verified; research forecast only.'}
    except Exception as exc:
        with connect() as c:
            c.execute("UPDATE training_runs SET status='failed',finished_at=?,errors_json=? WHERE id=?",(datetime.now(timezone.utc).isoformat(),json.dumps([str(exc)[:500]]),run_id))
            if version_id: c.execute("UPDATE model_versions SET status='failed' WHERE id=?",(version_id,))
        raise


def train_universe():
    run_id=_new_training_run('pooled',0)
    panel,meta=build_panel()
    if panel.empty:
        message='No eligible universe data. Import real data, confirm permitted research use, and provide benchmark and historical membership coverage.'
        with connect() as c: c.execute("UPDATE training_runs SET status='failed',finished_at=?,errors_json=? WHERE id=?",(datetime.now(timezone.utc).isoformat(),json.dumps([message]),run_id))
        raise ValueError(message)
    cols=meta.get('feature_columns') or [c for c in FEATURES if c in panel]
    if len(panel)<MIN_TRAINING_ROWS:
        message=f'{len(panel)} pooled rows available; {MIN_TRAINING_ROWS} required.'
        with connect() as c: c.execute("UPDATE training_runs SET status='failed',finished_at=?,total=?,errors_json=? WHERE id=?",(datetime.now(timezone.utc).isoformat(),len(panel),json.dumps([message]),run_id))
        raise ValueError(message)
    with connect() as c: c.execute('UPDATE training_runs SET total=? WHERE id=?',(len(panel),run_id))
    version_id=None
    try:
        oos,metrics=_walk_forward(panel,cols)
        x=panel[cols].to_numpy(float); y=panel.target.to_numpy(float)
        candidates={'ridge':_fit_ridge(x,y),'rbf_ridge':_fit_rbf(x,y),'zero_excess':{'kind':'constant','value':0.0},'historical_mean':{'kind':'constant','value':float(y.mean())}}
        selected=min(candidates,key=lambda m:metrics[m]['selection_fold']['mae'])
        version_id,digest,path=_save_artifact(run_id,'pooled',selected,candidates[selected],hashlib.sha256('|'.join(meta.get('data_versions',[])).encode()).hexdigest(),cols,metrics,{'start':panel.origin_date.min(),'end':panel.origin_date.max(),'symbols':meta['symbols_included'],'price_basis':meta['price_basis']})
        with connect() as c:
            for p in oos:
                if p['model_key']==selected and p['origin_date'] not in set(metrics[selected]['selection_dates']):
                    c.execute('INSERT OR IGNORE INTO oos_predictions(training_run_id,model_key,symbol,origin_date,target_end_date,prediction,actual_excess,stock_return,benchmark_return,data_version,price_basis) VALUES(?,?,?,?,?,?,?,?,?,?,?)',(run_id,'pooled:'+selected,p['symbol'],p['origin_date'],p['target_end_date'],p['prediction'],p['actual_excess'],p['stock_return'],p['benchmark_return'],hashlib.sha256('|'.join(meta.get('data_versions',[])).encode()).hexdigest(),meta['price_basis']))
        for symbol in meta['symbols_included']:
            try:
                latest,origin=_current_feature_row(symbol,cols); forecast=float(_predict(candidates[selected],latest)[0])
                input_version,input_fingerprint=_forecast_lineage()
                with connect() as c: c.execute("INSERT INTO current_forecasts(model_version_id,symbol,origin_date,horizon_sessions,prediction,price_basis,created_at,version_id,input_fingerprint,status,stale_reason) VALUES(?,?,?,?,?,?,?,?,?,'current',NULL) ON CONFLICT(model_version_id,symbol,origin_date) DO UPDATE SET prediction=excluded.prediction,created_at=excluded.created_at,version_id=excluded.version_id,input_fingerprint=excluded.input_fingerprint,status='current',stale_reason=NULL",(version_id,symbol,origin,5,forecast,meta['price_basis'],datetime.now(timezone.utc).isoformat(),input_version,input_fingerprint))
            except ValueError: pass
        with connect() as c:
            c.execute("UPDATE training_runs SET status='succeeded',finished_at=?,progress=?,summary_json=? WHERE id=?",(datetime.now(timezone.utc).isoformat(),len(panel),json.dumps({'model_version_id':version_id,'selected_model':selected,'metrics':metrics,'data':meta}),run_id))
        return {'run_id':run_id,'status':'succeeded','model_version_id':version_id,'selected_model':selected,'metrics':metrics,'data':meta,'artifact_sha256':digest,'artifact_path':str(path)}
    except Exception as exc:
        with connect() as c:
            c.execute("UPDATE training_runs SET status='failed',finished_at=?,errors_json=? WHERE id=?",(datetime.now(timezone.utc).isoformat(),json.dumps([str(exc)[:500]]),run_id))
            if version_id: c.execute("UPDATE model_versions SET status='failed' WHERE id=?",(version_id,))
        raise


def train_aggregator():
    run_id=_new_training_run('aggregator',0)
    with connect() as c:
        rows=c.execute("SELECT symbol,origin_date,target_end_date,model_key,prediction,actual_excess,stock_return,benchmark_return,price_basis FROM oos_predictions WHERE model_key<>'master_aggregator' ORDER BY origin_date,symbol,model_key").fetchall()
    if not rows:
        message='Aggregator not trainable: no base-model out-of-fold predictions are recorded.'
        with connect() as c: c.execute("UPDATE training_runs SET status='failed',finished_at=?,errors_json=? WHERE id=?",(datetime.now(timezone.utc).isoformat(),json.dumps([message]),run_id))
        raise ValueError(message)
    frame=pd.DataFrame([dict(r) for r in rows])
    if frame.price_basis.nunique()>1:
        message='Aggregator refuses OOS streams with mixed price bases.'
        with connect() as c: c.execute("UPDATE training_runs SET status='failed',finished_at=?,errors_json=? WHERE id=?",(datetime.now(timezone.utc).isoformat(),json.dumps([message]),run_id))
        raise ValueError(message)
    wide=frame.pivot_table(index=['symbol','origin_date','target_end_date'],columns='model_key',values='prediction',aggfunc='first').reset_index()
    actual=frame.drop_duplicates(['symbol','origin_date'])[['symbol','origin_date','target_end_date','actual_excess','stock_return','benchmark_return']]
    data=wide.merge(actual,on=['symbol','origin_date','target_end_date']).dropna()
    model_cols=[c for c in data.columns if c not in {'symbol','origin_date','target_end_date','actual_excess','stock_return','benchmark_return'}]
    dates=sorted(data.origin_date.unique())
    if len(data)<100 or len(dates)<30 or len(model_cols)<2:
        message=f'Aggregator needs >=100 aligned rows, >=30 dates and >=2 base OOS prediction streams; has {len(data)} rows, {len(dates)} dates, {len(model_cols)} streams.'
        with connect() as c: c.execute("UPDATE training_runs SET status='failed',finished_at=?,total=?,errors_json=? WHERE id=?",(datetime.now(timezone.utc).isoformat(),len(data),json.dumps([message]),run_id))
        raise ValueError(message)
    split=int(len(dates)*.7); train_dates=dates[:max(0,split-5)]; test_dates=dates[split:]
    tr=data[data.origin_date.isin(train_dates)]; te=data[data.origin_date.isin(test_dates)]
    if len(tr)<60 or te.empty:
        message='Aggregator chronological holdout is insufficient after five-session purging.'
        with connect() as c: c.execute("UPDATE training_runs SET status='failed',finished_at=?,total=?,errors_json=? WHERE id=?",(datetime.now(timezone.utc).isoformat(),len(data),json.dumps([message]),run_id))
        raise ValueError(message)
    model=_fit_ridge(tr[model_cols].to_numpy(float),tr.actual_excess.to_numpy(float)); pred=_predict(model,te[model_cols].to_numpy(float))
    metrics={'ridge_aggregator':_metrics(te.actual_excess,pred),'zero_baseline':_metrics(te.actual_excess,np.zeros(len(te))),'historical_mean':_metrics(te.actual_excess,np.full(len(te),float(tr.actual_excess.mean()))),'base_streams':model_cols,'out_of_fold_only':True,'purge_sessions':5,'evaluation_start':min(test_dates),'evaluation_end':max(test_dates)}
    with connect() as c: c.execute('UPDATE training_runs SET total=? WHERE id=?',(len(data),run_id))
    version_id=None
    try:
        if metrics['ridge_aggregator']['mae'] < metrics['zero_baseline']['mae']:
            version_id,digest,path=_save_artifact(run_id,'master_aggregator','ridge_aggregator',model,hashlib.sha256('|'.join(sorted(set(frame.model_key))).encode()).hexdigest(),model_cols,metrics,{'start':min(train_dates),'end':max(test_dates),'price_basis':frame.price_basis.iloc[0]})
            with connect() as c:
                for _,r in te.iterrows(): c.execute('INSERT OR IGNORE INTO oos_predictions(training_run_id,model_key,symbol,origin_date,target_end_date,prediction,actual_excess,stock_return,benchmark_return,data_version,price_basis) VALUES(?,?,?,?,?,?,?,?,?,?,?)',(run_id,'master_aggregator',r.symbol,r.origin_date,r.target_end_date,float(_predict(model,r[model_cols].to_numpy(float).reshape(1,-1))[0]),float(r.actual_excess),float(r.stock_return),float(r.benchmark_return),'aggregated_base_oos',r.price_basis))
                forecast_rows=c.execute("SELECT f.symbol,f.origin_date,f.prediction,f.price_basis,v.model_key,v.model_type FROM current_forecasts f JOIN model_versions v ON v.id=f.model_version_id WHERE v.status='eligible'").fetchall()
                by_symbol={}
                for f in forecast_rows:
                    stream=('company:' if f['model_key'].startswith('company:') else 'pooled:')+f['model_type']
                    by_symbol.setdefault(f['symbol'],{})[stream]=dict(f)
                for symbol,streams in by_symbol.items():
                    if not set(model_cols).issubset(streams): continue
                    price_bases={streams[k]['price_basis'] for k in model_cols}
                    if len(price_bases)!=1 or next(iter(price_bases))!=frame.price_basis.iloc[0]: continue
                    origins={streams[k]['origin_date'] for k in model_cols}
                    if len(origins)!=1: continue
                    vector=np.asarray([[streams[k]['prediction'] for k in model_cols]],float)
                    prediction=float(_predict(model,vector)[0]); origin=next(iter(origins))
                    input_version,input_fingerprint=_forecast_lineage()
                    c.execute("INSERT INTO current_forecasts(model_version_id,symbol,origin_date,horizon_sessions,prediction,price_basis,created_at,version_id,input_fingerprint,status,stale_reason) VALUES(?,?,?,?,?,?,?,?,?,'current',NULL) ON CONFLICT(model_version_id,symbol,origin_date) DO UPDATE SET prediction=excluded.prediction,created_at=excluded.created_at,version_id=excluded.version_id,input_fingerprint=excluded.input_fingerprint,status='current',stale_reason=NULL",(version_id,symbol,origin,5,prediction,next(iter(price_bases)),datetime.now(timezone.utc).isoformat(),input_version,input_fingerprint))
        with connect() as c: c.execute("UPDATE training_runs SET status=?,finished_at=?,progress=?,summary_json=? WHERE id=?",('succeeded' if version_id else 'evaluated_no_eligible_model',datetime.now(timezone.utc).isoformat(),len(data),json.dumps({'metrics':metrics,'model_version_id':version_id}),run_id))
        return {'status':'succeeded' if version_id else 'evaluated_no_eligible_model','run_id':run_id,'model_version_id':version_id,'metrics':metrics,'note':'Aggregator uses only stored out-of-fold base-model predictions and is retained only when it beats the zero baseline on a purged chronological holdout.'}
    except Exception as exc:
        with connect() as c:
            c.execute("UPDATE training_runs SET status='failed',finished_at=?,errors_json=? WHERE id=?",(datetime.now(timezone.utc).isoformat(),json.dumps([str(exc)[:500]]),run_id))
            if version_id: c.execute("UPDATE model_versions SET status='failed' WHERE id=?",(version_id,))
        raise


def train_all_eligible_companies(run_id=None, symbols=None):
    """Train eligible company models one at a time and persist every outcome."""
    if symbols is None:
        with connect() as c:
            latest=c.execute('SELECT MAX(as_of_date) FROM published_instrument_universe').fetchone()[0]
            symbols=[r[0] for r in c.execute("SELECT symbol FROM published_instrument_universe WHERE as_of_date=? AND exchange='NSE' AND quality<>'DEMO_SYNTHETIC' ORDER BY symbol",(latest,))] if latest else []
    decisions=[]
    for symbol in symbols:
        try:
            eligibility=training_eligibility(symbol)
            decisions.append((symbol,eligibility['eligible'],eligibility['reason']))
        except Exception as exc:
            decisions.append((symbol,False,f'Eligibility check failed: {str(exc)[:300]}'))
    eligible_symbols=[s for s,ok,_ in decisions if ok]
    now=datetime.now(timezone.utc).isoformat()
    with connect() as c:
        if run_id is None:
            cur=c.execute("INSERT INTO training_runs(mode,status,started_at,total) VALUES('company_batch','running',?,?)",(now,len(symbols))); run_id=cur.lastrowid
        for symbol,ok,reason in decisions:
            c.execute('INSERT INTO training_run_items(run_id,symbol,status,reason,updated_at) VALUES(?,?,?,?,?) ON CONFLICT(run_id,symbol) DO UPDATE SET status=excluded.status,reason=excluded.reason,updated_at=excluded.updated_at',(run_id,symbol,'pending' if ok else 'skipped',None if ok else reason,now))
        summary={'discovered':len(symbols),'eligible':len(eligible_symbols),'completed':0,'results':[]}
        skipped_count=len(symbols)-len(eligible_symbols)
        c.execute('UPDATE training_runs SET status=?,progress=?,summary_json=? WHERE id=?',('running' if eligible_symbols else 'no_eligible_companies',skipped_count,json.dumps(summary),run_id))
    _job.update({'progress':skipped_count,'total':len(symbols),'results':[{'symbol':s,'status':'skipped','reason':reason} for s,ok,reason in decisions if not ok]})
    if not eligible_symbols:
        with connect() as c: c.execute("UPDATE training_runs SET finished_at=?,errors_json=? WHERE id=?",(datetime.now(timezone.utc).isoformat(),json.dumps(['No company has sufficient user-confirmed eligible price and benchmark history.']),run_id))
        return {'run_id':run_id,'status':'no_eligible_companies','discovered':len(symbols),'eligible':0,'succeeded':0,'failed':0,'skipped':len(symbols),'results':[]}
    outcomes=[]; succeeded=failed=0
    for position,symbol in enumerate(eligible_symbols,1):
        try:
            result=train_company(symbol); succeeded+=1
            item={'symbol':symbol,'status':'succeeded','model_version_id':result['model_version_id'],'metrics':result['metrics'],'training_run_id':result['run_id']}
            with connect() as c: c.execute("UPDATE training_run_items SET status='succeeded',model_version_id=?,metrics_json=?,updated_at=? WHERE run_id=? AND symbol=?",(result['model_version_id'],json.dumps(result['metrics']),datetime.now(timezone.utc).isoformat(),run_id,symbol))
        except Exception as exc:
            failed+=1; item={'symbol':symbol,'status':'failed','reason':str(exc)[:500]}
            with connect() as c: c.execute("UPDATE training_run_items SET status='failed',reason=?,updated_at=? WHERE run_id=? AND symbol=?",(item['reason'],datetime.now(timezone.utc).isoformat(),run_id,symbol))
        outcomes.append(item); summary={'discovered':len(symbols),'eligible':len(eligible_symbols),'completed':position,'succeeded':succeeded,'failed':failed,'skipped':len(symbols)-len(eligible_symbols),'results':outcomes}
        with connect() as c: c.execute('UPDATE training_runs SET progress=?,summary_json=? WHERE id=?',(skipped_count+position,json.dumps(summary),run_id))
        _job.update({'progress':skipped_count+position,'total':len(symbols),'current_symbol':symbol,'results':[{'symbol':s,'status':'skipped','reason':reason} for s,ok,reason in decisions if not ok]+outcomes})
    status='succeeded' if failed==0 else 'partial_success' if succeeded else 'failed'
    with connect() as c: c.execute('UPDATE training_runs SET status=?,finished_at=?,progress=?,summary_json=?,errors_json=? WHERE id=?',(status,datetime.now(timezone.utc).isoformat(),len(symbols),json.dumps(summary),json.dumps([x['symbol']+': '+x['reason'] for x in outcomes if x['status']=='failed']),run_id))
    return {'run_id':run_id,'status':status,'discovered':len(symbols),'eligible':len(eligible_symbols),'succeeded':succeeded,'failed':failed,'skipped':len(symbols)-len(eligible_symbols),'results':outcomes}


def start_training(mode, symbol=None):
    if not _job_lock.acquire(blocking=False): return {'status':'already_running','job':dict(_job)}
    if mode not in ('aggregator','company_batch','company','pooled'):
        _job_lock.release(); raise ValueError('Unsupported local training mode.')
    batch_run_id=None; batch_symbols=None
    if mode=='company_batch':
        with connect() as c:
            latest=c.execute('SELECT MAX(as_of_date) FROM published_instrument_universe').fetchone()[0]
            batch_symbols=[r[0] for r in c.execute("SELECT symbol FROM published_instrument_universe WHERE as_of_date=? AND exchange='NSE' AND quality<>'DEMO_SYNTHETIC' ORDER BY symbol",(latest,))] if latest else []
        batch_run_id=_new_training_run('company_batch',len(batch_symbols))
        with connect() as c:
            for company_symbol in batch_symbols: c.execute("INSERT INTO training_run_items(run_id,symbol,status,updated_at) VALUES(?,?,'pending',?)",(batch_run_id,company_symbol,datetime.now(timezone.utc).isoformat()))
    _job.update({'status':'queued','mode':mode,'symbol':symbol,'started_at':datetime.now(timezone.utc).isoformat(),'run_id':batch_run_id,'error':None,'progress':0,'total':len(batch_symbols) if batch_symbols is not None else 0,'results':[]})
    def work():
        try:
            _job['status']='running'
            result=train_company(symbol) if mode=='company' else train_universe() if mode=='pooled' else train_aggregator() if mode=='aggregator' else train_all_eligible_companies(batch_run_id,batch_symbols)
            _job.update({'status':result.get('status','succeeded'),'run_id':result.get('run_id'),'finished_at':datetime.now(timezone.utc).isoformat(),'result':result})
        except Exception as exc:
            _job.update({'status':'failed','finished_at':datetime.now(timezone.utc).isoformat(),'error':str(exc)[:500]})
            if batch_run_id:
                with connect() as c: c.execute("UPDATE training_runs SET status='failed',finished_at=?,errors_json=? WHERE id=?",(datetime.now(timezone.utc).isoformat(),json.dumps([str(exc)[:500]]),batch_run_id))
        finally: _job_lock.release()
    threading.Thread(target=work,name='nivesh-local-training',daemon=True).start()
    return {'status':'queued','mode':mode,'symbol':symbol,'run_id':batch_run_id}


def training_status():
    with connect() as c:
        runs=[dict(r) for r in c.execute('SELECT * FROM training_runs ORDER BY id DESC LIMIT 25')]
        versions=[dict(r) for r in c.execute('SELECT id,training_run_id,model_key,model_type,data_version,metrics_json,training_period_json,status,created_at FROM model_versions ORDER BY id DESC LIMIT 25')]
        last_batch=c.execute("SELECT id FROM training_runs WHERE mode='company_batch' ORDER BY id DESC LIMIT 1").fetchone()
        batch_items=[dict(r) for r in c.execute('SELECT i.*,v.model_type AS selected_model,v.metrics_json AS registered_metrics_json FROM training_run_items i LEFT JOIN model_versions v ON v.id=i.model_version_id WHERE i.run_id=? ORDER BY i.symbol',(last_batch['id'],))] if last_batch else []
        last_success=c.execute("SELECT id,mode,finished_at,summary_json FROM training_runs WHERE status='succeeded' ORDER BY id DESC LIMIT 1").fetchone()
    eligible=False
    with connect() as c: symbols=[r[0] for r in c.execute("SELECT DISTINCT symbol FROM published_prices WHERE quality='USER_CONFIRMED_UNVERIFIED'")]
    for symbol in symbols:
        if training_eligibility(symbol)['eligible']: eligible=True; break
    with connect() as c: oos=[dict(r) for r in c.execute("SELECT symbol,origin_date,target_end_date,model_key,prediction,actual_excess FROM oos_predictions WHERE model_key<>'master_aggregator'")]
    aggregator={'eligible':False,'reason':'Requires at least two sufficiently long, aligned base-model OOS prediction streams.'}
    if oos:
        f=pd.DataFrame(oos); wide=f.pivot_table(index=['symbol','origin_date','target_end_date'],columns='model_key',values='prediction',aggfunc='first').dropna()
        aggregator={'eligible':len(wide)>=100 and wide.index.get_level_values('origin_date').nunique()>=30 and wide.shape[1]>=2,'aligned_rows':len(wide),'streams':wide.shape[1],'distinct_dates':wide.index.get_level_values('origin_date').nunique(),'reason':'Ready for aggregator evaluation.' if len(wide)>=100 and wide.index.get_level_values('origin_date').nunique()>=30 and wide.shape[1]>=2 else 'Insufficient aligned out-of-fold prediction streams.'}
    calibration={'status':'not_calibrated','reason':'No OOS prediction stream is available.'}
    if oos:
        best=max(set(r['model_key'] for r in oos),key=lambda k:sum(r['model_key']==k for r in oos))
        from .calibration import calibration_status
        calibration=calibration_status(best)
    return {'job':dict(_job),'runs':runs,'model_versions':versions,'training_available':eligible,'latest_batch':{'run_id':last_batch['id'],'items':batch_items} if last_batch else None,'last_successful_run':dict(last_success) if last_success else None,'aggregator_eligibility':aggregator,'calibration':calibration,'note':'Training requires at least 750 eligible aligned observations plus user-confirmed provenance and benchmark data; any attestation remains user-declared, not independently verified.'}


def latest_model(symbol=None):
    with connect() as c:
        if symbol:
            row=c.execute('SELECT f.*,v.model_key,v.model_type,v.metrics_json,v.data_version,v.feature_schema_json,v.training_period_json FROM current_forecasts f JOIN model_versions v ON v.id=f.model_version_id WHERE f.symbol=? AND v.status=\'eligible\' ORDER BY f.origin_date DESC,f.id DESC LIMIT 1',(symbol.strip().upper(),)).fetchone()
        else:
            row=c.execute('SELECT f.*,v.model_key,v.model_type,v.metrics_json,v.data_version,v.feature_schema_json,v.training_period_json FROM current_forecasts f JOIN model_versions v ON v.id=f.model_version_id WHERE v.status=\'eligible\' ORDER BY f.created_at DESC,f.id DESC LIMIT 1').fetchone()
    if not row: return {'available':False,'symbol':symbol.strip().upper() if symbol else None,'message':'No eligible current forecast; real validated observations and a trained local model are required.'}
    oos_key='master_aggregator' if row['model_key']=='master_aggregator' else ('company:' if row['model_key'].startswith('company:') else 'pooled:')+row['model_type']
    from .calibration import calibrated_probability
    calibrated=calibrated_probability(oos_key,row['prediction'])
    try: stale=(datetime.now().date()-datetime.strptime(row['origin_date'],'%Y-%m-%d').date()).days>5
    except ValueError: stale=True
    return {'available':True,'symbol':row['symbol'],'model_version_id':row['model_version_id'],'model_key':row['model_key'],'model_type':row['model_type'],'latest_prediction':row['prediction'],'prediction_origin_date':row['origin_date'],'forecast_created_at':row['created_at'],'stale':stale,'stale_warning':'Forecast origin is more than five calendar days old; do not treat as current.' if stale else None,'horizon_sessions':row['horizon_sessions'],'price_basis':row['price_basis'],'data_version':row['data_version'],'feature_schema':json.loads(row['feature_schema_json']),'training_period':json.loads(row['training_period_json']),'metrics':json.loads(row['metrics_json']),'calibration':calibrated,'provenance':'user_confirmed_unverified_local_dataset','quality':'USER_CONFIRMED_UNVERIFIED'}




