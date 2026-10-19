from datetime import date, timedelta

import numpy as np
import pandas as pd

from .market import LocalDatabaseProvider

PERIODS={'1d':1,'5d':5,'1m':30,'3m':90,'6m':180,'1y':365,'prev_month':0}

def _price_history(symbol:str) -> pd.DataFrame:
    end=date.today(); start=end-timedelta(days=3655)
    df=LocalDatabaseProvider().history(symbol,start.isoformat(),end.isoformat())
    if not df.empty:
        df['date']=pd.to_datetime(df.date); df=df.sort_values('date').drop_duplicates('date',keep='last').set_index('date')
    return df

def _period_pair(df:pd.DataFrame,period:str):
    if df.empty: return None
    n=PERIODS.get(period)
    if n is None: raise ValueError('period must be one of '+', '.join(PERIODS))
    if period=='5d':
        return (df.iloc[-6] if len(df)>=6 else None, df.iloc[-1])
    if period=='1d': return (df.iloc[-2] if len(df)>=2 else None,df.iloc[-1])
    if period=='prev_month':
        current_start=date.today().replace(day=1); month_end=current_start-timedelta(days=1); month_start=month_end.replace(day=1)
        eligible=df[(df.index.date>=month_start)&(df.index.date<=month_end)]
        prior=df[df.index.date<month_start]
        return (prior.iloc[-1] if not prior.empty else None,eligible.iloc[-1] if not eligible.empty else None)
    cutoff=df.index[-1]-pd.Timedelta(days=n)
    eligible=df[df.index<=cutoff]
    return (eligible.iloc[-1] if not eligible.empty else None, df.iloc[-1])

def _metrics_for_frame(symbol:str,period:str,df:pd.DataFrame) -> dict:
    if df.empty: return {'symbol':symbol,'available':False,'reason':'No price observations are loaded.'}
    pair=_period_pair(df,period)
    if pair[0] is None: return {'symbol':symbol,'available':False,'reason':f'Insufficient history for {period}.'}
    a,b=pair; start,end=str(a.name.date()),str(b.name.date())
    raw_unadjusted=str(getattr(b,'price_basis',''))=='raw_unadjusted'
    basis='close' if raw_unadjusted else 'adjusted_close'
    ret=float(getattr(b,basis)/getattr(a,basis)-1)
    window=df.loc[a.name:b.name]
    daily=window[basis].pct_change().dropna()
    vol=float(daily.std()*np.sqrt(252)) if len(daily)>1 else None
    path=(1+daily).cumprod(); peak=path.cummax(); dd=path/peak-1
    m20=float(df[basis].tail(20).mean()); m50=float(df[basis].tail(50).mean())
    volume=float(window.volume.mean()) if 'volume' in window and window.volume.notna().any() else None
    first_v=float(window.volume.iloc[0]) if volume is not None else None
    last_v=float(window.volume.iloc[-1]) if volume is not None else None
    return {'symbol':symbol,'available':True,'period':period,'start_date':start,'end_date':end,'start_price':float(getattr(a,basis)),'end_price':float(getattr(b,basis)),'price_basis':'raw_close_unadjusted' if raw_unadjusted else 'adjusted_close','start_adjusted_close':None if raw_unadjusted else float(a.adjusted_close),'end_adjusted_close':None if raw_unadjusted else float(b.adjusted_close),'return':ret,'annualized_volatility':vol,'max_drawdown':float(dd.min()) if not dd.empty else 0,'moving_average_20':m20,'moving_average_50':m50,'momentum_20':float(df[basis].iloc[-1]/df[basis].iloc[-21]-1) if len(df)>=21 else None,'average_volume':volume,'volume_change':(last_v/first_v-1) if first_v and last_v else None,'source':str(b.source),'quality':str(b.quality),'retrieved_at':str(b.retrieved_at)}

def metrics(symbol:str,period:str='1m') -> dict:
    return _metrics_for_frame(symbol,period,_price_history(symbol))

def compare_benchmark(symbol:str,period:str='1m',benchmark:str='NIFTY50') -> dict:
    sh=_price_history(symbol); bh=_price_history(benchmark)
    if sh.empty or bh.empty:
        s={'symbol':symbol,'available':False,'reason':'No stock or benchmark observations are loaded.'}; b={'available':False}
    else:
        common=sh.index.intersection(bh.index)
        s=_metrics_for_frame(symbol,period,sh.loc[common]) if len(common) else {'symbol':symbol,'available':False,'reason':'No aligned stock/benchmark sessions.'}
        b=_metrics_for_frame(benchmark,period,bh.loc[common]) if len(common) else {'available':False}
    if not s.get('available') or not b.get('available'):
        s.update({'benchmark':benchmark,'benchmark_return':None,'simple_excess_return':None,'compounded_relative_return':None,'benchmark_quality':b.get('quality')}); return s
    s.update({'benchmark':benchmark,'benchmark_return':b['return'],'simple_excess_return':s['return']-b['return'],'compounded_relative_return':(1+s['return'])/(1+b['return'])-1,'benchmark_quality':b['quality']})
    return s

def fundamentals(symbol:str) -> dict:
    from .db import connect
    with connect() as c:
        query="SELECT * FROM published_fundamentals WHERE symbol=? AND quality<>'DEMO_SYNTHETIC' ORDER BY period DESC LIMIT 1"
        row=c.execute(query,(symbol.upper(),)).fetchone()
    if not row: return {'symbol':symbol.upper(),'available':False,'quality':'NO_DATA'}
    return dict(row)

def company_research(symbol:str,period='3m') -> dict:
    from .db import connect
    s=symbol.upper(); m=compare_benchmark(s,period)
    with connect() as c:
        query="SELECT date,kind,title,source,quality FROM published_events WHERE symbol=? AND quality<>'DEMO_SYNTHETIC' ORDER BY date DESC LIMIT 20"
        events=[dict(r) for r in c.execute(query,(s,))]
    with connect() as c: active=c.execute("SELECT value FROM app_state WHERE key='active_release_id'").fetchone()
    return {'symbol':s,'release_id':int(active['value']) if active else None,'performance':m,'fundamentals':fundamentals(s),'events':events,'disclaimer':'Research data may be synthetic, imported, incomplete, or unverified. Not investment advice.'}

def screen(period='1m',min_return=None,min_excess=None,max_volatility=None,max_drawdown=None,universe='nifty50') -> dict:
    from .db import connect
    with connect() as c:
        selected=universe.strip().lower()
        if selected in {'all','nse','inventory'}:
            latest=c.execute('SELECT MAX(as_of_date) FROM published_instrument_universe').fetchone()[0]
            rows=c.execute("SELECT symbol,name,sector FROM published_instrument_universe WHERE as_of_date=? AND exchange='NSE' AND quality<>'DEMO_SYNTHETIC' ORDER BY symbol",(latest,)).fetchall() if latest else []
        elif selected=='nifty50':
            rows=c.execute("SELECT s.symbol,s.name,s.sector FROM published_securities s JOIN published_constituent_snapshots cs ON cs.symbol=s.symbol WHERE cs.as_of_date=(SELECT MAX(as_of_date) FROM published_constituent_snapshots) AND cs.quality<>'DEMO_SYNTHETIC' ORDER BY s.symbol").fetchall()
        else: raise ValueError("universe must be 'nifty50' or 'all' (the latest imported NSE inventory).")
    out=[]
    for sec in rows:
        m=compare_benchmark(sec['symbol'],period)
        m.update({'name':sec['name'],'sector':sec['sector']})
        checks=[]
        for key,label,lim,op in [('return','minimum return',min_return,lambda a,b:a>=b),('simple_excess_return','minimum simple excess return',min_excess,lambda a,b:a>=b),('annualized_volatility','maximum annualized volatility',max_volatility,lambda a,b:a<=b),('max_drawdown','maximum drawdown',max_drawdown,lambda a,b:a>=b)]:
            if lim is None: continue
            val=m.get(key); passed=val is not None and op(val,lim); checks.append({'condition':label,'threshold':lim,'value':val,'passed':passed})
        m['checks']=checks; m['passes']=bool(m.get('available')) and all(x['passed'] for x in checks)
        out.append(m)
    eligible=[x for x in out if x['passes']]
    eligible.sort(key=lambda x:x.get('simple_excess_return') if x.get('simple_excess_return') is not None else -999,reverse=True)
    with connect() as c: active=c.execute("SELECT value FROM app_state WHERE key='active_release_id'").fetchone()
    return {'release_id':int(active['value']) if active else None,'period':period,'universe':universe,'benchmark':'NIFTY50','results':eligible,'evaluated':len(out),'criteria':{'min_return':min_return,'min_excess':min_excess,'max_volatility':max_volatility,'max_drawdown':max_drawdown},'data_note':'Every row includes provenance and quality. NIFTY 50 screening uses the latest dated constituent snapshot; broad screening uses only the latest imported NSE instrument inventory. With either input missing, that universe is empty. Synthetic demo observations are not market data.'}
