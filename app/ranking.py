"""Transparent factor ranking; forecast rows require an explicit OOS provenance record."""
from .analytics import compare_benchmark, fundamentals
from .db import connect

def rank(period='3m',universe='nifty50',weights=None):
    weights=weights or {'excess_return':.35,'forecast':.15,'low_volatility':.15,'shallow_drawdown':.15,'fundamentals':.20}
    with connect() as c:
        if universe=='nifty50':
            sec=c.execute("SELECT s.symbol,s.name,s.sector FROM published_securities s JOIN published_constituent_snapshots cs ON cs.symbol=s.symbol WHERE cs.as_of_date=(SELECT MAX(as_of_date) FROM published_constituent_snapshots) AND cs.quality<>'DEMO_SYNTHETIC' ORDER BY s.symbol").fetchall()
        elif universe in {'all','nse','inventory'}:
            latest=c.execute('SELECT MAX(as_of_date) FROM published_instrument_universe').fetchone()[0]
            sec=c.execute("SELECT symbol,name,sector FROM published_instrument_universe WHERE as_of_date=? AND exchange='NSE' AND quality<>'DEMO_SYNTHETIC' ORDER BY symbol",(latest,)).fetchall() if latest else []
        else: raise ValueError("universe must be 'nifty50' or 'all' (the latest imported NSE inventory).")
        models={r['symbol']:r for r in c.execute("SELECT f.symbol,f.prediction,f.origin_date,f.price_basis,v.model_key,v.model_type,v.metrics_json FROM current_forecasts f JOIN model_versions v ON v.id=f.model_version_id WHERE v.status='eligible' AND f.id IN (SELECT MAX(f2.id) FROM current_forecasts f2 JOIN model_versions v2 ON v2.id=f2.model_version_id WHERE v2.status='eligible' GROUP BY f2.symbol)")}
    rows=[]
    for s in sec:
        m=compare_benchmark(s['symbol'],period); f=fundamentals(s['symbol']); model=models.get(s['symbol'])
        pred=None; provenance=None
        if model:
            pred=model['prediction']; provenance='current_forecast:'+model['model_key']+'; origin='+model['origin_date']+'; basis='+model['price_basis']
        rows.append({'symbol':s['symbol'],'name':s['name'],'sector':s['sector'],'metrics':m,'fundamentals':f,'forecast':pred,'forecast_provenance':provenance})
    def norm(values,higher=True):
        vals=[v for v in values if v is not None]
        if not vals: return [None]*len(values)
        lo,hi=min(vals),max(vals)
        return [None if v is None else ((v-lo)/(hi-lo) if hi!=lo else .5) if higher else ((hi-v)/(hi-lo) if hi!=lo else .5) for v in values]
    dimensions={
      'excess_return':norm([r['metrics'].get('simple_excess_return') for r in rows]),
      'forecast':norm([r['forecast'] for r in rows]),
      'low_volatility':norm([r['metrics'].get('annualized_volatility') for r in rows],False),
      'shallow_drawdown':norm([r['metrics'].get('max_drawdown') for r in rows]),
      'fundamentals':norm([((r['fundamentals'].get('profit_growth') or 0)*.5+(r['fundamentals'].get('roe') or 0)*.5-(r['fundamentals'].get('debt_equity') or 0)*2) if r['fundamentals'].get('available') else None for r in rows])}
    for i,r in enumerate(rows):
        terms={k:v[i] for k,v in dimensions.items() if v[i] is not None and weights.get(k,0)>0}; total=sum(weights[k]*v for k,v in terms.items())
        mass=sum(weights[k] for k in terms); r['score']=round(total/mass,4) if mass else None
        r['factor_scores']=terms; r['score_coverage']=round(mass,3); r['rank']=None
    rows.sort(key=lambda r:r['score'] if r['score'] is not None else -1,reverse=True)
    for i,r in enumerate(rows,1): r['rank']=i
    with connect() as c: active=c.execute("SELECT value FROM app_state WHERE key='active_release_id'").fetchone()
    return {'release_id':int(active['value']) if active else None,'period':period,'weights':weights,'results':rows,'method':'Cross-sectional min-max normalization and weighted mean of available factors. Missing model forecasts are omitted, with coverage shown. This is a research ranking, not suitability or expected return.','forecast_policy':'Only registered locally trained current forecasts are shown; historical evaluation uses separate out-of-sample prediction records.'}
