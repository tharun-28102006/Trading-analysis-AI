from datetime import datetime, timezone
from .analytics import compare_benchmark
from .db import connect

def add_watch(symbol):
    symbol=symbol.upper()
    with connect() as c:
        if not c.execute('SELECT 1 FROM published_securities WHERE symbol=?',(symbol,)).fetchone(): raise ValueError('Unknown symbol. Import price data first.')
        c.execute('INSERT OR IGNORE INTO watchlist(symbol,added_at) VALUES(?,?)',(symbol,datetime.now(timezone.utc).isoformat()))
    return {'symbol':symbol,'added':True}

def get_watchlist():
    with connect() as c: return [dict(r) for r in c.execute('SELECT w.symbol,s.name,s.sector,w.added_at FROM watchlist w JOIN published_securities s USING(symbol) ORDER BY w.added_at DESC')]

def delete_watch(symbol):
    with connect() as c: c.execute('DELETE FROM watchlist WHERE symbol=?',(symbol.upper(),))
    return {'deleted':symbol.upper()}

def add_alert(body):
    symbol=body['symbol'].upper(); condition=body['condition']; period=body.get('period','1m')
    if condition not in {'return_above','excess_above','volatility_below'}: raise ValueError('Unsupported alert condition.')
    with connect() as c:
        if not c.execute('SELECT 1 FROM published_securities WHERE symbol=?',(symbol,)).fetchone(): raise ValueError('Unknown symbol.')
        cur=c.execute('INSERT INTO alerts(name,symbol,condition,threshold,period,created_at) VALUES(?,?,?,?,?,?)',(body.get('name') or f'{symbol} alert',symbol,condition,float(body['threshold']),period,datetime.now(timezone.utc).isoformat()))
        return dict(c.execute('SELECT * FROM alerts WHERE id=?',(cur.lastrowid,)).fetchone())

def list_alerts():
    with connect() as c: return [dict(r) for r in c.execute('SELECT * FROM alerts ORDER BY id DESC')]

def delete_alert(alert_id):
    with connect() as c: c.execute('DELETE FROM alerts WHERE id=?',(alert_id,)); c.execute('DELETE FROM alert_history WHERE alert_id=?',(alert_id,))
    return {'deleted':alert_id}

def evaluate_alerts():
    fired=[]
    for a in list_alerts():
        if not a['enabled']: continue
        m=compare_benchmark(a['symbol'],a['period'])
        key={'return_above':'return','excess_above':'simple_excess_return','volatility_below':'annualized_volatility'}[a['condition']]; value=m.get(key)
        passed=value is not None and (value<a['threshold'] if a['condition']=='volatility_below' else value>=a['threshold'])
        if passed:
            day=datetime.now(timezone.utc).date().isoformat(); msg=f"{a['symbol']} {key} is {value:.2%} for {a['period']} (threshold {a['threshold']:.2%})"
            with connect() as c: c.execute('INSERT OR IGNORE INTO alert_history(alert_id,triggered_at,message,value) VALUES(?,?,?,?)',(a['id'],day,msg,value))
            fired.append({'alert_id':a['id'],'symbol':a['symbol'],'message':msg,'value':value,'quality':m.get('quality')})
    return {'evaluated':len(list_alerts()),'triggered':fired,'note':'Evaluation is on-demand. Scheduled delivery requires an external scheduler and notification provider.'}

def alert_history():
    with connect() as c: return [dict(r) for r in c.execute('SELECT h.*,a.symbol FROM alert_history h JOIN alerts a ON a.id=h.alert_id ORDER BY h.id DESC LIMIT 100')]
