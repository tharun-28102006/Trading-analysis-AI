"""Small allow-listed parser: natural language becomes a typed screen request."""
import re

def parse_command(text:str) -> dict:
    q=text.strip().lower()
    if not q: return {'status':'clarification_required','questions':['What market analysis should I run?']}
    symbol_match=re.search(r'\b(?:compare|analyze|analyse|risk|forecast|prediction|predict)\s+(?:for\s+)?([a-z][a-z0-9&.-]{0,19})\b',q)
    if symbol_match:
        symbol=symbol_match.group(1).upper()
        if symbol in {'THE','WITH','STOCK','STOCKS','NIFTY','NIFTY50','MARKET'}:
            return {'status':'clarification_required','questions':['Name an NSE symbol, such as RELIANCE, for a local-data query.']}
        operation='risk' if re.search(r'\brisk\b',q) else 'forecast' if re.search(r'\b(forecast|prediction|predict)\b',q) else 'compare'
        return {'status':'ready','operation':operation,'request':{'symbol':symbol,'period':'3m'},'interpretation':{'symbol':symbol,'benchmark':'NIFTY50','period':'3m'},'notes':['Numerical answers use only local database records; missing or ineligible data is reported as unavailable.']}
    if 'backtest' in q or 'back test' in q:
        return {'status':'ready','operation':'backtest','request':{},'interpretation':{'source':'stored out-of-sample predictions'},'notes':['Backtests are historical simulations and require eligible OOS predictions.']}
    if 'model' in q and any(word in q for word in ('evaluation','performance','validation','compare')):
        return {'status':'ready','operation':'model_evaluation','request':{},'interpretation':{'source':'registered model validation metrics'},'notes':['Reported metrics are chronological holdout measurements, not probabilities or profitability claims.']}
    pct=re.search(r'(?:at least|more than|over|above|greater than)\s+(\d+(?:\.\d+)?)\s*(?:%|percent)',q)
    min_return=float(pct.group(1))/100 if pct else None
    period='1m'
    for key,pat in [('5d',r'five[- ]day|5[- ]day|5 trading day'),('1d',r'one[- ]day|today|daily'),('3m',r'three[- ]month|quarter'),('6m',r'six[- ]month'),('1y',r'one[- ]year|12[- ]month|yearly'),('prev_month',r'previous completed calendar month|previous month|last completed month|last month'),('1m',r'monthly|one[- ]month')]:
        if re.search(pat,q): period=key; break
    min_excess=0.0 if ('outperform' in q or 'beat the nifty' in q or 'outperformed' in q) else None
    if min_return is None and min_excess is None and not any(w in q for w in ['screen','find','identify','companies','stocks','shares','market']):
        return {'status':'clarification_required','questions':['I can run a historical stock screen. What return threshold or comparison should I apply?']}
    if 'nifty' not in q and 'nse' not in q and 'indian' not in q:
        return {'status':'clarification_required','questions':['Should I use the NIFTY 50 universe? This version only supports that index universe or the imported local universe.']}
    vol=re.search(r'(?:volatility|vol)\s*(?:below|under|less than|at most)\s*(\d+(?:\.\d+)?)\s*(?:%|percent)',q)
    return {'status':'ready','operation':'screen','request':{'period':period,'min_return':min_return,'min_excess':min_excess,'max_volatility':float(vol.group(1))/100 if vol else None,'universe':'nifty50'},'interpretation':{'benchmark':'NIFTY50','return_definition':'simple adjusted-price return','period':period},'notes':['The parser only creates an allow-listed screen request; numerical results are calculated by deterministic application code.']}

def execute_command(text:str):
    parsed=parse_command(text)
    if parsed['status']!='ready': return parsed
    operation=parsed['operation']; request=parsed['request']
    if operation=='screen':
        from .analytics import screen
        result=screen(**request)
    elif operation=='compare':
        from .analytics import compare_benchmark
        result=compare_benchmark(request['symbol'],request['period'])
    elif operation=='risk':
        from .analytics import metrics
        result=metrics(request['symbol'],request['period'])
    elif operation=='forecast':
        from .db import connect
        with connect() as c:
            row=c.execute("SELECT f.*,v.model_key,v.model_type,v.metrics_json FROM current_forecasts f JOIN model_versions v ON v.id=f.model_version_id WHERE f.symbol=? AND v.status='eligible' ORDER BY f.id DESC LIMIT 1",(request['symbol'],)).fetchone()
        result={'available':False,'symbol':request['symbol'],'reason':'No eligible current local forecast is available.'} if not row else {'available':True,**dict(row)}
    elif operation=='model_evaluation':
        import json
        from .db import connect
        with connect() as c:
            rows=c.execute('SELECT id,model_key,model_type,metrics_json,training_period_json,status,created_at FROM model_versions ORDER BY id DESC').fetchall()
        result={'models':[{'id':r['id'],'model_key':r['model_key'],'model_type':r['model_type'],'metrics':json.loads(r['metrics_json']),'training_period':json.loads(r['training_period_json']),'status':r['status'],'created_at':r['created_at']} for r in rows]}
    elif operation=='backtest':
        from .backtest import backtest_history
        result={'runs':backtest_history()}
    else: return {'status':'unsupported','message':'This request is not in the supported local analysis set.'}
    parsed['result']=result
    parsed['status']='complete'
    return parsed
