"""Non-overlapping, next-session-open research backtests from OOS predictions."""
import json
from datetime import datetime, timezone

import numpy as np

from .db import connect


def run_backtest(*, model_key='pooled:ridge', top_n=5, bottom_n=0, max_weight=.20, transaction_cost_bps=5.0, slippage_bps=5.0):
    if top_n<1 or bottom_n<0 or max_weight<=0 or top_n*max_weight<.5:
        raise ValueError('Invalid portfolio sizing. top_n must be positive and the per-name cap must permit at least 50% long exposure.')
    if transaction_cost_bps<0 or slippage_bps<0: raise ValueError('Cost and slippage assumptions cannot be negative.')
    with connect() as c:
        predictions=[dict(r) for r in c.execute('SELECT symbol,origin_date,target_end_date,prediction FROM oos_predictions WHERE model_key=? ORDER BY origin_date,symbol',(model_key,))]
        if not predictions and model_key=='pooled:ridge':
            latest=c.execute("SELECT model_key FROM oos_predictions WHERE model_key LIKE 'pooled:%' GROUP BY model_key ORDER BY MAX(id) DESC LIMIT 1").fetchone()
            if latest:
                model_key=latest['model_key']
                predictions=[dict(r) for r in c.execute('SELECT symbol,origin_date,target_end_date,prediction FROM oos_predictions WHERE model_key=? ORDER BY origin_date,symbol',(model_key,))]
        prices={(r['symbol'],r['date']):dict(r) for r in c.execute("SELECT symbol,date,open,close,quality,price_basis FROM published_prices WHERE quality='USER_CONFIRMED_UNVERIFIED' ORDER BY symbol,date")}
        dates_by_symbol={}
        for (symbol,date),row in prices.items(): dates_by_symbol.setdefault(symbol,[]).append(date)
    if not predictions:
        return {'available':False,'status':'unavailable','reason':'No historical out-of-sample predictions exist for the selected model. No equity curve was created.'}
    # An execution result requires raw OHLC so the signal uses only information
    # through the close and enters at the next session open. Adjusted close alone
    # cannot support a defensible intraday execution price.
    for row in prices.values():
        if row['price_basis']!='raw_unadjusted':
            return {'available':False,'status':'unavailable','reason':'This execution policy requires raw OHLC prices. Adjusted-close-only files do not provide an adjusted next-session open.'}
    grouped={}
    for p in predictions: grouped.setdefault(p['origin_date'],[]).append(p)
    origins=sorted(grouped); selected_origins=[]; last_pos=-10
    # Use NIFTY valid sessions to keep five-session cohorts non-overlapping.
    benchmark_dates=dates_by_symbol.get('NIFTY50',[])
    date_pos={d:i for i,d in enumerate(benchmark_dates)}
    for origin in origins:
        pos=date_pos.get(origin)
        if pos is not None and pos+5<len(benchmark_dates) and pos-last_pos>=5:
            selected_origins.append(origin); last_pos=pos
    curves=[]; trades=[]; previous_weights={}
    for origin in selected_origins:
        candidates=sorted(grouped[origin],key=lambda r:r['prediction'],reverse=True)
        longs=candidates[:top_n]
        shorts=list(reversed(candidates[-bottom_n:])) if bottom_n else []
        if len(longs)<top_n or (bottom_n and len(shorts)<bottom_n): continue
        next_date=benchmark_dates[date_pos[origin]+1]
        exit_date=benchmark_dates[date_pos[origin]+5]
        exposure=.5 if bottom_n else 1.0
        weights={r['symbol']:exposure/len(longs) for r in longs}
        if bottom_n: weights.update({r['symbol']:-.5/len(shorts) for r in shorts})
        if max(abs(w) for w in weights.values())>max_weight+1e-12: continue
        stock_returns=[]; benchmark_row=prices.get(('NIFTY50',next_date)); benchmark_exit=prices.get(('NIFTY50',exit_date))
        if not benchmark_row or not benchmark_exit or not benchmark_row['open'] or not benchmark_exit['close']: continue
        benchmark_return=benchmark_exit['close']/benchmark_row['open']-1
        valid=True
        for symbol,weight in weights.items():
            entry=prices.get((symbol,next_date)); exit_=prices.get((symbol,exit_date))
            if not entry or not exit_ or not entry['open'] or not exit_['close']:
                valid=False; break
            stock_returns.append((weight,exit_['close']/entry['open']-1))
        if not valid: continue
        gross=sum(w*r for w,r in stock_returns)
        # Every cohort is fully closed at its exit, so charge both its entry
        # and exit turnover even when the next cohort holds the same names.
        turnover=2*sum(abs(w) for w in weights.values())
        cost=turnover*(transaction_cost_bps+slippage_bps)/10000
        net=gross-cost
        curves.append({'origin_date':origin,'entry_date':next_date,'exit_date':exit_date,'strategy_return':net,'gross_return':gross,'benchmark_return':benchmark_return,'turnover':turnover,'cost_return':cost,'holdings':weights})
        trades.append({'origin_date':origin,'entry_date':next_date,'exit_date':exit_date,'longs':[r['symbol'] for r in longs],'shorts':[r['symbol'] for r in shorts],'gross_return':gross,'net_return':net,'benchmark_return':benchmark_return,'turnover':turnover,'cost_return':cost})
        previous_weights=weights
    if not curves:
        return {'available':False,'status':'unavailable','reason':'No complete raw-OHLC five-session cohorts satisfy the execution, coverage, and position-limit rules.'}
    strategy=np.array([r['strategy_return'] for r in curves]); bench=np.array([r['benchmark_return'] for r in curves])
    equity=np.cumprod(1+strategy); peak=np.maximum.accumulate(equity); dd=equity/peak-1
    vol=float(np.std(strategy,ddof=1)*np.sqrt(252/5)) if len(strategy)>1 else None
    mean=float(strategy.mean()); risk_adjusted=float(mean/np.std(strategy,ddof=1)*np.sqrt(252/5)) if len(strategy)>1 and np.std(strategy,ddof=1)>0 else None
    cumulative=float(equity[-1]-1); elapsed_sessions=5*len(curves)
    annualized_return=float((1+cumulative)**(252/elapsed_sessions)-1) if cumulative>-1 else -1.0
    wins=strategy[strategy>0]; losses=strategy[strategy<0]
    turnover=float(np.mean([r['turnover'] for r in curves]))
    sensitivity=[]
    for total_bps in (0,5,10,20,50):
        adjusted=np.array([r['gross_return']-r['turnover']*total_bps/10000 for r in curves])
        sensitivity.append({'combined_cost_and_slippage_bps':total_bps,'cumulative_return':float(np.prod(1+adjusted)-1)})
    results={'periods':len(curves),'trade_count':sum(len(t['longs'])+len(t['shorts']) for t in trades),'start_date':curves[0]['entry_date'],'end_date':curves[-1]['exit_date'],'equity_curve':equity.tolist(),'strategy_returns':strategy.tolist(),'benchmark_returns':bench.tolist(),'cumulative_return':cumulative,'annualized_return':annualized_return,'benchmark_cumulative_return':float(np.prod(1+bench)-1),'excess_cumulative_return':cumulative-float(np.prod(1+bench)-1),'maximum_drawdown':float(dd.min()),'annualized_volatility':vol,'risk_adjusted_ratio':risk_adjusted,'win_rate':float(np.mean(strategy>0)),'average_gain':float(wins.mean()) if len(wins) else None,'average_loss':float(losses.mean()) if len(losses) else None,'average_period_return':mean,'average_turnover':turnover,'exposure':float(np.mean([sum(abs(w) for w in c['holdings'].values()) for c in curves])),'total_cost_return':float(sum(r['cost_return'] for r in curves)),'transaction_cost_bps':transaction_cost_bps,'slippage_bps':slippage_bps,'cost_sensitivity':sensitivity,'trades':trades,'execution_policy':'Signal after origin close; enter next common session open; exit at fifth subsequent session close; non-overlapping five-session cohorts; raw OHLC only. Annualization assumes 252 sessions per year; cost sensitivity applies the listed combined round-trip bps to each cohort.'}
    with connect() as c:
        c.execute('INSERT INTO backtest_runs(created_at,status,config_json,results_json) VALUES(?,?,?,?)',(datetime.now(timezone.utc).isoformat(),'completed',json.dumps({'model_key':model_key,'top_n':top_n,'bottom_n':bottom_n,'max_weight':max_weight,'transaction_cost_bps':transaction_cost_bps,'slippage_bps':slippage_bps}),json.dumps(results)))
    return {'available':True,'status':'completed','simulation_only':True,'results':results,'disclaimer':'Historical simulation using out-of-sample model predictions. Not actual trading performance or a guarantee of future results.'}


def backtest_history():
    with connect() as c:
        return [dict(r) for r in c.execute('SELECT id,created_at,status,config_json,results_json,errors_json FROM backtest_runs ORDER BY id DESC LIMIT 25')]
