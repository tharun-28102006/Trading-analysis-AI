from app import backtest, db


def prepare(tmp_path, monkeypatch):
    monkeypatch.setattr(db, 'DATA_DIR', tmp_path)
    monkeypatch.setattr(db, 'DB_PATH', tmp_path / 'backtest.sqlite3')
    db.init_db()


def seed_one_cohort(basis='raw_unadjusted'):
    dates=['2026-01-05','2026-01-06','2026-01-07','2026-01-08','2026-01-09','2026-01-12','2026-01-13']
    with db.connect() as c:
        c.executemany("INSERT INTO securities(symbol,name,exchange) VALUES(?,?,'NSE')", [('AAA','AAA'),('NIFTY50','NIFTY 50')])
        for symbol,base in [('AAA',100.0),('NIFTY50',200.0)]:
            for i,date in enumerate(dates):
                open_=base+i
                close=base+i+0.5
                c.execute("INSERT INTO prices(symbol,date,open,high,low,close,adjusted_close,volume,source,retrieved_at,quality,price_basis) VALUES(?,?,?,?,?,?,?,1000,'test','now','USER_CONFIRMED_UNVERIFIED',?)",
                          (symbol,date,open_,close+0.5,open_-0.5,close,close,basis))
        run=c.execute("INSERT INTO training_runs(mode,status,started_at) VALUES('test','succeeded','now')").lastrowid
        c.execute("INSERT INTO oos_predictions(training_run_id,model_key,symbol,origin_date,target_end_date,prediction,actual_excess,stock_return,benchmark_return,data_version,price_basis) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                  (run,'pooled:ridge','AAA',dates[0],dates[5],.02,.03,.1,.07,'fixture','raw_unadjusted'))


def test_backtest_uses_next_open_and_reports_cost_sensitivity(tmp_path, monkeypatch):
    prepare(tmp_path, monkeypatch)
    seed_one_cohort()
    result=backtest.run_backtest(model_key='pooled:ridge',top_n=1,max_weight=1,transaction_cost_bps=5,slippage_bps=5)
    assert result['available'] is True
    metrics=result['results']
    assert metrics['trades'][0]['entry_date']=='2026-01-06'
    assert metrics['trades'][0]['exit_date']=='2026-01-12'
    assert metrics['trade_count']==1
    assert len(metrics['cost_sensitivity'])==5
    assert metrics['cost_sensitivity'][0]['cumulative_return']>metrics['cost_sensitivity'][-1]['cumulative_return']


def test_adjusted_close_only_data_cannot_be_used_as_execution_open(tmp_path, monkeypatch):
    prepare(tmp_path, monkeypatch)
    seed_one_cohort('adjusted_close')
    result=backtest.run_backtest(model_key='pooled:ridge',top_n=1,max_weight=1)
    assert result['available'] is False
    assert 'raw OHLC' in result['reason']
