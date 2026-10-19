"""Synthetic fixtures here test math only; they are never written as market data."""
import numpy as np
import pandas as pd

from app import db, features
from app.models import _fit_ridge, _fit_rbf, _predict, _walk_forward


def test_target_uses_five_next_common_sessions_and_features_stop_at_origin(monkeypatch,tmp_path):
    monkeypatch.setattr(db,'DATA_DIR',tmp_path); monkeypatch.setattr(db,'DB_PATH',tmp_path/'features.sqlite3'); db.init_db()
    dates=pd.bdate_range('2020-01-01',periods=130)
    stock_values=100*np.exp(np.arange(len(dates))*.002+np.sin(np.arange(len(dates))*.11)*.02)
    bench_values=100*np.exp(np.arange(len(dates))*.001+np.sin(np.arange(len(dates))*.08)*.01)
    def frame(values):
        return pd.DataFrame({'close':values,'adjusted_close':values,'volume':np.full(len(values),1000),
            'price_basis':'raw_unadjusted','quality':'SYNTHETIC_TEST_FIXTURE','source':'unit_test'},index=dates)
    stock=frame(stock_values); bench=frame(bench_values)
    monkeypatch.setattr(features,'eligible_price_history',lambda symbol, **kwargs: stock.copy() if symbol=='ABC' else bench.copy())
    baseline,meta=features.build_dataset('ABC')
    origin=dates[80]
    row=baseline.loc[baseline.origin_date==origin.strftime('%Y-%m-%d')].iloc[0]
    expected=(stock_values[85]/stock_values[80]-1)-(bench_values[85]/bench_values[80]-1)
    assert abs(row.target-expected)<1e-12
    assert row.target_end_date==dates[85].strftime('%Y-%m-%d')
    changed=stock.copy(); changed.loc[dates[85],'close']*=1.5; changed.loc[dates[85],'adjusted_close']*=1.5
    old=stock; stock=changed
    future_changed,_=features.build_dataset('ABC')
    row2=future_changed.loc[future_changed.origin_date==origin.strftime('%Y-%m-%d')].iloc[0]
    assert row2.ret_5==row.ret_5
    assert row2.target!=row.target
    assert meta['target_horizon_sessions']==5
    assert meta['provenance']=='user_confirmed_unverified'


def test_local_models_and_walk_forward_are_deterministic_and_chronological():
    rng=np.random.default_rng(731)
    x=rng.normal(size=(180,4)); y=.3*x[:,0]-.15*x[:,1]+rng.normal(scale=.01,size=len(x))
    ridge=_fit_ridge(x[:140],y[:140]); rbf=_fit_rbf(x[:140],y[:140]);
    assert np.allclose(_predict(ridge,x[140:]),_predict(ridge,x[140:]))
    assert np.isfinite(_predict(rbf,x[140:])).all()
    dates=pd.bdate_range('2018-01-01',periods=765)
    panel=pd.DataFrame(x[:160].tolist()*5,columns=['a','b','c','d'])
    # Deterministic test-only panel for fold scheduling. No market facts or
    # persisted observations are produced.
    n=760
    panel=pd.DataFrame({'origin_date':[d.strftime('%Y-%m-%d') for d in dates[:n]],
        'target_end_date':[dates[i+5].strftime('%Y-%m-%d') for i in range(n)],
        'target':np.sin(np.arange(n)*.03),'stock_return':np.sin(np.arange(n)*.03),
        'benchmark_return':np.zeros(n),**{f'r{i}':np.cos(np.arange(n)*.01*(i+1)) for i in range(4)}})
    oos,metrics=_walk_forward(panel,['r0','r1','r2','r3'])
    assert oos and set(metrics)=={'zero_excess','historical_mean','ridge','rbf_ridge'}
    assert all(p['origin_date']<p['target_end_date'] for p in oos)
    first_test=min(p['origin_date'] for p in oos)
    assert first_test>dates[300].strftime('%Y-%m-%d')


def test_constant_baselines_reload_through_same_prediction_path():
    values=_predict({'kind':'constant','value':0.0},np.zeros((3,4)))
    mean=_predict({'kind':'constant','value':0.125},np.ones((2,4)))
    assert np.array_equal(values,np.zeros(3))
    assert np.array_equal(mean,np.full(2,.125))


def test_five_session_target_keeps_benchmark_grid_when_stock_session_is_missing(monkeypatch,tmp_path):
    monkeypatch.setattr(db,'DATA_DIR',tmp_path); monkeypatch.setattr(db,'DB_PATH',tmp_path/'benchmark-grid.sqlite3'); db.init_db()
    dates=pd.bdate_range('2021-01-01',periods=150)
    def mk(values): return pd.DataFrame({'close':values,'adjusted_close':values,'volume':1000,'price_basis':'raw_unadjusted','quality':'USER_CONFIRMED_UNVERIFIED','source':'test'},index=dates)
    stock=mk(100+np.arange(150)); bench=mk(200+np.arange(150))
    stock=stock.drop(dates[82])
    monkeypatch.setattr(features,'eligible_price_history',lambda symbol, **kwargs:stock.copy() if symbol=='ABC' else bench.copy())
    data,meta=features.build_dataset('ABC')
    row=data.loc[data.origin_date==dates[80].strftime('%Y-%m-%d')].iloc[0]
    assert row.target_end_date==dates[85].strftime('%Y-%m-%d')
    expected=(stock.loc[dates[85],'close']/stock.loc[dates[80],'close']-1)-(bench.loc[dates[85],'close']/bench.loc[dates[80],'close']-1)
    assert abs(row.target-expected)<1e-12
    assert meta['missing_stock_benchmark_sessions']==1


def test_event_feature_respects_publication_cutoff(monkeypatch,tmp_path):
    from app.features import _known_event_features
    monkeypatch.setattr(db,'DATA_DIR',tmp_path); monkeypatch.setattr(db,'DB_PATH',tmp_path/'events.sqlite3'); db.init_db()
    with db.connect() as c:
        c.executemany('INSERT INTO events(symbol,date,kind,title,source,quality,published_at) VALUES(?,?,?,?,?,?,?)',[
            ('ABC','2026-01-03','corporate','known','test','USER_CONFIRMED_UNVERIFIED','2026-01-02'),
            ('ABC','2026-01-02','corporate','same-day date-only','test','USER_CONFIRMED_UNVERIFIED','2026-01-02'),
            ('ABC','2026-01-01','corporate','announced later','test','USER_CONFIRMED_UNVERIFIED','2026-01-05'),
            ('ABC','2026-01-02','corporate','synthetic','test','DEMO_SYNTHETIC','2026-01-01')])
    dates=pd.date_range('2026-01-01',periods=7,freq='D')
    frame,coverage=_known_event_features('ABC',dates)
    assert frame.loc['2026-01-02','event_count_20_sessions']==0
    assert frame.loc['2026-01-04','event_count_20_sessions']==2
    assert frame.loc['2026-01-04','days_since_known_event']==1
    assert coverage['rows']==3
