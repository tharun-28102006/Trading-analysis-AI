"""Leakage-aware shared five-session dataset builder for local observations."""
import hashlib
import json
import re

import numpy as np
import pandas as pd

from .db import connect

FEATURES = [
    'ret_1', 'ret_5', 'ret_20', 'ret_60', 'relative_ret_5', 'relative_ret_20',
    'relative_ret_60', 'volatility_20', 'volatility_60', 'ma20_distance',
    'ma50_distance', 'drawdown_60', 'market_ret_20', 'market_volatility_20',
    'volume_change_5', 'average_turnover_20',
]
HORIZON = 5
FUNDAMENTAL_FEATURES=['fundamental_revenue_growth','fundamental_profit_growth','fundamental_debt_equity','fundamental_roe']
EVENT_FEATURES=['event_count_5_sessions','event_count_20_sessions','days_since_known_event']


def _parse_published(values):
    parsed=[]
    for value in values:
        raw=str(value).strip()
        if re.fullmatch(r'\d{4}-\d{2}-\d{2}',raw): raw+='T23:59:59+00:00'
        parsed.append(pd.to_datetime(raw,errors='coerce',utc=True))
    return pd.to_datetime(parsed,utc=True)


def _known_event_features(symbol, sessions, *, published=True):
    """Events enter only after publication and event dates are both observable."""
    with connect() as c:
        events=pd.read_sql_query(f"SELECT date,published_at FROM {'published_events' if published else 'events'} WHERE symbol=? AND quality='USER_CONFIRMED_UNVERIFIED' AND published_at IS NOT NULL",c,params=(symbol.upper(),))
    if events.empty: return None,{'rows':0,'available':False}
    events['date']=pd.to_datetime(events.date,errors='coerce'); events['published_at']=_parse_published(events.published_at)
    events=events.dropna(subset=['date','published_at'])
    if events.empty: return None,{'rows':0,'available':False}
    event_dates=events.date.to_numpy(dtype='datetime64[ns]'); published=events.published_at.to_numpy(dtype='datetime64[ns]')
    values=[]; session_index=pd.DatetimeIndex(sessions)
    for i,origin in enumerate(session_index):
        session_close=(origin.tz_localize('Asia/Kolkata')+pd.Timedelta(hours=15,minutes=30)).tz_convert('UTC')
        known=(event_dates<=origin.to_datetime64())&(published<=session_close.to_datetime64())
        prior=event_dates[known]
        in5=session_index[max(0,i-4)]
        in20=session_index[max(0,i-19)]
        values.append([int(((prior>=in5.to_datetime64())&(prior<=origin.to_datetime64())).sum()),
                       int(((prior>=in20.to_datetime64())&(prior<=origin.to_datetime64())).sum()),
                       float((origin.to_datetime64()-prior.max())/np.timedelta64(1,'D')) if len(prior) else np.nan])
    return pd.DataFrame(values,index=session_index,columns=EVENT_FEATURES),{'rows':len(events),'available':True}


def eligible_price_history(symbol: str, *, published=True) -> pd.DataFrame:
    """Return only explicitly user-confirmed, non-synthetic local observations."""
    with connect() as c:
        frame = pd.read_sql_query(
            "SELECT date,open,high,low,close,adjusted_close,volume,source,quality,price_basis "
            f"FROM {'published_prices' if published else 'prices'} WHERE symbol=? AND quality='USER_CONFIRMED_UNVERIFIED' ORDER BY date",
            c, params=(symbol.upper(),))
    if frame.empty:
        return frame
    frame['date'] = pd.to_datetime(frame.date, errors='coerce')
    return frame.dropna(subset=['date']).drop_duplicates('date').set_index('date').sort_index()


def _price_column(frame: pd.DataFrame) -> tuple[str, str]:
    bases = set(frame.price_basis.dropna().astype(str))
    if len(bases) != 1 or not bases.issubset({'adjusted_close', 'raw_unadjusted'}):
        raise ValueError('Each instrument must have one consistent, explicitly supported price basis.')
    basis = next(iter(bases))
    return ('adjusted_close' if basis == 'adjusted_close' else 'close', basis)


def build_dataset(symbol: str, *, benchmark='NIFTY50', published=True) -> tuple[pd.DataFrame, dict]:
    """Build one shared feature/target definition from common valid sessions.

    Features are computed with prices available at the origin close. Target is
    the next five common valid trading sessions, not five calendar days.
    """
    symbol = symbol.upper()
    stock = eligible_price_history(symbol, published=published)
    market = eligible_price_history(benchmark, published=published)
    if stock.empty or market.empty:
        raise ValueError(f'{symbol}: requires user-confirmed stock and {benchmark} observations.')
    stock_col, stock_basis = _price_column(stock)
    bench_col, bench_basis = _price_column(market)
    if stock_basis != bench_basis:
        raise ValueError('Stock and benchmark price bases must match; raw and adjusted series cannot be mixed.')
    # The imported benchmark dates define market sessions. Preserve missing
    # stock observations on that grid so t+5 always means five benchmark
    # sessions, rather than silently skipping stock gaps via an inner join.
    joined = pd.DataFrame({'benchmark': market[bench_col]}, index=market.index)
    joined['price']=stock[stock_col].reindex(joined.index)
    joined['volume']=stock.volume.reindex(joined.index)
    if joined.index.has_duplicates:
        raise ValueError('Duplicate aligned observation dates are not permitted.')
    ret = joined.price.pct_change(fill_method=None)
    bench_ret = joined.benchmark.pct_change(fill_method=None)
    features = pd.DataFrame(index=joined.index)
    for n in (1, 5, 20, 60):
        features[f'ret_{n}'] = joined.price.pct_change(n,fill_method=None)
    for n in (5, 20, 60):
        features[f'relative_ret_{n}'] = joined.price.pct_change(n,fill_method=None) - joined.benchmark.pct_change(n,fill_method=None)
    features['volatility_20'] = ret.rolling(20, min_periods=20).std()
    features['volatility_60'] = ret.rolling(60, min_periods=60).std()
    features['ma20_distance'] = joined.price / joined.price.rolling(20, min_periods=20).mean() - 1
    features['ma50_distance'] = joined.price / joined.price.rolling(50, min_periods=50).mean() - 1
    rolling_peak = joined.price.rolling(60, min_periods=20).max()
    features['drawdown_60'] = joined.price / rolling_peak - 1
    features['market_ret_20'] = joined.benchmark.pct_change(20,fill_method=None)
    features['market_volatility_20'] = bench_ret.rolling(20, min_periods=20).std()
    vol_change = joined.volume.pct_change(5,fill_method=None).replace([np.inf, -np.inf], np.nan)
    features['volume_change_5'] = vol_change
    features['average_turnover_20'] = (joined.price * joined.volume).rolling(20, min_periods=20).mean()
    with connect() as c:
        fundamentals=pd.read_sql_query(f"SELECT published_at,revenue_growth,profit_growth,debt_equity,roe FROM {'published_fundamental_observations' if published else 'fundamental_observations'} WHERE symbol=? AND quality='USER_CONFIRMED_UNVERIFIED' ORDER BY published_at",c,params=(symbol,))
    fundamental_coverage={'rows':0,'available':False}
    if not fundamentals.empty:
        fundamentals['published_at']=_parse_published(fundamentals.published_at)
        fundamentals=fundamentals.dropna(subset=['published_at']).drop_duplicates('published_at',keep='last')
        origins=pd.DataFrame({'origin_time':(pd.DatetimeIndex(joined.index).tz_localize('Asia/Kolkata')+pd.Timedelta(hours=15,minutes=30)).tz_convert('UTC')}).sort_values('origin_time')
        asof=pd.merge_asof(origins,fundamentals.sort_values('published_at'),left_on='origin_time',right_on='published_at',direction='backward')
        asof.index=origins.index
        for source_col,feature_col in zip(('revenue_growth','profit_growth','debt_equity','roe'),FUNDAMENTAL_FEATURES):
            features[feature_col]=asof[source_col].to_numpy()
        fundamental_coverage={'rows':len(fundamentals),'available':True,'point_in_time_rows':int(asof.published_at.notna().sum())}
    event_features,event_coverage=_known_event_features(symbol,joined.index,published=published)
    if event_features is not None: features=features.join(event_features)

    target_stock = joined.price.shift(-HORIZON) / joined.price - 1
    target_bench = joined.benchmark.shift(-HORIZON) / joined.benchmark - 1
    out = features.copy()
    out['stock_return'] = target_stock
    out['benchmark_return'] = target_bench
    out['target'] = target_stock - target_bench
    out['target_end_date'] = pd.Series(joined.index, index=joined.index).shift(-HORIZON)
    # Volume-derived columns are omitted from the fixed schema when the source
    # does not supply usable volume; never impute an invented liquidity signal.
    columns = list(FEATURES)
    if not fundamentals.empty: columns.extend(FUNDAMENTAL_FEATURES)
    if event_features is not None: columns.extend(EVENT_FEATURES)
    if joined.volume.notna().sum() < max(61, len(joined) // 2):
        columns.remove('volume_change_5')
        columns.remove('average_turnover_20')
    valid = out[columns + ['stock_return', 'benchmark_return', 'target', 'target_end_date']].replace([np.inf, -np.inf], np.nan)
    before = len(valid)
    valid = valid.dropna(subset=columns + ['target', 'target_end_date'])
    valid['target_end_date'] = pd.to_datetime(valid.target_end_date).dt.strftime('%Y-%m-%d')
    valid.index.name = 'origin_date'
    valid['origin_date'] = valid.index.strftime('%Y-%m-%d')
    with connect() as c:
        rows = []
        fundamental_versions=c.execute(f"SELECT published_at,source,revenue_growth,profit_growth,debt_equity,roe FROM {'published_fundamental_observations' if published else 'fundamental_observations'} WHERE symbol=? AND quality='USER_CONFIRMED_UNVERIFIED' ORDER BY published_at",(symbol,)).fetchall()
        event_versions=c.execute(f"SELECT date,published_at,kind,title,source FROM {'published_events' if published else 'events'} WHERE symbol=? AND quality='USER_CONFIRMED_UNVERIFIED' AND published_at IS NOT NULL ORDER BY date,published_at,source,title",(symbol,)).fetchall()
        price_basis_versions=c.execute(f"SELECT symbol,price_basis,quality,COUNT(*) FROM {'published_prices' if published else 'prices'} WHERE symbol IN (?,?) GROUP BY symbol,price_basis,quality ORDER BY symbol,price_basis,quality",(symbol,benchmark)).fetchall()
    version_bytes = ('|'.join(f'{r[0]}:{r[1]}:{r[2]}:{r[3]}' for r in rows)+'#'+'|'.join(json.dumps(tuple(r),default=str) for r in fundamental_versions)+'#'+'|'.join(json.dumps(tuple(r),default=str) for r in event_versions)+'#'+'|'.join(json.dumps(tuple(r),default=str) for r in price_basis_versions)).encode()
    data_version = hashlib.sha256(version_bytes).hexdigest()
    with connect() as c:
        active=c.execute("SELECT value FROM app_state WHERE key='active_release_id'").fetchone()
    meta = {'symbol': symbol, 'benchmark': benchmark, 'release_id':int(active['value']) if active and published else None, 'price_basis': stock_basis,
            'target_name': 'five_session_raw_price_excess_return' if stock_basis == 'raw_unadjusted' else 'five_session_adjusted_price_excess_return',
            'target_horizon_sessions': HORIZON, 'aligned_sessions': int(joined.price.notna().sum()),
            'benchmark_sessions':len(joined),'missing_stock_benchmark_sessions':int(joined.price.isna().sum()),
            'rows': len(valid), 'rows_excluded_insufficient_history_or_missing_features': before-len(valid),
            'feature_columns': columns, 'data_version': data_version,
            'feature_coverage':{'fundamentals':fundamental_coverage,'events':event_coverage},
            'provenance': 'user_confirmed_unverified', 'independently_verified': False}
    return valid, meta


def build_panel(symbols: list[str] | None = None, *, benchmark='NIFTY50') -> tuple[pd.DataFrame, dict]:
    """Shared pooled dataset; all symbols call the exact same builder."""
    if symbols is None:
        with connect() as c:
            symbols = [r[0] for r in c.execute("SELECT DISTINCT p.symbol FROM published_prices p JOIN published_securities s USING(symbol) WHERE p.quality='USER_CONFIRMED_UNVERIFIED' AND p.symbol<>? ORDER BY p.symbol", (benchmark,))]
    pieces, excluded, versions, schemas, bases = [], {}, set(), [], set()
    for symbol in symbols:
        try:
            frame, meta = build_dataset(symbol, benchmark=benchmark, published=True)
            with connect() as c:
                membership=c.execute("SELECT valid_from,valid_to FROM published_constituent_membership WHERE symbol=? AND quality='USER_CONFIRMED_UNVERIFIED' ORDER BY valid_from",(symbol,)).fetchall()
            if not membership:
                excluded[symbol] = 'Historical index-membership intervals are unavailable; current constituents are not projected backward.'
                continue
            member_mask=pd.Series(False,index=frame.index)
            for m in membership:
                interval=frame.origin_date>=m['valid_from']
                if m['valid_to'] is not None: interval &= frame.origin_date<=m['valid_to']
                member_mask |= interval
            frame=frame[member_mask]
            if len(frame) < 750:
                excluded[symbol] = f'Only {len(frame)} eligible rows; 750 required.'
                continue
            frame['symbol'] = symbol
            pieces.append(frame)
            versions.add(meta['data_version'])
            schemas.append(set(meta['feature_columns']))
            bases.add(meta['price_basis'])
        except ValueError as exc:
            excluded[symbol] = str(exc)
    if not pieces:
        return pd.DataFrame(), {'symbols_included': [], 'excluded': excluded, 'rows': 0}
    common_features=sorted(set.intersection(*schemas))
    if len(bases)>1:
        return pd.DataFrame(), {'symbols_included': [], 'excluded': {**excluded,'_universe':'Mixed raw and adjusted price bases cannot be pooled.'}, 'rows': 0}
    metadata_cols={'origin_date','target_end_date','stock_return','benchmark_return','target','symbol'}
    panel = pd.concat(pieces, ignore_index=True).sort_values(['origin_date', 'symbol'])
    panel=panel[common_features+sorted(metadata_cols)]
    return panel, {'symbols_included': sorted(panel.symbol.unique()), 'excluded': excluded,
                   'rows': len(panel), 'data_versions': sorted(versions),'feature_columns':common_features,'price_basis':next(iter(bases)),
                   'survivorship_note': 'Pooled index-universe modeling requires historical constituent membership. Without it, eligibility is withheld rather than assuming current membership historically.'}


def training_eligibility(symbol: str) -> dict:
    if symbol.upper()=='NIFTY50': return {'eligible':False,'rows':0,'reason':'NIFTY50 is the benchmark, not an eligible stock target.'}
    with connect() as c:
        rows = c.execute("SELECT quality,price_basis,COUNT(*) n,MIN(date) first_date,MAX(date) last_date FROM published_prices WHERE symbol=? GROUP BY quality,price_basis", (symbol.upper(),)).fetchall()
    eligible_rows = [r for r in rows if r['quality'] == 'USER_CONFIRMED_UNVERIFIED']
    if not eligible_rows:
        return {'eligible': False, 'rows': 0, 'reason': 'No user-confirmed, non-synthetic local observations.'}
    count = sum(r['n'] for r in eligible_rows)
    if count < 750:
        return {'eligible': False, 'rows': count, 'reason': f'{count} rows available; at least 750 aligned observations and benchmark coverage are required.'}
    try:
        dataset, meta = build_dataset(symbol)
    except ValueError as exc:
        return {'eligible': False, 'rows': count, 'reason': str(exc)}
    if len(dataset) < 750:
        return {'eligible': False, 'rows': len(dataset), 'reason': f'{len(dataset)} aligned, complete rows available; at least 750 required.'}
    return {'eligible': True, 'rows': len(dataset), 'reason': 'User-confirmed provenance, common benchmark sessions, consistent price basis, and 750+ rows. The source/license attestation is not independently verified.', 'price_basis': meta['price_basis']}
