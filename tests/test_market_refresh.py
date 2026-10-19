import pandas as pd
import pytest
from datetime import date

from app import db, market


@pytest.fixture(autouse=True)
def isolated_database(tmp_path, monkeypatch):
    monkeypatch.setattr(db, 'DATA_DIR', tmp_path)
    monkeypatch.setattr(db, 'DB_PATH', tmp_path / 'nivesh.sqlite3')
    db.init_db()


def instrument(symbol='RELIANCE', ticker='RELIANCE.NS'):
    return {'instrument_id':'NSE_EQ_'+symbol,'symbol':symbol,'name':symbol,'sector':'Test','exchange':'NSE','index_member':symbol!='NIFTY50','provider':'yahoo_finance','provider_symbol':ticker}


def frame(dates=('2026-10-14','2026-10-15')):
    return pd.DataFrame({
        'date':pd.to_datetime(list(dates)), 'open':[100.0]*len(dates), 'high':[103.0]*len(dates),
        'low':[98.0]*len(dates), 'close':[101.0]*len(dates), 'adjusted_close':[100.5]*len(dates),
        'volume':[1000]*len(dates), 'dividends':[0.0]*len(dates), 'stock_splits':[0.0]*len(dates),
    })


class FakeProvider:
    name=market.KITE_SOURCE
    quality=market.KITE_QUALITY
    def __init__(self, fail=()): self.calls=[]; self.fail=set(fail)
    def assert_allowed(self): pass
    def history(self,symbol,start,end):
        self.calls.append((symbol,start,end))
        if symbol in self.fail: raise RuntimeError('simulated provider failure')
        return frame()


def test_first_refresh_saves_normalized_and_raw_data(monkeypatch):
    monkeypatch.setattr(market,'load_instruments',lambda:[instrument()])
    provider=FakeProvider()
    result=market.refresh_market_data(provider)
    assert result['status']=='succeeded'
    assert result['rows_written']==2
    assert provider.calls[0][0]=='RELIANCE.NS'
    assert provider.calls[0][1] <= date.today().isoformat()
    with db.connect() as c:
        prices=c.execute('SELECT source,quality,adjusted_close FROM prices').fetchall()
        raw=c.execute('SELECT COUNT(*),MIN(validation_status) FROM raw_price_observations').fetchone()
    assert len(prices)==2
    assert prices[0]['source']==market.KITE_SOURCE
    assert prices[0]['quality']==market.KITE_QUALITY
    assert prices[0]['adjusted_close']==100.5
    assert raw[0]==2 and raw[1]=='STRUCTURALLY_VALID_UNVERIFIED_SOURCE'


def test_incremental_refresh_uses_overlap_and_deduplicates_exact_raw(monkeypatch):
    monkeypatch.setattr(market,'load_instruments',lambda:[instrument()])
    market.refresh_market_data(FakeProvider())
    provider=FakeProvider()
    result=market.refresh_market_data(provider)
    assert result['status']=='succeeded'
    assert provider.calls[0][1] > '2016-10-18'
    with db.connect() as c:
        assert c.execute('SELECT COUNT(*) FROM prices').fetchone()[0]==2
        assert c.execute('SELECT COUNT(*) FROM raw_price_observations').fetchone()[0]==2


def test_provider_failure_preserves_last_good_observations(monkeypatch):
    monkeypatch.setattr(market,'load_instruments',lambda:[instrument()])
    market.refresh_market_data(FakeProvider())
    result=market.refresh_market_data(FakeProvider(fail={'RELIANCE.NS'}))
    assert result['status']=='failed'
    assert result['errors'][0]['symbol']=='RELIANCE'
    with db.connect() as c:
        assert c.execute('SELECT COUNT(*) FROM prices').fetchone()[0]==2


def test_partial_failure_keeps_successful_symbol(monkeypatch):
    monkeypatch.setattr(market,'load_instruments',lambda:[instrument(),instrument('TCS','TCS.NS')])
    result=market.refresh_market_data(FakeProvider(fail={'TCS.NS'}))
    assert result['status']=='partial_success'
    assert result['instruments_updated']==1
    assert result['errors'][0]['symbol']=='TCS'


def test_imported_records_are_not_overwritten_by_provider(monkeypatch):
    monkeypatch.setattr(market,'load_instruments',lambda:[instrument()])
    with db.connect() as c:
        c.execute("INSERT INTO securities(symbol,name,index_member) VALUES('RELIANCE','Reliance',1)")
        c.execute("INSERT INTO prices(symbol,date,close,adjusted_close,source,retrieved_at,quality) VALUES('RELIANCE','2026-10-14',77,76,'user_csv:prices.csv','now','USER_IMPORTED_UNVERIFIED')")
    result=market.refresh_market_data(FakeProvider())
    assert result['status']=='succeeded'
    with db.connect() as c:
        row=c.execute("SELECT close,source,quality FROM prices WHERE date='2026-10-14'").fetchone()
    assert tuple(row)==(77.0,'user_csv:prices.csv','USER_IMPORTED_UNVERIFIED')


def test_invalid_ohlc_is_rejected():
    bad=frame(); bad.loc[0,'high']=97
    with pytest.raises(ValueError,match='OHLC'):
        market.normalize_yahoo_frame(bad,'RELIANCE')


def test_missing_adjusted_close_is_reported():
    bad=frame().drop(columns=['adjusted_close'])
    with pytest.raises(ValueError,match='adjusted_close'):
        market.normalize_yahoo_frame(bad,'RELIANCE')


def test_yahoo_collection_is_disabled_even_if_env_flag_is_set(monkeypatch):
    monkeypatch.setenv('YAHOO_FINANCE_PERMISSION_CONFIRMED','1')
    with pytest.raises(RuntimeError,match='collection is disabled'):
        market.YahooFinanceProvider().history('RELIANCE.NS','2026-10-01','2026-10-17')


def test_kite_collection_is_disabled_without_touching_network(monkeypatch):
    provider=market.KiteConnectProvider()
    with pytest.raises(RuntimeError,match='collection is disabled'):
        provider.history('RELIANCE','2026-10-15','2026-10-15')


def test_kite_adapter_does_not_request_broker_credentials(monkeypatch):
    monkeypatch.delenv('KITE_API_KEY',raising=False)
    monkeypatch.delenv('KITE_ACCESS_TOKEN',raising=False)
    with pytest.raises(RuntimeError,match='collection is disabled'):
        market.KiteConnectProvider().assert_allowed()


def test_status_reports_provenance_and_missing_market_data():
    result=market.get_data_status()
    assert result['latest_market_observation_date'] is None
    assert result['quality']=='NO_MARKET_DATA'
    assert result['instrument_coverage']==[]
