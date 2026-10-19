from app import analytics, db, market, ranking


def prepare(tmp_path, monkeypatch):
    monkeypatch.setattr(db, 'DATA_DIR', tmp_path)
    monkeypatch.setattr(db, 'DB_PATH', tmp_path / 'universe.sqlite3')
    db.init_db()


def test_broad_screen_and_ranking_use_only_latest_imported_instrument_snapshot(tmp_path, monkeypatch):
    prepare(tmp_path, monkeypatch)
    with db.connect() as c:
        c.executemany("INSERT INTO securities(symbol,name,sector,exchange) VALUES(?,?,?,'NSE')", [
            ('OLD_DEMO', 'Old demo', 'Demo'), ('IMPORTED', 'Imported', 'Finance')])
        c.execute("INSERT INTO instrument_universe(symbol,as_of_date,name,sector,exchange,quality) VALUES('IMPORTED','2026-10-18','Imported','Finance','NSE','USER_IMPORTED_UNVERIFIED')")
    monkeypatch.setattr(analytics, 'compare_benchmark', lambda symbol, period: {
        'symbol': symbol, 'available': True, 'return': .01, 'simple_excess_return': .002,
        'annualized_volatility': .2, 'max_drawdown': -.1, 'quality': 'USER_IMPORTED_UNVERIFIED'})
    monkeypatch.setattr(ranking, 'compare_benchmark', analytics.compare_benchmark)
    result = analytics.screen(universe='all')
    ranked = ranking.rank(universe='all')
    assert result['evaluated'] == 1 and [r['symbol'] for r in result['results']] == ['IMPORTED']
    assert [r['symbol'] for r in ranked['results']] == ['IMPORTED']


def test_broad_screen_is_empty_without_imported_inventory(tmp_path, monkeypatch):
    prepare(tmp_path, monkeypatch)
    with db.connect() as c:
        c.execute("INSERT INTO securities(symbol,name,exchange) VALUES('OLD_DEMO','Old demo','NSE')")
    result = analytics.screen(universe='all')
    assert result['evaluated'] == 0 and result['results'] == []
