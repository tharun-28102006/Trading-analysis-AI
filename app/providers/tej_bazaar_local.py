"""Local Tej Bazaar Parquet discovery and safe keyless snapshot updates."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable

import polars as pl

from .tej import TejClient, TejError, retain_payload

# Tej Bazaar's raw BSE symbol includes ``#`` for some valid equity symbols
# (for example the source's SBIN# row); preserve it as provider attribution.
_SYMBOL = re.compile(r'^[A-Z0-9][A-Z0-9&#-]{0,29}$')
_SERIES = {'NSE': {'EQ', 'BE', 'BZ'}, 'BSE': {'A', 'B', 'T'}}
_COLUMNS = {'date','symbol','series','isin','name','open','high','low','close','last','prev_close','volume','turnover','trades'}


@dataclass(frozen=True)
class MarketFile:
    exchange: str
    path: Path
    relative_path: str
    observation_date: str
    sha256: str | None
    size_bytes: int
    modified_ns: int


def default_data_root() -> Path:
    configured = os.getenv('NIVESH_TEJ_DATA_DIR', '').strip()
    return Path(configured).expanduser() if configured else Path(__file__).resolve().parents[2] / 'data' / 'out'


def overlap_days() -> int:
    try:
        value = int(os.getenv('NIVESH_TEJ_OVERLAP_DAYS', '7'))
    except ValueError as exc:
        raise ValueError('NIVESH_TEJ_OVERLAP_DAYS must be an integer.') from exc
    if not 1 <= value <= 60:
        raise ValueError('NIVESH_TEJ_OVERLAP_DAYS must be between 1 and 60.')
    return value


def _path_date(path: Path) -> str:
    match = re.search(r'(^|[\\/])date=(\d{4}-\d{2}-\d{2})\.parquet$', str(path))
    if match:
        return date.fromisoformat(match.group(2)).isoformat()
    match = re.search(r'(^|[\\/])year=(\d{4})[\\/]([^\\/]+)\.parquet$', str(path))
    if match:
        return match.group(2) + '-01-01'
    raise ValueError(f'Unsupported Tej Bazaar Parquet path: {path}')


def discover_files(root: Path, exchange: str, *, known: dict[str, tuple[str,int,int]] | None = None,
                   hash_from: date | None = None, hash_files: bool = True, force_hash: bool = False) -> list[MarketFile]:
    exchange = exchange.lower()
    folder = root / exchange
    if not folder.is_dir():
        return []
    # Prefer daily partitions when both the daily cache and compacted rollups exist.
    daily = sorted(folder.glob('year=*/month=*/date=*.parquet'))
    paths = daily or sorted(folder.glob(f'year=*/{exchange}_*.parquet'))
    result = []
    for path in paths:
        relative = path.relative_to(root).as_posix()
        obs_date = _path_date(path); info=path.stat(); prior=(known or {}).get(relative)
        should_hash=hash_files and (force_hash or prior is None or info.st_size != prior[1] or info.st_mtime_ns != prior[2]
            or (hash_from is not None and obs_date >= hash_from.isoformat()))
        digest=None
        if should_hash:
            sha = hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    sha.update(block)
            digest=sha.hexdigest()
        result.append(MarketFile(exchange.upper(), path, relative, obs_date, digest, info.st_size, info.st_mtime_ns))
    return result


def read_and_validate_file(source: MarketFile) -> pl.DataFrame:
    frame = pl.read_parquet(source.path, hive_partitioning=False)
    missing = _COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f'{source.relative_path} is missing columns: {", ".join(sorted(missing))}.')
    frame = frame.select(sorted(_COLUMNS))
    if frame.is_empty():
        raise ValueError(f'{source.relative_path} contains no market observations.')
    if frame.schema['date'] != pl.Date:
        frame = frame.with_columns(pl.col('date').cast(pl.Date, strict=True))
    required_null=frame.select(pl.any_horizontal([pl.col(x).is_null() for x in ('date','symbol','series','open','high','low','close','volume')]).any()).item()
    if required_null:
        raise ValueError(f'{source.relative_path} contains null required market fields.')
    if frame.filter(~pl.col('series').is_in(sorted(_SERIES[source.exchange]))).height:
        raise ValueError(f'{source.relative_path} contains a non-equity or unsupported {source.exchange} series.')
    if frame.filter(pl.col('symbol').is_null() | (pl.col('symbol').str.to_uppercase() != pl.col('symbol'))).height:
        raise ValueError(f'{source.relative_path} contains an invalid or non-uppercase symbol.')
    for row in frame.select(['symbol']).iter_rows():
        if not _SYMBOL.fullmatch(row[0]):
            raise ValueError(f'{source.relative_path} contains an invalid instrument symbol.')
    invalid = frame.filter(
        (pl.col('open') <= 0) | (pl.col('high') <= 0) | (pl.col('low') <= 0) | (pl.col('close') <= 0)
        | (pl.col('volume') < 0)
        | (pl.col('high') < pl.max_horizontal('open','close','low'))
        | (pl.col('low') > pl.min_horizontal('open','close','high'))
    )
    if invalid.height:
        raise ValueError(f'{source.relative_path} contains {invalid.height} invalid OHLCV row(s).')
    if frame.select((pl.col('volume') % 1 != 0).any()).item():
        raise ValueError(f'{source.relative_path} contains fractional share volume.')
    if frame.select(pl.struct(['symbol','date']).is_duplicated().any()).item():
        raise ValueError(f'{source.relative_path} contains duplicate symbol/date observations.')
    if source.observation_date.endswith('-01-01') and source.path.name.endswith('.parquet') and 'date=' not in source.path.name:
        year = int(source.observation_date[:4])
        if frame.filter(pl.col('date').dt.year() != year).height:
            raise ValueError(f'{source.relative_path} contains rows outside its declared year.')
    else:
        expected = date.fromisoformat(source.observation_date)
        if frame.filter(pl.col('date') != expected).height:
            raise ValueError(f'{source.relative_path} contains rows outside its declared session date.')
    return frame.sort(['date','symbol'])


def _validate_snapshot(exchange: str, day: date, rows: list[dict]) -> pl.DataFrame:
    if not isinstance(rows, list):
        raise TejError(f'TejHQ returned a malformed {exchange} snapshot.')
    normalized = []
    seen = set()
    required = ('symbol','series','open','high','low','close','volume')
    for raw in rows:
        if not isinstance(raw, dict) or any(key not in raw for key in required):
            raise TejError(f'TejHQ {exchange} snapshot contains an incomplete observation.')
        symbol = str(raw['symbol']).strip().upper()
        series = str(raw['series']).strip().upper()
        if not _SYMBOL.fullmatch(symbol) or series not in _SERIES[exchange]:
            raise TejError(f'TejHQ returned an invalid {exchange} instrument or equity series.')
        if symbol in seen:
            raise TejError(f'TejHQ returned duplicate {exchange} symbol {symbol} for {day}.')
        seen.add(symbol)
        try:
            values = {k: float(raw[k]) for k in ('open','high','low','close','volume')}
        except (TypeError, ValueError) as exc:
            raise TejError(f'TejHQ returned non-numeric {exchange} OHLCV for {symbol}.') from exc
        if not all(math.isfinite(v) for v in values.values()):
            raise TejError(f'TejHQ returned non-finite {exchange} OHLCV for {symbol}.')
        if min(values[k] for k in ('open','high','low','close')) <= 0 or values['volume'] < 0 or values['volume'] % 1:
            raise TejError(f'TejHQ returned invalid positive OHLC or whole-share volume for {symbol}.')
        if values['high'] < max(values['open'],values['close'],values['low']) or values['low'] > min(values['open'],values['close'],values['high']):
            raise TejError(f'TejHQ returned invalid OHLC relationships for {symbol}.')
        normalized.append({'date':day,'symbol':symbol,'series':series,'isin':raw.get('isin'),'name':raw.get('name'),
            'open':values['open'],'high':values['high'],'low':values['low'],'close':values['close'],
            'last':raw.get('last'),'prev_close':raw.get('prev_close'),'volume':int(values['volume']),
            'turnover':raw.get('turnover'),'trades':raw.get('trades')})
    if not normalized:
        return pl.DataFrame(schema={'date':pl.Date,'symbol':pl.String,'series':pl.String,'isin':pl.String,'name':pl.String,
            'open':pl.Float64,'high':pl.Float64,'low':pl.Float64,'close':pl.Float64,'last':pl.Float64,'prev_close':pl.Float64,
            'volume':pl.Int64,'turnover':pl.Float64,'trades':pl.Int64})
    return pl.DataFrame(normalized).select(sorted(_COLUMNS))


def fetch_incremental_snapshots(root: Path, *, latest_by_exchange: dict[str, str | None], today: date,
                                status_date: str, client: TejClient | None = None,
                                progress: Callable | None = None) -> dict:
    """Fetch only recent exchange sessions, atomically writing validated daily Parquet files."""
    client = client or TejClient()
    overlap = overlap_days()
    status_day = date.fromisoformat(status_date)
    if status_day > today + timedelta(days=1):
        raise TejError('TejHQ status reports a future trading date.')
    results = {}
    try:
        from pipeline.holidays import trading_days_between
    except ImportError as exc:
        raise RuntimeError('The installed tej-bazaar Python package is required for its exchange-session calendar.') from exc
    for exchange in ('NSE','BSE'):
        latest = latest_by_exchange.get(exchange)
        # Local Parquet is the bootstrap; remote requests cover its trailing overlap and any missing sessions.
        if not latest:
            raise ValueError(f'No local {exchange} history is available for incremental refresh.')
        start = max(date.fromisoformat(latest) - timedelta(days=overlap),
                    date(2010,1,4) if exchange=='NSE' else date(2024,7,8))
        end = min(today, status_day)
        days = trading_days_between(start, end, exchange) if end >= start else []
        inserted = 0; files = []; errors = []; empty = []
        for day in days:
            try:
                payload, rows, meta = client.snapshot(exchange,day.isoformat())
                if not rows:
                    raise TejError(f'TejHQ reported {status_date} as available but returned an empty {exchange} session for {day}.')
                frame = _validate_snapshot(exchange,day,rows)
                digest, raw_path = retain_payload(payload)
                folder = root / exchange.lower() / f'year={day.year}' / f'month={day.month:02d}'
                folder.mkdir(parents=True,exist_ok=True)
                target = folder / f'date={day.isoformat()}.parquet'
                fd, temp = tempfile.mkstemp(prefix='.tej-snapshot-',suffix='.parquet',dir=folder)
                os.close(fd)
                try:
                    frame.write_parquet(temp,compression='zstd')
                    os.replace(temp,target)
                finally:
                    if os.path.exists(temp): os.unlink(temp)
                files.append({'date':day.isoformat(),'rows':frame.height,'payload_sha256':digest,'payload_path':raw_path,
                    'meta':meta,'path':str(target)})
                inserted += frame.height
            except Exception as exc:
                errors.append({'date':day.isoformat(),'error':f'{type(exc).__name__}: {str(exc)[:300]}'})
                # A failed date must not prevent the other exchange from being attempted.
                if progress: progress(exchange,day.isoformat(),len(files),inserted)
                break
            if progress: progress(exchange,day.isoformat(),len(files),inserted)
        results[exchange]={'status':'failed' if errors else 'succeeded','source_trading_date':status_date,
            'latest_fetched_date':max((f['date'] for f in files),default=None),'rows_fetched':inserted,
            'files':files,'empty_sessions':empty,'errors':errors}
    return results


def frame_rows(frame: pl.DataFrame):
    """Yield normalized source rows in bounded chunks for database staging."""
    columns=['date','symbol','series','isin','name','open','high','low','close','last','prev_close','volume','turnover','trades']
    selected=frame.select(columns)
    return selected.iter_rows(named=True)
