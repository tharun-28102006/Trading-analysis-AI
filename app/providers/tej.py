"""TejHQ keyless end-of-day API client for the NSE bhavcopy dataset."""
from __future__ import annotations

import json
import hashlib
import os
import tempfile
import threading
import time
import queue
import urllib.error
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path
from typing import Callable

BASE_URL = 'https://api.tejhq.dev/v1'
SOURCE = 'TejHQ Indian Markets (NSE EOD)'
MAX_RESPONSE_BYTES = 25 * 1024 * 1024
DEFAULT_TIMEOUT_SECONDS = 8
_REQUEST_LOCK = threading.Lock()
_LAST_REQUEST = 0.0
_NETWORK_GATE = threading.Lock()


class TejError(RuntimeError):
    pass


class TejTimeoutError(TejError):
    pass


def _bounded_request(request: Callable, url: str, timeout: int) -> bytes:
    if not _NETWORK_GATE.acquire(blocking=False):
        raise TejError('A previous TejHQ request is still blocked by network name resolution.')
    result=queue.Queue(maxsize=1)
    def worker():
        try: result.put((True,request(url,timeout=timeout)))
        except BaseException as exc: result.put((False,exc))
        finally: _NETWORK_GATE.release()
    threading.Thread(target=worker,name='nivesh-tejhq-request',daemon=True).start()
    try: succeeded,value=result.get(timeout=timeout+1)
    except queue.Empty: raise TejTimeoutError(f'TejHQ request exceeded its {timeout + 1}-second total timeout.') from None
    if succeeded: return value
    raise value


def _http_get(url: str, *, timeout: int = 20) -> bytes:
    request = urllib.request.Request(url, headers={
        'Accept': 'application/json',
        'User-Agent': 'Nivesh local research client/1.0',
    })
    with urllib.request.urlopen(request, timeout=timeout) as response:
        host = urllib.parse.urlsplit(response.geturl()).hostname
        if response.status != 200:
            raise TejError(f'TejHQ API returned HTTP {response.status}.')
        if host != 'api.tejhq.dev':
            raise TejError('TejHQ API redirected outside its documented API host.')
        if response.headers.get_content_type() != 'application/json':
            raise TejError('TejHQ API returned a non-JSON response.')
        payload = response.read(MAX_RESPONSE_BYTES + 1)
        if len(payload) > MAX_RESPONSE_BYTES:
            raise TejError('TejHQ response exceeded the 25 MiB safety limit.')
        return payload


class TejClient:
    def __init__(self, request: Callable = _http_get, sleeper: Callable = time.sleep):
        self._request = request
        self._sleep = sleeper

    def _get(self, url: str) -> bytes:
        global _LAST_REQUEST
        last_error = None
        for attempt in range(3):
            try:
                with _REQUEST_LOCK:
                    delay = max(0.0, 1.0 - (time.monotonic() - _LAST_REQUEST))
                    if delay:
                        self._sleep(delay)
                    _LAST_REQUEST = time.monotonic()
                    try: timeout=max(2,min(int(os.getenv('NIVESH_TEJ_TIMEOUT_SECONDS',str(DEFAULT_TIMEOUT_SECONDS))),30))
                    except ValueError: timeout=DEFAULT_TIMEOUT_SECONDS
                    return _bounded_request(self._request,url,timeout)
            except TejTimeoutError:
                raise
            except urllib.error.HTTPError as exc:
                last_error = TejError(f'TejHQ API returned HTTP {exc.code}.')
                if exc.code not in (408, 429, 500, 502, 503, 504):
                    raise last_error from None
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last_error = TejError(f'TejHQ request failed ({type(exc).__name__}).')
            if attempt < 2:
                self._sleep(0.5 * (2 ** attempt))
        raise last_error or TejError('TejHQ request failed.')

    def status(self) -> dict:
        payload = self._get(BASE_URL + '/status')
        obj = _decode_json(payload)
        data = obj.get('data', obj) if isinstance(obj, dict) else None
        if not isinstance(data, dict):
            raise TejError('TejHQ status response has an invalid shape.')
        trading_date = data.get('trading_date') or data.get('latest_trading_date')
        try:
            trading_date = date.fromisoformat(str(trading_date)).isoformat()
        except (TypeError, ValueError) as exc:
            raise TejError('TejHQ status response does not provide a valid trading_date.') from exc
        return {**data, 'trading_date': trading_date}

    def history(self, symbol: str, start: str, end: str) -> tuple[bytes, list[dict]]:
        if not symbol or len(symbol) > 30 or not all(c.isalnum() or c in '-&' for c in symbol):
            raise TejError('Invalid NSE symbol for TejHQ API.')
        try:
            start_date, end_date = date.fromisoformat(start), date.fromisoformat(end)
        except ValueError as exc:
            raise TejError('Invalid requested date range.') from exc
        if start_date < date(2010, 1, 4) or end_date < start_date:
            raise TejError('Requested date range is outside supported NSE coverage.')
        query = urllib.parse.urlencode({'from': start, 'to': end})
        url = f'{BASE_URL}/ohlcv/nse/{urllib.parse.quote(symbol, safe="-&")}?{query}'
        payload = self._get(url)
        obj = _decode_json(payload)
        rows = obj.get('data') if isinstance(obj, dict) else None
        if not isinstance(rows, list):
            raise TejError(f'TejHQ returned an invalid OHLCV response for {symbol}.')
        return payload, rows

    def snapshot(self, exchange: str, trading_date: str) -> tuple[bytes, list[dict], dict]:
        """Fetch one public full-market EOD snapshot, using TejHQ's keyless API."""
        exchange = str(exchange).upper()
        if exchange not in {'NSE', 'BSE'}:
            raise TejError('Exchange must be NSE or BSE.')
        try:
            day = date.fromisoformat(trading_date).isoformat()
        except ValueError as exc:
            raise TejError('Invalid snapshot date.') from exc
        query = urllib.parse.urlencode({'date': day})
        payload = self._get(f'{BASE_URL}/snapshot/{exchange.lower()}?{query}')
        obj = _decode_json(payload)
        rows = obj.get('data') if isinstance(obj, dict) else None
        meta = obj.get('meta', {}) if isinstance(obj, dict) else {}
        if not isinstance(rows, list) or not isinstance(meta, dict):
            raise TejError(f'TejHQ returned an invalid {exchange} snapshot response for {day}.')
        meta_date = meta.get('date') or meta.get('trading_date')
        if meta_date is not None and str(meta_date) != day:
            raise TejError(f'TejHQ snapshot date mismatch for {exchange}: requested {day}, received {meta_date}.')
        return payload, rows, meta


def retain_payload(payload: bytes) -> tuple[str, str]:
    from ..config import DATA_DIR
    digest = hashlib.sha256(payload).hexdigest()
    folder = DATA_DIR / 'provider_payloads' / 'tejhq'
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f'{digest}.json'
    if not target.exists():
        fd, temp = tempfile.mkstemp(prefix='.tej-', suffix='.tmp', dir=folder)
        try:
            with os.fdopen(fd, 'wb') as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp, target)
        finally:
            if os.path.exists(temp):
                os.unlink(temp)
    return digest, str(target)


def _decode_json(payload: bytes):
    try:
        return json.loads(payload)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise TejError('TejHQ response is not valid JSON.') from exc
