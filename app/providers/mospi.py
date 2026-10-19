"""Free MoSPI e-Sankhyiki current CPI API adapter. This source is macro-only."""
from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

API_BASE = 'https://api.mospi.gov.in'
API_URL = API_BASE + '/api/cpi/getCPIData'
SOURCE = 'MoSPI CPI API'
MAX_RESPONSE_BYTES = 40 * 1024 * 1024
_REQUEST_LOCK = threading.Lock()
_LAST_REQUEST = 0.0


class MoSPIError(RuntimeError):
    pass


def _request(url: str, *, headers: dict, method='GET', body=None, timeout: int = 30):
    request = urllib.request.Request(url, headers=headers, data=body, method=method)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        if response.status != 200:
            raise MoSPIError(f'MoSPI API returned HTTP {response.status}.')
        if urllib.parse.urlsplit(response.geturl()).hostname not in {'www.api.mospi.gov.in', 'api.mospi.gov.in'}:
            raise MoSPIError('MoSPI API redirected outside its official host.')
        content_type = response.headers.get_content_type()
        if content_type not in {'application/json', 'text/json'}:
            raise MoSPIError(f'MoSPI API returned unexpected content type {content_type}.')
        data = response.read(MAX_RESPONSE_BYTES + 1)
        if len(data) > MAX_RESPONSE_BYTES:
            raise MoSPIError('MoSPI API response exceeded the 40 MiB size limit.')
        return data


class MoSPIClient:
    def __init__(self, request: Callable = _request, sleeper: Callable = time.sleep):
        self._request, self._sleep = request, sleeper

    def _call(self, url, *, method='GET', payload=None, token=None):
        global _LAST_REQUEST
        headers={'Accept':'application/json','User-Agent':'Nivesh local research client/1.0'}
        body=None
        if payload is not None:
            headers['Content-Type']='application/json'
            body=json.dumps(payload,separators=(',',':')).encode()
        if token:
            headers['Authorization']=f'Bearer {token}'
        error=None
        for attempt in range(3):
            try:
                with _REQUEST_LOCK:
                    delay=max(0.0,1.05-(time.monotonic()-_LAST_REQUEST))
                    if delay: self._sleep(delay)
                    _LAST_REQUEST=time.monotonic()
                    return self._request(url,headers=headers,method=method,body=body,timeout=30)
            except urllib.error.HTTPError as exc:
                error=MoSPIError(f'MoSPI API returned HTTP {exc.code}.')
                if exc.code not in (408,429,500,502,503,504): raise error from None
            except (urllib.error.URLError,TimeoutError,OSError) as exc:
                error=MoSPIError(f'MoSPI request failed ({type(exc).__name__}).')
            if attempt<2: self._sleep(0.75*(2**attempt))
        raise error or MoSPIError('MoSPI request failed.')

    def fetch_cpi(self) -> tuple[bytes, list[dict]]:
        query = urllib.parse.urlencode({'base_year': '2024', 'year': str(datetime.now(timezone.utc).year),
                                        'series': 'Current', 'Format': 'JSON'})
        url = f'{API_URL}?{query}'
        payload=self._call(url)
        retain_payload(payload)
        return payload,parse_cpi(payload)


def _number(value, label: str, *, positive=False):
    if value in (None, ''):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise MoSPIError(f'MoSPI CPI row has an invalid {label}.') from exc
    if not math.isfinite(result) or (positive and result <= 0):
        raise MoSPIError(f'MoSPI CPI row has an invalid {label}.')
    return result


def parse_cpi(payload: bytes) -> list[dict]:
    if not payload or len(payload) > MAX_RESPONSE_BYTES:
        raise MoSPIError('MoSPI CPI response is empty or too large.')
    try:
        obj = json.loads(payload)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise MoSPIError('MoSPI CPI response is not valid JSON.') from exc
    rows = obj.get('data') if isinstance(obj, dict) else obj
    if isinstance(rows, dict):
        rows = rows.get('data') or rows.get('records') or rows.get('rows')
    if not isinstance(rows, list) or not rows:
        raise MoSPIError('MoSPI CPI response has no data rows.')
    months = {name.lower(): number for number, name in enumerate(
        ('January','February','March','April','May','June','July','August','September','October','November','December'), 1)}
    parsed = []
    for row in rows:
        if not isinstance(row, dict):
            raise MoSPIError('MoSPI CPI response contains a non-object row.')
        try:
            year = int(row.get('year'))
            month_value = row.get('month', row.get('month_name', row.get('month_code')))
            month = int(month_value) if str(month_value).isdigit() else months[str(month_value).strip().lower()]
            if year < 1900 or year > datetime.now(timezone.utc).year + 1 or not 1 <= month <= 12:
                raise ValueError()
        except (TypeError, ValueError, KeyError) as exc:
            raise MoSPIError('MoSPI CPI row has an invalid year or month.') from exc
        index_value = _number(row.get('index', row.get('index_value', row.get('cpi_index'))), 'index value', positive=True)
        inflation = _number(row.get('inflation', row.get('inflation_rate')), 'inflation')
        if index_value is None and inflation is None:
            raise MoSPIError('MoSPI CPI row has neither an index value nor inflation.')
        published = row.get('published_at') or row.get('publication_date')
        if published:
            try:
                published = datetime.fromisoformat(str(published).replace('Z', '+00:00')).isoformat()
            except ValueError as exc:
                raise MoSPIError('MoSPI CPI row has an invalid publication timestamp.') from exc
        parsed.append({
            'year': year, 'month': month,
            'geography': str(row.get('state') or row.get('state_name') or row.get('geography') or 'All India').strip(),
            'sector': str(row.get('sector') or '').strip(),
            'group_name': str(row.get('group') or row.get('group_name') or row.get('item_group') or '').strip(),
            'subgroup': str(row.get('subgroup') or '').strip(),
            'index_value': index_value, 'inflation': inflation,
            'status': str(row.get('status') or '').strip() or None,
            'published_at': published,
            'record_json': json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(',', ':')),
        })
    return parsed


def retain_payload(payload: bytes) -> tuple[str, str]:
    from ..config import DATA_DIR
    digest = hashlib.sha256(payload).hexdigest()
    folder = DATA_DIR / 'provider_payloads' / 'mospi'
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f'{digest}.json'
    if not target.exists():
        fd, temp = tempfile.mkstemp(prefix='.mospi-', suffix='.tmp', dir=folder)
        try:
            with os.fdopen(fd, 'wb') as handle:
                handle.write(payload); handle.flush(); os.fsync(handle.fileno())
            os.replace(temp, target)
        finally:
            if os.path.exists(temp): os.unlink(temp)
    return digest, str(target)


def store_cpi(payload: bytes, rows: list[dict]) -> dict:
    from ..db import connect
    digest, raw_path = retain_payload(payload)
    retrieved = datetime.now(timezone.utc).isoformat()
    inserted = 0
    with connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        for row in rows:
            identity = {key: row[key] for key in ('year','month','geography','sector','group_name','subgroup','index_value','inflation','status','published_at')}
            record_hash = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
            cursor = conn.execute('''INSERT OR IGNORE INTO macro_observations
                (source,dataset,year,month,geography,sector,group_name,subgroup,index_value,inflation,status,
                 published_at,retrieved_at,payload_hash,record_hash,record_json)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
                (SOURCE,'cpi_current_base_2024',row['year'],row['month'],row['geography'],row['sector'],
                 row['group_name'],row['subgroup'],row['index_value'],row['inflation'],row['status'],
                 row['published_at'],retrieved,digest,record_hash,row['record_json']))
            inserted += cursor.rowcount
    return {'rows_received':len(rows),'new_rows':inserted,'duplicate_rows':len(rows)-inserted,
            'payload_sha256':digest,'raw_payload_path':raw_path,
            'publication_time_complete':all(row['published_at'] for row in rows)}
