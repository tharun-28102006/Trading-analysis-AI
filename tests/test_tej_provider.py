import json
import urllib.error

import pytest

from app.providers import tej


def test_history_builds_documented_keyless_request_and_parses_rows():
    calls = []
    payload = json.dumps({'data': [{'date': '2026-10-15', 'open': 100, 'high': 103,
        'low': 99, 'close': 102, 'volume': 1200}]}).encode()
    def request(url, timeout):
        calls.append((url, timeout))
        return payload
    client = tej.TejClient(request=request, sleeper=lambda _: None)
    raw, rows = client.history('RELIANCE', '2026-10-01', '2026-10-15')
    assert rows[0]['close'] == 102
    assert raw == payload
    assert 'ohlcv/nse/RELIANCE?from=2026-10-01&to=2026-10-15' in calls[0][0]
    assert calls[0][1] == 8


@pytest.mark.parametrize('payload', [b'not json', b'{"data":{}}', b'{"error":"no access"}'])
def test_history_rejects_malformed_provider_response(payload):
    client = tej.TejClient(request=lambda *_args, **_kwargs: payload, sleeper=lambda _: None)
    with pytest.raises(tej.TejError):
        client.history('RELIANCE', '2026-10-01', '2026-10-15')


def test_history_rejects_invalid_symbols_and_ranges_without_network():
    client = tej.TejClient(request=lambda *_args, **_kwargs: pytest.fail('network called'))
    with pytest.raises(tej.TejError, match='Invalid NSE symbol'):
        client.history('../RELIANCE', '2026-10-01', '2026-10-15')
    with pytest.raises(tej.TejError, match='supported NSE coverage'):
        client.history('RELIANCE', '2009-12-31', '2026-10-15')


def test_retry_is_bounded_and_does_not_log_response_or_credentials():
    calls = []
    def request(_url, timeout):
        calls.append(timeout)
        raise urllib.error.URLError('offline')
    client = tej.TejClient(request=request, sleeper=lambda _: None)
    with pytest.raises(tej.TejError, match='request failed'):
        client.status()
    assert calls == [8, 8, 8]
