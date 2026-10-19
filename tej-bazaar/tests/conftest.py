import pytest
import requests

from pipeline.actions import fetch as af


@pytest.fixture(autouse=True)
def _plain_bse_session(monkeypatch):
    monkeypatch.setattr(af, "bse_session", requests.Session)
