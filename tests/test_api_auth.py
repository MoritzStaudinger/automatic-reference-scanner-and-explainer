import datetime
import importlib

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def api(tmp_path, monkeypatch):
    monkeypatch.setenv("ARES_JOBS_DB", str(tmp_path / "jobs.sqlite"))
    monkeypatch.setenv("ARES_API_KEYS_FILE", str(tmp_path / "keys.json"))
    monkeypatch.delenv("ARES_API_KEY", raising=False)
    monkeypatch.delenv("ARES_API_ALLOW_UNAUTHENTICATED", raising=False)
    import job_store
    monkeypatch.setattr(job_store, "_singleton", None)
    import api
    return importlib.reload(api), tmp_path


def _post(client, key=None, data=b"%PDF-1.4\n%%EOF\n"):
    return client.post("/api/v1/process", headers={"X-API-Key": key} if key else {},
                       files={"file": ("x.pdf", data, "application/pdf")},
                       data={"parser": "pymupdf4llm"})


def test_no_keys_means_disabled(api):
    mod, _ = api
    c = TestClient(mod.app)
    assert c.get("/healthz").status_code == 200
    assert _post(c).status_code == 503


def test_keys_file_and_expiry(api):
    mod, tmp = api
    past = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=1)).isoformat()
    (tmp / "keys.json").write_text(
        '{"good": {"label": "a", "expires_at": "2999-01-01T00:00:00+02:00"},'
        ' "old": {"label": "b", "expires_at": "%s"},'
        ' "typo": {"label": "c", "expires_at": "next tuesday"}}' % past)
    c = TestClient(mod.app)
    assert _post(c, "nope").status_code == 401
    assert _post(c, "old").status_code == 401
    assert _post(c, "typo").status_code == 401          # malformed expiry fails closed
    assert _post(c, "good").status_code == 400          # authed; corrupt PDF → 400, not 500


def test_parse_expiry_naive_is_utc(api):
    mod, _ = api
    assert mod.parse_expiry("2030-01-01").tzinfo is not None
    assert mod.parse_expiry(None) is None
