"""The ARES → ARSE rename must not break an existing .env."""

import importlib
import os

import arse_env


def test_old_ares_settings_are_accepted(monkeypatch):
    monkeypatch.setenv("ARES_API_KEY", "old-key")
    monkeypatch.setenv("ARES_CONTACT_EMAIL", "a@b.c")
    monkeypatch.setenv("ARSE_CONTACT_EMAIL", "new@b.c")      # new name wins
    monkeypatch.delenv("ARSE_API_KEY", raising=False)
    importlib.reload(arse_env)
    assert os.environ["ARSE_API_KEY"] == "old-key"
    assert os.environ["ARSE_CONTACT_EMAIL"] == "new@b.c"
    monkeypatch.delenv("ARSE_API_KEY", raising=False)
