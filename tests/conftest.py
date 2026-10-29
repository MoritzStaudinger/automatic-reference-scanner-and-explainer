import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def fresh_cache(tmp_path, monkeypatch):
    """Point the lookup cache at an empty temp DB for one test."""
    import cache
    monkeypatch.setattr(cache, "CACHE_PATH", tmp_path / "cache.sqlite")
    monkeypatch.setattr(cache, "_cache_singleton", None)
    yield cache.get_cache()
    monkeypatch.setattr(cache, "_cache_singleton", None)
