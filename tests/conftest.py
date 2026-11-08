import pytest

from musegadget import config


@pytest.fixture
def paired_inbox(tmp_path, monkeypatch):
    monkeypatch.setenv(config.STATE_DIR_ENV, str(tmp_path))
    config.save_json(config.PAIRING_FILE, {"username": "owner", "access_token": "private-token"})
