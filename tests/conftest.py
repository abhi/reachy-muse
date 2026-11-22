import os

import pytest

from musegadget import config


@pytest.fixture(autouse=True)
def cli_environment():
    """Restores the state directory and SDK token that ``cli.main()`` exports to the process."""
    saved = {name: os.environ.get(name) for name in (config.STATE_DIR_ENV, config.SDK_TOKEN_ENV)}
    yield
    for name, value in saved.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


@pytest.fixture
def paired_inbox(tmp_path, monkeypatch):
    monkeypatch.setenv(config.STATE_DIR_ENV, str(tmp_path))
    config.save_json(config.PAIRING_FILE, {"username": "owner", "access_token": "private-token"})
