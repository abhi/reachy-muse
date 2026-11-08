"""Local notifications survive restarts without repeating completed IDs."""

import json
import os
import stat

import pytest

from musegadget import config

from reachy_muse.inbox import Inbox

pytestmark = pytest.mark.usefixtures("paired_inbox")


def notice(id="n1", text="Call Mom.", *, ttl=14400, kind="notification"):
    return {"id": id, "text": text, "mode": "when_present", "expires_in_s": ttl, "kind": kind}


def test_queue_and_receipt_survive_restart_without_retaining_delivered_text(tmp_path):
    path = tmp_path / "inbox.json"
    inbox = Inbox(path, clock=lambda: 100)
    assert inbox.add(notice()) == {"ok": True, "status": "queued", "reason": "awaiting_presence"}
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    inbox = Inbox(path, clock=lambda: 101)
    assert inbox.pending()["text"] == "Call Mom."
    item = inbox.pending()
    assert inbox.claim(item) is True
    inbox.finish(item, spoken=True)
    assert "Call Mom." not in path.read_text()
    assert json.loads(path.read_text())["items"][0]["delivered_at"] == 101
    restarted = Inbox(path, clock=lambda: 102)
    assert restarted.add(notice()) == {"ok": True, "status": "delivered", "reason": "spoken"}
    assert restarted.pending() is None
    assert restarted.add(notice(text="Different.")) == {"ok": True, "status": "refused", "reason": "id_conflict"}


def test_twenty_ids_include_receipts_until_expiry_and_overflow_is_refused(tmp_path):
    now = [100]
    inbox = Inbox(tmp_path / "inbox.json", clock=lambda: now[0])
    for index in range(20):
        assert inbox.add(notice(str(index), ttl=10))["status"] == "queued"
    assert inbox.add(notice("overflow")) == {"ok": True, "status": "refused", "reason": "inbox_full"}
    assert inbox.add(notice("0", ttl=500))["status"] == "queued"
    assert inbox.pending()["expires_at"] == 110
    assert inbox.pending()["id"] == "0"
    item = inbox.pending()
    inbox.claim(item)
    inbox.finish(item, spoken=True)
    assert inbox.pending()["id"] == "1"
    assert inbox.add(notice("overflow"))["reason"] == "inbox_full"
    now[0] = 110
    assert inbox.result("1") == {"ok": True, "status": "expired", "reason": "ttl_elapsed"}
    assert inbox.pending() is None
    assert inbox.add(notice("new"))["status"] == "queued"


def test_zero_ttl_and_expired_retry_never_extend_delivery_lifetime():
    now = [100]
    inbox = Inbox(clock=lambda: now[0])
    assert inbox.add(notice(ttl=0)) == {"ok": True, "status": "expired", "reason": "ttl_elapsed"}
    inbox.add(notice(ttl=5))
    now[0] = 106
    assert inbox.add(notice()) == {"ok": True, "status": "expired", "reason": "ttl_elapsed"}
    assert inbox.pending() is None


def test_crash_during_delivery_does_not_replay_an_uncertain_line(tmp_path):
    path = tmp_path / "inbox.json"
    inbox = Inbox(path, clock=lambda: 100)
    inbox.add(notice())
    assert inbox.claim(inbox.pending()) is True
    restarted = Inbox(path, clock=lambda: 101)
    assert restarted.add(notice()) == {"ok": True, "status": "refused", "reason": "delivery_uncertain"}
    assert restarted.pending() is None
    assert "Call Mom." not in path.read_text()


def test_claim_syncs_the_replaced_directory_entry_before_returning(tmp_path, monkeypatch):
    path = tmp_path / "inbox.json"
    inbox = Inbox(path)
    inbox.add(notice())
    synced = []
    fsync = os.fsync

    def sync(fd):
        fsync(fd)
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            item, = json.loads(path.read_text())["items"]
            synced.append((item["status"], item["reason"], item["text"]))

    monkeypatch.setattr(os, "fsync", sync)
    assert inbox.claim(inbox.pending()) is True
    assert synced == [("refused", "delivery_uncertain", None)]


def test_unpair_removes_the_entire_ledger(tmp_path):
    path = tmp_path / "inbox.json"
    inbox = Inbox(path)
    inbox.add(notice())
    path.with_suffix(".tmp").write_text("Call Mom.")
    config.delete_json(config.PAIRING_FILE)
    assert inbox.pending() is None
    assert path.with_suffix(".tmp").exists() is False
    assert path.exists() is False
    assert Inbox(path).pending() is None


def test_storage_failure_never_accepts_an_unpersisted_notification(tmp_path, monkeypatch):
    inbox = Inbox(tmp_path / "inbox.json")

    def fail():
        raise OSError("disk full")

    monkeypatch.setattr(inbox, "_save", fail)
    assert inbox.add(notice()) == {"ok": True, "status": "refused", "reason": "storage_unavailable"}
    assert inbox.pending() is None


@pytest.mark.parametrize("change", ["account", "server", "missing", "legacy"])
def test_open_discards_items_without_a_matching_owner(tmp_path, change):
    path = tmp_path / "inbox.json"
    inbox = Inbox(path)
    assert inbox.add(notice())["status"] == "queued"
    if change == "missing":
        config.delete_json(config.PAIRING_FILE)
    elif change == "legacy":
        data = json.loads(path.read_text())
        del data["owner"]
        path.write_text(json.dumps(data))
    else:
        pairing = config.load_json(config.PAIRING_FILE)
        pairing.update({"username": "next-owner"} if change == "account" else
                       {"api_url_v2": "https://other.example"})
        config.save_json(config.PAIRING_FILE, pairing)
    assert Inbox(path).pending() is None
    assert path.exists() is False
    assert inbox.pending() is None


def test_rotating_tokens_preserves_the_owner_and_queue_without_storing_tokens(tmp_path):
    path = tmp_path / "inbox.json"
    inbox = Inbox(path)
    assert inbox.add(notice())["status"] == "queued"
    pairing = config.load_json(config.PAIRING_FILE)
    config.save_json(config.PAIRING_FILE, {**pairing, "access_token": "rotated-access",
                                         "refresh_token": "rotated-refresh", "access_token_saved_at": 200})
    reopened = Inbox(path)
    assert reopened.pending()["text"] == "Call Mom."
    item = reopened.pending()
    assert reopened.claim(item) is True
    reopened.finish(item, spoken=True)
    assert inbox.result("n1") == {"ok": True, "status": "delivered", "reason": "spoken"}
    assert "private-token" not in path.read_text()
    assert "rotated-access" not in path.read_text()
    assert "rotated-refresh" not in path.read_text()
