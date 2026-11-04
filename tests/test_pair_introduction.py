# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0 (the "License").

import io
import json
from types import SimpleNamespace

import pytest

from musegadget import config
from musegadget.identity import Identity
from musegadget.pair import PairOutcome, PairResult

from reachy_muse import cli, introduction
from test_introduction import Session


INTRODUCTION = (
    "I have a Reachy Mini robot at home that talks with you by voice. "
    "Reachy is one of my persistent surfaces, like the Muse app. "
    "When I talk through Reachy, your replies are spoken aloud, so keep them short. "
    "You can speak to me in the room with reachy.say and check whether someone is there "
    "with reachy.presence, for example to deliver a reminder when I'm home. Please remember this."
)
OFFER = "\n" + INTRODUCTION + "\n\nSend this introduction to your main Muse chat? [Y/n] "


class Terminal(io.StringIO):
    def __init__(self, text, *, interactive=True):
        super().__init__(text)
        self.interactive = interactive
        self.reads = 0

    def isatty(self):
        return self.interactive

    def readline(self, *args):
        self.reads += 1
        return super().readline(*args)


class PairedSession(Session):
    async def wait_registered(self):
        await self.registered.wait()


@pytest.fixture
def pairing_flow(tmp_path, monkeypatch):
    flow = SimpleNamespace(vm_id="vm-one", outcome=PairOutcome.PAIRED,
                           pair_calls=[], fetched=[], sessions=[], connected_vms=[])

    def pair(*, on_open, force, timeout_s):
        flow.pair_calls.append((force, timeout_s))
        if flow.outcome is PairOutcome.PAIRED:
            config.save_json(config.PAIRING_FILE, {
                "access_token": "fake-token", "username": "paired-user"}, tmp_path)
        return PairResult(flow.outcome, sdk_token_problem="invalid SDK token")

    def fetch(token, api):
        assert config.load_json(config.PAIRING_FILE, tmp_path) == {
            "access_token": "fake-token", "username": "paired-user"}
        flow.fetched.append(token)
        return [{"vm_id": flow.vm_id, "is_default": True,
                 "vm_auth_token": "fake-vm-token"}], 200

    def connect(**kwargs):
        session = PairedSession()
        flow.sessions.append(session)
        flow.connected_vms.append(kwargs["vm_id"])
        return session

    monkeypatch.setenv(config.STATE_DIR_ENV, str(tmp_path))
    monkeypatch.setattr("musegadget.pair.pair", pair)
    monkeypatch.setattr(introduction.identity, "load_or_create",
                        lambda: Identity("02:00:00:ab:cd:ef"))
    monkeypatch.setattr(introduction.muse_api, "fetch_vms_with_status", fetch)
    monkeypatch.setattr(introduction, "LinkSession", connect)
    monkeypatch.setattr(cli, "_hardware", lambda *args: pytest.fail("pairing started hardware"))
    return flow


@pytest.mark.parametrize("answer, sends", [
    ("yes\n", True), ("Y\n", True), ("\n", True),
    ("n\n", False), ("no\n", False), ("maybe\n", False), ("", False),
])
def test_interactive_pair_shows_exact_text_and_asks_once(
        tmp_path, monkeypatch, capsys, pairing_flow, answer, sends):
    terminal = Terminal(answer)
    monkeypatch.setattr(cli.sys, "stdin", terminal)

    assert cli.main(["--state-dir", str(tmp_path), "pair"]) == 0

    assert capsys.readouterr() == (
        f"Reachy is paired with Muse. The pairing is saved in {tmp_path}.\n" + OFFER, "")
    assert terminal.reads == 1
    assert pairing_flow.pair_calls == [(False, 600)]
    assert [session.posts for session in pairing_flow.sessions] == ([[INTRODUCTION]] if sends else [])
    assert (tmp_path / "reachy-introduction.json").exists() is sends
    assert config.load_json(config.PAIRING_FILE, tmp_path) == {
        "access_token": "fake-token", "username": "paired-user"}


@pytest.mark.parametrize("interactive, flags, sends", [
    (False, [], False),
    (False, ["--introduce"], True),
    (False, ["--no-introduce"], False),
    (True, ["--introduce"], True),
    (True, ["--no-introduce"], False),
])
def test_pair_flags_and_noninteractive_default_never_prompt(
        tmp_path, monkeypatch, capsys, pairing_flow, interactive, flags, sends):
    terminal = Terminal("yes\n", interactive=interactive)
    monkeypatch.setattr(cli.sys, "stdin", terminal)

    assert cli.main(["--state-dir", str(tmp_path), "pair", *flags]) == 0

    assert capsys.readouterr() == (
        f"Reachy is paired with Muse. The pairing is saved in {tmp_path}.\n", "")
    assert terminal.reads == 0
    assert [session.posts for session in pairing_flow.sessions] == ([[INTRODUCTION]] if sends else [])
    assert pairing_flow.fetched == (["fake-token"] if sends else [])
    assert (tmp_path / "reachy-introduction.json").exists() is sends


def test_pair_introduction_flags_are_mutually_exclusive(tmp_path, monkeypatch, capsys, pairing_flow):
    monkeypatch.setattr(cli.sys, "stdin", Terminal("yes\n"))

    with pytest.raises(SystemExit) as error:
        cli.main(["--state-dir", str(tmp_path), "pair", "--introduce", "--no-introduce"])

    assert error.value.code == 2
    assert "not allowed with argument --introduce" in capsys.readouterr().err
    assert not (tmp_path / config.PAIRING_FILE).exists()


def test_pair_reoffers_only_for_a_new_vm(tmp_path, monkeypatch, capsys, pairing_flow):
    terminal = Terminal("\nyes\n")
    monkeypatch.setattr(cli.sys, "stdin", terminal)
    command = ["--state-dir", str(tmp_path), "pair"]
    paired = f"Reachy is paired with Muse. The pairing is saved in {tmp_path}.\n"

    assert cli.main(command) == 0
    assert cli.main([*command, "--force"]) == 0
    pairing_flow.vm_id = "vm-two"
    assert cli.main([*command, "--force"]) == 0
    pairing_flow.vm_id = "vm-one"
    assert cli.main([*command, "--force"]) == 0

    assert capsys.readouterr() == (paired + OFFER + paired + paired + OFFER + paired, "")
    assert terminal.reads == 2
    assert pairing_flow.pair_calls == [(False, 600), (True, 600), (True, 600), (True, 600)]
    assert pairing_flow.connected_vms == ["vm-one", "vm-two"]
    assert [session.posts for session in pairing_flow.sessions] == [[INTRODUCTION], [INTRODUCTION]]
    receipts = json.loads((tmp_path / "reachy-introduction.json").read_text())["receipts"]
    assert [(receipt["vm_id"], receipt["status"]) for receipt in receipts] == [
        ("vm-one", "sent"), ("vm-two", "sent")]


def test_declined_introduction_can_be_accepted_on_repair(
        tmp_path, monkeypatch, capsys, pairing_flow):
    terminal = Terminal("no\nyes\n")
    monkeypatch.setattr(cli.sys, "stdin", terminal)
    command = ["--state-dir", str(tmp_path), "pair"]
    paired = f"Reachy is paired with Muse. The pairing is saved in {tmp_path}.\n"

    assert cli.main(command) == 0
    assert not (tmp_path / "reachy-introduction.json").exists()
    assert cli.main([*command, "--force"]) == 0

    assert capsys.readouterr() == ((paired + OFFER) * 2, "")
    assert terminal.reads == 2
    assert [session.posts for session in pairing_flow.sessions] == [[INTRODUCTION]]


@pytest.mark.parametrize("outcome, message", [
    (PairOutcome.ALREADY_PAIRED, "already paired"),
    (PairOutcome.WINDOW_CLOSED, "Setup window closed without pairing. Run `muse-reachy pair` to try again.\n"),
    (PairOutcome.INVALID_SDK_TOKEN, "Reachy: invalid SDK token\n"),
])
@pytest.mark.parametrize("flags", [[], ["--introduce"]])
def test_failed_pair_does_not_offer_or_send(
        tmp_path, monkeypatch, capsys, pairing_flow, outcome, message, flags):
    pairing_flow.outcome = outcome
    terminal = Terminal("yes\n")
    monkeypatch.setattr(cli.sys, "stdin", terminal)

    assert cli.main(["--state-dir", str(tmp_path), "pair", *flags]) == 1

    if outcome is PairOutcome.ALREADY_PAIRED:
        message = (f"Reachy is already paired. The pairing is saved in {tmp_path}. "
                   "Pass --force to pair again.\n")
    assert capsys.readouterr() == ("", message)
    assert terminal.reads == 0
    assert pairing_flow.fetched == []
    assert pairing_flow.sessions == []
    assert not (tmp_path / "reachy-introduction.json").exists()


@pytest.mark.parametrize("manual_first", [False, True])
def test_pair_and_manual_introduction_share_receipts(
        tmp_path, monkeypatch, capsys, pairing_flow, manual_first):
    terminal = Terminal("yes\n")
    monkeypatch.setattr(cli.sys, "stdin", terminal)
    command = ["--state-dir", str(tmp_path)]
    paired = f"Reachy is paired with Muse. The pairing is saved in {tmp_path}.\n"
    if manual_first:
        config.save_json(config.PAIRING_FILE, {
            "access_token": "fake-token", "username": "paired-user"}, tmp_path)
        assert cli.main([*command, "introduce", "--yes"]) == 0
        assert cli.main([*command, "pair", "--force"]) == 0
        expected = INTRODUCTION + "\n" + paired
    else:
        assert cli.main([*command, "pair"]) == 0
        assert cli.main([*command, "introduce", "--yes"]) == 0
        expected = paired + OFFER + INTRODUCTION + "\n"

    assert capsys.readouterr() == (expected, "")
    assert terminal.reads == (0 if manual_first else 1)
    assert pairing_flow.connected_vms == ["vm-one"]
    assert [session.posts for session in pairing_flow.sessions] == [[INTRODUCTION]]


def test_pairing_changed_during_confirmation_does_not_send(
        tmp_path, monkeypatch, capsys, pairing_flow):
    terminal = Terminal("yes\n")
    readline = terminal.readline

    def replace_pairing(*args):
        config.save_json(config.PAIRING_FILE, {
            "access_token": "second-token", "username": "second-account"}, tmp_path)
        return readline(*args)

    monkeypatch.setattr(terminal, "readline", replace_pairing)
    monkeypatch.setattr(cli.sys, "stdin", terminal)

    assert cli.main(["--state-dir", str(tmp_path), "pair"]) == 1

    assert capsys.readouterr() == (
        f"Reachy is paired with Muse. The pairing is saved in {tmp_path}.\n" + OFFER,
        "Reachy: Reachy's introduction did not finish\n")
    assert terminal.reads == 1
    assert pairing_flow.sessions == []
    assert not (tmp_path / "reachy-introduction.json").exists()
    assert config.load_json(config.PAIRING_FILE, tmp_path) == {
        "access_token": "second-token", "username": "second-account"}
