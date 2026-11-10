# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0 (the "License").

import json
from pathlib import Path

from musegadget.identity import Identity
from musegadget.link_client import encode_message
from reachy_muse.conversation import ReachyService
from reachy_muse.contract.persona import REACHY_PERSONA


def test_exact_sdk_registration_and_byte_budget():
    service = ReachyService(identity=Identity("02:00:00:ab:cd:ef"), executor=object(), backends=lambda _: None)
    message = service.describe().register_message("00000000-0000-0000-0000-000000000000")
    expected = json.loads((Path(__file__).parent / "pins" / "registration.json").read_text())
    assert message == expected
    assert message["params"]["display_name"] == "Reachy Mini"
    commands = message["params"]["commands_v2"]
    assert set(commands) == {"reachy.say", "reachy.presence", "reachy.notify", "reachy.expression", "reachy.move",
                             "reachy.music.play", "reachy.music.stop", "reachy.music.status", "reachy.dance"}
    assert REACHY_PERSONA in commands["reachy.say"]["description"]
    assert {name: len(spec["description"].encode("utf-8")) for name, spec in commands.items()} == {
        "reachy.say": 1471, "reachy.presence": 537, "reachy.notify": 475, "reachy.expression": 446, "reachy.move": 393,
        "reachy.music.play": 178, "reachy.music.stop": 54, "reachy.music.status": 84, "reachy.dance": 120,
    }
    wire = encode_message(message)
    assert int.from_bytes(wire[:4], "little") == len(wire[4:]) == 6749
    assert len(wire) == 6753
    assert len(wire) < 8192


def test_longest_escaped_owner_place_stays_within_registration_budget():
    service = ReachyService(identity=Identity("02:00:00:ab:cd:ef"), executor=object(),
                           backends=lambda _: None, place='"' * 64)
    message = service.describe().register_message("00000000-0000-0000-0000-000000000000")
    assert len(encode_message(message)) == 7039
    assert len(encode_message(message)) < 8192
