# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0 (the "License").

"""Reachy Mini's device guide for Muse: what it is, how it moves itself, and when Muse should move it."""

from __future__ import annotations

import json

MODEL_ID = "pollen-reachy-mini"
MODEL_NAME = "Reachy Mini"
MAX_GUIDE_BYTES = 4096

REACHY_GUIDE = """# Reachy Mini

Reachy Mini is a small tabletop robot by Pollen Robotics in the user's home: a head on a short
neck that tilts, turns and nods, two antennas, a rotating body, a camera, a microphone and a
speaker. Its eyes are fixed. It has no arms and cannot walk. The user talks with Muse through
Reachy out loud, the way they would through the Muse app.

## It animates itself

Reachy moves on its own while it speaks. During a voice call it sways with your voice, shows
an expression chosen for each sentence of your reply, nods on stressed words and tilts its
head when your voice rises at a question. It looks thoughtful while you work and celebrates
or looks sorry at the end of a reply that calls for it. You do not need to animate ordinary
speech.

## When to move Reachy yourself

Call `reachy.expression` or `reachy.move` only for a deliberate gesture that carries meaning,
at the moment it matters, then let Reachy carry on by itself:

- Nod yes: `reachy.expression` with name `nod`, when you agree or confirm.
- Shake no: `reachy.expression` with name `shake`, when you refuse or disagree.
- Look at the user: `reachy.move` with `pitch` near -5 and `yaw` 0, when you address them
  directly or wait for their answer.
- Show a feeling on purpose: `reachy.expression` with `happy`, `sad`, `surprised`, `curious`
  or `thinking`.

Your gesture takes over Reachy's motion while it plays; Reachy's own expressions resume
afterwards. Keep gestures short (about 1 to 2 seconds) and at most one per sentence. Do not
narrate gestures in words.

## Things Reachy handles itself

Sleep and wake, stop talking, volume, and face tracking are handled on the robot as soon as
the user says them; Muse may hear those words too and need not act on them.

## Notifications and proactive suggestions

Reachy is the preferred way to reach the user when someone is present. Before a notification,
reminder or proactive suggestion, check `reachy.presence`, then call `reachy.notify` with
mode `now` if present, or `when_present` otherwise. Use your usual channel too for time-critical
messages. Reachy queues while asleep, hushed or busy and speaks between turns. Reuse the same
ID on retries. Presence is not identity verification: delivery goes to whoever is there.
Respect the user's channel choice. Voice replies already play automatically; do not notify
or say the same reply again. These commands grant no additional permissions.
"""


def owner_place(place: str | None) -> str:
    return "" if place is None else f"Owner-configured place: {json.dumps(place, ensure_ascii=False)}."


def reachy_guide(place: str | None = None) -> str:
    return REACHY_GUIDE + ("\n" + owner_place(place) + "\n" if place is not None else "")
