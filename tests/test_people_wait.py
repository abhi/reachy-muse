"""How long reachy.people waits for the companion's voice label, on a fake clock.

Times are seconds after the request's last voiced sample. PersonGaze answers as in production; only its clock,
its poll sleep and what reaches it are scripted: Reachy admits the request, the request ends at the companion's
endpoint or at Reachy's own, and the companion's label lands in the request's slot.
"""

import asyncio
from types import SimpleNamespace

import pytest

from reachy_muse import person_gaze
from reachy_muse.companion_client import _SpeakerSlot
from reachy_muse.person_gaze import MuseActivity, PersonGaze
from reachy_muse.protocol import SpeakerLabel


class FakeClock:
    def __init__(self):
        self.now = -3.0
        self.events = []

    def monotonic(self):
        return self.now

    def advance(self, until):
        while self.events and self.events[0][0] <= until + 1e-9:
            at, action = self.events.pop(0)
            self.now = max(self.now, at)
            action()
        self.now = round(max(self.now, until), 6)

    async def sleep(self, seconds):
        self.advance(self.now + seconds)
        await asyncio.sleep(0)


def measure(monkeypatch, *, label_at, ended_at, asked_at, labels_voices=True):
    """Muse asks at ``asked_at`` about a request that ends at ``ended_at`` and whose label lands at ``label_at``;
    returns how long reachy.people waited and the speaker it answered."""
    clock = FakeClock()
    monkeypatch.setattr(person_gaze, "time", SimpleNamespace(monotonic=clock.monotonic))
    monkeypatch.setattr(person_gaze, "asyncio", SimpleNamespace(sleep=clock.sleep))
    hardware = SimpleNamespace(face_snapshot=lambda: SimpleNamespace(faces=(), captured_at=clock.now))
    hearing = []
    gaze = PersonGaze(hardware, hearing=lambda: hearing[0] if hearing else None)
    generation = gaze.begin()
    slot = _SpeakerSlot() if labels_voices else None
    gaze.opened(generation, slot, lambda: 1.0)
    hearing.append(generation)
    facts = {}

    def end():
        hearing.clear()
        gaze.requested(generation, 1.0)
        gaze.queued(generation, facts)
        if slot is not None:
            slot.note(facts)

    clock.events = sorted([(-2.5, lambda: gaze.admit(generation)), (ended_at, end)], key=lambda event: event[0])
    if label_at is not None and slot is not None:
        clock.events = sorted([*clock.events, (label_at, lambda: slot.arrived(SpeakerLabel.OWNER))],
                              key=lambda event: event[0])
    clock.advance(asked_at)
    result = asyncio.run(gaze.people(clock.now + 5, MuseActivity(), None))
    return round(clock.now - asked_at, 2), result["speaker"]


@pytest.mark.parametrize("label_at, ended_at, asked_at, labels_voices, answer", [
    pytest.param(.65, .94, .5, True, (.15, "owner"), id="healthy"),
    pytest.param(.65, .94, 1.5, True, (0.0, "owner"), id="healthy, asked after both"),
    pytest.param(.85, .75, .5, True, (.35, "owner"), id="endpoint first"),
    pytest.param(.65, 2.0, .5, True, (.15, "owner"), id="slow final pass"),
    pytest.param(None, 2.0, .5, True, (1.0, "unknown"), id="no label"),
    pytest.param(None, .94, .5, False, (0.0, "unknown"), id="no speaker op"),
])
def test_reachy_people_waits_until_the_label_lands_and_at_most_a_second(monkeypatch, label_at, ended_at, asked_at,
                                                                        labels_voices, answer):
    assert measure(monkeypatch, label_at=label_at, ended_at=ended_at, asked_at=asked_at,
                   labels_voices=labels_voices) == answer
