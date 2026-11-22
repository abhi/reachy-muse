"""A spoken photo request through Reachy's whole conversation: capture, send, answer, and every failure."""

import asyncio
import contextlib
import functools
import logging
import os
import sys
import time
from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("av")

import reachy_muse.conversation
from reachy_muse.hardware import ReachyController
from reachy_muse.photo import JPEG_MAGIC, PhotoCapture, shutter_cue
from test_device_command_turns import SpeakerHardware, played, run, sent_to_muse, service
from test_person_gaze import people  # noqa: F401
from test_photo_capture import Encoder
from test_pins import FakeHardware, PinSession, Rig, Transcripts
from reachy_muse.realtime_call import realtime_supported
from test_realtime_turns import (CallHardware, ScriptedCalls, StreamingTranscripts, realtime_service,
                                 reply_events, voiced)

PHOTO = PhotoCapture(jpeg=JPEG_MAGIC + bytes(120_000 - len(JPEG_MAGIC)), width=1280, height=720, quality=80,
                     capture_ms=40, encode_ms=12)
realtime = pytest.mark.skipif(not realtime_supported(), reason="the Muse SDK has no voice call events")
PHOTO_MESSAGE = ("Reachy took this photo with its camera just now because the user asked. "
                 "The user's spoken request is: {}")


class PhotoCamera:
    """A camera that returns ``photos`` in turn, and a speaker that plays the shutter as itself."""

    def __init__(self, recording, photos=()):
        super().__init__(recording)
        self.photos = list(photos)
        self.captures = []

    def capture_photo(self, timeout_s):
        self.captures.append(timeout_s)
        photo = self.photos.pop(0)
        if callable(photo):
            return photo()
        return photo

    def play_audio(self, samples, **motion):
        cue = shutter_cue(self.output_sample_rate)
        if any(np.array_equal(cue[start:start + len(samples)], samples) for start in np.flatnonzero(cue == samples[0])):
            event = {"robot": "play", "audio": "shutter"}
            if self.recording.events[-1:] != [event]:
                self.recording.add(event)
            FakeHardware.play_audio(self, samples)
        else:
            super().play_audio(samples, **motion)


class PhotoHardware(PhotoCamera, SpeakerHardware):
    pass


class PhotoCallHardware(PhotoCamera, CallHardware):
    pass


def photo_rig(monkeypatch, photos=()):
    rig = Rig(monkeypatch)
    rig.hardware = PhotoHardware(rig.recording, photos)
    return rig


def sent(events):
    return [event for event in events if event.get("muse") in ("send_chat", "send_image", "send_voice")]


def photo_lines(caplog):
    return [record.getMessage() for record in caplog.records if record.getMessage().startswith("Reachy photo ")]


def fields(line):
    return dict(field.split("=", 1) for field in line.split()[2:])


def answers(*replies):
    async def muse(session, request):
        await session.answer(request, replies[request - 1])
    return muse


def test_a_spoken_photo_request_sends_one_photo_with_the_words_and_speaks_the_answer(monkeypatch, caplog):
    caplog.set_level(logging.DEBUG)
    rig = photo_rig(monkeypatch, [PHOTO])
    conversation = service(rig, answers("A red mug with a cat on it."), Transcripts("Reachy, what am I holding?"),
                           photo_requests=True)

    async def user():
        rig.say()

    events = run(rig, conversation, user)
    assert sent(events) == [{
        "muse": "send_image", "session_id": "robot-chat", "options": {},
        "text": PHOTO_MESSAGE.format("Reachy, what am I holding?"), "mime_type": "image/jpeg",
        "filename": "reachy-photo.jpg", "jpeg": True, "bytes": 120_000}]
    assert rig.hardware.captures == [2.0]
    assert played(events) == ["shutter", "tts:A red mug with a cat on it."]
    [line] = photo_lines(caplog)
    assert {key: value for key, value in fields(line).items() if key != "upload_ms"} == {
        "outcome": "sent", "path": "chat", "width": "1280", "height": "720", "quality": "80", "bytes": "120000",
        "capture_ms": "40", "encode_ms": "12"}
    assert fields(line)["upload_ms"].isdigit()
    assert any("photo=sent" in message for message in caplog.messages if message.startswith("Reachy turn timeline"))
    logged = "\n".join(record.getMessage() for record in caplog.records)
    for secret in ("holding", "Holding", "red mug", "/9j/", "\\xff\\xd8", "\xff\xd8"):
        assert secret not in logged


def test_the_controller_takes_the_photo_mid_turn_and_muse_gets_a_jpeg_within_its_limit(monkeypatch):
    rig = photo_rig(monkeypatch)
    reads = []
    controller = ReachyController(enable_motion=False,
                                  photo_encoder=Encoder({(1280, 80): 400_000, (1280, 70): 140_000}))
    controller._robot = SimpleNamespace(media=SimpleNamespace(
        get_frame=lambda: reads.append(1) or np.zeros((720, 1280, 3), dtype=np.uint8)))
    rig.hardware.vision_budget = controller.vision_budget
    rig.hardware.capture_photo = controller.capture_photo
    conversation = service(rig, answers("A blank wall."),
                           Transcripts("Muse, take a photo, then tell me what you see."), photo_requests=True)

    async def user():
        rig.say()

    events = run(rig, conversation, user)
    assert [(event["muse"], event["jpeg"], event["bytes"]) for event in sent(events)] == [
        ("send_image", True, 140_000)]
    assert reads == [1]
    assert played(events) == ["shutter", "tts:A blank wall."]


def test_without_the_flag_a_photo_request_is_an_ordinary_request(monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    rig = photo_rig(monkeypatch, [PHOTO])
    conversation = service(rig, answers("I can't see anything."), Transcripts("Reachy, what do you see?"))

    async def user():
        rig.say()

    events = run(rig, conversation, user)
    assert sent_to_muse(events) == ["Reachy, what do you see?"]
    assert rig.hardware.captures == []
    assert photo_lines(caplog) == []


def test_a_voice_other_than_the_owner_s_may_still_ask_for_a_photo(monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    monkeypatch.setattr("reachy_muse.local_backends._LocalEndedTurn.note_speaker",
                        lambda self, facts: facts.update(speaker="other"))
    rig = photo_rig(monkeypatch, [PHOTO])
    conversation = service(rig, answers("A plant."), Transcripts("What do you see, Reachy?"), photo_requests=True)

    async def user():
        rig.say()

    events = run(rig, conversation, user)
    assert [event["muse"] for event in sent(events)] == ["send_image"]
    assert any("speaker=other" in message and "photo=sent" in message for message in caplog.messages)


def test_a_second_photo_request_while_muse_answers_the_first_gets_the_busy_line(monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    rig = photo_rig(monkeypatch, [PHOTO, PHOTO])

    async def muse(session, request):
        await rig.wait_until_or_timeout(lambda: "tts:One at a time, please." in played(rig.recording.events), 10)
        await session.answer(request, "A red mug.")

    conversation = service(rig, muse, Transcripts("Reachy, what am I holding?", "Reachy, take a photo."),
                           photo_requests=True)

    async def user():
        rig.say()
        await rig.quiet("thinking")
        rig.say()

    events = run(rig, conversation, user)
    assert [event["muse"] for event in sent(events)] == ["send_image"]
    assert rig.hardware.captures == [2.0]
    assert played(events) == ["shutter", "tts:One at a time, please.", "tts:A red mug."]
    assert [fields(line)["outcome"] for line in photo_lines(caplog)] == ["busy", "sent"]


def test_a_photo_request_while_muse_works_on_another_request_gets_the_busy_line(monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    rig = photo_rig(monkeypatch, [PHOTO])

    async def muse(session, request):
        await rig.wait_until_or_timeout(lambda: "tts:One at a time, please." in played(rig.recording.events), 10)
        await session.answer(request, "Sunny tomorrow.")

    conversation = service(rig, muse, Transcripts("What's the weather tomorrow?", "Reachy, what do you see?"),
                           photo_requests=True)

    async def user():
        rig.say()
        await rig.quiet("thinking")
        rig.say()

    events = run(rig, conversation, user)
    assert sent_to_muse(events) == ["What's the weather tomorrow?"]
    assert rig.hardware.captures == []
    assert played(events) == ["tts:Hmm, good question.", "tts:One at a time, please.", "tts:Sunny tomorrow."]
    assert [fields(line) for line in photo_lines(caplog)] == [{
        "outcome": "busy", "path": "chat", "width": "none", "height": "none", "quality": "none", "bytes": "none",
        "capture_ms": "none", "encode_ms": "none", "upload_ms": "none"}]


class RejectsPhotos(PinSession):
    async def send_image(self, image, mime_type, session_id=None, **options):
        self.recording.add({"muse": "send_image", "bytes": len(image)})
        return {"ok": False, "status": 413}


class PhotosUnacknowledged(PinSession):
    async def send_image(self, image, mime_type, session_id=None, **options):
        self.recording.add({"muse": "send_image", "bytes": len(image)})
        return {"ok": True, "status": 200, "response": None}


class PhotosNeverArrive(PinSession):
    async def send_image(self, image, mime_type, session_id=None, **options):
        self.recording.add({"muse": "send_image", "bytes": len(image)})
        await asyncio.Event().wait()


def slowly():
    time.sleep(1)
    return PHOTO


@pytest.mark.parametrize("case, photos, line, shutter", [
    ("camera_busy", [PhotoCapture("camera_busy")], "I couldn't take a photo.", False),
    ("no_frame", [PhotoCapture("no_frame", capture_ms=500)], "I couldn't take a photo.", False),
    ("camera_failed", [PhotoCapture("camera_failed", capture_ms=3)], "I couldn't take a photo.", False),
    ("too_large", [PhotoCapture("too_large", capture_ms=40, encode_ms=90)], "That photo was too large to send.", False),
    ("slow_camera", [slowly], "I couldn't take a photo.", False),
    ("rejected", [PHOTO], "Muse didn't get the photo.", True),
    ("unacknowledged", [PHOTO], "Muse didn't get the photo.", True),
    ("send_timeout", [PHOTO], "Muse didn't get the photo.", True),
    ("no_answer", [PHOTO], "Muse didn't answer about the photo.", True),
    ("expired", [PHOTO], "Please ask me again for the photo.", False),
])
def test_each_failure_gets_one_line_and_the_conversation_goes_on(monkeypatch, caplog, case, photos, line, shutter):
    caplog.set_level(logging.INFO)
    rig = photo_rig(monkeypatch, photos)
    if case == "slow_camera":
        monkeypatch.setattr(reachy_muse.conversation, "PHOTO_CAPTURE_S", .05)
    if case == "send_timeout":
        monkeypatch.setattr(reachy_muse.conversation, "PHOTO_SEND_TIMEOUT_S", .2)
    if case == "no_answer":
        monkeypatch.setattr(reachy_muse.conversation, "VoiceConversation",
                            functools.partial(reachy_muse.conversation.VoiceConversation, reply_timeout_s=.5))
    if case == "expired":
        monkeypatch.setattr("reachy_muse.photo.PHOTO_REQUEST_TTL_S", 0.0)

    async def muse(session, request):
        if not (case == "no_answer" and request == 1):
            await session.answer(request, "It's noon.")

    conversation = service(rig, muse, Transcripts("Reachy, what do you see?", "What time is it?"), photo_requests=True)
    if case in ("rejected", "unacknowledged", "send_timeout"):
        rig.session = {"rejected": RejectsPhotos, "unacknowledged": PhotosUnacknowledged,
                       "send_timeout": PhotosNeverArrive}[case](rig.recording, muse)

    async def user():
        rig.say()
        await rig.wait_until_or_timeout(lambda: f"tts:{line}" in played(rig.recording.events), 5)
        await rig.quiet()
        rig.say()

    events = run(rig, conversation, user)
    assert played(events) == ["shutter"] * shutter + [f"tts:{line}", "tts:It's noon."]
    assert sent_to_muse(events) == ["What time is it?"]
    assert rig.hardware.captures == ([] if case == "expired" else [.05 if case == "slow_camera" else 2.0])
    assert [fields(entry)["outcome"] for entry in photo_lines(caplog)] == [
        {"slow_camera": "camera_busy", "unacknowledged": "rejected"}.get(case, case)]


def test_a_photo_turn_writes_no_file(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    rig = photo_rig(monkeypatch, [PHOTO])
    conversation = service(rig, answers("A desk."), Transcripts("Reachy, what do you see?"), photo_requests=True)

    async def user():
        rig.say()

    with writes_recorded() as writes:
        events = run(rig, conversation, user)
    assert [event["muse"] for event in sent(events)] == ["send_image"]
    assert writes == []
    assert list(tmp_path.iterdir()) == []


_WRITES = []
_RECORDING = []


def _record_write(event, args):
    if _RECORDING and event == "open":
        path, mode, flags = args
        if any(char in (mode or "") for char in "wax+") or (flags or 0) & (os.O_WRONLY | os.O_RDWR | os.O_CREAT):
            _WRITES.append(path)


@contextlib.contextmanager
def writes_recorded():
    """Every file this process opens for writing meanwhile; an audit hook cannot be removed, so it idles after."""
    if not _RECORDING:
        sys.addaudithook(_record_write)
    _WRITES.clear()
    _RECORDING.append(True)
    try:
        yield _WRITES
    finally:
        _RECORDING.pop()


@pytest.mark.usefixtures("paired_inbox")
def test_muse_commands_presence_briefings_and_notifications_never_take_a_photo(monkeypatch):
    from reachy_muse.arrival import PersonBox
    from test_morning_turns import Camera, briefing, morning

    rig = photo_rig(monkeypatch)
    controller = ReachyController(enable_motion=False, photo_encoder=Encoder())
    controller._robot = SimpleNamespace(media=SimpleNamespace(get_frame=lambda: np.zeros((24, 32, 3), np.uint8)))
    controller._presence_detector = SimpleNamespace(detect=lambda image: [PersonBox(0, 0, .2, .3, .9)])
    for name in ("presence", "presence_observations", "run_command"):
        setattr(rig.hardware, name, getattr(controller, name))
    camera = Camera()
    conversation = service(rig, briefing, Transcripts(), morning=morning(camera), photo_requests=True)
    results = {}

    async def user():
        await rig.wait_until_or_timeout(lambda: sent_to_muse(rig.recording.events), 5)
        camera.arrive()
        await rig.wait_until_or_timeout(lambda: camera.calls == ["start", "stop"], 10)
        await rig.quiet()
        for name, params in (("camera.capture", {}), ("reachy.photo", {"text": "What do you see?"}),
                             ("reachy.presence", {}),
                             ("reachy.notify", {"id": "n1", "text": "Your package arrived.", "mode": "now",
                                                "kind": "notification"})):
            results[name] = await asyncio.to_thread(conversation.run_command, name, params, 5000)
        await rig.quiet()

    events = run(rig, conversation, user)
    assert rig.hardware.captures == []
    assert {name: result.get("error") or result["payload"].get("status") or result["payload"]["present"]
            for name, result in results.items()} == {
        "camera.capture": "unsupported command: camera.capture", "reachy.photo": "unsupported command: reachy.photo",
        "reachy.presence": True, "reachy.notify": "delivered"}
    assert sent_to_muse(events) == ["Brief me."]
    assert played(events)[-1] == "tts:Your package arrived."


def call_rig(monkeypatch, photos=()):
    rig = Rig(monkeypatch)
    rig.hardware = PhotoCallHardware(rig.recording, photos)
    return rig


def answers_the_next_call_turn(spoken):
    """Muse's call answer to the photo request is the SDK's to drop after the barge-in; this one answers later."""
    async def answer(sdk, emit, script):
        await spoken.wait()
        reply_events(sdk, emit, "r2", "It will be sunny.", voiced(.4))
    return answer


def realtime_command_lines(caplog):
    return [message for message in caplog.messages if message.startswith("Realtime command ")]


@realtime
def test_a_photo_request_in_a_call_takes_the_turn_from_muse_and_the_call_stays_open(monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    rig = call_rig(monkeypatch, [PHOTO])
    asked_again = asyncio.Event()
    calls = ScriptedCalls(answer=answers_the_next_call_turn(asked_again))
    conversation = realtime_service(rig, StreamingTranscripts("Hey Muse, what am I holding?",
                                                              "Will it be sunny tomorrow?"),
                                    calls, muse=answers("A red mug."), photo_requests=True, wake_timeout_s=5)
    observed = []

    async def user():
        rig.say(wake=True)
        await rig.wait_until_or_timeout(lambda: "tts:A red mug." in played(rig.recording.events), 5)
        voice = conversation._voice[1]
        call, frames = voice.call, len(calls.frames)
        await rig.quiet("listening", "thinking")
        observed.append((call.turns, list(call.pending)))
        rig.say()
        await rig.wait_until_or_timeout(lambda: len(calls.frames) > frames + 20, 3)
        asked_again.set()
        await rig.wait_until_or_timeout(lambda: "muse" in played(rig.recording.events), 5)
        observed.append((voice.call is call, call.done, call.turns, list(call.pending)))

    events = run(rig, conversation, user, rest=("listening", "idle"))
    assert calls.barge_ins == 1
    assert [(event["muse"], event["text"]) for event in sent(events)] == [
        ("send_image", PHOTO_MESSAGE.format("what am I holding?"))]
    assert played(events) == ["shutter", "tts:A red mug.", "muse"]
    assert observed == [(0, []), (True, False, 1, [])]
    assert realtime_command_lines(caplog) == [
        "Realtime command transcript=arrived words=6 wake_prefix=exact outcome=PHOTO drop=none",
        "Realtime command transcript=arrived words=5 wake_prefix=none outcome=none drop=none"]
    assert [(fields(line)["outcome"], fields(line)["path"]) for line in photo_lines(caplog)] == [("sent", "realtime")]


@realtime
def test_a_second_photo_request_in_a_call_gets_the_busy_line(monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    rig = call_rig(monkeypatch, [PHOTO, PHOTO])

    async def muse(session, request):
        await rig.wait_until_or_timeout(lambda: "tts:One at a time, please." in played(rig.recording.events), 10)
        await session.answer(request, "A red mug.")

    calls = ScriptedCalls(reply=None)
    conversation = realtime_service(rig, StreamingTranscripts("Hey Muse, what am I holding?", "Reachy, take a photo."),
                                    calls, muse=muse, photo_requests=True, wake_timeout_s=5)

    async def user():
        rig.say(wake=True)
        await rig.quiet("thinking")
        rig.say()

    events = run(rig, conversation, user, rest=("listening", "idle"))
    assert calls.barge_ins == 2
    assert rig.hardware.captures == [2.0]
    assert [event["muse"] for event in sent(events)] == ["send_image"]
    assert played(events) == ["shutter", "tts:One at a time, please.", "tts:A red mug."]
    assert [(fields(line)["outcome"], fields(line)["path"]) for line in photo_lines(caplog)] == [
        ("busy", "realtime"), ("sent", "realtime")]
    assert [line.split(" outcome=")[1] for line in realtime_command_lines(caplog)] == [
        "PHOTO drop=none", "PHOTO drop=none"]


@realtime
@pytest.mark.parametrize("case, photos, line", [
    ("camera_failed", [PhotoCapture("camera_failed", capture_ms=3)], "I couldn't take a photo."),
    ("too_large", [PhotoCapture("too_large", capture_ms=40, encode_ms=90)], "That photo was too large to send."),
    ("slow_camera", [slowly], "I couldn't take a photo."),
    ("rejected", [PHOTO], "Muse didn't get the photo."),
    ("send_timeout", [PHOTO], "Muse didn't get the photo."),
    ("no_answer", [PHOTO], "Muse didn't answer about the photo."),
    ("expired", [PHOTO], "Please ask me again for the photo."),
])
def test_each_failure_in_a_call_gets_one_line_and_the_call_stays_open(monkeypatch, caplog, case, photos, line):
    caplog.set_level(logging.INFO)
    rig = call_rig(monkeypatch, photos)
    if case == "slow_camera":
        monkeypatch.setattr(reachy_muse.conversation, "PHOTO_CAPTURE_S", .05)
    if case == "send_timeout":
        monkeypatch.setattr(reachy_muse.conversation, "PHOTO_SEND_TIMEOUT_S", .2)
    if case == "no_answer":
        monkeypatch.setattr(reachy_muse.conversation, "VoiceConversation",
                            functools.partial(reachy_muse.conversation.VoiceConversation, reply_timeout_s=.5))
    if case == "expired":
        monkeypatch.setattr("reachy_muse.photo.PHOTO_REQUEST_TTL_S", 0.0)
    asked_again = asyncio.Event()
    calls = ScriptedCalls(answer=answers_the_next_call_turn(asked_again))

    async def muse(session, request):
        if case != "no_answer":
            raise AssertionError("only the photo request goes to Muse by chat")

    conversation = realtime_service(rig, StreamingTranscripts("Hey Muse, what do you see?", "Will it be sunny?"),
                                    calls, muse=muse, photo_requests=True, wake_timeout_s=5)
    if case in ("rejected", "send_timeout"):
        rig.session = (RejectsPhotos if case == "rejected" else PhotosNeverArrive)(rig.recording, muse)
    observed = []

    async def user():
        rig.say(wake=True)
        await rig.wait_until_or_timeout(lambda: f"tts:{line}" in played(rig.recording.events), 5)
        voice = conversation._voice[1]
        call, frames = voice.call, len(calls.frames)
        await rig.quiet("listening", "thinking")
        rig.say()
        await rig.wait_until_or_timeout(lambda: len(calls.frames) > frames + 20, 3)
        asked_again.set()
        await rig.wait_until_or_timeout(lambda: "muse" in played(rig.recording.events), 5)
        observed.append((voice.call is call, call.done))

    events = run(rig, conversation, user, rest=("listening", "idle"))
    assert calls.barge_ins == 1
    assert played(events) == ["shutter"] * (case in ("rejected", "send_timeout", "no_answer")) + [f"tts:{line}", "muse"]
    assert observed == [(True, False)]
    assert sent_to_muse(events) == []
    assert [fields(entry)["outcome"] for entry in photo_lines(caplog)] == [
        "camera_busy" if case == "slow_camera" else case]


@realtime
def test_a_photo_request_whose_words_arrive_too_late_stays_with_muse(monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    rig = call_rig(monkeypatch, [PHOTO])

    class LateWords(StreamingTranscripts):
        def open_turn(self):
            turn = super().open_turn()
            text = turn.partial.text

            def finish():
                async def recognize():
                    await asyncio.sleep(1)
                    return text
                return asyncio.ensure_future(recognize())

            turn.finish = finish
            return turn

    calls = ScriptedCalls(reply="I can't see, but I can listen.")
    conversation = realtime_service(rig, LateWords("Hey Muse, what do you see?"), calls, photo_requests=True)

    async def user():
        rig.say(wake=True)
        await rig.wait_until_or_timeout(lambda: "muse" in played(rig.recording.events), 5)

    events = run(rig, conversation, user)
    assert rig.hardware.captures == []
    assert sent(events) == []
    assert set(played(events)) == {"muse"}
    assert calls.barge_ins == 0
    assert realtime_command_lines(caplog) == [
        "Realtime command transcript=deadline words=0 wake_prefix=none outcome=none drop=deadline"]
    assert photo_lines(caplog) == []


@realtime
def test_a_photo_request_releases_a_held_person_gaze_and_shows_what_reachy_faces_then(people, caplog):
    import math
    import threading
    from test_person_gaze import DelayedWake, say_live
    from test_pins import CHUNK, WAKE_SAMPLE
    from test_realtime_turns import room

    rig, controller, robot, scene = people
    caplog.set_level(logging.INFO)
    controller._photo_encoder = Encoder()
    photo_yaws, capturing = [], threading.Event()
    read_frame = robot.media.get_frame

    def get_frame():
        if capturing.is_set():
            photo_yaws.append(round(math.degrees(controller.head_yaw)))
        return read_frame()

    def capture_photo(timeout_s):
        capturing.set()
        try:
            return controller.capture_photo(timeout_s)
        finally:
            capturing.clear()

    robot.media.get_frame = get_frame
    rig.hardware.capture_photo = capture_photo

    async def greets(sdk, emit, script):
        await rig.wait_until_or_timeout(lambda: controller._person_hold, 3)
        reply_events(sdk, emit, "r1", "Hello there!", voiced(.3))

    calls = ScriptedCalls(answer=greets)
    conversation = service(rig, answers("A desk and a lamp."), StreamingTranscripts("Say hi to my friend.",
                                                                                     "What do you see, Reachy?"),
                           wake_detector=DelayedWake(), wake_timeout_s=5, muse_path="realtime",
                           voice_link=calls.link, voice_call=calls, photo_requests=True)
    conversation.silence_s = .8

    def facing():
        return round(math.degrees(controller.head_yaw)), round(math.degrees(robot.body_yaw))

    async def user():
        for _ in range(30):
            rig.hardware.samples.put(np.full(CHUNK, WAKE_SAMPLE, np.float32))
            await asyncio.sleep(.02)
        await rig.wait_until_or_timeout(lambda: facing() == (30, 10))
        await say_live(rig, paced_silence_s=.9)
        await rig.wait_until_or_timeout(lambda: facing() == (-25, -5))
        assert facing() == (-25, -5)
        await rig.wait_until_or_timeout(lambda: "muse" in played(rig.recording.events))
        await rig.wait_until_or_timeout(lambda: not conversation._voice[1].call.audible, 4)
        await room(rig, .3)
        await say_live(rig, paced_silence_s=.9)
        await rig.wait_until_or_timeout(lambda: "tts:A desk and a lamp." in played(rig.recording.events), 5)

    events = run(rig, conversation, user)
    assert photo_yaws == [30]
    assert [event["muse"] for event in sent(events)] == ["send_image"]
    assert played(events)[-1] == "tts:A desk and a lamp."
    assert [line for line in caplog.messages if line.startswith("Reachy gaze request")] == [
        "Reachy gaze request outcome=turned source=fresh faces=2"]


def test_reachy_s_own_words_heard_back_never_take_a_photo(monkeypatch, caplog):
    from test_device_command_turns import LongVoice

    caplog.set_level(logging.INFO)
    rig = photo_rig(monkeypatch, [PHOTO])
    rig.hardware.chunk_s = .1
    rig.hardware.echo_cancelled_input = True
    conversation = service(rig, answers("I can't tell from here. Ask me: Reachy, what do you see?"),
                           Transcripts("Is it sunny outside?", "Reachy, what do you see?"),
                           voice=LongVoice(rig.recording, chunks=20), photo_requests=True)

    async def user():
        rig.say()
        await rig.wait_until_or_timeout(lambda: played(rig.recording.events), 5)
        rig.say()

    events = run(rig, conversation, user)
    assert rig.hardware.captures == []
    assert [event["muse"] for event in sent(events)] == ["send_chat"]
    assert photo_lines(caplog) == []
    assert any("ignored" in message and "heard over its own speech" in message for message in caplog.messages)


def test_a_photo_turn_cut_short_when_the_session_ends_still_logs_its_one_line(monkeypatch, caplog):
    import threading

    caplog.set_level(logging.INFO)
    capturing = threading.Event()

    def until_the_session_ends():
        capturing.set()
        time.sleep(1.5)
        return PHOTO

    rig = photo_rig(monkeypatch, [until_the_session_ends])
    conversation = service(rig, answers("A lamp."), Transcripts("Reachy, what do you see?"), photo_requests=True)

    async def user():
        rig.say()
        await rig.wait_until_or_timeout(capturing.is_set, 5)

    events = run(rig, conversation, user, rest=("thinking",))
    assert sent(events) == []
    assert played(events) == []
    assert [fields(line)["outcome"] for line in photo_lines(caplog)] == ["interrupted"]


@realtime
def test_never_mind_in_a_call_drops_the_photo_muse_is_still_answering(monkeypatch, caplog):
    from reachy_muse.contract.prompts import STOP_REQUEST

    caplog.set_level(logging.INFO)
    rig = call_rig(monkeypatch, [PHOTO])

    async def muse(session, request):
        if request == 1:
            await rig.wait_until_or_timeout(lambda: session.requests == 2, 3)
            await session.answer(request, "A red mug.")
        else:
            await session.answer(request, "Okay, I stopped.")

    asked_again = asyncio.Event()

    async def answers_the_cancel(sdk, emit, script):
        await asked_again.wait()
        reply_events(sdk, emit, "r2", "Okay, never mind.", voiced(.4))

    calls = ScriptedCalls(answer=answers_the_cancel)
    conversation = realtime_service(rig, StreamingTranscripts("Hey Muse, what am I holding?", "Never mind."), calls,
                                    muse=muse, photo_requests=True, wake_timeout_s=5)

    async def user():
        rig.say(wake=True)
        await rig.quiet("thinking")
        frames = len(calls.frames)
        rig.say()
        await rig.wait_until_or_timeout(lambda: len(calls.frames) > frames + 20, 3)
        asked_again.set()
        await rig.wait_until_or_timeout(lambda: "muse" in played(rig.recording.events), 5)

    events = run(rig, conversation, user, rest=("listening", "idle"))
    assert [(event["muse"], event["text"]) for event in sent(events)] == [
        ("send_image", PHOTO_MESSAGE.format("what am I holding?")), ("send_chat", STOP_REQUEST)]
    assert played(events) == ["shutter", "muse"]
    assert [fields(line)["outcome"] for line in photo_lines(caplog)] == ["sent"]


@pytest.mark.parametrize("said, photo_words", [
    ("What do you see?", None),
    ("Hey Muse, what do you see?", "what do you see?"),
    ("What do you see, Reachy?", "What do you see, Reachy?"),
])
def test_inside_an_open_window_only_words_spoken_to_reachy_take_a_photo(monkeypatch, said, photo_words):
    from test_pins import Wake

    rig = photo_rig(monkeypatch, [PHOTO])
    conversation = service(rig, answers("Sunny tomorrow.", "A desk."),
                           Transcripts("Hey Muse, what's the weather tomorrow?", said),
                           wake_detector=Wake(), wake_timeout_s=30, photo_requests=True)

    async def user():
        rig.say(wake=True)
        await rig.wait_until_or_timeout(lambda: "tts:Sunny tomorrow." in played(rig.recording.events), 5)
        await rig.quiet("listening")
        rig.say()

    events = run(rig, conversation, user, rest=("listening",))
    second = ("send_image", PHOTO_MESSAGE.format(photo_words)) if photo_words else ("send_chat", said)
    assert [(event["muse"], event["text"]) for event in sent(events)] == [
        ("send_chat", "what's the weather tomorrow?"), second]
    assert rig.hardware.captures == ([2.0] if photo_words else [])


@realtime
def test_in_a_call_a_photo_phrase_not_spoken_to_reachy_stays_with_muse(monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    rig = call_rig(monkeypatch, [PHOTO])
    calls = ScriptedCalls(reply=None)
    conversation = realtime_service(rig, StreamingTranscripts("Hey Muse, tell me a joke.", "What do you see?"), calls,
                                    photo_requests=True, wake_timeout_s=5)

    async def user():
        rig.say(wake=True)
        await rig.wait_until_or_timeout(lambda: len(realtime_command_lines(caplog)) == 1, 5)
        await rig.quiet("thinking", "listening")
        rig.say()
        await rig.wait_until_or_timeout(lambda: len(realtime_command_lines(caplog)) == 2, 5)

    events = run(rig, conversation, user, rest=("thinking", "listening", "idle"))
    assert (rig.hardware.captures, sent(events), calls.barge_ins) == ([], [], 0)
    assert [line.split(" outcome=")[1] for line in realtime_command_lines(caplog)] == [
        "none drop=none", "none drop=none"]
    assert photo_lines(caplog) == []


def capture_held_until(released):
    """A capture that lasts, as a slow head settle does, until ``released``; and when it began."""
    import threading

    started = threading.Event()

    def capture():
        started.set()
        released.wait(5)
        return PHOTO
    return started, capture


@pytest.mark.parametrize("cancel", ["Never mind.", "Stop talking."])
def test_a_cancel_heard_while_the_photo_is_taken_keeps_it_home(monkeypatch, caplog, cancel):
    import threading

    caplog.set_level(logging.INFO)
    released = threading.Event()
    capturing, capture = capture_held_until(released)
    rig = photo_rig(monkeypatch, [capture])
    conversation = service(rig, answers("A red mug."), Transcripts("Reachy, what am I holding?", cancel),
                           photo_requests=True)

    async def user():
        rig.say()
        await rig.wait_until_or_timeout(capturing.is_set, 5)
        rig.say()
        await rig.wait_until_or_timeout(lambda: "tts:Okay, I'll drop that." in played(rig.recording.events), 5)
        released.set()

    try:
        events = run(rig, conversation, user)
    finally:
        released.set()
    assert sent(events) == []
    assert played(events) == ["tts:Okay, I'll drop that."]
    assert [fields(line)["outcome"] for line in photo_lines(caplog)] == ["cancelled"]


@realtime
@pytest.mark.parametrize("cancel, command", [
    ("Never mind.", "none"), ("Go to sleep.", "SLEEP"), ("Stop listening.", "STOP_LISTENING")])
def test_a_cancel_said_in_a_call_while_the_photo_is_taken_keeps_it_home(monkeypatch, caplog, cancel, command):
    import threading

    caplog.set_level(logging.INFO)
    released = threading.Event()
    capturing, capture = capture_held_until(released)
    rig = call_rig(monkeypatch, [capture])
    calls = ScriptedCalls(reply=None)
    conversation = realtime_service(rig, StreamingTranscripts("Hey Muse, what am I holding?", cancel), calls,
                                    muse=answers("A red mug."), photo_requests=True, wake_timeout_s=5)

    async def user():
        rig.say(wake=True)
        await rig.wait_until_or_timeout(capturing.is_set, 5)
        rig.say()
        await rig.wait_until_or_timeout(lambda: len(realtime_command_lines(caplog)) == 2, 5)
        released.set()
        await rig.wait_until_or_timeout(lambda: photo_lines(caplog), 5)

    try:
        events = run(rig, conversation, user, rest=("listening", "idle", "sleeping", "thinking"))
    finally:
        released.set()
    assert sent(events) == []
    assert "shutter" not in played(events)
    assert realtime_command_lines(caplog)[1].endswith(f"outcome={command} drop=none")
    assert [fields(line)["outcome"] for line in photo_lines(caplog)] == ["cancelled"]


# What the user said, and the companion's partial at the robot's endpoint, which lacks the tail.
LAGGING = [
    ("take a photo of me in five minutes", "take a photo of me"),
    ("take a photo and send it to my wife", "take a photo"),
    ("take a picture of this for my landlord", "take a picture of this"),
    ("take a photo, no wait, don't", "take a photo"),
]


def test_each_lagging_partial_would_take_a_photo_that_the_whole_request_refuses():
    from reachy_muse.device_command import photo_requested

    assert [(photo_requested(said), photo_requested(partial)) for said, partial in LAGGING] == [(False, True)] * 4


@realtime
@pytest.mark.parametrize("said, partial", LAGGING)
def test_a_call_photo_needs_the_companion_s_final_not_a_lagging_partial(monkeypatch, caplog, said, partial):
    from dataclasses import replace
    from reachy_muse.companion_client import CompanionHearing, CompanionLink, LinkState, _Connection
    from reachy_muse.protocol import HearEndpoint, HearPartial, Op, encode

    caplog.set_level(logging.INFO)
    rig = call_rig(monkeypatch, [PHOTO])
    calls = ScriptedCalls(reply=None)
    link = CompanionLink("ws://localhost:9", token="test", out_rate=16000, insecure=True)
    link._connection = connection = _Connection(None, (Op.HEAR,))
    link._set(LinkState.UP)
    opened = link.open_hear

    def open_hear(outcomes=None):
        stream = opened(outcomes)
        link._receive(connection, encode(HearPartial(stream.id, 0, partial)))
        finish = stream.finish

        def finish_with_the_final():
            finish()
            asyncio.get_running_loop().call_later(
                .3, lambda: link._receive(connection, encode(HearEndpoint(stream.id, said, True))))
        stream.finish = finish_with_the_final
        return stream

    link.open_hear = open_hear
    conversation = realtime_service(rig, None, calls, muse=answers("A face."), photo_requests=True, wake_timeout_s=5)
    backends = conversation.backends
    conversation.backends = lambda session: replace(backends(session), hearing=CompanionHearing(link))

    async def user():
        rig.say(wake=True)
        await rig.wait_until_or_timeout(lambda: realtime_command_lines(caplog), 5)
        await asyncio.sleep(1)

    events = run(rig, conversation, user, rest=("thinking", "listening", "idle"))
    assert (rig.hardware.captures, sent(events), photo_lines(caplog)) == ([], [], [])
    assert realtime_command_lines(caplog)[0].endswith("outcome=none drop=none")


def test_a_photo_request_decided_while_the_user_still_speaks_is_an_ordinary_request(monkeypatch):
    from test_pins import CHUNK, VOICE_SAMPLE

    waiting, released = asyncio.Event(), asyncio.Event()

    class FirstWordsLate(Transcripts):
        async def transcribe(self, wav):
            if len(self.texts) == 2:
                waiting.set()
                await released.wait()
            return await super().transcribe(wav)

    rig = photo_rig(monkeypatch, [PHOTO])
    conversation = service(rig, answers("I can't see it.", "Okay."),
                           FirstWordsLate("Reachy, what do you see?", "It is on the table."), photo_requests=True)

    async def user():
        rig.say()
        await asyncio.wait_for(waiting.wait(), 5)
        for index in range(40):
            rig.hardware.samples.put(np.full(CHUNK, VOICE_SAMPLE, np.float32))
            await asyncio.sleep(CHUNK / 16000)
            if index == 20:
                released.set()
        rig.say(voiced_s=0)

    events = run(rig, conversation, user)
    assert rig.hardware.captures == []
    assert [(event["muse"], event["text"]) for event in sent(events)][0] == ("send_chat", "Reachy, what do you see?")
