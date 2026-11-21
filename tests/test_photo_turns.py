"""A spoken photo request through Reachy's whole conversation: capture, send, answer, and every failure."""

import asyncio
import functools
import logging
import time
from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("av")

import reachy_muse.conversation
from reachy_muse.hardware import ReachyController
from reachy_muse.photo import JPEG_MAGIC, PhotoCapture, shutter_cue
from test_device_command_turns import SpeakerHardware, played, run, sent_to_muse, service
from test_photo_capture import Encoder
from test_pins import FakeHardware, PinSession, Rig, Transcripts

PHOTO = PhotoCapture(jpeg=JPEG_MAGIC + bytes(120_000 - len(JPEG_MAGIC)), width=1280, height=720, quality=80,
                     capture_ms=40, encode_ms=12)
PHOTO_MESSAGE = ("Reachy took this photo with its camera just now because the user asked. "
                 "The user's spoken request is: {}")


class PhotoHardware(SpeakerHardware):
    """The rig's microphone and speaker, with a camera that returns ``photos`` in turn; the shutter plays as itself."""

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

    def play_audio(self, samples):
        cue = shutter_cue(self.output_sample_rate)
        if any(np.array_equal(cue[start:start + len(samples)], samples) for start in np.flatnonzero(cue == samples[0])):
            event = {"robot": "play", "audio": "shutter"}
            if self.recording.events[-1:] != [event]:
                self.recording.add(event)
            FakeHardware.play_audio(self, samples)
        else:
            super().play_audio(samples)


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
    conversation = service(rig, answers("A red mug with a cat on it."), Transcripts("What am I holding?"),
                           photo_requests=True)

    async def user():
        rig.say()

    events = run(rig, conversation, user)
    assert sent(events) == [{
        "muse": "send_image", "session_id": "robot-chat", "options": {},
        "text": PHOTO_MESSAGE.format("What am I holding?"), "mime_type": "image/jpeg",
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
    conversation = service(rig, answers("A blank wall."), Transcripts("Take a photo, then tell me what you see."),
                           photo_requests=True)

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
    conversation = service(rig, answers("I can't see anything."), Transcripts("What do you see?"))

    async def user():
        rig.say()

    events = run(rig, conversation, user)
    assert sent_to_muse(events) == ["What do you see?"]
    assert rig.hardware.captures == []
    assert photo_lines(caplog) == []


def test_a_voice_other_than_the_owner_s_may_still_ask_for_a_photo(monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    monkeypatch.setattr("reachy_muse.local_backends._LocalEndedTurn.note_speaker",
                        lambda self, facts: facts.update(speaker="other"))
    rig = photo_rig(monkeypatch, [PHOTO])
    conversation = service(rig, answers("A plant."), Transcripts("What do you see?"), photo_requests=True)

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

    conversation = service(rig, muse, Transcripts("What am I holding?", "Take a photo."), photo_requests=True)

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

    conversation = service(rig, muse, Transcripts("What's the weather tomorrow?", "What do you see?"),
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

    conversation = service(rig, muse, Transcripts("What do you see?", "What time is it?"), photo_requests=True)
    if case in ("rejected", "send_timeout"):
        rig.session = (RejectsPhotos if case == "rejected" else PhotosNeverArrive)(rig.recording, muse)

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
        "camera_busy" if case == "slow_camera" else case]


def test_a_photo_turn_writes_no_file(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    rig = photo_rig(monkeypatch, [PHOTO])
    conversation = service(rig, answers("A desk."), Transcripts("What do you see?"), photo_requests=True)

    async def user():
        rig.say()

    events = run(rig, conversation, user)
    assert [event["muse"] for event in sent(events)] == ["send_image"]
    assert list(tmp_path.iterdir()) == []


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
