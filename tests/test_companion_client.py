"""The companion link's boundary: URL and pin parsing, pinned TLS, and the handshake."""

import asyncio
import datetime
import hashlib
import logging
import ssl

import pytest

from fake_companion import TOKEN, FakeCompanion
from reachy_muse.contract.replies import ReplyStyle
from reachy_muse.capabilities import CapabilityUnavailable
from reachy_muse.companion_client import CompanionLink, CompanionNarrator, LinkState, parse_pin, parse_url
from reachy_muse.protocol import Narrate, Sender, decode

DIGEST = "4f" * 32


@pytest.mark.parametrize("chunk_samples", [40000, 1600])
def test_buffered_wake_audio_waits_for_a_responsive_companion(chunk_samples):
    from websockets.asyncio.server import serve
    from reachy_muse.companion_client import CompanionHearing
    from reachy_muse.protocol import (
        VERSION, HearClose, HearEndpoint, HearPartial, Op, Welcome, decode_audio, encode,
    )
    import numpy as np

    class Vad:
        def is_speech(self, pcm, rate):
            return bool(np.any(np.frombuffer(pcm, dtype="<i2")))

    async def scenario():
        received = bytearray()

        async def responsive(ws):
            await ws.recv()
            await ws.send(encode(Welcome(VERSION, (Op.HEAR, Op.SPEAK, Op.NARRATE), ())))
            async for frame in ws:
                if isinstance(frame, bytes):
                    audio = decode_audio(frame)
                    if not received:
                        await asyncio.sleep(.1)
                        await ws.send(encode(HearPartial(audio.stream, 1, "hello there")))
                    received.extend(audio.pcm)
                else:
                    message = decode(frame, sender=Sender.ROBOT)
                    if isinstance(message, HearClose):
                        await ws.send(encode(HearEndpoint(message.stream, "hello there", True)))

        async with serve(responsive, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            link = CompanionLink(f"ws://127.0.0.1:{port}", token=TOKEN, out_rate=16000,
                                 insecure=True, backoff_s=(5, 5))
            running = asyncio.create_task(link.run())
            try:
                await asyncio.wait_for(link.settled(), 2)
                turn = CompanionHearing(link).open_turn(16000, silence_s=.3, vad=Vad())
                for _ in range(40000 // chunk_samples):
                    turn.feed(np.full(chunk_samples, .1, dtype=np.float32))
                    assert turn.take().ended is None
                    assert link.up
                    await asyncio.sleep(0)
                async with asyncio.timeout(1):
                    while turn.partial is None:
                        turn.take()
                        await asyncio.sleep(.01)
                assert turn.partial.text == "hello there"
                turn.feed(np.zeros(6400, dtype=np.float32))
                async with asyncio.timeout(1):
                    while (ended := turn.take().ended) is None:
                        await asyncio.sleep(.01)
                assert (await ended.endpoint()).text == "hello there"
                assert len(received) >= 80000
                assert link.can(Op.SPEAK) and link.up
                assert link.take_outage() is False
            finally:
                running.cancel()
                await asyncio.gather(running, return_exceptions=True)
    asyncio.run(scenario())


@pytest.mark.parametrize("options", [
    {"silence_s": 3},
    {"silence_s": .3, "short_utterance_silence_s": 3},
])
@pytest.mark.parametrize("partial", [False, True])
def test_a_permitted_pause_keeps_hearing_and_speech_on_the_companion(monkeypatch, options, partial):
    from types import SimpleNamespace
    import numpy as np
    import reachy_muse.companion_client as client
    from reachy_muse.protocol import HearEndpoint, HearPartial, Op, encode

    class Vad:
        def is_speech(self, pcm, rate):
            return bool(np.any(np.frombuffer(pcm, dtype="<i2")))

    now = 100.0
    monkeypatch.setattr(client, "time", SimpleNamespace(monotonic=lambda: now))
    link = CompanionLink("ws://127.0.0.1:9000", token=TOKEN, out_rate=16000, insecure=True)
    link._connection = client._Connection(None, (Op.HEAR, Op.SPEAK, Op.NARRATE))
    link._set(LinkState.UP)
    turn = client.CompanionHearing(link).open_turn(16000, vad=Vad(), **options)
    turn.feed(np.full(8000, .1, dtype=np.float32))
    assert turn.take().ended is None
    if partial:
        link._receive(link._connection, encode(HearPartial(1, 0, "hello")))
        assert turn.take().ended is None
    for _ in range(125):
        now += .02
        turn.feed(np.zeros(320, dtype=np.float32))
        assert turn.take().ended is None
        assert link.up and link.can(Op.SPEAK)
    now += .1
    turn.feed(np.full(1600, .1, dtype=np.float32))
    assert turn.take().ended is None
    link._receive(link._connection, encode(HearEndpoint(1, "hello there", False)))
    ended = turn.take().ended
    assert asyncio.run(ended.endpoint()).text == "hello there"
    assert link.up and link.can(Op.SPEAK)
    assert link.take_outage() is False


def _finished_hearing_turn(monkeypatch):
    """A turn the robot's VAD has just ended, on a link whose clock the test moves."""
    from types import SimpleNamespace
    import numpy as np
    import reachy_muse.companion_client as client
    from reachy_muse.protocol import HearPartial, Op, encode

    clock = SimpleNamespace(now=100.0)
    monkeypatch.setattr(client, "time", SimpleNamespace(monotonic=lambda: clock.now))
    link = CompanionLink("ws://127.0.0.1:9000", token=TOKEN, out_rate=16000, insecure=True)
    link._connection = client._Connection(None, (Op.HEAR, Op.SPEAK, Op.NARRATE))
    link._set(LinkState.UP)
    turn = client.CompanionHearing(link).open_turn(
        16000, silence_s=.3, vad=SimpleNamespace(is_speech=lambda pcm, rate: any(pcm)))
    turn.feed(np.full(8000, .1, dtype=np.float32))
    assert turn.take().ended is None
    link._receive(link._connection, encode(HearPartial(1, 0, "hello")))
    turn.feed(np.zeros(6400, dtype=np.float32))
    ended = turn.take().ended
    assert ended is not None and not turn.active
    return link, ended, clock


def _companion_endpoint(link):
    from reachy_muse.protocol import HearEndpoint, encode

    link._receive(link._connection, encode(HearEndpoint(1, "hello there", True)))


@pytest.mark.parametrize("delay", [0.662, 1.122])
def test_the_companion_transcribes_a_turn_the_robot_ended(monkeypatch, delay):
    link, ended, clock = _finished_hearing_turn(monkeypatch)
    clock.now += delay
    _companion_endpoint(link)
    endpoint = asyncio.run(ended.endpoint())
    assert (endpoint.text, endpoint.forced, link.up, link.take_outage()) == ("hello there", True, True, False)


def test_a_companion_endpoint_that_lands_just_before_the_robots_keeps_its_words(monkeypatch):
    from types import SimpleNamespace
    import numpy as np
    import reachy_muse.companion_client as client
    from reachy_muse.protocol import HearEndpoint, HearPartial, Op, encode

    clock = SimpleNamespace(now=100.0)
    monkeypatch.setattr(client, "time", SimpleNamespace(monotonic=lambda: clock.now))
    link = CompanionLink("ws://127.0.0.1:9000", token=TOKEN, out_rate=16000, insecure=True)
    link._connection = client._Connection(None, (Op.HEAR, Op.SPEAK, Op.NARRATE))
    link._set(LinkState.UP)
    turn = client.CompanionHearing(link).open_turn(
        16000, silence_s=.3, vad=SimpleNamespace(is_speech=lambda pcm, rate: any(pcm)))
    turn.feed(np.full(8000, .1, dtype=np.float32))
    assert turn.take().ended is None
    link._receive(link._connection, encode(HearPartial(1, 0, "hello")))
    link._receive(link._connection, encode(HearEndpoint(1, "hello there", False)))
    turn.feed(np.zeros(6400, dtype=np.float32))
    ended = turn.take().ended
    clock.now += 1.6
    endpoint = asyncio.run(ended.endpoint())
    assert (endpoint.text, endpoint.forced, link.up, link.take_outage()) == ("hello there", False, True, False)


@pytest.mark.parametrize("delay", [1.6, 3.641, 37.28])
def test_a_late_companion_endpoint_fails_only_that_turn(monkeypatch, delay):
    from reachy_muse.protocol import CloseHow, HearClose

    link, ended, clock = _finished_hearing_turn(monkeypatch)
    clock.now += delay
    with pytest.raises(CapabilityUnavailable, match="did not end the turn in time") as raised:
        asyncio.run(ended.endpoint())
    _companion_endpoint(link)
    assert raised.value.link_down is False
    assert (link.up, link.take_outage(), link._connection.hearing) == (True, False, {})
    outbox = link._connection.outbox
    sent = [decode(frame, sender=Sender.ROBOT) for frame in (outbox.get_nowait() for _ in range(outbox.qsize()))
            if isinstance(frame, str)]
    assert [message.how for message in sent if isinstance(message, HearClose)] == [CloseHow.FINISH, CloseHow.ABORT]


def test_speech_waits_for_a_slow_worker_while_pongs_continue():
    from websockets.asyncio.server import serve
    from reachy_muse.protocol import (
        VERSION, AudioKind, Op, SpeakStart, SpeakDone, VoiceRole, Welcome, encode, encode_audio,
    )

    async def scenario():
        async def slow(ws):
            await ws.recv()
            await ws.send(encode(Welcome(VERSION, (Op.SPEAK, Op.NARRATE), ())))
            async for frame in ws:
                message = decode(frame, sender=Sender.ROBOT)
                if isinstance(message, SpeakStart):
                    await asyncio.sleep(2.1)
                    await ws.send(encode_audio(AudioKind.SPEECH, message.stream, 0, b"\x01\x00" * 160))
                    await ws.send(encode(SpeakDone(message.stream, 1)))

        async with serve(slow, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            link = CompanionLink(f"ws://127.0.0.1:{port}", token=TOKEN, out_rate=16000,
                                 insecure=True, ping_interval_s=.02, ping_timeout_s=.1)
            running = asyncio.create_task(link.run())
            try:
                await asyncio.wait_for(link.settled(), 1)
                chunks = [pcm async for pcm in link.speak("Hello.", VoiceRole.MAIN)]
                assert chunks == [b"\x01\x00" * 160]
                assert (link.up, link.responsive, link.take_outage()) == (True, True, False)
            finally:
                running.cancel()
                await asyncio.gather(running, return_exceptions=True)
    asyncio.run(scenario())


def test_no_audio_or_pong_fails_the_pending_sentence_as_a_dead_link():
    from websockets.asyncio.server import serve
    from reachy_muse.protocol import VERSION, Op, VoiceRole, Welcome, encode

    async def scenario():
        release = asyncio.Event()

        async def stalled(ws):
            await ws.recv()
            await ws.send(encode(Welcome(VERSION, (Op.SPEAK, Op.NARRATE), ())))
            ws.transport.pause_reading()
            await release.wait()
            ws.transport.abort()

        async with serve(stalled, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            link = CompanionLink(f"ws://127.0.0.1:{port}", token=TOKEN, out_rate=16000,
                                 insecure=True, backoff_s=(5, 5), ping_interval_s=.01, ping_timeout_s=.04)
            running = asyncio.create_task(link.run())
            try:
                await asyncio.wait_for(link.settled(), 1)
                with pytest.raises(CapabilityUnavailable, match="no frames or pong") as failed:
                    await asyncio.wait_for(anext(link.speak("Hello.", VoiceRole.MAIN)), .3)
                assert failed.value.link_down is True
                assert link.can(Op.SPEAK) is False
            finally:
                release.set()
                running.cancel()
                await asyncio.gather(running, return_exceptions=True)
    asyncio.run(scenario())


def test_jitter_bursts_do_not_drop_the_companion_link():
    from bench.turn2 import jitter_replay

    result = asyncio.run(jitter_replay())
    assert result["bursts"] == [
        {"jitter_s": seconds, "drops": 0} for seconds in (0, .5, 1, 1.5, 2, 3, 4, 6, 8)
    ]


def test_idle_link_reconnects_before_another_turn_needs_it():
    from bench.turn2 import reconnect_replay

    result = asyncio.run(reconnect_replay())
    assert result["reply"] == ["Should I work out today?"]
    assert result["idle_recovery_ms"] < 700


def test_a_long_outage_does_not_exhaust_the_reconnect_loop():
    from bench.turn2 import outage_replay

    assert asyncio.run(outage_replay()) == {
        "attempts": 1101, "max_retry_delay_s": 5.0,
        "first_retry_delays_s": [.2, .4, .8, 1.6, 3.2, 5.0, 5.0, 5.0],
        "reply": ["The companion is back."],
    }


def test_hearing_and_classification_wait_while_the_socket_is_responsive():
    from reachy_muse.companion_client import CompanionHearing
    from reachy_muse.device_command import DeviceCommand
    from reachy_muse.protocol import Classify, ClassifyResult, HearEndpoint, Op, encode
    import numpy as np

    async def scenario():
        async with FakeCompanion(stall=True, ops=(Op.HEAR, Op.SPEAK, Op.NARRATE, Op.CLASSIFY)) as companion:
            link = CompanionLink(companion.url[:-3], token=TOKEN, out_rate=16000, insecure=True,
                                 ping_interval_s=.02, ping_timeout_s=.1)
            running = asyncio.create_task(link.run())
            classify = None
            try:
                await asyncio.wait_for(link.settled(), 1)
                from types import SimpleNamespace
                turn = CompanionHearing(link).open_turn(
                    16000, silence_s=.3, vad=SimpleNamespace(is_speech=lambda *args: True))
                turn.feed(np.full(3200, .1, dtype=np.float32))
                turn.take()
                classify = asyncio.create_task(link.classify("sleep", frozenset({DeviceCommand.SLEEP})))
                await asyncio.sleep(2.1)
                assert turn.take().ended is None
                assert classify.done() is False
                for ws in companion._server.connections:
                    await ws.send(encode(HearEndpoint(1, "hello there", True)))
                    request = next(decode(frame, sender=Sender.ROBOT) for frame in companion.received
                                   if isinstance(frame, str) and isinstance(decode(frame, sender=Sender.ROBOT), Classify))
                    await ws.send(encode(ClassifyResult(request.id, DeviceCommand.SLEEP)))
                assert await asyncio.wait_for(classify, 1) is DeviceCommand.SLEEP
                async with asyncio.timeout(1):
                    while (ended := turn.take().ended) is None:
                        await asyncio.sleep(.01)
                assert ((await ended.endpoint()).text, link.up, link.take_outage()) == ("hello there", True, False)
            finally:
                if classify is not None:
                    classify.cancel()
                running.cancel()
                await asyncio.gather(running, *([classify] if classify is not None else []), return_exceptions=True)
    asyncio.run(scenario())


@pytest.mark.parametrize("url, insecure, expected", [
    ("wss://studio.local:8765", False, "wss://studio.local:8765"),
    ("wss://studio.local:8765/", False, "wss://studio.local:8765"),
    ("ws://127.0.0.1:9000", True, "ws://127.0.0.1:9000"),
])
def test_a_companion_url_is_its_base_address(url, insecure, expected):
    assert parse_url(url, insecure=insecure) == expected


@pytest.mark.parametrize("url", [
    "ws://studio.local:8765", "https://studio.local:8765", "wss://studio.local:8765/v1",
    "wss://user:pw@studio.local:8765", "wss://:8765", "studio.local:8765",
])
def test_a_companion_url_with_a_path_credentials_or_plain_ws_is_refused(url):
    with pytest.raises(ValueError, match="companion URL"):
        parse_url(url)


@pytest.mark.parametrize("pin, expected", [
    (f"sha256:{DIGEST}", DIGEST), (f"sha256:{DIGEST.upper()}", None), (DIGEST, None),
    (f"sha256:{DIGEST[:-2]}", None), (f"sha1:{DIGEST}", None),
])
def test_a_pin_is_sha256_and_64_lowercase_hex_digits(pin, expected):
    if expected is None:
        with pytest.raises(ValueError, match="pin"):
            parse_pin(pin)
    else:
        assert parse_pin(pin) == expected


def test_a_wss_companion_needs_a_pin():
    with pytest.raises(ValueError, match="pin"):
        CompanionLink("wss://studio.local:8765", token=TOKEN, out_rate=16000)


@pytest.fixture(scope="module")
def certificate(tmp_path_factory):
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID

    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "companion")])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(key.public_key())
            .serial_number(x509.random_serial_number()).not_valid_before(now - datetime.timedelta(days=1))
            .not_valid_after(now + datetime.timedelta(days=1)).sign(key, hashes.SHA256()))
    directory = tmp_path_factory.mktemp("companion-cert")
    (directory / "cert.pem").write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    (directory / "key.pem").write_bytes(key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(directory / "cert.pem", directory / "key.pem")
    pin = "sha256:" + hashlib.sha256(cert.public_bytes(serialization.Encoding.DER)).hexdigest()
    return context, pin


async def first_outcome(companion, **options):
    link = CompanionLink(companion.url[:-3], token=TOKEN, out_rate=16000, backoff_s=(5, 5), **options)
    running = asyncio.ensure_future(link.run())
    try:
        await asyncio.wait_for(link.settled(), 5)
        return link.state, link.models
    finally:
        running.cancel()
        await asyncio.gather(running, return_exceptions=True)


def test_the_pinned_certificate_is_accepted_and_the_link_comes_up(certificate):
    context, pin = certificate

    async def run():
        async with FakeCompanion(ssl=context) as companion:
            return await first_outcome(companion, pin=pin), companion.hellos
    outcome, hellos = asyncio.run(run())
    assert outcome == (LinkState.UP, (("hear", "fake"), ("speak", "tone"), ("narrate", "echo")))
    assert hellos == [TOKEN]


def test_another_certificate_is_refused_before_the_token_is_sent(certificate):
    context, _ = certificate

    async def run():
        async with FakeCompanion(ssl=context) as companion:
            return await first_outcome(companion, pin=f"sha256:{DIGEST}"), companion.hellos
    assert asyncio.run(run()) == ((LinkState.DOWN, ()), [])


def test_a_refused_token_keeps_the_link_down_and_logs_the_companions_reason(caplog):
    async def run():
        async with FakeCompanion(token="rc1_another_robot") as companion:
            return await first_outcome(companion, insecure=True)
    with caplog.at_level(logging.WARNING):
        assert asyncio.run(run()) == (LinkState.DOWN, ())
    assert "the companion refused hello: auth: bad token" in caplog.text


def test_a_pin_for_a_plain_ws_companion_is_refused():
    with pytest.raises(ValueError, match="a pinned companion needs wss://"):
        CompanionLink("ws://127.0.0.1:9000", token=TOKEN, pin=f"sha256:{DIGEST}", insecure=True, out_rate=16000)


def narrated(call):
    """What the fake's echo narrator answers ``call``, and the narrate request Reachy sent it."""
    async def run():
        async with FakeCompanion() as companion:
            link = CompanionLink(companion.url[:-3], token=TOKEN, insecure=True, out_rate=16000, backoff_s=(5, 5))
            running = asyncio.ensure_future(link.run())
            try:
                await asyncio.wait_for(link.settled(), 5)
                said = await call(CompanionNarrator(link))
            finally:
                running.cancel()
                await asyncio.gather(running, return_exceptions=True)
            sent = [decode(frame, sender=Sender.ROBOT) for frame in companion.received if isinstance(frame, str)]
            return said, [message for message in sent if isinstance(message, Narrate)][-1]
    return asyncio.run(run())


def test_a_progress_label_longer_than_the_wire_allows_is_cut_at_a_word():
    status = "Earlier from Muse: " + "checking the castle archives " * 8
    said, sent = narrated(lambda narrator: narrator.say_progress("castles", status, ()))
    expected = ("Earlier from Muse: checking the castle archives checking the castle archives "
                "checking the castle archives checking the")
    assert (sent.status, said.text) == (expected, expected)


def test_a_reply_longer_than_the_wire_allows_never_reaches_the_companion():
    reply = " ".join(f"Line {n:03d} is here." for n in range(600))

    async def run():
        async with FakeCompanion() as companion:
            link = CompanionLink(companion.url[:-3], token=TOKEN, insecure=True, out_rate=16000, backoff_s=(5, 5))
            running = asyncio.ensure_future(link.run())
            try:
                await asyncio.wait_for(link.settled(), 5)
                with pytest.raises(CapabilityUnavailable) as refused:
                    await CompanionNarrator(link).lines("castles", reply, ReplyStyle.PLAIN_SHORT)
            finally:
                running.cancel()
                await asyncio.gather(running, return_exceptions=True)
            sent = [decode(frame, sender=Sender.ROBOT) for frame in companion.received if isinstance(frame, str)]
            return refused.value.link_down, [message for message in sent if isinstance(message, Narrate)]
    assert asyncio.run(run()) == (False, [])


def test_partial_progress_keeps_a_missing_pong_alive_until_all_progress_stops():
    from websockets.asyncio.server import serve
    from reachy_muse.protocol import VERSION, HearPartial, Op, Welcome, encode

    async def scenario():
        start, frames_done, release = asyncio.Event(), asyncio.Event(), asyncio.Event()

        async def progressing(ws):
            await ws.recv()
            await ws.send(encode(Welcome(VERSION, (Op.HEAR, Op.SPEAK, Op.NARRATE), ())))
            await start.wait()
            ws.transport.pause_reading()
            for revision in range(6):
                await ws.send(encode(HearPartial(1, revision, "still listening")))
                await asyncio.sleep(.02)
            frames_done.set()
            await release.wait()
            ws.transport.abort()

        async with serve(progressing, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            link = CompanionLink(f"ws://127.0.0.1:{port}", token=TOKEN, out_rate=16000,
                                 insecure=True, backoff_s=(5, 5), ping_interval_s=.01, ping_timeout_s=.05)
            running = asyncio.create_task(link.run())
            try:
                await asyncio.wait_for(link.settled(), 1)
                stream = link.open_hear()
                start.set()
                await asyncio.wait_for(frames_done.wait(), 1)
                assert [item.text for item in stream.inbox] == ["still listening"] * 6
                assert link.can(Op.SPEAK) and link.take_outage() is False
                async with asyncio.timeout(.3):
                    while stream.failure is None:
                        await asyncio.sleep(.005)
                assert str(stream.failure) == "the companion sent no frames or pong"
                assert stream.failure.link_down is True
            finally:
                release.set()
                running.cancel()
                await asyncio.gather(running, return_exceptions=True)
    asyncio.run(scenario())


def test_long_unicode_speech_is_split_without_changing_voice_or_losing_text():
    from reachy_muse.capabilities import SpokenLine
    from reachy_muse.companion_client import CompanionVoice

    class Link:
        out_rate = 16000

        def __init__(self):
            self.requests = []

        async def speak(self, text, role):
            self.requests.append(text)
            yield b"\x00\x40"

    async def scenario():
        link = Link()
        voice = CompanionVoice(link)
        chunks = [chunk.tolist() async for chunk in voice.stream(
            SpokenLine("é" * 121 + " hello there", None, "answer"), 16000)]
        return link.requests, chunks

    assert asyncio.run(scenario()) == (["é" * 120, "é hello there"], [[.5], [.5]])


def test_a_blocked_ping_write_still_expires_on_no_progress():
    from types import SimpleNamespace
    from reachy_muse.companion_client import _Connection
    from reachy_muse.protocol import Op

    async def scenario():
        aborted = []

        async def blocked_ping():
            await asyncio.Event().wait()

        ws = SimpleNamespace(ping=blocked_ping, transport=SimpleNamespace(abort=lambda: aborted.append(True)))
        link = CompanionLink("ws://127.0.0.1:9000", token=TOKEN, out_rate=16000, insecure=True,
                             ping_interval_s=.01, ping_timeout_s=.04)
        connection = _Connection(ws, (Op.HEAR, Op.SPEAK))
        link._connection = connection
        link._set(LinkState.UP)
        stream = link.open_hear()
        await asyncio.wait_for(link._heartbeat(connection), .3)
        return str(stream.failure), stream.failure.link_down, aborted

    assert asyncio.run(scenario()) == ("the companion sent no frames or pong", True, [True])


def test_ping_send_and_pong_wait_share_one_inactivity_budget():
    from types import SimpleNamespace
    from reachy_muse.companion_client import _Connection
    from reachy_muse.protocol import Op

    async def scenario():
        aborted = []

        async def slow_ping():
            await asyncio.sleep(.15)
            return asyncio.get_running_loop().create_future()

        ws = SimpleNamespace(ping=slow_ping, transport=SimpleNamespace(abort=lambda: aborted.append(True)))
        link = CompanionLink("ws://127.0.0.1:9000", token=TOKEN, out_rate=16000, insecure=True,
                             ping_interval_s=.01, ping_timeout_s=.2)
        connection = _Connection(ws, (Op.HEAR, Op.SPEAK))
        link._connection = connection
        link._set(LinkState.UP)
        stream = link.open_hear()
        await asyncio.wait_for(link._heartbeat(connection), .28)
        return str(stream.failure), stream.failure.link_down, aborted

    assert asyncio.run(scenario()) == ("the companion sent no frames or pong", True, [True])


def test_speech_failure_details_cannot_enter_reconnect_logs():
    from reachy_muse.protocol import Error, ErrorCode
    link = CompanionLink("ws://127.0.0.1:9000", token=TOKEN, insecure=True, out_rate=16000)
    with pytest.raises(ConnectionError, match="^the companion failed: model$"):
        link._error(None, Error(ErrorCode.MODEL, "A private reminder.", stream=2))
