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


def test_pauses_do_not_erase_time_waiting_for_transcription(monkeypatch):
    from types import SimpleNamespace
    import numpy as np
    import reachy_muse.companion_client as client
    from reachy_muse.protocol import Op

    class Vad:
        def is_speech(self, pcm, rate):
            return bool(np.any(np.frombuffer(pcm, dtype="<i2")))

    now = 100.0
    monkeypatch.setattr(client, "time", SimpleNamespace(monotonic=lambda: now))
    link = CompanionLink("ws://127.0.0.1:9000", token=TOKEN, out_rate=16000, insecure=True)
    link._connection = client._Connection(None, (Op.HEAR, Op.SPEAK))
    link._set(LinkState.UP)
    turn = client.CompanionHearing(link).open_turn(16000, silence_s=3, vad=Vad())
    turn.feed(np.full(8000, .1, dtype=np.float32))
    assert turn.take().ended is None
    now = 101.0
    turn.feed(np.full(16000, .1, dtype=np.float32))
    assert turn.take().ended is None
    now = 101.5
    turn.feed(np.zeros(8000, dtype=np.float32))
    assert turn.take().ended is None
    now = 102.6
    turn.feed(np.full(17600, .1, dtype=np.float32))
    with pytest.raises(CapabilityUnavailable, match="no transcript"):
        turn.take()
    assert not link.can(Op.SPEAK)


@pytest.mark.parametrize("quiet_frame", [0, 2, 4])
def test_mixed_batches_exclude_only_quiet_audio_from_the_response_deadline(monkeypatch, quiet_frame):
    from types import SimpleNamespace
    import numpy as np
    import reachy_muse.companion_client as client
    from reachy_muse.protocol import Op

    class Vad:
        def is_speech(self, pcm, rate):
            return bool(np.any(np.frombuffer(pcm, dtype="<i2")))

    now = 100.0
    monkeypatch.setattr(client, "time", SimpleNamespace(monotonic=lambda: now))
    link = CompanionLink("ws://127.0.0.1:9000", token=TOKEN, out_rate=16000, insecure=True)
    link._connection = client._Connection(None, (Op.HEAR, Op.SPEAK))
    link._set(LinkState.UP)
    turn = client.CompanionHearing(link).open_turn(16000, silence_s=3, vad=Vad())
    turn.feed(np.full(1600, .1, dtype=np.float32))
    assert turn.take().ended is None
    batch = np.full(1600, .1, dtype=np.float32)
    batch[quiet_frame * 320:(quiet_frame + 1) * 320] = 0
    for index in range(1, 25):
        now = 100.0 + index * .1
        turn.feed(batch)
        assert turn.take().ended is None
        assert link.can(Op.SPEAK)
    now = 102.6
    turn.feed(np.tile(batch, 2))
    with pytest.raises(CapabilityUnavailable, match="no transcript"):
        turn.take()
    assert not link.can(Op.SPEAK)


def test_polling_does_not_reuse_quiet_audio(monkeypatch):
    from types import SimpleNamespace
    import numpy as np
    import reachy_muse.companion_client as client
    from reachy_muse.protocol import Op

    class Vad:
        def is_speech(self, pcm, rate):
            return bool(np.any(np.frombuffer(pcm, dtype="<i2")))

    now = 100.0
    monkeypatch.setattr(client, "time", SimpleNamespace(monotonic=lambda: now))
    link = CompanionLink("ws://127.0.0.1:9000", token=TOKEN, out_rate=16000, insecure=True)
    link._connection = client._Connection(None, (Op.HEAR, Op.SPEAK))
    link._set(LinkState.UP)
    turn = client.CompanionHearing(link).open_turn(16000, silence_s=3, vad=Vad())
    turn.feed(np.full(1600, .1, dtype=np.float32))
    assert turn.take().ended is None
    now = 101.5
    turn.feed(np.full(24000, .1, dtype=np.float32))
    assert turn.take().ended is None
    now = 101.52
    turn.feed(np.zeros(320, dtype=np.float32))
    assert turn.take().ended is None
    now = 101.8
    assert turn.take().ended is None
    now = 102.03
    with pytest.raises(CapabilityUnavailable, match="no transcript"):
        turn.take()
    assert not link.can(Op.SPEAK)


def test_partial_progress_restarts_the_hearing_response_deadline(monkeypatch):
    from types import SimpleNamespace
    import numpy as np
    import reachy_muse.companion_client as client
    from reachy_muse.protocol import HearPartial, Op, encode

    now = 100.0
    monkeypatch.setattr(client, "time", SimpleNamespace(monotonic=lambda: now))
    link = CompanionLink("ws://127.0.0.1:9000", token=TOKEN, out_rate=16000, insecure=True)
    link._connection = client._Connection(None, (Op.HEAR, Op.SPEAK))
    link._set(LinkState.UP)
    turn = client.CompanionHearing(link).open_turn(
        16000, silence_s=.3, vad=SimpleNamespace(is_speech=lambda *args: True))
    turn.feed(np.full(1600, .1, dtype=np.float32))
    assert turn.take().ended is None
    now = 101.5
    link._receive(link._connection, encode(HearPartial(1, 0, "hello")))
    assert turn.take().ended is None
    now = 102.5
    turn.feed(np.full(1600, .1, dtype=np.float32))
    assert turn.take().ended is None
    assert link.can(Op.SPEAK)
    now = 103.5
    with pytest.raises(CapabilityUnavailable, match="no transcript"):
        turn.take()
    assert not link.can(Op.SPEAK)


@pytest.mark.parametrize("code", ["overloaded", "gap"])
def test_a_terminal_hearing_probe_error_reconnects_and_restores_speech(code):
    from reachy_muse.protocol import CloseHow, Error, ErrorCode, Op, VoiceRole, encode

    async def scenario():
        async with FakeCompanion(stall=True) as companion:
            link = CompanionLink(companion.url[:-3], token=TOKEN, out_rate=16000, insecure=True,
                                 backoff_s=(.01, .01), healthy_s=0)
            running = asyncio.create_task(link.run())
            try:
                await asyncio.wait_for(link.settled(), 2)
                stream = link.open_hear()
                stream.send(b"\0\0" * 1600)
                link.hearing_stalled(stream, "no hearing response")
                stream.close(CloseHow.ABORT)
                assert not link.can(Op.SPEAK)
                companion.stall = False
                for ws in companion._server.connections:
                    await ws.send(encode(Error(ErrorCode(code), "hearing stopped", stream=stream.id)))
                async with asyncio.timeout(1):
                    while not link.up:
                        await asyncio.sleep(.01)
                assert companion.hellos == [TOKEN, TOKEN]
                assert link.can(Op.HEAR) and link.can(Op.SPEAK)
                async with asyncio.timeout(1):
                    chunks = [pcm async for pcm in link.speak("Hello.", VoiceRole.MAIN)]
                assert len(chunks) == 3
                assert all(chunks)
            finally:
                running.cancel()
                await asyncio.gather(running, return_exceptions=True)
    asyncio.run(scenario())


def test_a_socket_that_never_pongs_fails_pending_work_within_two_seconds():
    from websockets.asyncio.server import serve
    from reachy_muse.protocol import VERSION, Op, VoiceRole, Welcome, encode

    async def scenario():
        release = asyncio.Event()

        async def stalled(ws):
            await ws.recv()
            await ws.send(encode(Welcome(VERSION, (Op.SPEAK, Op.NARRATE), ())))
            ws.transport.pause_reading()  # no pongs or close handshake; the event loop stays responsive
            await release.wait()
            ws.transport.abort()

        async with serve(stalled, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            link = CompanionLink(f"ws://127.0.0.1:{port}", token=TOKEN, out_rate=16000,
                                 insecure=True, backoff_s=(5, 5))
            running = asyncio.create_task(link.run())
            try:
                await asyncio.wait_for(link.settled(), 2)
                start = asyncio.get_running_loop().time()
                with pytest.raises(CapabilityUnavailable):
                    await asyncio.wait_for(anext(link.speak("Hello.", VoiceRole.MAIN)), 2.2)
                assert asyncio.get_running_loop().time() - start < 2.2
                assert not link.can(Op.HEAR) and not link.can(Op.SPEAK) and not link.up
                assert link.take_outage() is True
                assert link.take_outage() is False
            finally:
                release.set()
                running.cancel()
                await asyncio.gather(running, return_exceptions=True)
    asyncio.run(scenario())


def test_hearing_stall_with_working_pongs_stays_local_until_a_real_endpoint():
    from reachy_muse.companion_client import CompanionHearing
    from reachy_muse.protocol import Op
    import numpy as np

    class Vad:
        def is_speech(self, *args):
            return True

    async def scenario():
        async with FakeCompanion(stall=True) as companion:
            link = CompanionLink(companion.url[:-3], token=TOKEN, out_rate=16000, insecure=True,
                                 backoff_s=(5, 5), healthy_s=0)
            running = asyncio.create_task(link.run())
            try:
                await asyncio.wait_for(link.settled(), 2)
                turn = CompanionHearing(link).open_turn(16000, silence_s=.3, vad=Vad())
                turn.feed(np.full(3200, .1, dtype=np.float32))
                assert turn.take().ended is None
                start = asyncio.get_running_loop().time()
                while link.up:
                    try:
                        turn.take()
                    except CapabilityUnavailable:
                        break
                    await asyncio.sleep(.01)
                assert asyncio.get_running_loop().time() - start < 2.2
                turn.abort()
                for _ in range(5):
                    with pytest.raises(CapabilityUnavailable):
                        link.open_hear()
                assert not link.can(Op.SPEAK)
                assert link.take_outage() is True
                assert link.take_outage() is False
                companion.stall = False
                # The fake delayed the endpoint. Deliver it on the same live socket.
                from reachy_muse.protocol import HearEndpoint, encode
                for ws in companion._server.connections:
                    await ws.send(encode(HearEndpoint(1, "", True)))
                async with asyncio.timeout(2):
                    while not link.up:
                        await asyncio.sleep(.01)
                assert link.open_hear().id == 3
                assert link.can(Op.SPEAK)
                assert link.take_outage() is False
            finally:
                running.cancel()
                await asyncio.gather(running, return_exceptions=True)
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
