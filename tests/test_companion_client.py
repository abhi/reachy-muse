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
                turn.feed(np.full(3200, .1, dtype=np.float32))  # just 200 ms, below the byte-gap limit
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
