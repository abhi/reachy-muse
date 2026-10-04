import asyncio
import base64
import dataclasses
import hashlib
import json
import re
import ssl
import subprocess
import sys
from enum import Enum
from importlib.resources import files

import pytest
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed, InvalidStatus

from musegadget.reachy_companion_protocol import (
    MESSAGES,
    AudioFrame,
    AudioKind,
    CloseHow,
    Error,
    ErrorCode,
    HearClose,
    HearEndpoint,
    HearOpen,
    HearPartial,
    Hello,
    Narrate,
    NarratedLine,
    NarrateOp,
    NarrateResult,
    Op,
    ProtocolError,
    Sender,
    SpeakCancel,
    SpeakDone,
    SpeakStart,
    VoiceRole,
    Welcome,
    decode,
    decode_audio,
    encode,
    encode_audio,
    message_type,
)

from reachy_companion.engines import EchoNarrator, ScriptedHearing, ToneVoice
from reachy_companion.pair import pair
from reachy_companion.server import CompanionServer, Engines

VECTORS = files("musegadget") / "data" / "companion_protocol_v1"
VALID = json.loads((VECTORS / "valid.json").read_text(encoding="utf-8"))
INVALID = json.loads((VECTORS / "invalid.json").read_text(encoding="utf-8"))

TOKEN = "rc1_test_token_0123456789abcdef0123456789"
HELLO = Hello(versions=(1,), token=TOKEN, ops=(Op.HEAR, Op.SPEAK, Op.NARRATE), out_rate=24000, robot="test")
STUBS = Engines(hearing=ScriptedHearing(), voice=ToneVoice(), narrator=EchoNarrator())
SILENCE = b"\x00\x00" * 320


def plain(value):
    if dataclasses.is_dataclass(value):
        return {f.name: plain(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (tuple, list)):
        return [plain(item) for item in value]
    return value


@pytest.mark.parametrize("vector", VALID, ids=[v["name"] for v in VALID])
def test_sdk_valid_vector_round_trips_through_the_imported_codec(vector):
    if "audio" in vector:
        frame = base64.b64decode(vector["audio"])
        decoded = decode_audio(frame)
        assert {**plain(decoded), "pcm": base64.b64encode(decoded.pcm).decode()} == vector["expect"]
        assert encode_audio(decoded.kind, decoded.stream, decoded.seq, decoded.pcm) == frame
        return
    sender = Sender(vector["sender"])
    message = decode(vector["text"], sender=sender)
    assert {"t": message_type(message), **plain(message)} == vector["expect"]
    assert decode(encode(message), sender=sender) == message


@pytest.mark.parametrize("vector", INVALID, ids=[v["name"] for v in INVALID])
def test_sdk_invalid_vector_raises_through_the_imported_codec(vector):
    with pytest.raises(ProtocolError) as caught:
        if "audio" in vector:
            decode_audio(base64.b64decode(vector["audio"]))
        else:
            decode(vector["text"], sender=Sender(vector["sender"]))
    assert (caught.value.code.value, caught.value.reason.value) == (vector["code"], vector["reason"])


def test_sdk_vectors_cover_every_message_type():
    assert {json.loads(v["text"])["t"] for v in VALID if "text" in v} == set(MESSAGES)


# ---- the real server, driven by a raw websockets client


class HangingVoice:
    name = "hanging"

    async def speak(self, text, role, out_rate):
        yield b"\x01\x00" * (out_rate // 10)
        await asyncio.Event().wait()


class HangingNarrator(EchoNarrator):
    name = "hanging"

    async def acknowledge(self, request):
        await asyncio.Event().wait()


class FailingNarrator(EchoNarrator):
    name = "failing"

    async def lines(self, request, reply):
        raise RuntimeError("boom")


def scenario(script, *, engines=STUBS, **server):
    async def run():
        async with CompanionServer(engines, TOKEN, **server) as companion:
            await script(companion)

    asyncio.run(asyncio.wait_for(run(), 10))


async def connected(companion, hello=HELLO, **client):
    ws = await connect(companion.url, **client)
    await ws.send(encode(hello))
    welcome = await recv(ws)
    assert isinstance(welcome, Welcome), welcome
    return ws


async def recv(ws):
    frame = await asyncio.wait_for(ws.recv(), 2)
    return decode_audio(frame) if isinstance(frame, bytes) else decode(frame, sender=Sender.COMPANION)


async def send(ws, message):
    await ws.send(encode(message))


async def nothing_else_arrived(ws):
    """An echo narration answered next proves no other frame was queued before it."""
    await send(ws, Narrate(id=999, op=NarrateOp.ACKNOWLEDGE, request="Barrier."))
    assert await recv(ws) == NarrateResult(999, (NarratedLine("Barrier.", None),))


async def mic(ws, stream, seqs):
    for seq in seqs:
        await ws.send(encode_audio(AudioKind.MIC, stream, seq, SILENCE))


def test_handshake_welcomes_with_every_offered_op_and_model_name():
    async def script(companion):
        async with connect(companion.url) as ws:
            await send(ws, HELLO)
            assert await recv(ws) == Welcome(1, (Op.HEAR, Op.SPEAK, Op.NARRATE), (("hear", "scripted"), ("speak", "tone"), ("narrate", "echo")))

    scenario(script)


def test_handshake_picks_v1_and_grants_only_ops_both_sides_have():
    async def script(companion):
        hello = dataclasses.replace(HELLO, versions=(2, 1), ops=(Op.SPEAK, Op.NARRATE))
        async with connect(companion.url) as ws:
            await send(ws, hello)
            assert await recv(ws) == Welcome(1, (Op.NARRATE,), (("narrate", "echo"),))

    scenario(script, engines=Engines(hearing=ScriptedHearing(), narrator=EchoNarrator()))


@pytest.mark.parametrize(
    "first, expected",
    [
        (encode(dataclasses.replace(HELLO, token="rc1_wrong")), Error(ErrorCode.AUTH, "the first frame must be hello with the paired token")),
        (encode(Narrate(id=1, op=NarrateOp.ACKNOWLEDGE, request="hi")), Error(ErrorCode.AUTH, "the first frame must be hello with the paired token")),
        (encode(dataclasses.replace(HELLO, versions=(2,))), Error(ErrorCode.VERSION, "this companion speaks versions [1]")),
    ],
    ids=["wrong_token", "not_hello", "no_common_version"],
)
def test_refused_handshake_gets_one_error_then_a_policy_close(first, expected):
    async def script(companion):
        async with connect(companion.url) as ws:
            await ws.send(first)
            assert await recv(ws) == expected
            with pytest.raises(ConnectionClosed):
                await recv(ws)
            assert ws.close_code == 1008

    scenario(script)


def test_hear_stream_sends_growing_partials_then_one_forced_endpoint():
    async def script(companion):
        ws = await connected(companion)
        await send(ws, HearOpen(stream=1, pre_roll_frames=0))
        await mic(ws, 1, range(12))
        await send(ws, HearClose(stream=1, how=CloseHow.FINISH))
        assert [await recv(ws) for _ in range(3)] == [
            HearPartial(1, 0, "what's"),
            HearPartial(1, 1, "what's the weather"),
            HearEndpoint(1, "what's the weather", True),
        ]
        await nothing_else_arrived(ws)

    scenario(script, engines=Engines(hearing=ScriptedHearing(("what's", "what's the weather")), narrator=EchoNarrator()))


def test_hear_endpoint_found_by_the_companion_is_the_only_endpoint():
    async def script(companion):
        ws = await connected(companion)
        await send(ws, HearOpen(stream=3, pre_roll_frames=2))
        await mic(ws, 3, range(8))
        await send(ws, HearClose(stream=3, how=CloseHow.FINISH))
        assert [await recv(ws) for _ in range(2)] == [HearPartial(3, 0, "lights off"), HearEndpoint(3, "lights off", False)]
        await nothing_else_arrived(ws)

    scenario(script, engines=Engines(hearing=ScriptedHearing(("lights off",), endpoint_after=6), narrator=EchoNarrator()))


def test_aborted_hear_stream_gets_no_endpoint():
    async def script(companion):
        ws = await connected(companion)
        await send(ws, HearOpen(stream=1, pre_roll_frames=0))
        await mic(ws, 1, range(3))
        await send(ws, HearClose(stream=1, how=CloseHow.ABORT))
        await nothing_else_arrived(ws)

    scenario(script)


def test_microphone_gap_fails_only_that_stream():
    async def script(companion):
        ws = await connected(companion)
        await send(ws, HearOpen(stream=1, pre_roll_frames=0))
        await mic(ws, 1, [0, 2, 3])
        assert await recv(ws) == Error(ErrorCode.GAP, "gap/gap: stream 1 expected seq 1, got 2", stream=1)
        await send(ws, HearClose(stream=1, how=CloseHow.FINISH))
        await nothing_else_arrived(ws)

    scenario(script)


def test_speak_reframes_engine_audio_into_100ms_frames_in_seq_order_then_done():
    async def script(companion):
        ws = await connected(companion)
        await send(ws, SpeakStart(stream=2, text="Hello there.", voice=VoiceRole.MAIN))
        frames = [await recv(ws) for _ in range(3)]
        assert [(f.kind, f.stream, f.seq, len(f.pcm)) for f in frames] == [
            (AudioKind.SPEECH, 2, 0, 4800),
            (AudioKind.SPEECH, 2, 1, 4800),
            (AudioKind.SPEECH, 2, 2, 2400),
        ]
        assert b"".join(f.pcm for f in frames) == b"".join([chunk async for chunk in voice.speak("Hello there.", VoiceRole.MAIN, 24000)])
        assert await recv(ws) == SpeakDone(2, 3)

    voice = ToneVoice(chunks=1, chunk_ms=250)
    scenario(script, engines=Engines(voice=voice, narrator=EchoNarrator()))


def test_speak_cancel_stops_with_one_done_and_repeats_are_ignored():
    async def script(companion):
        ws = await connected(companion)
        await send(ws, SpeakStart(stream=4, text="A long answer.", voice=VoiceRole.PROGRESS))
        assert await recv(ws) == AudioFrame(AudioKind.SPEECH, 4, 0, b"\x01\x00" * 2400)
        await send(ws, SpeakCancel(stream=4))
        assert await recv(ws) == SpeakDone(4, 1)
        await send(ws, SpeakCancel(stream=4))
        await send(ws, SpeakCancel(stream=6))
        await nothing_else_arrived(ws)

    scenario(script, engines=Engines(voice=HangingVoice(), narrator=EchoNarrator()))


def test_narrate_echo_answers_each_op_by_id():
    async def script(companion):
        ws = await connected(companion)
        await send(ws, Narrate(id=1, op=NarrateOp.LINES, request="Weather?", reply="Sunny today. High of 20! Low of 9? Windy."))
        await send(ws, Narrate(id=2, op=NarrateOp.PROGRESS, request="Weather?", status="Checking the forecast"))
        results = {}
        for _ in range(2):
            result = await recv(ws)
            results[result.id] = result.lines
        assert results == {
            1: (NarratedLine("Sunny today.", None), NarratedLine("High of 20!", None), NarratedLine("Low of 9?", None)),
            2: (NarratedLine("Checking the forecast", None),),
        }

    scenario(script)


def test_fifth_outstanding_narration_is_overloaded():
    async def script(companion):
        ws = await connected(companion)
        for id in range(1, 6):
            await send(ws, Narrate(id=id, op=NarrateOp.ACKNOWLEDGE, request="hi"))
        assert await recv(ws) == Error(ErrorCode.OVERLOADED, "4 narrations are outstanding", id=5)

    scenario(script, engines=Engines(narrator=HangingNarrator()))


def test_narrator_failure_is_error_model_for_that_id():
    async def script(companion):
        ws = await connected(companion)
        await send(ws, Narrate(id=7, op=NarrateOp.LINES, request="hi", reply="Hello."))
        assert await recv(ws) == Error(ErrorCode.MODEL, "RuntimeError: boom", id=7)

    scenario(script, engines=Engines(narrator=FailingNarrator()))


@pytest.mark.parametrize(
    "frame, expected",
    [
        ('{"t":"dance"}', Error(ErrorCode.UNKNOWN_TYPE, "unknown_type/unknown_type: dance")),
        ('{"t":"welcome","version":1,"ops":[]}', Error(ErrorCode.UNKNOWN_TYPE, "unknown_type/wrong_direction: welcome is not sent by the robot")),
        ('{"t":"speak_start","stream":3,"text":"hi","voice":"main"}', Error(ErrorCode.BAD_REQUEST, "bad_request/bad_stream: stream 3 has the wrong parity")),
        ('{"t":"' + "x" * 600 + '"}', Error(ErrorCode.UNKNOWN_TYPE, "unknown_type/unknown_type: " + "x" * 473)),
    ],
    ids=["unknown_type", "wrong_direction", "bad_field", "huge_type_name"],
)
def test_refused_frame_gets_its_error_code_and_the_link_stays_up(frame, expected):
    async def script(companion):
        ws = await connected(companion)
        await ws.send(frame)
        assert await recv(ws) == expected
        await nothing_else_arrived(ws)

    scenario(script)


def test_op_missing_from_welcome_is_bad_request():
    async def script(companion):
        ws = await connected(companion, dataclasses.replace(HELLO, ops=(Op.NARRATE,)))
        await send(ws, HearOpen(stream=1, pre_roll_frames=0))
        assert await recv(ws) == Error(ErrorCode.BAD_REQUEST, "op hear was not granted in welcome", stream=1)

    scenario(script)


def test_paths_other_than_v1_are_not_found():
    async def script(companion):
        with pytest.raises(InvalidStatus) as caught:
            await connect(companion.url.replace("/v1", "/v2"))
        assert caught.value.response.status_code == 404

    scenario(script)


def test_tls_serves_the_paired_certificate_the_robot_pins(tmp_path):
    paired = pair(tmp_path / "state", hosts=["studio.local"])
    client = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    client.check_hostname = False
    client.verify_mode = ssl.CERT_NONE

    async def script(companion):
        assert companion.url.startswith("wss://127.0.0.1:")
        ws = await connected(companion, ssl=client)
        der = ws.transport.get_extra_info("ssl_object").getpeercert(binary_form=True)
        assert "sha256:" + hashlib.sha256(der).hexdigest() == paired.pin
        await nothing_else_arrived(ws)

    scenario(script, ssl_context=paired.tls_context())


def test_shutdown_closes_the_robot_with_going_away():
    async def script():
        async with CompanionServer(Engines(voice=HangingVoice()), TOKEN) as companion:
            ws = await connected(companion)
            await send(ws, SpeakStart(stream=2, text="Talking.", voice=VoiceRole.MAIN))
            await recv(ws)
        with pytest.raises(ConnectionClosed):
            await recv(ws)
        return ws.close_code

    assert asyncio.run(asyncio.wait_for(script(), 10)) == 1001


def test_cli_serve_answers_a_robot_and_stops_on_sigint(tmp_path):
    state = ["--state-dir", str(tmp_path)]
    subprocess.run([sys.executable, "-m", "reachy_companion", *state, "pair"], check=True, capture_output=True)
    token = (tmp_path / "token").read_text()
    command = [sys.executable, "-m", "reachy_companion", *state, "serve", "--bind", "127.0.0.1", "--port", "0", "--insecure"]
    with subprocess.Popen(command, stderr=subprocess.PIPE, text=True) as server:
        try:
            url = next(match.group(1) for line in server.stderr if (match := re.search(r"serving (ws://\S+)", line)))

            async def robot():
                async with connect(url) as ws:
                    await send(ws, dataclasses.replace(HELLO, token=token))
                    return await recv(ws)

            assert asyncio.run(robot()) == Welcome(1, (Op.HEAR, Op.SPEAK, Op.NARRATE), (("hear", "scripted"), ("speak", "tone"), ("narrate", "echo")))
        finally:
            server.send_signal(2)
        assert server.wait(5) == 0
        assert server.stderr.read().strip().endswith("stopped")
