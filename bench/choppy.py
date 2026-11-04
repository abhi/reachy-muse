"""Replay speech incidents through production routing and sentence playback.

No models, network, devices, or services are used. Run this file with another
checkout's src on PYTHONPATH to measure that revision with identical inputs.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import replace
import hashlib
import inspect
import json
from pathlib import Path
import time
import wave

import numpy as np

from reachy_muse.capabilities import Mode, SpokenLine
from reachy_muse.conversation import VoiceConversation
from reachy_muse.failover import CompanionRoute, FailoverVoice
from reachy_muse.local_backends import LocalVoice, backends_for
from reachy_muse.local_speech import PreparedSpeech


ROOT = Path(__file__).resolve().parents[1]
TEXTS = ("Remind me at eight to call Maya, not at nine.",
         "Actually, make that tomorrow afternoon.",
         "Do not turn off the kitchen lights.")
RECORDINGS = ("reminder", "correction", "negation")
FRAMES = (48000, 64000, 56000)


class RecordedPCM(np.ndarray):
    """Keep source identity beside untouched samples through slicing/copying."""

    def __array_finalize__(self, source):
        self.voice = getattr(source, "voice", "unknown")


def recordings():
    result = []
    for name, frames in zip(RECORDINGS, FRAMES):
        path = ROOT / "tests" / "companion_server" / "data" / f"{name}.wav"
        with wave.open(str(path)) as audio:
            assert audio.getframerate() == 16000 and audio.getnchannels() == 1
            result.append(np.frombuffer(audio.readframes(frames), dtype="<i2").astype(np.float32) / 32768)
    return result


class RecordedSpeech:
    def __init__(self, audio, delays, voice):
        self.audio, self.delays, self.voice = audio, delays, voice
        self._lock = asyncio.Lock()

    def prepare(self, text, rate):
        stream = self.stream(text, rate)
        if "sample_rate" in inspect.signature(PreparedSpeech).parameters:
            return PreparedSpeech(stream, rate)
        return PreparedSpeech(stream)

    async def stream(self, text, rate):
        assert rate == 16000
        index = TEXTS.index(text)
        async with self._lock:
            await asyncio.sleep(self.delays[index])
            pcm = self.audio[index].view(RecordedPCM)
            pcm.voice = self.voice
            # The packet size matches the Piper child's 4096-byte PCM16 writes.
            for offset in range(0, len(pcm), 2048):
                yield pcm[offset:offset + 2048]


class CompanionVoice(LocalVoice):
    available = True


class Link:
    up = True


class MeasuredHardware:
    sample_rate = output_sample_rate = 16000

    def __init__(self):
        self.accepted = []
        self.cleared = 0

    def play_audio(self, pcm):
        self.accepted.append((time.monotonic(), pcm.copy()))

    def clear_audio(self):
        self.cleared += 1

    def set_state(self, state, *, expression=None):
        pass


async def replay(name, *, short=False):
    """Measure actual output acceptance, including starvation between packets."""
    audio = recordings()
    if short:
        audio = [pcm[:1600] for pcm in audio]
    if name == "morning_slow_companion":
        remote_delays, local_delays, initially_up, reply_up = (.2, 2.2, .2), (.8, 3.5, .2), True, True
    elif name == "hearing_recovers_before_reply":
        remote_delays, local_delays, initially_up, reply_up = (.2, .2, .2), (4.831, 13.150, .870), False, True
    elif name == "cpu_fallback_pipeline":
        remote_delays, local_delays, initially_up, reply_up = (0, 0, 0), (.2, 1.3, .4), False, False
    elif name == "cpu_fallback_incident_stall":
        remote_delays, local_delays, initially_up, reply_up = (0, 0, 0), (4.831, 13.150, .870), False, False
    else:
        raise ValueError(name)
    if short:
        remote_delays = local_delays = (.01, .04, .02)
    link = Link()
    link.up = initially_up
    route = CompanionRoute(link)
    route.begin_turn()
    local = LocalVoice(RecordedSpeech(audio, local_delays, "piper"))
    companion = CompanionVoice(RecordedSpeech(audio, remote_delays, "companion"))
    voice = FailoverVoice(companion, local, route)
    hardware = MeasuredHardware()
    backends = replace(backends_for(Mode.ON_ROBOT, None, speech=local.speech), voice=voice)
    conversation = VoiceConversation(None, hardware, backends=backends)
    link.up = reply_up
    started = time.monotonic()
    try:
        await conversation._speak_lines("recorded-reply", [SpokenLine(text, None, "answer") for text in TEXTS])
    finally:
        await conversation._moments.close()
    if not hardware.accepted:
        raise AssertionError("replay delivered no PCM")
    actual = np.concatenate([pcm for _, pcm in hardware.accepted])
    expected = np.concatenate(audio)
    if not np.array_equal(actual, expected):
        raise AssertionError("replay dropped, repeated, changed, or reordered recorded PCM")
    first = hardware.accepted[0][0]
    end = first
    gaps = []
    voices = []
    for accepted, pcm in hardware.accepted:
        if accepted - end > .01:
            gaps.append(accepted - end)
        end = max(end, accepted) + len(pcm) / hardware.output_sample_rate
        if not voices or voices[-1] != pcm.voice:
            voices.append(pcm.voice)
    return {
        "first_audio_s": round(first - started, 3),
        "gaps_over_10ms_s": [round(gap, 3) for gap in gaps],
        "max_gap_s": round(max(gaps, default=0), 3),
        "voice_switches": len(voices) - 1,
        "voices": voices,
        "pcm_frames": len(actual),
        "pcm_sha256": hashlib.sha256(actual.tobytes()).hexdigest(),
        "queue_clears": hardware.cleared,
    }


CASES = ("morning_slow_companion", "hearing_recovers_before_reply",
         "cpu_fallback_pipeline", "cpu_fallback_incident_stall")


async def run(names=CASES):
    return {name: await replay(name) for name in names}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--case", choices=CASES)
    args = parser.parse_args()
    result = asyncio.run(run((args.case,) if args.case else CASES))
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
