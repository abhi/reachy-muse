"""Offline measurements of Reachy's motion for Muse realtime replies; no devices, network, or models.

    PYTHONPATH=src:tests python bench/realtime_expressive.py [--replies 24] [--json out.json]

Replies are synthesized speech-like PCM (voiced syllables, word gaps, comma and sentence pauses,
stressed words, rising questions) with a known start sample for every sentence. Each reply is
replayed in virtual time in the order a call delivers it: transcript deltas and 24 kHz chunks
arrive ahead of playback, and the speaker asks for each chunk's motion marks when it plays the
chunk. Two speech profiles are synthesized: ``clear`` pauses longer between sentences than at
commas over a quiet floor; ``tight`` makes sentence and comma pauses overlap over a louder floor.
Three delivery regimes are measured for each:

- ``ahead``: words and audio arrive in bursts at 2.5x real time, words just before their audio.
- ``with_audio``: each word arrives with the audio chunk that holds its start.
- ``late``: each word arrives when playback reaches it.

First audio and speaker CPU use the real ``RealtimeSpeaker`` with its PyAV resampler.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import time

import numpy as np

from reachy_muse.contract.expressions import Expression
from reachy_muse.motion.reply_motion import ReplyMotion
from reachy_muse.realtime_call import RealtimeSpeaker

RATE = 24000
CHUNK = 4800
REPLIES = (
    "Sure, I can help with that. Your flight leaves at nine tomorrow morning. Want me to set an alarm?",
    "Great news! The package arrived this afternoon, and it's waiting by the door.",
    "I'm sorry, the restaurant is fully booked tonight. There's a table free tomorrow at seven. Should I take it?",
    "It's sunny and warm today, around seventy degrees. Perfect weather for a walk!",
    "Here's the plan. First, we pick up groceries. Then we stop at the pharmacy. Finally, we head home.",
    "Wow, that's a big number! The total comes to four hundred and twelve dollars.",
    "Yes, absolutely. I added milk, eggs, and bread to your list. Anything else?",
    "Unfortunately, your train was cancelled. The next one leaves in forty minutes, from platform six.",
)


PROFILES = {
    "clear": {"sentence_pause": (.25, .6), "comma_pause": (.18, .3), "noise": 40},
    "tight": {"sentence_pause": (.12, .3), "comma_pause": (.12, .25), "noise": 150},
}


def synthesize(text: str, rng: np.random.Generator, profile=PROFILES["clear"]):
    """Speech-like int16 PCM for ``text`` and each sentence's true first sample."""
    sentences = re.findall(r"[^.!?]+[.!?]", text)
    parts, starts, at = [], [], 0
    for index, sentence in enumerate(sentences):
        sentence = sentence.strip()
        starts.append(at)
        chars_per_s = rng.uniform(12, 18)
        words = sentence.split()
        for position, word in enumerate(words):
            core = word.strip(",.!?")
            seconds = max(.12, len(core) / chars_per_s)
            t = np.arange(int(seconds * RATE)) / RATE
            rising = word.endswith("?")
            f0 = 140 + (np.linspace(0, 90, len(t)) if rising else -np.linspace(0, 20, len(t)))
            syllables = max(1, round(len(core) / 3))
            envelope = np.sin(np.pi * np.clip(t * syllables / seconds % 1, 0, 1)) ** .6
            amp = rng.uniform(3000, 5000) * (2.4 if rng.random() < .12 else 1)
            phase = 2 * np.pi * np.cumsum(f0) / RATE
            voice = amp * envelope * (np.sin(phase) + .4 * np.sin(2 * phase) + .2 * np.sin(3 * phase))
            parts.append(voice)
            at += len(voice)
            if position < len(words) - 1:
                gap = rng.uniform(*profile["comma_pause"]) if word.endswith(",") else rng.uniform(.03, .08)
                parts.append(rng.normal(0, profile["noise"], int(gap * RATE)))
                at += parts[-1].size
        if index < len(sentences) - 1:
            parts.append(rng.normal(0, profile["noise"], int(rng.uniform(*profile["sentence_pause"]) * RATE)))
            at += parts[-1].size
    pcm = np.clip(np.concatenate(parts), -32768, 32767).astype("<i2")
    return pcm, [sentence.strip() for sentence in sentences], starts


def word_starts(text: str, sentences, starts, pcm):
    """The sample each word's audio starts at, by its share of its sentence's characters."""
    bounds = starts[1:] + [len(pcm)]
    positions, cursor = [], 0
    for sentence, start, end in zip(sentences, starts, bounds):
        offset = 0
        for word in sentence.split():
            position = text.index(word, cursor)
            positions.append((position, start + int((end - start) * offset / max(1, len(sentence)))))
            offset += len(word) + 1
            cursor = position + len(word)
    return positions


def replay(text: str, pcm, sentences, starts, regime: str, *, muse_moving=lambda: False):
    """Deliver one reply in virtual time; return the scheduled sentences and marks."""
    motion = ReplyMotion(muse_moving=muse_moving)
    events = []
    words = word_starts(text, sentences, starts, pcm)
    playback_start = .4
    for index, (position, sample) in enumerate(words):
        end = words[index + 1][0] if index + 1 < len(words) else len(text)
        delta = text[position:end]
        if index == 0:
            delta = text[:end]
        if regime == "ahead":
            at = sample / RATE / 2.5 - .02
        elif regime == "with_audio":
            at = (sample // CHUNK * CHUNK) / RATE / 2.5
        else:
            at = playback_start + sample / RATE
        events.append((at, 0, "words", delta))
    for offset in range(0, len(pcm), CHUNK):
        events.append((offset / RATE / 2.5, 1, "audio", offset))
        events.append((playback_start + offset / RATE, 2, "play", offset))
    transcript_end = max(event[0] for event in events if event[2] == "words")
    audio_end = len(pcm) / RATE / 2.5 + .01
    events.append((max(transcript_end, audio_end), 3, "done", None))
    events.append((audio_end, 4, "audio_done", None))
    marks = []
    cpu = 0.0
    transcript_done = False
    received_text = ""
    for _, _, kind, value in sorted(events, key=lambda item: (item[0], item[1])):
        began = time.process_time()
        if kind == "words":
            assert not transcript_done
            received_text += value
            motion.words("r1", value)
        elif kind == "audio":
            motion.audio("r1", value, pcm[value:value + CHUNK].tobytes())
        elif kind == "done":
            assert received_text == text
            transcript_done = True
            motion.transcript("r1", text)
        elif kind == "audio_done":
            motion.audio_done("r1")
        else:
            count = len(pcm[value:value + CHUNK])
            marks += [(value + mark.sample, mark) for mark in motion.marks("r1", value, count)]
        cpu += time.process_time() - began
    return motion.scheduled("r1"), marks, cpu, motion.closing("r1")


def quantiles(values):
    if not values:
        return {}
    return {name: round(float(np.percentile(values, q)), 1) for name, q in (("p50", 50), ("p95", 95), ("max", 100))}


def alignment(replies: int, seed: int, profile):
    rng = np.random.default_rng(seed)
    report = {}
    for regime in ("ahead", "with_audio", "late"):
        errors, confirmed, shown, late, expressive = [], 0, 0, 0, 0
        total = accents = 0
        audio_s = cpu = 0.0
        closings = 0
        local = np.random.default_rng(rng.integers(1 << 32))
        for index in range(replies):
            text = REPLIES[index % len(REPLIES)]
            pcm, sentences, starts = synthesize(text, local, profile)
            scheduled, marks, spent, closing = replay(text, pcm, sentences, starts, regime)
            audio_s += len(pcm) / RATE
            cpu += spent
            closings += closing is not None
            for item in scheduled:
                if item.index >= len(starts):
                    continue
                total += 1
                errors.append(abs(item.sample - starts[item.index]) / RATE * 1000)
                confirmed += item.confirmed
                shown += item.shown
                late += item.late
            for _, mark in marks:
                accents += mark.accent is not None
                expressive += mark.sentence is not None and mark.sentence.expression is not Expression.NEUTRAL
        report[regime] = {
            "sentences": total,
            "start_error_ms": quantiles(errors),
            "within_150ms": round(sum(error <= 150 for error in errors) / max(1, len(errors)), 3),
            "pause_confirmed": round(confirmed / max(1, total), 3),
            "shown": round(shown / max(1, total), 3),
            "late": round(late / max(1, total), 3),
            "non_neutral_expressions": expressive,
            "accents_per_s": round(accents / audio_s, 2),
            "replies_with_closing": f"{closings}/{replies}",
            "motion_cpu_ms_per_audio_s": round(cpu / audio_s * 1000, 3),
        }
    return report


class Hardware:
    sample_rate = 16000
    output_sample_rate = 16000

    def __init__(self):
        self.pushed = []

    def play_audio(self, samples, *, sentence=None, accents=()):
        self.pushed.append(time.perf_counter())

    def finish_sentence(self):
        pass

    def clear_audio(self):
        pass


def first_audio(trials: int):
    """Chunk arrival to the first speaker push, through the real speaker, resampler and marks."""
    rng = np.random.default_rng(7)
    pcm, _, _ = synthesize(REPLIES[0], rng)

    async def one():
        hardware = Hardware()
        motion = ReplyMotion()

        async def nothing():
            pass

        speaker = RealtimeSpeaker(hardware, motion, on_audible=lambda text, until: None,
                                  on_start=nothing, on_idle=nothing)
        motion.words("r1", REPLIES[0])
        first = pcm[:CHUNK].tobytes()
        accepted = time.perf_counter()
        motion.audio("r1", 0, first)
        speaker.chunk("r1", 0, first)
        while not hardware.pushed:
            await asyncio.sleep(0)
        latency = (hardware.pushed[0] - accepted) * 1000
        await speaker.close()
        return latency

    async def cpu():
        hardware = Hardware()
        motion = ReplyMotion()

        async def nothing():
            pass

        speaker = RealtimeSpeaker(hardware, motion, on_audible=lambda text, until: None,
                                  on_start=nothing, on_idle=nothing)
        began = time.process_time()
        motion.words("r1", REPLIES[0])
        for offset in range(0, len(pcm), CHUNK):
            motion.audio("r1", offset, pcm[offset:offset + CHUNK].tobytes())
            speaker.chunk("r1", offset, pcm[offset:offset + CHUNK].tobytes())
        await asyncio.sleep(len(pcm) / RATE + .5)
        spent = time.process_time() - began
        await speaker.close()
        return spent / (len(pcm) / RATE) * 1000

    latencies = [asyncio.run(one()) for _ in range(trials)]
    return {"first_push_ms": quantiles(latencies), "speaker_and_motion_cpu_ms_per_audio_s": round(asyncio.run(cpu()), 2)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--replies", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20261105)
    parser.add_argument("--json")
    args = parser.parse_args()
    report = {"alignment": {name: alignment(args.replies, args.seed, profile) for name, profile in PROFILES.items()},
              "speaker": first_audio(30)}
    text = json.dumps(report, indent=2)
    print(text)
    if args.json:
        with open(args.json, "w") as output:
            output.write(text + "\n")


if __name__ == "__main__":
    main()
