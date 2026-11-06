"""Content-free wake decisions shared by live capture and offline replay."""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import asdict, dataclass
import io
import json
import logging
import math
import re
import time
import wave

log = logging.getLogger(__name__)
RATE = 16000


def contains_phrase(text: str, phrase: str = "hey muse") -> bool:
    words = re.findall(r"[a-z]+", text.lower())
    wanted = re.findall(r"[a-z]+", phrase.lower())
    if wanted == ["hey", "muse"]:
        # The local streaming model spells the identical /heɪ/ sound HAY.
        # Keep the noun exact: NEWS, MUSIC, MOOSE and MUSEUM remain competitors.
        words = ["hey" if word == "hay" else word for word in words]
    return any(words[i:i + len(wanted)] == wanted for i in range(len(words) - len(wanted) + 1))


def pcm_wav(pcm: bytes) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as recording:
        recording.setparams((1, 2, RATE, 0, "NONE", "not compressed"))
        recording.writeframes(pcm)
    return output.getvalue()


@dataclass(frozen=True)
class WakeContext:
    armed: bool = True
    blocked: bool = False
    speaking: bool = False
    playback: tuple[str, ...] = ()


@dataclass(frozen=True)
class Verification:
    result: str
    source: str


@dataclass(frozen=True)
class WakeDecision:
    candidate: int
    accepted: bool
    reason: str
    score: float | None
    score_kind: str
    threshold: float | None
    boost: float | None
    replay_floor: float | None
    verifier: str
    verification: str
    inference_ms: float
    verification_ms: float
    alignment_ms: float
    alignment_available: bool


class WakeVerifier:
    """Race warmed local hearing against a short companion allowance.

    A companion NO_MATCH is authoritative. Local work starts immediately, but
    is used only after companion unavailability or its allowance has elapsed.
    No call creates a connection or sends anything to Muse.
    """

    def __init__(self, *, local=None, companion=lambda: None,
                 budget_s: float = .18, companion_s: float = .06):
        if not all(math.isfinite(x) for x in (budget_s, companion_s)) or not 0 < companion_s <= budget_s <= .3:
            raise ValueError("Wake verification needs 0 < companion allowance <= budget <= .3 seconds")
        self.local, self.companion = local, companion
        self.budget_s, self.companion_s = budget_s, companion_s

    async def _companion_text(self, pcm: bytes) -> str:
        link = self.companion()
        if link is None:
            raise RuntimeError("No companion hearing")
        return await link.transcribe_wake(pcm)

    async def verify(self, pcm: bytes, phrase: str) -> Verification:
        async def recognize(work, source):
            try:
                text = await work
                if not isinstance(text, str):
                    return Verification("unavailable", source)
                return Verification("match" if contains_phrase(text, phrase) else "no_match", source)
            except Exception:
                # Exception strings can include hypotheses or remote payloads.
                return Verification("unavailable", source)

        deadline = time.monotonic() + self.budget_s
        tasks = []
        local = None
        if self.local is not None:
            local = asyncio.create_task(recognize(self.local.transcribe(pcm_wav(pcm)), "local"))
            tasks.append(local)
        companion = asyncio.create_task(recognize(self._companion_text(pcm), "companion"))
        tasks.append(companion)
        try:
            done, _ = await asyncio.wait({companion}, timeout=self.companion_s)
            if done and companion.result().result != "unavailable":
                return companion.result()
            if not done:
                companion.cancel()
            if local is not None:
                done, _ = await asyncio.wait({local}, timeout=max(0, deadline - time.monotonic()))
                if done:
                    return local.result()
            return Verification("timeout" if any(not task.done() for task in tasks) else "unavailable", "none")
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)


class MicrophoneLevels:
    """One audio-minute window, independent of recognition and wake resets."""

    def __init__(self):
        self.samples = self.clipped = 0
        self.energy = self.peak = 0.0
        self.started = None

    def feed(self, audio):
        import numpy as np

        if self.started is None:
            self.started = time.monotonic()
        for start in range(0, len(audio), RATE):
            part = audio[start:start + RATE]
            self.samples += len(part)
            self.energy += float(np.dot(part, part))
            self.peak = max(self.peak, float(np.max(np.abs(part))))
            self.clipped += int(np.count_nonzero(np.abs(part) >= .999))
            if self.samples >= 60 * RATE:
                elapsed = max(.000001, time.monotonic() - self.started)
                log.info("Wake levels %s", json.dumps({
                    "audio_s": round(self.samples / RATE, 3), "wall_s": round(elapsed, 3),
                    "rms": round(math.sqrt(self.energy / self.samples), 6),
                    "peak": round(self.peak, 6), "clipped": round(self.clipped / self.samples, 6),
                }, sort_keys=True))
                self.samples = self.clipped = 0
                self.energy = self.peak = 0.0
                self.started = time.monotonic()


class WakePolicy:
    """Decide one candidate at a time; the caller retains capture/handoff ownership."""

    def __init__(self, verifier: WakeVerifier | None = None, *, fallback_threshold: float = .25,
                 cooldown_s: float = .8, verification_window_s: float = 1.5):
        if not math.isfinite(fallback_threshold) or not 0 < fallback_threshold <= 1:
            raise ValueError("Fallback threshold must be between zero and one")
        if not math.isfinite(cooldown_s) or cooldown_s < 0:
            raise ValueError("Wake cooldown must be finite and nonnegative")
        if not math.isfinite(verification_window_s) or not 1 <= verification_window_s <= 2:
            raise ValueError("Wake verification audio must cover between one and two seconds")
        self.verifier = verifier or WakeVerifier()
        self.fallback_threshold = fallback_threshold
        self.cooldown_samples = int(cooldown_s * RATE)
        self.verification_samples = int(verification_window_s * RATE)
        self.levels = MicrophoneLevels()
        self.last_decision: WakeDecision | None = None
        self._sequence = self._samples = 0
        self._last_accept = -self.cooldown_samples
        self._audio = deque()
        self._audio_samples = 0
        self._echo_sample = 0

    def reset(self, *, new_trial=False):
        """Discard candidate context after a gap; keep exposure and cooldown clocks."""
        self._audio.clear()
        self._audio_samples = 0
        self._echo_sample = 0
        self.last_decision = None
        if new_trial:
            self._last_accept = self._samples - self.cooldown_samples

    async def feed(self, detector, samples, *, context: WakeContext = WakeContext(),
                   valid=lambda: True) -> bool:
        import numpy as np

        audio = np.asarray(samples, dtype=np.float32)
        if audio.ndim != 1 or not np.isfinite(audio).all() or (np.abs(audio) > 1).any():
            raise ValueError("Wake audio must be finite normalized mono floats")
        self.last_decision = None
        if not len(audio):
            return False
        self.levels.feed(audio)
        self._samples += len(audio)
        echo = contains_phrase(" ".join(context.playback), detector.phrase)
        if echo:
            self._echo_sample = self._samples
        started = time.monotonic()
        replay = getattr(detector, "candidate_replay", False)
        found = await asyncio.to_thread(detector.feed, audio, **({"align": False} if replay else {}))
        inference_ms = (time.monotonic() - started) * 1000
        candidate = getattr(detector, "last_candidate", None) if found else None
        consumed = len(audio) - getattr(candidate, "remaining", 0)
        self._audio.append(audio[:consumed].copy())
        self._audio_samples += consumed
        while self._audio_samples > self.verification_samples:
            first = self._audio.popleft()
            excess = self._audio_samples - self.verification_samples
            taken = min(excess, len(first))
            if taken < len(first):
                self._audio.appendleft(first[taken:])
            self._audio_samples -= taken
        if not found:
            return False
        self._sequence += 1
        reason, source, floor = "accepted", "none", None
        verification = "not_run"
        verification_ms = alignment_ms = 0.0
        if echo or self._echo_sample > self._samples - getattr(candidate, "segment_lookback_samples", 0):
            reason = "echo"
        elif context.blocked or context.speaking and not context.playback:
            reason = "speaking"
        elif not context.armed or self._samples - self._last_accept < self.cooldown_samples:
            reason = "cooldown"
        elif getattr(candidate, "phrase_verified", False):
            source, verification = "native", "match"
        else:
            pcm = np.rint(np.concatenate(tuple(self._audio)) * 32767).astype("<i2").tobytes()
            started = time.monotonic()
            verified = await self.verifier.verify(pcm, detector.phrase)
            verification = verified.result
            verification_ms = (time.monotonic() - started) * 1000
            source = verified.source
            if verified.result == "no_match":
                reason = "verification_rejected"
            elif verified.result != "match":
                if context.speaking:
                    reason = "speaking"
                elif replay:
                    strong = await asyncio.to_thread(detector.replay_candidate, self.fallback_threshold)
                    floor = self.fallback_threshold if strong else None
                    reason = "accepted" if strong else "below_threshold"
                    source = "conservative_replay"
                else:
                    # A non-permissive backend retains its original standalone behavior.
                    source = "native"
            if reason == "accepted" and not valid():
                reason = "capture_gap"
            if reason == "accepted" and replay:
                started = time.monotonic()
                await asyncio.to_thread(detector.align_candidate)
                alignment_ms = (time.monotonic() - started) * 1000
        if reason == "accepted" and not valid():
            reason = "capture_gap"
        accepted = reason == "accepted"
        if accepted:
            self._last_accept = self._samples
        candidate = getattr(detector, "last_candidate", None)
        self.last_decision = WakeDecision(
            self._sequence, accepted, reason, getattr(candidate, "score", None),
            getattr(candidate, "score_kind", "unavailable"),
            getattr(detector, "threshold", None), getattr(detector, "boost", None), floor, source,
            verification,
            round(inference_ms, 3), round(verification_ms, 3), round(alignment_ms, 3),
            accepted and getattr(detector, "last_detection", None) is not None)
        log.info("Wake candidate %s", json.dumps(asdict(self.last_decision), sort_keys=True))
        self._audio.clear()
        self._audio_samples = 0
        return accepted
