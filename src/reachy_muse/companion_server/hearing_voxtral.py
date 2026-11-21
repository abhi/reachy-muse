"""Hearing for C2: Voxtral Mini 4B Realtime streams the partials, Silero VAD and Smart Turn v3
decide when the turn is over, and Parakeet TDT v2 can re-transcribe the whole utterance at the end.

One set of models serves every hear stream. Their own thread runs each call in turn,
so the event loop never waits on inference and two streams never share a model at once.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from concurrent.futures import Future
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Sequence, TypeVar

import numpy as np

from reachy_muse.companion_server.watchdog import EngineExecutor, watchdog
from reachy_muse.companion_server.engines import HearEvent, Partial, SpeechEnded

log = logging.getLogger("reachy_muse.companion_server")

RATE = 16_000
VOXTRAL = "mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit"
PARAKEET = "mlx-community/parakeet-tdt-0.6b-v2"
SILERO = ("onnx-community/silero-vad", "onnx/model.onnx")
SMART_TURN = ("pipecat-ai/smart-turn-v3", "smart-turn-v3.2-cpu.onnx")
VAD_WINDOW = 512  # Silero's window at 16 kHz, 32 ms
VAD_CONTEXT = 64  # Silero expects the last 64 samples of the previous window in front of each one
TURN_SECONDS = 8  # Smart Turn reads the last 8 s of the turn
# Parakeet's final pass is one native call that reports no progress, and it grows with the utterance:
# 2.32 / 2.77 s (p50 / p95) for an 8.2 s clip while Kokoro spoke, 0.34 s per second of audio at p95.
# A real-time factor of 1.0 is about 3x that; the server's 3 s streaming deadline is its floor.
FINAL_PASS_RTF = 1.0

T = TypeVar("T")
_timing = threading.local()


@contextmanager
def _stage(name: str):
    started = time.perf_counter()
    try:
        yield
        watchdog.progress()
    finally:
        stages = getattr(_timing, "stages", None)
        if stages is not None:
            stages[name] = stages.get(name, 0.0) + time.perf_counter() - started


@dataclass(frozen=True)
class HearingConfig:
    delay_ms: int = 480
    final_pass: bool = True
    speech_threshold: float = 0.5
    # Pauses inside a sentence ran to 0.38 s on the bench corpus, and Smart Turn called most of them complete.
    turn_check_after_s: float = 0.5
    turn_threshold: float = 0.5
    silence_endpoint_s: float = 3.0  # a turn Smart Turn calls unfinished still ends after this much silence


@dataclass(frozen=True)
class _Models:
    voxtral: Any
    parakeet: Any
    vad: Any
    turn: Any
    features: Any


class VoxtralHearing:
    name = "voxtral-mini-4b-realtime"

    def __init__(self, config: HearingConfig = HearingConfig()) -> None:
        self.config = config
        self._thread = EngineExecutor("hearing")
        self._models: Optional[_Models] = None

    async def load(self) -> None:
        self._models = await asyncio.wrap_future(self._thread.submit(self._load))

    async def run(self, work: Callable[..., T], *args: Any) -> T:
        return await asyncio.wrap_future(self.submit(work, *args))

    def submit(self, work: Callable[..., T], *args: Any) -> "Future[T]":
        """Queue one call on the hearing worker, from any thread, logging its stages if it is slow."""
        queued = time.perf_counter()

        def timed():
            started = time.perf_counter()
            _timing.stages = {}
            try:
                return work(*args)
            finally:
                stages = _timing.stages
                del _timing.stages
                elapsed = time.perf_counter() - queued
                if elapsed > 1.0:
                    log.warning("hearing call=%s total=%.3fs queue=%.3fs stages=%s", work.__name__.lstrip("_"),
                                elapsed, started - queued, " ".join(f"{name}:{seconds:.3f}s" for name, seconds in stages.items()))

        return self._thread.submit(timed)

    def open(self, pre_roll_frames: int) -> _Session:
        # The pre-roll is transcribed like any other audio: it can hold the start of the request.
        assert self._models is not None, "load() first"
        return _Session(self)

    def _load(self) -> _Models:
        import onnxruntime
        from huggingface_hub import hf_hub_download
        from mlx_audio.stt.utils import load
        from parakeet_mlx import from_pretrained
        from transformers import WhisperFeatureExtractor

        started = time.perf_counter()
        options = onnxruntime.SessionOptions()
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = 1
        models = _Models(
            voxtral=load(VOXTRAL),
            parakeet=from_pretrained(PARAKEET) if self.config.final_pass else None,
            vad=onnxruntime.InferenceSession(hf_hub_download(*SILERO), options, providers=["CPUExecutionProvider"]),
            turn=onnxruntime.InferenceSession(hf_hub_download(*SMART_TURN), options, providers=["CPUExecutionProvider"]),
            features=WhisperFeatureExtractor(chunk_length=TURN_SECONDS),
        )
        warm = _Utterance(models, self.config)
        noise = (np.random.default_rng(0).standard_normal(RATE) * 0.05).astype(np.float32)
        warm.feed(noise)
        warm.turn_complete()
        warm.final()
        log.info("hearing %s ready in %.1f s (final pass %s)", VOXTRAL, time.perf_counter() - started, "parakeet" if self.config.final_pass else "off")
        return models


class _Session:
    def __init__(self, hearing: VoxtralHearing) -> None:
        self.hearing = hearing
        self.utterance: Optional[_Utterance] = None
        self.said = ""
        self._closed = threading.Event()
        self._final: Optional[Future] = None

    async def feed(self, pcm: bytes) -> Sequence[HearEvent]:
        return await self.hearing.run(self._feed, pcm)

    def _feed(self, pcm: bytes) -> Sequence[HearEvent]:
        try:
            _check_open(self._closed)
            # Creating a streaming session can evaluate MLX arrays, including after idle.
            # Construction, inference and release all belong to the same model worker.
            if self.utterance is None:
                self.utterance = _Utterance(self.hearing._models, self.hearing.config, cancelled=self._closed)
            _check_open(self._closed)
            samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768
            ended = self.utterance.feed(samples)
            events: List[HearEvent] = []
            if self.utterance.text != self.said:
                self.said = self.utterance.text
                events.append(Partial(self.said))
            if ended:
                # Queued from this call, the final pass runs next on this worker, as early as it ran inline,
                # while the server acts on SpeechEnded.
                self._final = self.hearing.submit(self._finish)
                events.append(SpeechEnded())
            return events
        except _Abandoned:
            return []

    async def finish(self) -> str:
        return await asyncio.wrap_future(self._final or self.hearing.submit(self._finish))

    def _finish(self) -> str:
        try:
            _check_open(self._closed)
            return self.utterance.final() if self.utterance is not None else ""
        except _Abandoned:
            return ""

    def cancel(self) -> None:
        self._closed.set()

    async def close(self) -> None:
        self.cancel()
        # Native work cannot be interrupted; release its arrays on the worker when it returns.
        self.hearing._thread.submit(self._release)

    def _release(self) -> None:
        self.utterance = None


class _Utterance:
    """One turn's state. Every method runs on the hearing thread."""

    def __init__(self, models: _Models, config: HearingConfig, *, cancelled=None) -> None:
        self.models = models
        self.config = config
        self.cancelled = cancelled if cancelled is not None else threading.Event()
        with _stage("voxtral_open"):
            self.stream = models.voxtral.create_streaming_session(transcription_delay_ms=config.delay_ms)
        self.raw = ""
        self.audio: List[np.ndarray] = []
        self.unread = np.zeros(0, np.float32)
        self.context = np.zeros(VAD_CONTEXT, np.float32)
        self.vad_state = np.zeros((2, 1, 128), np.float32)
        self.heard_speech = False
        self.silence_s = 0.0
        self.turn_checked = False

    @property
    def text(self) -> str:
        return self.raw.strip()

    def feed(self, samples: np.ndarray) -> bool:
        """Whether this audio ends the turn."""
        _check_open(self.cancelled)
        self.audio.append(samples)
        with _stage("voxtral_feed"):
            self.stream.feed(samples)
        for _ in range(16):
            _check_open(self.cancelled)
            with _stage("voxtral_step"):
                self.raw += "".join(self.stream.step(max_decode_tokens=1))
        _check_open(self.cancelled)
        return self._turn_over(samples)

    def final(self) -> str:
        _check_open(self.cancelled)
        if self.config.final_pass:
            from mlx.core import array
            from parakeet_mlx.audio import get_logmel

            parakeet = self.models.parakeet
            with _stage("parakeet_features"):
                mel = get_logmel(array(np.concatenate(self.audio)), parakeet.preprocessor_config)
            _check_open(self.cancelled)
            watchdog.progress(quiet_s=FINAL_PASS_RTF * sum(map(len, self.audio)) / RATE)
            with _stage("parakeet_final"):
                return parakeet.generate(mel)[0].text.strip()
        with _stage("voxtral_close"):
            self.stream.close()
        while not self.stream.done:
            _check_open(self.cancelled)
            with _stage("voxtral_final"):
                self.raw += "".join(self.stream.step(max_decode_tokens=1))
        return self.text

    def _turn_over(self, samples: np.ndarray) -> bool:
        self.unread = np.concatenate((self.unread, samples))
        while len(self.unread) >= VAD_WINDOW:
            _check_open(self.cancelled)
            window, self.unread = self.unread[:VAD_WINDOW], self.unread[VAD_WINDOW:]
            if self._speech_probability(window) >= self.config.speech_threshold:
                self.heard_speech, self.silence_s, self.turn_checked = True, 0.0, False
                continue
            self.silence_s += VAD_WINDOW / RATE
            if not self.heard_speech:
                continue
            if self.silence_s >= self.config.silence_endpoint_s:
                return True
            if not self.turn_checked and self.silence_s >= self.config.turn_check_after_s:
                self.turn_checked = True
                if self.turn_complete():
                    return True
        return False

    def _speech_probability(self, window: np.ndarray) -> float:
        framed = np.concatenate((self.context, window))[None, :]
        self.context = window[-VAD_CONTEXT:]
        with _stage("vad"):
            probability, self.vad_state = self.models.vad.run(None, {"input": framed, "state": self.vad_state, "sr": np.array(RATE, np.int64)})
        return float(probability[0][0])

    def turn_complete(self) -> bool:
        _check_open(self.cancelled)
        tail = np.concatenate(self.audio)[-TURN_SECONDS * RATE :]
        tail = np.pad(tail, (TURN_SECONDS * RATE - len(tail), 0))
        with _stage("turn_features"):
            features = self.models.features(tail, sampling_rate=RATE, return_tensors="np", padding="max_length", max_length=TURN_SECONDS * RATE, truncation=True, do_normalize=True)
        _check_open(self.cancelled)
        with _stage("turn"):
            (probability,) = self.models.turn.run(None, {"input_features": features.input_features.astype(np.float32)})
        return float(probability[0][0]) >= self.config.turn_threshold


class _Abandoned(Exception):
    pass


def _check_open(cancelled: threading.Event) -> None:
    if cancelled.is_set():
        raise _Abandoned()
