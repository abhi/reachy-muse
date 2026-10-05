"""Hearing for C2: Voxtral Mini 4B Realtime streams the partials, Silero VAD and Smart Turn v3
decide when the turn is over, and Parakeet TDT v2 can re-transcribe the whole utterance at the end.

One set of models serves every hear stream. Their own thread runs each call in turn,
so the event loop never waits on inference and two streams never share a model at once.
"""

from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Sequence, TypeVar

import numpy as np

from reachy_muse.companion_server.engines import Endpoint, HearEvent, Partial

log = logging.getLogger("reachy_muse.companion_server")

RATE = 16_000
VOXTRAL = "mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit"
PARAKEET = "mlx-community/parakeet-tdt-0.6b-v2"
SILERO = ("onnx-community/silero-vad", "onnx/model.onnx")
SMART_TURN = ("pipecat-ai/smart-turn-v3", "smart-turn-v3.2-cpu.onnx")
VAD_WINDOW = 512  # Silero's window at 16 kHz, 32 ms
VAD_CONTEXT = 64  # Silero expects the last 64 samples of the previous window in front of each one
TURN_SECONDS = 8  # Smart Turn reads the last 8 s of the turn

T = TypeVar("T")


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
        self._thread = ThreadPoolExecutor(1, thread_name_prefix="hearing")
        self._models: Optional[_Models] = None

    async def load(self) -> None:
        self._models = await self.run(self._load)

    async def run(self, work: Callable[..., T], *args: Any) -> T:
        return await asyncio.get_running_loop().run_in_executor(self._thread, work, *args)

    def open(self, pre_roll_frames: int) -> _Session:
        # The pre-roll is transcribed like any other audio: it can hold the start of the request.
        assert self._models is not None, "load() first"
        return _Session(self, _Utterance(self._models, self.config))

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
    def __init__(self, hearing: VoxtralHearing, utterance: _Utterance) -> None:
        self.hearing = hearing
        self.utterance = utterance
        self.said = ""

    async def feed(self, pcm: bytes) -> Sequence[HearEvent]:
        samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768
        final = await self.hearing.run(self.utterance.feed, samples)
        events: List[HearEvent] = []
        if self.utterance.text != self.said:
            self.said = self.utterance.text
            events.append(Partial(self.said))
        if final is not None:
            events.append(Endpoint(final))
        return events

    async def finish(self) -> str:
        return await self.hearing.run(self.utterance.final)

    async def close(self) -> None:
        pass


class _Utterance:
    """One turn's state. Every method runs on the hearing thread."""

    def __init__(self, models: _Models, config: HearingConfig) -> None:
        self.models = models
        self.config = config
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

    def feed(self, samples: np.ndarray) -> Optional[str]:
        """Return the final transcript when this audio ends the turn."""
        self.audio.append(samples)
        self.stream.feed(samples)
        for delta in self.stream.step(max_decode_tokens=16):
            self.raw += delta
        return self.final() if self._turn_over(samples) else None

    def final(self) -> str:
        if self.config.final_pass:
            from mlx.core import array
            from parakeet_mlx.audio import get_logmel

            parakeet = self.models.parakeet
            return parakeet.generate(get_logmel(array(np.concatenate(self.audio)), parakeet.preprocessor_config))[0].text.strip()
        self.stream.close()
        while not self.stream.done:
            for delta in self.stream.step(max_decode_tokens=64):
                self.raw += delta
        return self.text

    def _turn_over(self, samples: np.ndarray) -> bool:
        self.unread = np.concatenate((self.unread, samples))
        while len(self.unread) >= VAD_WINDOW:
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
        probability, self.vad_state = self.models.vad.run(None, {"input": framed, "state": self.vad_state, "sr": np.array(RATE, np.int64)})
        return float(probability[0][0])

    def turn_complete(self) -> bool:
        tail = np.concatenate(self.audio)[-TURN_SECONDS * RATE :]
        tail = np.pad(tail, (TURN_SECONDS * RATE - len(tail), 0))
        features = self.models.features(tail, sampling_rate=RATE, return_tensors="np", padding="max_length", max_length=TURN_SECONDS * RATE, truncation=True, do_normalize=True)
        (probability,) = self.models.turn.run(None, {"input_features": features.input_features.astype(np.float32)})
        return float(probability[0][0]) >= self.config.turn_threshold
