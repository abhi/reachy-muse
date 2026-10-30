"""Speech for C3: Kokoro-82M on MLX, one sentence per call, resampled to the robot's rate.

One model serves every speak stream. Its own thread runs each call in turn, so the
event loop never waits on synthesis.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator, List, Optional

import numpy as np

from reachy_muse.companion_server.watchdog import EngineExecutor
from reachy_muse.contract.expressions import Expression
from reachy_muse.protocol import VoiceRole

from reachy_muse.companion_server.voices import KOKORO_DEFAULT

log = logging.getLogger("reachy_muse.companion_server")

REPO = "mlx-community/Kokoro-82M-bf16"
VOICED_FRAME_S = 0.02
VOICED_FLOOR = 0.01  # 40 dB under the loudest frame


@dataclass(frozen=True)
class Loudness:
    """Raise each sentence to one level, and limit its peaks under a ceiling with a short look-ahead.

    The level counts voiced frames only, so a pause does not make a sentence louder.
    The same sentence always comes out the same.
    """

    target_dbfs: float = -12.0
    ceiling_dbfs: float = -4.0

    def apply(self, audio: np.ndarray, rate: int) -> np.ndarray:
        if audio.size == 0 or float(np.abs(audio).max()) < 1e-6:
            return audio
        louder = audio * (10 ** (self.target_dbfs / 20) / _voiced_rms(audio, rate))
        return _limit(louder, 10 ** (self.ceiling_dbfs / 20), reach=rate // 500)


def _limit(audio: np.ndarray, ceiling: float, reach: int) -> np.ndarray:
    need = np.pad(np.minimum(1.0, ceiling / np.maximum(np.abs(audio), 1e-9)), reach, mode="edge")
    held = np.lib.stride_tricks.sliding_window_view(need, 2 * reach + 1).min(axis=1)
    eased = np.convolve(np.pad(held, reach // 2, mode="edge"), np.ones(reach) / reach, mode="valid")[: audio.size]
    return np.clip(audio * eased, -ceiling, ceiling).astype(np.float32)


class Limiter:
    """`_limit` for audio that arrives in pieces: the pieces it returns join into exactly `_limit` of the whole."""

    def __init__(self, ceiling: float, reach: int) -> None:
        self.ceiling = ceiling
        self.reach = reach
        self.settles_after = 2 * reach
        self._audio = np.zeros(0, np.float32)
        self._sent = 0

    def feed(self, chunk: np.ndarray) -> np.ndarray:
        self._audio = np.concatenate([self._audio, chunk])
        return self._take(self._audio.size - self.settles_after)

    def flush(self) -> np.ndarray:
        return self._take(self._audio.size)

    def _take(self, ready: int) -> np.ndarray:
        if ready <= self._sent:
            return np.zeros(0, np.float32)
        out = _limit(self._audio, self.ceiling, self.reach)[self._sent:ready]
        self._sent = ready
        return out


def _voiced_rms(audio: np.ndarray, rate: int) -> float:
    hop = int(rate * VOICED_FRAME_S)
    frames = np.pad(audio, (0, -audio.size % hop)).reshape(-1, hop).astype(np.float64)
    energy = np.sqrt(np.mean(np.square(frames), axis=1))
    return float(np.sqrt(np.mean(np.square(frames[energy >= energy.max() * VOICED_FLOOR]))))


class KokoroVoice:
    name = "kokoro-82m"

    def __init__(self, voice: str = KOKORO_DEFAULT, loudness: Loudness = Loudness()) -> None:
        self.voice = voice
        self.loudness = loudness
        self._thread = EngineExecutor("voice")
        self._model: Any = None

    async def load(self) -> None:
        self._model = await asyncio.get_running_loop().run_in_executor(self._thread, load_model, self.voice)

    async def speak(self, text: str, role: VoiceRole, out_rate: int, expression: Optional[Expression] = None) -> AsyncIterator[bytes]:
        loop = asyncio.get_running_loop()
        for sentence in sentences(text):
            yield await loop.run_in_executor(self._thread, self._synthesize, sentence, out_rate)

    def _synthesize(self, sentence: str, out_rate: int) -> bytes:
        return pcm16(self.loudness.apply(render(self._model, sentence, self.voice, out_rate), out_rate))


def render(model: Any, sentence: str, voice: str, out_rate: int) -> np.ndarray:
    """Kokoro's float audio for one sentence at out_rate, before any gain.

    A voice's first letter is its accent ("a" American, "b" British), and Kokoro pronounces the text in that accent.
    """
    parts = [np.asarray(result.audio, dtype=np.float32).reshape(-1) for result in model.generate(sentence, voice=voice, lang_code=voice[0])]
    audio = np.concatenate(parts) if parts else np.zeros(0, np.float32)
    if out_rate != model.sample_rate:
        import soxr

        audio = soxr.resample(audio, model.sample_rate, out_rate)
    return audio


def pcm16(audio: np.ndarray) -> bytes:
    return (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2").tobytes()


def sentences(text: str) -> List[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]


def load_model(voice: str) -> Any:
    from mlx_audio.tts.utils import load_model as load

    started = time.perf_counter()
    model = load(REPO)
    list(model.generate("Hello there.", voice=voice, lang_code=voice[0]))  # the first call builds the phonemizer pipeline, about 5 s
    log.info("voice %s %s ready in %.1f s", REPO, voice, time.perf_counter() - started)
    return model
