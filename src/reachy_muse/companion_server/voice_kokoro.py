"""Speech for C3: Kokoro-82M on MLX, one sentence per call, resampled to the robot's rate.

One model serves every speak stream. Its own thread runs each call in turn, so the
event loop never waits on synthesis.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, AsyncIterator, List

import numpy as np

from musegadget.reachy_companion_protocol import VoiceRole

log = logging.getLogger("reachy_companion")

REPO = "mlx-community/Kokoro-82M-bf16"
# The bench round-tripped af_heart with zero word errors. Every role speaks in the one voice.
DEFAULT_VOICE = "af_heart"


@dataclass(frozen=True)
class Loudness:
    """Gain every sentence to one RMS level, and soft-limit the peaks under a ceiling.

    Kokoro's raw sentences sit near -27 dBFS RMS, too quiet for the robot's small speaker.
    The limiter is a per-sample curve with no look-ahead, so the same sentence always comes out the same.
    """

    target_dbfs: float = -20.0
    ceiling_dbfs: float = -1.0
    knee_db: float = 6.0  # samples this far under the ceiling pass unchanged

    def apply(self, audio: np.ndarray) -> np.ndarray:
        rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)))) if audio.size else 0.0
        if rms < 1e-6:
            return audio
        louder = audio * np.float32(10 ** (self.target_dbfs / 20) / rms)
        ceiling = 10 ** (self.ceiling_dbfs / 20)
        knee = ceiling * 10 ** (-self.knee_db / 20)
        over = np.abs(louder) > knee
        room = ceiling - knee
        louder[over] = np.sign(louder[over]) * (knee + room * np.tanh((np.abs(louder[over]) - knee) / room))
        return louder


class KokoroVoice:
    name = "kokoro-82m"

    def __init__(self, voice: str = DEFAULT_VOICE, loudness: Loudness = Loudness()) -> None:
        self.voice = voice
        self.loudness = loudness
        self._thread = ThreadPoolExecutor(1, thread_name_prefix="voice")
        self._model: Any = None

    async def load(self) -> None:
        self._model = await asyncio.get_running_loop().run_in_executor(self._thread, load_model, self.voice)

    async def speak(self, text: str, role: VoiceRole, out_rate: int) -> AsyncIterator[bytes]:
        loop = asyncio.get_running_loop()
        for sentence in sentences(text):
            yield await loop.run_in_executor(self._thread, self._synthesize, sentence, out_rate)

    def _synthesize(self, sentence: str, out_rate: int) -> bytes:
        return pcm16(self.loudness.apply(render(self._model, sentence, self.voice, out_rate)))


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
