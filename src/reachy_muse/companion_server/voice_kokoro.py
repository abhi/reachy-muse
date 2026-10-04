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
from typing import Any, AsyncIterator, List

import numpy as np

from musegadget.reachy_companion_protocol import VoiceRole

log = logging.getLogger("reachy_companion")

REPO = "mlx-community/Kokoro-82M-bf16"
# The bench round-tripped af_heart with zero word errors. A second voice is untested, so both roles share it.
VOICES = {VoiceRole.MAIN: "af_heart", VoiceRole.PROGRESS: "af_heart"}


class KokoroVoice:
    name = "kokoro-82m"

    def __init__(self) -> None:
        self._thread = ThreadPoolExecutor(1, thread_name_prefix="voice")
        self._model: Any = None

    async def load(self) -> None:
        self._model = await asyncio.get_running_loop().run_in_executor(self._thread, _load)

    async def speak(self, text: str, role: VoiceRole, out_rate: int) -> AsyncIterator[bytes]:
        loop = asyncio.get_running_loop()
        for sentence in sentences(text):
            yield await loop.run_in_executor(self._thread, self._synthesize, sentence, VOICES[role], out_rate)

    def _synthesize(self, sentence: str, voice: str, out_rate: int) -> bytes:
        parts = [np.asarray(result.audio, dtype=np.float32).reshape(-1) for result in self._model.generate(sentence, voice=voice)]
        audio = np.concatenate(parts) if parts else np.zeros(0, np.float32)
        if out_rate != self._model.sample_rate:
            import soxr

            audio = soxr.resample(audio, self._model.sample_rate, out_rate)
        return (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2").tobytes()


def sentences(text: str) -> List[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]


def _load() -> Any:
    from mlx_audio.tts.utils import load_model

    started = time.perf_counter()
    model = load_model(REPO)
    list(model.generate("Hello there.", voice=VOICES[VoiceRole.MAIN]))  # the first call builds the phonemizer pipeline, about 5 s
    log.info("voice %s ready in %.1f s", REPO, time.perf_counter() - started)
    return model
