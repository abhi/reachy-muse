"""Speech for C3: Chatterbox Turbo on MLX in its built-in voice, streamed as it decodes.

mlx-audio's own streaming draws fresh noise for every decode and splices without a crossfade,
so its seams are audible.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, AsyncIterator, Callable, Dict, Iterator, List, Optional, Protocol, Sequence, Tuple

import numpy as np

from reachy_muse.companion_server.watchdog import EngineExecutor, watchdog
from reachy_muse.contract.expressions import Expression
from reachy_muse.protocol import VoiceRole

from reachy_muse.companion_server.voice_kokoro import Limiter, Loudness, _voiced_rms, pcm16, sentences
from reachy_muse.companion_server.voices import CHATTERBOX_VOICE

log = logging.getLogger("reachy_muse.companion_server")

REPO = "mlx-community/chatterbox-turbo-8bit"
SPEECH_TOKENS_PER_S = 25
DECODE_EVERY_S = 0.8
TOKENS_PER_DECODE = int(SPEECH_TOKENS_PER_S * DECODE_EVERY_S)
MEL_FRAMES_MAX = 2048
HOLD_S = 0.24
CROSSFADE_S = 0.06
# Event tags audibly add the sound and keep the words intact; style tags such as [happy] change nothing measurable.
TAGS = {Expression.HAPPY: "[chuckle]", Expression.SAD: "[sigh]", Expression.SURPRISED: "[gasp]"}
CALIBRATION = (
    "Your reminder is set for eight tonight.",
    "I could not find that file, so I did not send anything.",
    "The kitchen lights are off, and the door is locked.",
)


class SpeechModel(Protocol):
    sample_rate: int

    def tokens(self, text: str) -> Iterator[Tuple[List[int], bool]]:
        """Speech tokens for text, a few at a time, with True on the last batch."""

    def decode(self, tokens: Sequence[int], final: bool, seed: int) -> np.ndarray:
        """Float audio for all the tokens so far; one seed draws the same noise at the same position in every decode."""


@dataclass(frozen=True)
class Chunk:
    """Speech for sentence `sentence` of a line; `last` marks the sentence's final chunk."""

    sentence: int
    pcm: bytes
    last: bool


class Seams:
    """Joins decodes of a growing token prefix into one signal, emitting only what later decodes will not change.

    Each decode's last HOLD_S may still change, so it waits for the next decode. The samples
    held at each seam fade into the next decode over CROSSFADE_S.
    """

    def __init__(self, rate: int) -> None:
        self.hold = int(HOLD_S * rate)
        self.fade = int(CROSSFADE_S * rate)
        self._end = 0
        self._tail = np.zeros(0, np.float32)

    def feed(self, decoded: np.ndarray, final: bool) -> np.ndarray:
        start = self._end
        tail = self._tail
        if tail.size and decoded.size >= start:
            ramp = np.linspace(0, 1, tail.size, dtype=np.float32)
            tail = tail * (1 - ramp) + decoded[start - tail.size:start] * ramp
        stable = decoded.size if final else max(start, decoded.size - self.hold)
        fresh = np.concatenate([tail, decoded[start:stable]])
        self._end = max(start, stable)
        if final:
            self._tail = np.zeros(0, np.float32)
            return fresh
        keep = min(self.fade, fresh.size)
        self._tail = fresh[fresh.size - keep:]
        return fresh[:fresh.size - keep]


def decoded(model: SpeechModel, text: str, seed: int, stop: threading.Event) -> Iterator[Tuple[np.ndarray, bool]]:
    """The sentence's audio at the model's rate, in pieces as it decodes, with True on the last piece."""
    seams = Seams(model.sample_rate)
    tokens: List[int] = []
    for batch, final in model.tokens(text):
        if stop.is_set():
            return
        tokens += batch
        yield seams.feed(model.decode(tokens, final, seed), final), final
        if final:
            return
    # The token stream can end on a stop token without marking a last batch.
    yield seams.feed(model.decode(tokens, True, seed), True), True


def tagged(lines: Sequence[str], expression: Optional[Expression]) -> List[str]:
    """The text the model reads for each sentence: the expression's tag leads the first sentence only."""
    tag = TAGS.get(expression)
    return [f"{tag} {line}" if tag and index == 0 else line for index, line in enumerate(lines)]


class _SentencePcm:
    def __init__(self, in_rate: int, out_rate: int, gain: float, loudness: Loudness) -> None:
        import soxr

        self._resample = soxr.ResampleStream(in_rate, out_rate, 1, dtype="float32")
        self._gain = gain
        self._limiter = Limiter(10 ** (loudness.ceiling_dbfs / 20), reach=out_rate // 500)

    def feed(self, audio: np.ndarray, last: bool) -> bytes:
        louder = self._resample.resample_chunk(audio.astype(np.float32), last=last) * np.float32(self._gain)
        limited = self._limiter.feed(louder)
        if last:
            limited = np.concatenate([limited, self._limiter.flush()])
        return pcm16(limited)


_END = object()


class ChatterboxVoice:
    name = "chatterbox-turbo"

    def __init__(self, loudness: Loudness = Loudness(), *, open_model: Optional[Callable[[], SpeechModel]] = None) -> None:
        self.loudness = loudness
        self._open_model = open_model or _open_turbo
        self._thread = EngineExecutor("voice-chatterbox")
        self._model: Any = None
        self._gain = 1.0
        self._seed = 0

    async def load(self) -> None:
        await asyncio.get_running_loop().run_in_executor(self._thread, self._prepare)

    def _prepare(self) -> None:
        started = time.perf_counter()
        model = self._open_model()
        stop = threading.Event()
        levels = []
        for text in CALIBRATION:
            self._seed += 1
            audio = np.concatenate([piece for piece, _ in decoded(model, text, self._seed, stop)])
            levels.append(_voiced_rms(audio, model.sample_rate))
        level = float(np.mean(levels))
        self._gain = 10 ** (self.loudness.target_dbfs / 20) / level
        self._model = model
        log.info("voice %s %s ready in %.1f s; voiced level %.1f dBFS, gain %+.1f dB to %.1f dBFS", REPO, CHATTERBOX_VOICE,
                 time.perf_counter() - started, 20 * np.log10(level), 20 * np.log10(self._gain), self.loudness.target_dbfs)

    async def speak(self, text: str, role: VoiceRole, out_rate: int, expression: Optional[Expression] = None) -> AsyncIterator[bytes]:
        async for chunk in self.stream(sentences(text), out_rate, expression):
            if chunk.pcm:
                yield chunk.pcm

    async def stream(self, lines: Sequence[str], out_rate: int, expression: Optional[Expression] = None) -> AsyncIterator[Chunk]:
        """Each sentence's speech as it decodes. Closing the stream stops the worker at its next token batch."""
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        stop = threading.Event()

        def put(item: object) -> None:
            if not stop.is_set():
                loop.call_soon_threadsafe(queue.put_nowait, item)

        loop.run_in_executor(self._thread, self._run, tagged(lines, expression), out_rate, put, stop)
        try:
            while (item := await queue.get()) is not _END:
                if isinstance(item, Exception):
                    raise item
                yield item
        finally:
            stop.set()

    def _run(self, texts: Sequence[str], out_rate: int, put: Callable[[object], None], stop: threading.Event) -> None:
        try:
            for index, text in enumerate(texts):
                self._seed += 1
                output = _SentencePcm(self._model.sample_rate, out_rate, self._gain, self.loudness)
                for audio, last in decoded(self._model, text, self._seed, stop):
                    watchdog.progress()
                    put(Chunk(index, output.feed(audio, last), last))
                if stop.is_set():
                    return
        except Exception as error:
            put(error)
        finally:
            put(_END)


def _open_turbo() -> SpeechModel:
    from mlx_audio.tts.utils import load_model

    return _Turbo(load_model(REPO))


class _Turbo:
    MAX_GENERATION_STEPS = 1000
    SPEECH_TOKEN_LIMIT = 6561
    CLOSING_SILENCE_TOKENS = 5

    def __init__(self, model: Any) -> None:
        self.model = model
        self.sample_rate = model.sample_rate
        self._steady: Optional[_SteadyRandom] = None

    def tokens(self, text: str) -> Iterator[Tuple[List[int], bool]]:
        import mlx.core as mx
        from mlx_audio.tts.models.chatterbox_turbo.chatterbox_turbo import punc_norm

        ids = mx.array(self.model.tokenizer(punc_norm(text), return_tensors="np").input_ids)
        for batch, final in self.model.t3.inference_turbo_stream(t3_cond=self.model._conds.t3, text_tokens=ids, temperature=0.8,
                                                                 chunk_size=TOKENS_PER_DECODE, max_gen_len=self.MAX_GENERATION_STEPS):
            speech = np.array(batch).reshape(-1)
            yield speech[speech < self.SPEECH_TOKEN_LIMIT].tolist(), final

    def decode(self, tokens: Sequence[int], final: bool, seed: int) -> np.ndarray:
        import mlx.core as mx
        from mlx_audio.tts.models.chatterbox_turbo.models.s3gen import flow_matching, hifigan, s3gen

        if self._steady is None or self._steady.seed != seed:
            self._steady = _SteadyRandom(seed, self.sample_rate)
            mx.clear_cache()
        self._steady.calls = 0
        steady = _WithRandom(self._steady)
        # Only these modules see the steady noise, and only on this thread while it decodes: the
        # other models draw from mlx.core.random on their own threads. flow_matching also prints
        # a line and a progress bar on every decode, which would flood the server's log.
        with _replaced([(s3gen, "mx", steady), (hifigan, "mx", steady), (flow_matching, "mx", steady),
                        (flow_matching, "print", lambda *args, **kwargs: None), (flow_matching, "tqdm", lambda steps, **kwargs: steps)]):
            ids = mx.array(list(tokens) + [s3gen.S3GEN_SIL] * self.CLOSING_SILENCE_TOKENS * final, dtype=mx.int32)[None, :]
            mel = self.model.s3gen(ids, ref_dict=self.model._conds.gen, n_cfm_timesteps=2, finalize=final)
            wav, _ = self.model.s3gen.mel2wav.inference(mel.transpose(0, 2, 1), None)
            wav = wav.reshape(-1)
            fade = self.model.s3gen.trim_fade
            wav = mx.concatenate([wav[:len(fade)] * fade, wav[len(fade):]])
            mx.eval(wav)
            return np.array(wav, dtype=np.float32)


@contextmanager
def _replaced(replacements: Sequence[Tuple[Any, str, Any]]) -> Iterator[None]:
    missing = object()
    saved = [(module, name, module.__dict__.get(name, missing)) for module, name, _ in replacements]
    for module, name, value in replacements:
        setattr(module, name, value)
    try:
        yield
    finally:
        for module, name, original in saved:
            if original is missing:
                delattr(module, name)
            else:
                setattr(module, name, original)


class _SteadyRandom:
    def __init__(self, seed: int, rate: int) -> None:
        import mlx.core as mx

        self.mx = mx
        self.real = mx.random
        self.seed = seed
        self.rate = rate
        self.calls = 0
        self._drawn: Dict[Tuple[int, ...], Any] = {}

    def _key(self) -> Any:
        self.calls += 1
        return self.real.key(self.seed * 1000 + self.calls)

    def normal(self, shape: Sequence[int], *args: Any, **kwargs: Any) -> Any:
        # mlx-audio draws mel noise (b, 80, t), harmonic noise (b, 9, t) and source noise (b, t, 1).
        time_axis = 2 if shape[1] in (80, 9) else 1
        others = [size for axis, size in enumerate(shape) if axis not in (0, time_axis)]
        key = self._key()
        if (self.calls, shape[0], *others) not in self._drawn:
            longest = (MEL_FRAMES_MAX if others == [80] else
                       self.rate * (1 + _Turbo.MAX_GENERATION_STEPS + _Turbo.CLOSING_SILENCE_TOKENS) // SPEECH_TOKENS_PER_S)
            self._drawn[(self.calls, shape[0], *others)] = self.real.normal((shape[0], longest, *others), key=key)
        drawn = self._drawn[(self.calls, shape[0], *others)][:, :shape[time_axis]]
        return self.mx.moveaxis(drawn, 1, time_axis)

    def uniform(self, *args: Any, **kwargs: Any) -> Any:
        return self.real.uniform(*args, key=self._key(), **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.real, name)


class _WithRandom:
    def __init__(self, random: _SteadyRandom) -> None:
        import mlx.core as mx

        self.random = random
        self._mx = mx

    def __getattr__(self, name: str) -> Any:
        return getattr(self._mx, name)
