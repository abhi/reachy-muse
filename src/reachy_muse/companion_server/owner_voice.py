"""The owner's voice on the Mac: one voice print, its enrollment, and a label for each request's voice.

The print lives in the companion's state directory, readable only by its user. It never goes to
the robot or Muse, and no log holds it, the audio or any words. The encoder is the robot's own
sherpa-onnx speaker model, ``voice_verify.SpeakerEncoder``, on one CPU thread of its own. The
server asks for a label only after a request's endpoint has gone out, so hearing never waits.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator, Callable, List, Optional, Protocol, Sequence, Tuple

import numpy as np

from reachy_muse.companion_server.engines import Observation
from reachy_muse.companion_server.watchdog import EngineExecutor
from reachy_muse.protocol import MIC_RATE, SpeakerLabel
from reachy_muse.voice_verify import normalized

log = logging.getLogger("reachy_muse.companion_server")

PRINT_FILE = "voice-print.json"
MAX_PRINT_BYTES = 64 * 1024
FRAME = MIC_RATE // 50              # webrtcvad judges 20 ms frames, as SpeakerEncoder.embed does
WINDOW = MIC_RATE * 3 // 2          # embed scores at most 1.5 s of speech, the window it was calibrated on
# Shorter speech gives an embedding too noisy to label, so the request stays unknown.
MIN_SPEECH = MIC_RATE
# Averaging up to four windows, 6 s of speech, steadies a request's embedding. At the bench's
# 52 ms per window on one core, a request costs at most about 0.2 s of CPU, off the hearing thread.
MAX_WINDOWS = 4
# Five sentences with at least 2 s of speech each: 10 s or more, in ten or more windows. Averaged,
# the print's own noise is about a third of one window's, so a score mostly measures the request.
# Five also let the agreement check single out one take that does not match the rest.
ENROLL_UTTERANCES = 5
ENROLL_SPEECH_S = 2.0
# An enrollment left open would take whoever speaks next.
ENROLL_WITHIN_S = 180.0


class Encoder(Protocol):
    """``voice_verify.SpeakerEncoder``, or a test's stand-in."""

    model_id: str
    dimension: int

    def embed(self, samples: np.ndarray, rate: int) -> np.ndarray: ...


class Vad(Protocol):
    def is_speech(self, pcm: bytes, rate: int) -> bool: ...


def _webrtc_vad() -> Vad:
    import webrtcvad

    return webrtcvad.Vad(2)


@dataclass(frozen=True)
class VoicePrint:
    """The owner's mean speaker embedding, and the model that made it."""

    model_id: str
    embedding: Tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.model_id) != 64 or any(c not in "0123456789abcdef" for c in self.model_id):
            raise ValueError("it names no speaker model")
        normalized(self.embedding)

    def save(self, path: Path) -> None:
        """Replace the print atomically, readable only by this user."""
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=".voice-print-", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as target:
                os.fchmod(target.fileno(), 0o600)
                json.dump({"version": 1, "model_sha256": self.model_id, "embedding": self.embedding}, target,
                          allow_nan=False)
                target.flush()
                os.fsync(target.fileno())
            os.replace(temporary, path)
        finally:
            Path(temporary).unlink(missing_ok=True)

    @classmethod
    def load(cls, path: Path) -> VoicePrint:
        """Raises ValueError with a message safe to log."""
        try:
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        except OSError:
            raise ValueError("it cannot be opened") from None
        with os.fdopen(fd, "r", encoding="utf-8") as source:
            info = os.fstat(source.fileno())
            if (not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600
                    or info.st_uid != os.getuid() or info.st_size > MAX_PRINT_BYTES):
                raise ValueError("it must be a private mode-600 file")
            try:
                data = json.loads(source.read())
                if data["version"] != 1:
                    raise ValueError
                return cls(data["model_sha256"], tuple(float(value) for value in data["embedding"]))
            except (KeyError, TypeError, ValueError):
                raise ValueError("it is unreadable") from None


@dataclass(frozen=True)
class _Heard:
    """One request, judged on the speaker thread. Its embedding is kept only for an enrollment."""

    observation: Observation
    embedding: Optional[np.ndarray]


class OwnerVoice:
    """The ``Speaker`` engine: labels each request's voice against the print on disk, and enrolls it.

    Every request checks the file first, so ``forget-voice`` deleting it takes effect at the
    next request, and a label scored against a print that changed meanwhile becomes unknown.
    """

    def __init__(self, encoder: Encoder, print_path: Path, threshold: float, *, name: str = "speaker",
                 vad: Callable[[], Vad] = _webrtc_vad, thread: Optional[EngineExecutor] = None) -> None:
        self.encoder = encoder
        self.print_path = print_path
        self.threshold = threshold
        self.name = name
        self._vad = vad
        self._thread = thread if thread is not None else EngineExecutor("speaker")
        self._seen: Tuple[object, Optional[VoicePrint]] = (None, None)
        self._enrollment: Optional[asyncio.Queue] = None

    @classmethod
    async def load(cls, model: Path, print_path: Path, threshold: Optional[float]) -> OwnerVoice:
        """Load the speaker model on the speaker thread. Downloads nothing."""
        from reachy_muse.voice_verify import CALIBRATED_MODEL_SHA256, DEFAULT_THRESHOLD, SpeakerEncoder

        thread = EngineExecutor("speaker")
        try:
            encoder = await asyncio.wrap_future(thread.submit(SpeakerEncoder, model))
            if threshold is None and encoder.model_id != CALIBRATED_MODEL_SHA256:
                raise ValueError("this speaker model needs --speaker-threshold; see bench/voice_verify.md")
        except BaseException:
            thread.shutdown(wait=False)
            raise
        voice = cls(encoder, print_path, DEFAULT_THRESHOLD if threshold is None else threshold,
                    name=Path(model).stem.encode("utf-8")[:64].decode("utf-8", "ignore"), thread=thread)
        log.info("speaker recognition ready: model %s, threshold %.2f, %s", voice.name, voice.threshold,
                 "a voice print is enrolled" if print_path.exists() else "no voice print yet")
        return voice

    @property
    def enrolling(self) -> bool:
        return self._enrollment is not None

    async def observe(self, pcm: bytes) -> Observation:
        enrollment = self._enrollment
        heard = await asyncio.wrap_future(self._thread.submit(self._observe, pcm, enrollment is not None))
        if enrollment is not None and enrollment is self._enrollment:
            enrollment.put_nowait((heard.observation.voiced_s, heard.embedding))
        return heard.observation

    async def enroll(self) -> AsyncIterator[dict]:
        """Replace the print from the next utterances any robot sends. Yields one event per step for the CLI."""
        if self._enrollment is not None:
            yield {"event": "failed", "reason": "busy"}
            return
        heard: asyncio.Queue = asyncio.Queue()
        self._enrollment = heard
        try:
            loop = asyncio.get_running_loop()
            deadline = loop.time() + ENROLL_WITHIN_S
            yield {"event": "listening", "utterances": ENROLL_UTTERANCES, "speech_s": ENROLL_SPEECH_S,
                   "within_s": ENROLL_WITHIN_S}
            taken: List[Tuple[float, np.ndarray]] = []
            while len(taken) < ENROLL_UTTERANCES:
                try:
                    voiced_s, embedding = await asyncio.wait_for(heard.get(), deadline - loop.time())
                except asyncio.TimeoutError:
                    yield {"event": "failed", "reason": "timeout", "taken": len(taken), "of": ENROLL_UTTERANCES}
                    return
                if embedding is None or voiced_s < ENROLL_SPEECH_S:
                    yield {"event": "skipped", "voiced_s": round(voiced_s, 2), "speech_s": ENROLL_SPEECH_S}
                    continue
                taken.append((voiced_s, embedding))
                yield {"event": "taken", "taken": len(taken), "of": ENROLL_UTTERANCES, "voiced_s": round(voiced_s, 2)}
            embeddings = [embedding for _, embedding in taken]
            agreement, worst = _agreement(embeddings)
            if agreement < self.threshold:
                yield {"event": "failed", "reason": "mismatch", "utterance": worst + 1}
                return
            voice_print = VoicePrint(self.encoder.model_id,
                                     tuple(float(value) for value in normalized(np.mean(embeddings, axis=0))))
            try:
                await asyncio.wrap_future(self._thread.submit(voice_print.save, self.print_path))
            except OSError as error:
                log.warning("saving the voice print failed: %s", type(error).__name__)
                yield {"event": "failed", "reason": "save"}
                return
            voiced_s = sum(seconds for seconds, _ in taken)
            log.info("voice print enrolled from %d utterances with %.1f s of speech", len(taken), voiced_s)
            yield {"event": "enrolled", "taken": len(taken), "voiced_s": round(voiced_s, 2)}
        finally:
            if self._enrollment is heard:
                self._enrollment = None

    def _observe(self, pcm: bytes, enrolling: bool) -> _Heard:
        samples = np.frombuffer(pcm, dtype="<i2")
        vad = self._vad()
        voiced = [start for start in range(0, len(samples) - FRAME + 1, FRAME)
                  if vad.is_speech(samples[start:start + FRAME].tobytes(), MIC_RATE)]
        voiced_s = len(voiced) * FRAME / MIC_RATE
        unknown = Observation(SpeakerLabel.UNKNOWN, 0, len(samples), None, voiced_s)
        signature, voice_print = self._print()
        if (voice_print is None and not enrolling) or len(voiced) * FRAME < MIN_SPEECH:
            return _Heard(unknown, None)
        count = min(MAX_WINDOWS, len(voiced) * FRAME // MIN_SPEECH)
        used = voiced[:count * WINDOW // FRAME]
        speech = np.concatenate([samples[start:start + FRAME] for start in used]).astype(np.float32) / 32768
        embedding = self._embed(np.array_split(speech, count))
        if embedding is None or voice_print is None:
            return _Heard(unknown, embedding)
        score = float(np.clip(np.dot(np.asarray(voice_print.embedding, dtype=np.float32), embedding), -1, 1))
        if self._signature() != signature:
            return _Heard(unknown, embedding)
        label = SpeakerLabel.OWNER if score >= self.threshold else SpeakerLabel.OTHER
        return _Heard(Observation(label, used[0], used[-1] + FRAME, score, voiced_s), embedding)

    def _embed(self, windows: Sequence[np.ndarray]) -> Optional[np.ndarray]:
        vectors = []
        for window in windows:
            try:
                vectors.append(normalized(self.encoder.embed(window, MIC_RATE)))
            except ValueError:
                pass  # too little speech for embed's own voice detector, or no usable vector
        return normalized(np.mean(vectors, axis=0)) if vectors else None

    def _print(self) -> Tuple[object, Optional[VoicePrint]]:
        """The print on disk now. The file is read again only when it changes."""
        signature = self._signature()
        if signature != self._seen[0]:
            self._seen = (signature, self._read() if signature is not None else None)
        return self._seen

    def _signature(self) -> object:
        try:
            info = os.lstat(self.print_path)
        except FileNotFoundError:
            return None
        return info.st_ino, info.st_ctime_ns, info.st_mtime_ns, info.st_size

    def _read(self) -> Optional[VoicePrint]:
        try:
            voice_print = VoicePrint.load(self.print_path)
            if voice_print.model_id != self.encoder.model_id or len(voice_print.embedding) != self.encoder.dimension:
                raise ValueError("another speaker model made it")
        except ValueError as error:
            log.warning("ignoring the voice print: %s; run reachy-companion enroll-voice again", error)
            return None
        return voice_print


def _agreement(embeddings: Sequence[np.ndarray]) -> Tuple[float, int]:
    """The lowest score of one utterance against the print the others make, and that utterance's index."""
    scores = [float(np.dot(embedding, normalized(np.mean([other for j, other in enumerate(embeddings) if j != i], axis=0))))
              for i, embedding in enumerate(embeddings)]
    worst = int(np.argmin(scores))
    return scores[worst], worst
