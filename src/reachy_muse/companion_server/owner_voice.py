"""The owner's voice on the Mac: one voice print, its enrollment, and a label for each request's voice.

The print lives in the companion's state directory, readable only by its user. It never goes to
the robot or Muse, and no log holds it, the audio or any words. The encoder is the robot's own
sherpa-onnx speaker model, ``voice_verify.SpeakerEncoder``, on one CPU thread of its own. The
server asks for a label when a request's speech ends, beside the final pass, so hearing never waits.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import stat
import tempfile
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator, Callable, List, Optional, Protocol, Sequence, Tuple

import numpy as np

from reachy_muse.companion_server.engines import Observation
from reachy_muse.protocol import MIC_RATE, SpeakerLabel
from reachy_muse.voice_verify import normalized

log = logging.getLogger("reachy_muse.companion_server")

PRINT_FILE = "voice-print.json"
MAX_PRINT_BYTES = 64 * 1024
FRAME = MIC_RATE // 50              # webrtcvad judges 20 ms frames, as SpeakerEncoder.embed does
WINDOW = MIC_RATE * 3 // 2          # embed scores at most 1.5 s of speech
# Shorter speech gives an embedding too noisy to label, so the request stays unknown.
MIN_SPEECH = MIC_RATE
# Averaging up to four windows, 6 s of speech, steadies a request's embedding. At the bench's
# 52 ms per window on one core, a request costs at most about 0.2 s of CPU, off the hearing thread.
MAX_WINDOWS = 4
# One check running and one waiting. A check takes a fraction of a second, so more means the thread
# is stuck; later requests are then unknown instead of piling up audio behind it.
MAX_CHECKS = 2
# Five sentences with at least 2 s of speech each: 10 s or more, in ten or more windows. The print
# averages them, so it varies less than any one request's embedding. Five also let the agreement
# check single out one sentence that does not match the rest.
ENROLL_UTTERANCES = 5
ENROLL_SPEECH_S = 2.0
# Each sentence must score this against the print the other four make. It only has to catch a
# sentence in another voice, so it sits below the label threshold: an owner must be able to enroll
# before that threshold is tuned on real voices. Enrollment logs the five scores for that tuning.
ENROLL_AGREEMENT = 0.5
# An enrollment left open would take whoever speaks next.
ENROLL_WITHIN_S = 180.0
_UNCHECKED = object()   # what an enrollment hears of a request whose check could not run


class SpeakerBusy(Exception):
    """The speaker thread is behind; the server labels the request unknown."""


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
        """Whatever is at ``path``, raises only ValueError, with a message safe to log."""
        try:
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        except OSError:
            raise ValueError("it cannot be opened") from None
        try:
            info = os.fstat(fd)
            if (not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600
                    or info.st_uid != os.getuid() or info.st_size > MAX_PRINT_BYTES):
                raise ValueError("it must be a private mode-600 file")
            content = os.read(fd, MAX_PRINT_BYTES + 1)
        finally:
            os.close(fd)
        try:
            data = json.loads(content)
            if data["version"] != 1:
                raise ValueError
            return cls(data["model_sha256"], tuple(float(value) for value in normalized(data["embedding"])))
        except (KeyError, TypeError, ValueError, OverflowError, RecursionError):
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
                 vad: Callable[[], Vad] = _webrtc_vad, thread: Optional[ThreadPoolExecutor] = None,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.encoder = encoder
        self.print_path = print_path
        self.threshold = threshold
        self.name = name
        self._vad = vad
        self._thread = thread if thread is not None else _speaker_thread()
        self._clock = clock
        self._checks: List[Future] = []
        self._seen: Tuple[object, Optional[VoicePrint]] = (None, None)
        self._enrollment: Optional[asyncio.Queue] = None
        self._forgets = 0
        self._writing = threading.Lock()   # one save or delete of the print at a time

    @classmethod
    async def load(cls, model: Path, print_path: Path, threshold: Optional[float]) -> OwnerVoice:
        """Load the speaker model on the speaker thread. Downloads nothing."""
        from reachy_muse.voice_verify import CALIBRATED_MODEL_SHA256, DEFAULT_THRESHOLD, SpeakerEncoder

        thread = _speaker_thread()
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
        self._checks = [check for check in self._checks if not check.done()]
        if len(self._checks) >= MAX_CHECKS:
            if enrollment is not None:
                enrollment.put_nowait(_UNCHECKED)
            raise SpeakerBusy()
        check = self._thread.submit(self._observe, pcm, enrollment is not None)
        self._checks.append(check)
        if enrollment is not None:
            loop = asyncio.get_running_loop()

            def offer(done: Future) -> None:
                with contextlib.suppress(RuntimeError):
                    loop.call_soon_threadsafe(self._offer, enrollment, done)

            check.add_done_callback(offer)
        return (await asyncio.wrap_future(check)).observation

    def _offer(self, enrollment: asyncio.Queue, check: Future) -> None:
        """Give an enrollment its request once the check is done, even if the robot aborted the stream meanwhile."""
        if enrollment is self._enrollment:
            heard = None if check.cancelled() or check.exception() is not None else check.result()
            enrollment.put_nowait(_UNCHECKED if heard is None else (heard.observation.voiced_s, heard.embedding))

    async def enroll(self) -> AsyncIterator[dict]:
        """Replace the print from the next utterances any robot sends. Yields one event per step for the CLI."""
        if self._enrollment is not None:
            yield {"event": "failed", "reason": "busy"}
            return
        heard: asyncio.Queue = asyncio.Queue()
        self._enrollment = heard
        forgets = self._forgets
        try:
            deadline = self._clock() + ENROLL_WITHIN_S
            yield {"event": "listening", "utterances": ENROLL_UTTERANCES, "speech_s": ENROLL_SPEECH_S,
                   "within_s": ENROLL_WITHIN_S}
            taken: List[Tuple[float, np.ndarray]] = []
            while len(taken) < ENROLL_UTTERANCES:
                try:
                    item = await asyncio.wait_for(heard.get(), deadline - self._clock())
                except asyncio.TimeoutError:
                    item = None
                if self._forgets != forgets:
                    yield {"event": "failed", "reason": "forgotten"}
                    return
                if item is None or self._clock() >= deadline:
                    yield {"event": "failed", "reason": "timeout", "taken": len(taken), "of": ENROLL_UTTERANCES}
                    return
                voiced_s, embedding = item if item is not _UNCHECKED else (None, None)
                if voiced_s is not None and voiced_s < ENROLL_SPEECH_S:
                    yield {"event": "skipped", "voiced_s": round(voiced_s, 2), "speech_s": ENROLL_SPEECH_S}
                    continue
                if embedding is None:
                    yield {"event": "unchecked"}
                    continue
                taken.append((voiced_s, embedding))
                yield {"event": "taken", "taken": len(taken), "of": ENROLL_UTTERANCES, "voiced_s": round(voiced_s, 2)}
            embeddings = [embedding for _, embedding in taken]
            agreement = _agreement(embeddings)
            if min(agreement) < ENROLL_AGREEMENT:
                log.info("voice enrollment refused: agreement %s", " ".join(f"{score:.4f}" for score in agreement))
                yield {"event": "failed", "reason": "mismatch", "utterance": agreement.index(min(agreement)) + 1}
                return
            voice_print = VoicePrint(self.encoder.model_id,
                                     tuple(float(value) for value in normalized(np.mean(embeddings, axis=0))))
            try:
                await asyncio.wrap_future(self._thread.submit(self._save, voice_print, forgets))
            except OSError as error:
                log.warning("saving the voice print failed: %s", type(error).__name__)
                yield {"event": "failed", "reason": "save"}
                return
            if self._forgets != forgets:
                yield {"event": "failed", "reason": "forgotten"}
                return
            voiced_s = sum(seconds for seconds, _ in taken)
            log.info("voice print enrolled from %d utterances with %.1f s of speech; agreement %s", len(taken), voiced_s,
                     " ".join(f"{score:.4f}" for score in agreement))
            yield {"event": "enrolled", "taken": len(taken), "voiced_s": round(voiced_s, 2)}
        finally:
            if self._enrollment is heard:
                self._enrollment = None

    async def forget(self) -> Tuple[bool, bool]:
        """Stop any enrollment, then delete the print. Returns whether one was enrolling and a print was there."""
        self._forgets += 1
        enrollment = self._enrollment
        if enrollment is not None:
            enrollment.put_nowait(None)
        removed = await asyncio.to_thread(self._delete)
        return enrollment is not None, removed

    def _save(self, voice_print: VoicePrint, forgets: int) -> None:
        with self._writing:
            if self._forgets == forgets:
                voice_print.save(self.print_path)

    def _delete(self) -> bool:
        with self._writing:
            return delete_print(self.print_path)

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


def delete_print(path: Path) -> bool:
    """Delete the print and any copy an interrupted save left. Returns whether there was anything."""
    removed = False
    for copy in (path, *path.parent.glob(".voice-print-*")):
        with contextlib.suppress(FileNotFoundError):
            copy.unlink()
            removed = True
    return removed


def _speaker_thread() -> ThreadPoolExecutor:
    """Not an EngineExecutor: the watchdog would restart the whole companion for a stuck check,
    which only labels requests. A stuck check leaves later requests unknown instead."""
    return ThreadPoolExecutor(1, thread_name_prefix="speaker")


def _agreement(embeddings: Sequence[np.ndarray]) -> List[float]:
    """Each utterance's score against the print the others make."""
    return [float(np.dot(embedding, normalized(np.mean([other for j, other in enumerate(embeddings) if j != i],
                                                       axis=0))))
            for i, embedding in enumerate(embeddings)]
