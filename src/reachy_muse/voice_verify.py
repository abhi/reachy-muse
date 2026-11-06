# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0 (the "License").

"""Offline speaker enrollment and admission. No transport or model downloads."""

from __future__ import annotations

from dataclasses import dataclass
from collections import deque
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import stat
import tempfile
import threading

PROFILE_FILE = "voice-profile.json"
DEFAULT_THRESHOLD = 0.74
CALIBRATED_MODEL_SHA256 = "5ef208a9da1453335308a6b6f4e6dfbd7e183a38b604de0a57664f45d257fe94"
log = logging.getLogger(__name__)


def normalized(values):
    import numpy as np

    vector = np.asarray(values, dtype=np.float32)
    if vector.ndim != 1 or not 1 <= vector.size <= 4096 or not np.isfinite(vector).all():
        raise ValueError("Invalid speaker embedding")
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm < 1e-8:
        raise ValueError("Empty speaker embedding")
    return vector / norm


def resample(samples, source_rate: int, target_rate: int):
    import av
    import numpy as np

    audio = np.asarray(samples, dtype=np.float32)
    if (type(source_rate) is not int or not 8000 <= source_rate <= 192000
            or audio.ndim != 1 or not np.isfinite(audio).all()):
        raise ValueError("Invalid verification audio")
    if source_rate == target_rate or not len(audio):
        return np.ascontiguousarray(audio)
    converter = av.AudioResampler(format="flt", layout="mono", rate=target_rate)
    frame = av.AudioFrame.from_ndarray(np.ascontiguousarray(audio[None, :]), format="flt", layout="mono")
    frame.sample_rate = source_rate
    chunks = converter.resample(frame) + converter.resample(None)
    return np.concatenate([chunk.to_ndarray().reshape(-1) for chunk in chunks])


class SpeakerEncoder:
    """Sherpa's feature extraction and ONNX embedding, fixed to one CPU thread."""

    def __init__(self, model: Path):
        import av
        import numpy as np
        import sherpa_onnx
        import webrtcvad

        with Path(model).expanduser().open("rb") as source:
            digest = hashlib.sha256()
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        self.model_id = digest.hexdigest()
        config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=str(Path(model).expanduser()), num_threads=1, provider="cpu", debug=False)
        if not config.validate():
            raise ValueError("Invalid local speaker model")
        self._extractor = sherpa_onnx.SpeakerEmbeddingExtractor(config)
        self.dimension = self._extractor.dim
        self._lock = threading.Lock()
        # Pay session initialization costs before microphone admission.
        self._compute(np.zeros(16000, dtype=np.float32))

    def _compute(self, audio):
        with self._lock:
            stream = self._extractor.create_stream()
            stream.accept_waveform(sample_rate=16000, waveform=audio)
            stream.input_finished()
            if not self._extractor.is_ready(stream):
                raise ValueError("Insufficient speaker audio")
            return self._extractor.compute(stream)

    def embed(self, samples, rate: int):
        import numpy as np
        import webrtcvad

        audio = resample(samples, rate, 16000)
        if not 6400 <= len(audio) <= 16000 * 60:
            raise ValueError("Speaker audio needs at least 0.4 seconds")
        pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2")
        vad = webrtcvad.Vad(2)
        voiced = [i for i in range(0, len(pcm) - 319, 320)
                  if vad.is_speech(pcm[i:i + 320].tobytes(), 16000)]
        if len(voiced) < 15:
            raise ValueError("Speaker audio needs at least 0.3 seconds of speech")
        # Bound inference cost and use the same window during enrollment/evaluation.
        audio = audio[voiced[0]:min(voiced[-1] + 320, voiced[0] + 24000)]
        return normalized(self._compute(np.ascontiguousarray(audio)))


@dataclass(frozen=True)
class Profile:
    model_id: str
    embedding: tuple[float, ...]
    threshold: float

    def __post_init__(self):
        if (len(self.model_id) != 64 or any(c not in "0123456789abcdef" for c in self.model_id)
                or type(self.threshold) not in (float, int)
                or not math.isfinite(self.threshold) or not -1 < self.threshold < 1):
            raise ValueError("Invalid voice profile")
        normalized(self.embedding)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        fd, temporary = tempfile.mkstemp(prefix=".voice-", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as target:
                os.fchmod(target.fileno(), 0o600)
                json.dump({"version": 1, "model_sha256": self.model_id,
                           "embedding": self.embedding, "threshold": self.threshold}, target,
                          allow_nan=False)
                target.flush()
                os.fsync(target.fileno())
            os.replace(temporary, path)
        finally:
            Path(temporary).unlink(missing_ok=True)

    @classmethod
    def load(cls, path: Path):
        try:
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            with os.fdopen(fd, "r", encoding="utf-8") as source:
                info = os.fstat(source.fileno())
                if (not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600
                        or info.st_uid != os.getuid() or info.st_size > 100000):
                    raise ValueError("Voice profile must be a private mode-600 file")
                data = json.load(source)
            if data["version"] != 1:
                raise ValueError("Unsupported voice profile")
            return cls(data["model_sha256"], tuple(data["embedding"]), data["threshold"])
        except (OSError, KeyError, TypeError, ValueError) as exc:
            raise ValueError("Cannot load voice enrollment; enroll again or use --voice-verify off") from exc


def enroll(encoder, recordings, threshold=DEFAULT_THRESHOLD) -> Profile:
    import numpy as np

    vectors = [encoder.embed(audio, rate) for audio, rate in recordings]
    if len(vectors) < 3:
        raise ValueError("Record at least three enrollment phrases")
    # Prevent a silent or plainly inconsistent take from poisoning the template.
    if min(float(np.dot(a, b)) for i, a in enumerate(vectors) for b in vectors[i + 1:]) < threshold:
        raise ValueError("Enrollment phrases did not match; repeat in a quiet room")
    mean = normalized(np.mean(vectors, axis=0))
    return Profile(encoder.model_id, tuple(float(v) for v in mean), threshold)


@dataclass(frozen=True)
class Decision:
    score: float | None
    match: bool
    allowed: bool


class VoiceVerifier:
    def __init__(self, encoder, profile: Profile, mode: str, realtime: bool = False):
        if mode not in ("soft", "strict"):
            raise ValueError("Enabled voice verification must be soft or strict")
        if encoder.model_id != profile.model_id or encoder.dimension != len(profile.embedding):
            raise ValueError("Voice model changed; enroll again with this model")
        self.encoder = encoder
        self.profile = profile
        self.mode = mode
        self.realtime = realtime
        self._reference = normalized(profile.embedding)

    @property
    def enforcing(self):
        return self.mode == "strict" or self.realtime

    def verify(self, samples, rate: int) -> Decision:
        import numpy as np

        score = None
        try:
            vector = normalized(self.encoder.embed(samples, rate))
            score = float(np.clip(np.dot(self._reference, vector), -1, 1))
        except Exception:
            # Library exceptions can contain input details; never log them.
            pass
        match = score is not None and score >= self.profile.threshold
        allowed = match or not self.enforcing
        log.info("voice_verify score=%s decision=%s", "none" if score is None else f"{score:.4f}",
                 "accept" if match else "reject" if self.enforcing else "observe")
        return Decision(score, match, allowed)


class RequestGate:
    """Hold bounded speech segments locally; never inherit a prior speaker match.

    Accepted segments return at the input rate. Rejected segments become silence
    so downstream endpoint state advances without receiving their contents.
    """

    def __init__(self, verifier: VoiceVerifier, rate: int, *, vad=None):
        import webrtcvad

        self.verifier = verifier
        self.rate = rate
        self.vad = vad if vad is not None else webrtcvad.Vad(2)
        self.reset()

    @property
    def active(self):
        return bool(self._speech)

    def reset(self):
        import av
        import numpy as np

        self._resampler = av.AudioResampler(format="flt", layout="mono", rate=16000)
        self._pending = np.empty(0, dtype=np.float32)
        self._history = deque(maxlen=10)
        self._voice_tail = deque(maxlen=20)
        self._speech = []
        self._voiced_run = self._quiet = 0
        self.rejected = False
        self._enforcing = self.verifier.enforcing

    def feed(self, samples):
        import av
        import numpy as np

        self.rejected = False
        output = []
        samples = np.asarray(samples, dtype=np.float32)
        if samples.ndim != 1 or not np.isfinite(samples).all():
            raise ValueError("Invalid verification audio")
        if self._enforcing != self.verifier.enforcing:
            if self._enforcing:
                held = self._speech or list(self._history)
                tail = [chunk.to_ndarray().reshape(-1) for chunk in self._resampler.resample(None)]
                buffered = resample(np.concatenate([*held, self._pending, *tail]), 16000, self.rate)
                samples = np.concatenate((buffered, samples))
            self.reset()
        if len(samples):
            frame = av.AudioFrame.from_ndarray(np.ascontiguousarray(samples[None, :]), format="flt", layout="mono")
            frame.sample_rate = self.rate
            for converted in self._resampler.resample(frame):
                self._pending = np.concatenate((self._pending, converted.to_ndarray().reshape(-1)))
        offset = 0
        while offset + 320 <= len(self._pending):
            audio_frame = self._pending[offset:offset + 320].copy()
            offset += 320
            pcm = (np.clip(audio_frame, -1, 1) * 32767).astype("<i2").tobytes()
            voiced = self.vad.is_speech(pcm, 16000)
            self._voice_tail.append(bool(voiced))
            self._voiced_run = self._voiced_run + 1 if voiced else 0
            self._quiet = 0 if voiced else self._quiet + 1
            if not self._speech:
                if len(self._history) == self._history.maxlen:
                    output.append(np.zeros(320, np.float32))
                self._history.append(audio_frame)
                if self._voiced_run < 3:
                    continue
                self._speech = list(self._history)
                self._history.clear()
            else:
                self._speech.append(audio_frame)
            ended = self._quiet >= 15
            # Keep at least 0.3 voiced seconds in the unforwarded 0.4s suffix.
            # A recent pause can make elapsed time an inadequate evidence floor.
            cut = len(self._speech) >= 50 and self._quiet == 0 and sum(self._voice_tail) >= 15
            exhausted = len(self._speech) >= 75 and not (ended or cut)
            if ended or cut or exhausted:
                audio = np.concatenate(self._speech)
                # Sparse speech cannot grow the private buffer indefinitely.
                decision = self.verifier.verify(audio[:0] if exhausted else audio, 16000)
                self.rejected |= not decision.allowed
                retained = self._speech[-20:] if cut and not ended else []
                if retained:
                    audio = audio[:-6400]
                output.append(audio if decision.allowed else np.zeros_like(audio))
                self._speech = retained
                if ended or exhausted:
                    self._voiced_run = 0
        self._pending = self._pending[offset:].copy()
        if not self.verifier.enforcing:
            return samples
        if output:
            return resample(np.concatenate(output), 16000, self.rate)
        return np.empty(0, dtype=np.float32)


def rejection_cue(rate: int):
    """A quiet, 160 ms descending chime, generated locally."""
    import numpy as np

    count = round(rate * 0.16)
    t = np.arange(count) / rate
    return (0.04 * np.sin(2 * np.pi * (660 * t - 900 * t * t))
            * np.sin(np.linspace(0, np.pi, count)) ** 2).astype(np.float32)
