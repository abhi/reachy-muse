# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0 (the "License").

"""Optional offline acoustic keyword spotting for Reachy conversations."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import io
import math
import logging
from pathlib import Path
import re
import tempfile
import threading
import time

log = logging.getLogger(__name__)

MODEL_ARCHIVE_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/kws-models/"
    "sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01.tar.bz2"
)
MODEL_FILES = {
    name: f"{name}-epoch-12-avg-2-chunk-16-left-64.int8.onnx"
    for name in ("encoder", "decoder", "joiner")
}
MODEL_FILES.update(tokens="tokens.txt", bpe="bpe.model")
PHONETIC_MODEL_FILES = {
    "encoder": "encoder-epoch-13-avg-2-chunk-16-left-64.int8.onnx",
    "decoder": "decoder-epoch-13-avg-2-chunk-16-left-64.onnx",
    "joiner": "joiner-epoch-13-avg-2-chunk-16-left-64.int8.onnx",
    "tokens": "tokens.txt",
    "lexicon": "en.phone",
}


@dataclass(frozen=True)
class WakeDetection:
    """16 kHz caller tail from the last decoded token, not a word-end estimate."""

    post_wake_samples: int
    epoch: int
    feed_id: int


@dataclass(frozen=True)
class WakeCandidate:
    """One exported native match. Stock sherpa does not expose its posterior."""

    epoch: int
    feed_id: int
    threshold: float
    boost: float
    audio: object = field(repr=False, compare=False)
    remaining: int = 0
    start_sample: int = 0
    score: float | None = None


def _encoder_timing(path: Path) -> tuple[int, int] | None:
    """Read ONNX's small metadata entries, skipping its graph without loading it."""
    def integer(source):
        value = 0
        for shift in range(0, 70, 7):
            byte = source.read(1)
            if not byte:
                raise ValueError("Truncated ONNX metadata")
            value |= (byte[0] & 127) << shift
            if byte[0] < 128:
                return value
        raise ValueError("Invalid ONNX metadata integer")

    def fields(source, size):
        for _ in range(4096):
            if source.tell() == size:
                return
            tag = integer(source)
            field, kind = tag >> 3, tag & 7
            if field == 0:
                raise ValueError("Invalid ONNX metadata field")
            if kind == 0:
                integer(source)
            elif kind in (1, 5):
                source.seek(8 if kind == 1 else 4, io.SEEK_CUR)
            elif kind == 2:
                length = integer(source)
                if source.tell() + length > size:
                    raise ValueError("Truncated ONNX metadata field")
                yield field, length
            else:
                raise ValueError("Unsupported ONNX metadata field")
            if source.tell() > size:
                raise ValueError("Truncated ONNX metadata")
        raise ValueError("ONNX metadata has too many fields")

    try:
        size = path.stat().st_size
        if not 0 < size <= 64 * 1024 * 1024:
            return None
        metadata = {}
        with path.open("rb") as source:
            for field, length in fields(source, size):
                if field != 14:
                    source.seek(length, io.SEEK_CUR)
                    continue
                if length > 4096 or len(metadata) >= 128:
                    return None
                entry = io.BytesIO(source.read(length))
                strings = {}
                for number, count in fields(entry, length):
                    if number not in (1, 2) or number in strings:
                        return None
                    strings[number] = entry.read(count).decode("utf-8")
                if set(strings) != {1, 2} or strings[1] in metadata:
                    return None
                metadata[strings[1]] = strings[2]
        if metadata.get("model_type") != "zipformer2" or metadata.get("version") != "1":
            return None
        chunk, shift = int(metadata["T"]), int(metadata["decode_chunk_len"])
        if (chunk, shift) != (45, 32):
            return None
        return chunk, shift
    except (OSError, ValueError, KeyError, UnicodeError):
        return None


class WakeWordDetector:
    """Detect one phrase from 16 kHz mono floats without transcribing speech.

    Initialization and inference are synchronous. Lifecycle calls serialize
    with inference, including a worker that outlives a cancelled async call.
    Keep microphone pre-roll outside this detector.
    """

    sample_rate = 16000

    def __init__(self, model_dir: str | Path, phrase: str = "hey muse", *,
                 threshold: float = 0.25, score: float = 1.0,
                 candidate_replay: bool = False, token_variants: int = 1) -> None:
        if not isinstance(phrase, str):
            raise ValueError("Wake phrase must contain English words.")
        phrase = " ".join(phrase.lower().replace(",", " ").split())
        if len(phrase) > 80 or not re.fullmatch(r"[a-z]+(?:[' ][a-z]+)*", phrase):
            raise ValueError("Wake phrase must contain English words.")
        if not math.isfinite(threshold) or not 0 < threshold <= 1:
            raise ValueError("Wake threshold must be between zero and one.")
        if not math.isfinite(score) or score <= 0:
            raise ValueError("Wake keyword score must be positive.")
        if type(token_variants) is not int or not 1 <= token_variants <= 4:
            raise ValueError("Wake token variants must be between one and four.")
        self.phrase = phrase
        self.threshold, self.boost = threshold, score
        self.candidate_replay = candidate_replay
        self.last_candidate: WakeCandidate | None = None
        self._lock = threading.RLock()
        self._label = phrase.upper().replace(" ", "_")
        self._spotter = None
        self._stream = None
        self._epoch = 0
        self._feed_id = 0
        self._last_detection: WakeDetection | None = None
        self._history = deque()
        self._history_samples = 0
        self._diagnostic_windows = 0
        directory = Path(model_dir).expanduser()
        phonetic = any((directory / filename).is_file()
                       for name, filename in PHONETIC_MODEL_FILES.items() if name != "tokens")
        model_files = PHONETIC_MODEL_FILES if phonetic else MODEL_FILES
        files = {name: directory / filename for name, filename in model_files.items()}
        missing = [path.name for path in files.values() if not path.is_file() or not path.stat().st_size]
        if missing:
            raise ValueError("Wake-word model is missing: " + ", ".join(missing))
        dependencies = "sherpa-onnx" if phonetic else "sherpa-onnx and sentencepiece"
        try:
            import numpy as np
            import sherpa_onnx
        except ImportError as exc:
            raise RuntimeError(
                f"Offline wake detection requires {dependencies}; "
                "install the Reachy wake-word dependencies."
            ) from exc
        self._np = np
        table = self._read_tokens(files["tokens"])
        if phonetic:
            pieces = self._phonetic_pieces(files["lexicon"], phrase, table)
            sequences = [pieces]
        else:
            try:
                import sentencepiece
            except ImportError as exc:
                raise RuntimeError(
                    "Offline wake detection requires sherpa-onnx and sentencepiece; "
                    "install the Reachy wake-word dependencies."
                ) from exc
            tokenizer = sentencepiece.SentencePieceProcessor(model_file=str(files["bpe"]))
            pieces = tokenizer.encode(phrase.upper(), out_type=str)
            sequences = ([pieces] if token_variants == 1 else
                         tokenizer.nbest_encode(phrase.upper(), out_type=str, nbest_size=token_variants))
        if (not sequences or any(not sequence or any(
                not isinstance(piece, str) or piece not in table
                or piece == "<unk>" or any(char.isspace() for char in piece)
                for piece in sequence) for sequence in sequences)):
            raise ValueError("Wake phrase cannot be encoded by this model's vocabulary.")
        # Sherpa's symbol table renders a leading SentencePiece boundary as a space.
        self._piece_variants = {tuple(" " + piece[1:] if piece.startswith("▁") else piece for piece in sequence)
                                for sequence in sequences}
        self._timing = _encoder_timing(files["encoder"])
        if self._timing is not None:
            chunk, shift = self._timing
            self._alignment_steps = 150 // shift + 1
            self._history_limit = ((self._alignment_steps - 1) * shift + chunk + 4) * 160
        else:
            self._alignment_steps = self._history_limit = 0
        if candidate_replay:
            self._history_limit = max(self._history_limit, 2 * self.sample_rate)
        self._keyword_tokens = tuple(" ".join(sequence) for sequence in sequences)
        keyword = "".join(tokens + " @" + self._label + "\n" for tokens in self._keyword_tokens)
        # Sherpa loads the keyword graph during construction. Keep its one-phrase
        # file temporary so a read-only model directory remains usable.
        with tempfile.TemporaryDirectory(prefix="muse-wake-") as temporary:
            keywords = Path(temporary) / "keywords.txt"
            # Native defaults override duplicate per-stream entries. Keep the
            # default graph empty when candidate replay needs another threshold.
            keywords.write_text("" if candidate_replay else keyword, encoding="utf-8")
            self._spotter = sherpa_onnx.KeywordSpotter(
                tokens=str(files["tokens"]), encoder=str(files["encoder"]),
                decoder=str(files["decoder"]), joiner=str(files["joiner"]),
                keywords_file=str(keywords), sample_rate=self.sample_rate,
                feature_dim=80, num_threads=1, provider="cpu", max_active_paths=4,
                keywords_score=score, keywords_threshold=threshold,
                num_trailing_blanks=1,
            )
        self.reset()
        if self.timing_supported:
            log.info("Acoustic wake-tail timing ready")

    @property
    def timing_supported(self) -> bool:
        with self._lock:
            backend = getattr(self._spotter, "keyword_spotter", None)
            return self._timing is not None and callable(getattr(backend, "get_result", None))

    @property
    def epoch(self) -> int:
        with self._lock:
            return self._epoch

    @property
    def feed_id(self) -> int:
        with self._lock:
            return self._feed_id

    @property
    def last_detection(self) -> WakeDetection | None:
        with self._lock:
            return self._last_detection

    @staticmethod
    def _phonetic_pieces(path: Path, phrase: str, symbols: set[str]) -> list[str]:
        if path.stat().st_size > 8 * 1024 * 1024:
            raise ValueError("Wake-word pronunciation lexicon exceeds its size limit.")
        pronunciations = {}
        wanted = set(phrase.upper().split())
        for line in path.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if not fields:
                continue
            if len(fields) < 2:
                raise ValueError("Wake-word pronunciation lexicon has invalid fields.")
            word, phones = fields[0].upper(), fields[1:]
            if word in wanted:
                if any(phone not in symbols or phone == "<unk>" for phone in phones):
                    raise ValueError(f"Wake-word pronunciation for {word} is incompatible with the tokens.")
                pronunciations[word] = phones
        missing = wanted - pronunciations.keys()
        if missing:
            raise ValueError("Wake-word pronunciation lexicon is missing: " + ", ".join(sorted(missing)))
        return [phone for word in phrase.upper().split() for phone in pronunciations[word]]

    @staticmethod
    def _read_tokens(path: Path) -> set[str]:
        if path.stat().st_size > 65536:
            raise ValueError("Wake-word token table exceeds its size limit.")
        symbols = set()
        ids = set()
        for line in path.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if len(fields) != 2:
                raise ValueError("Wake-word token table has invalid fields.")
            symbol, identifier = fields
            try:
                number = int(identifier)
            except ValueError as exc:
                raise ValueError("Wake-word token table has an invalid token ID.") from exc
            if number < 0 or symbol in symbols or number in ids:
                raise ValueError("Wake-word token table has duplicate or invalid tokens.")
            symbols.add(symbol)
            ids.add(number)
        return symbols

    def reset(self) -> None:
        """Discard all queued features and hypotheses before listening again."""
        with self._lock:
            if self._spotter is None:
                raise RuntimeError("Wake-word detector is closed.")
            self._epoch += 1
            self._last_detection = None
            self.last_candidate = None
            self._reset_stream()

    def _new_stream(self, threshold=None):
        if not self.candidate_replay:
            return self._spotter.create_stream()
        floor = self.threshold if threshold is None else threshold
        return self._spotter.create_stream(
            keywords="\n".join(f"{tokens} :{self.boost} #{floor} @{self._label}" for tokens in self._keyword_tokens))

    def _reset_stream(self) -> None:
        self._stream = self._new_stream()
        self._stream_samples = 0
        self._history.clear()
        self._history_samples = 0
        self._diagnostic_start = None
        self._diagnostic_samples = 0
        self._diagnostic_energy = 0.0
        self._diagnostic_peak = 0.0
        self._diagnostic_clipped = 0
        self._diagnostic_max_feed = 0.0
        self._diagnostic_min_chunk = math.inf

    def _result(self, stream):
        backend = getattr(self._spotter, "keyword_spotter", None)
        if backend is None:
            return self._spotter.get_result(stream).strip(), None, None
        # GetResult consumes duplicate suppression state; query it exactly once.
        result = backend.get_result(stream)
        return result.keyword.strip(), tuple(result.tokens), tuple(result.timestamps)

    def _retain_audio(self, part) -> None:
        if not self._history_limit:
            return
        self._history.append(part.copy())
        self._history_samples += len(part)
        while self._history_samples > self._history_limit:
            count = self._history_samples - self._history_limit
            oldest = self._history.popleft()
            if count < len(oldest):
                self._history.appendleft(oldest[count:].copy())
                self._history_samples -= count
            else:
                self._history_samples -= len(oldest)

    def _phase_audio(self, audio, start_sample):
        if self._timing is None:
            return audio
        # Replaying at a different encoder phase can miss an identical phrase.
        # Trim pre-keyword history to the next original 320 ms chunk boundary.
        skip = -start_sample % (self._timing[1] * 160)
        return audio[skip:]

    def _align_detection(self, remaining: int, *, audio=None, start_sample=None) -> WakeDetection | None:
        if self._timing is None or getattr(self._spotter, "keyword_spotter", None) is None:
            return None
        started = time.monotonic()
        replay = self._new_stream()
        if audio is None:
            audio = self._np.concatenate(tuple(self._history))
            start_sample = self._stream_samples - self._history_samples
        audio = self._phase_audio(audio, start_sample)
        replay.accept_waveform(self.sample_rate, audio)
        aligned = None
        _, shift = self._timing
        # The qualified T45/stride32 model needs at most five replay decodes.
        # Sherpa checks its idle clock reset before the following decode.
        for step in range(1, self._alignment_steps + 1):
            if time.monotonic() - started > 0.7 or not self._spotter.is_ready(replay):
                break
            self._spotter.decode_stream(replay)
            keyword, tokens, timestamps = self._result(replay)
            if not keyword:
                continue
            if keyword != self._label or tokens not in self._piece_variants or len(timestamps) != len(tokens):
                return None
            if (any(isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value < 0 for value in timestamps)
                    or any(left > right for left, right in zip(timestamps, timestamps[1:]))
                    or timestamps[-1] <= 0
                    or timestamps[-1] >= step * shift / 100):
                return None
            cut = math.ceil(timestamps[-1] * self.sample_rate)
            post_wake = len(audio) - cut + remaining
            if not 0 <= cut <= len(audio) or not 0 <= post_wake <= 3 * self.sample_rate:
                return None
            aligned = WakeDetection(post_wake, self._epoch, self._feed_id)
        if time.monotonic() - started > 0.7:
            return None
        return aligned

    def replay_candidate(self, threshold: float) -> bool:
        """Check a fresh bounded replay, not the continuous stream's confidence."""
        if not math.isfinite(threshold) or not 0 < threshold <= 1:
            raise ValueError("Wake threshold must be between zero and one.")
        with self._lock:
            candidate = self.last_candidate
            if (not self.candidate_replay or candidate is None
                    or candidate.epoch != self._epoch or candidate.feed_id != self._feed_id):
                return False
            replay = self._new_stream(threshold)
            replay.accept_waveform(self.sample_rate, self._phase_audio(candidate.audio, candidate.start_sample))
            found = False
            while self._spotter.is_ready(replay):
                self._spotter.decode_stream(replay)
                result, _, _ = self._result(replay)
                found = found or result == self._label
            return found

    def align_candidate(self) -> None:
        """Resolve the accepted candidate without changing its epoch/feed identity."""
        with self._lock:
            candidate = self.last_candidate
            if (candidate is not None and candidate.epoch == self._epoch
                    and candidate.feed_id == self._feed_id):
                self._last_detection = self._align_detection(
                    candidate.remaining, audio=candidate.audio, start_sample=candidate.start_sample)

    def feed(self, samples, *, align: bool = True) -> bool:
        """Consume normalized mono 16 kHz audio, returning True for this phrase."""
        with self._lock:
            if self._stream is None:
                raise RuntimeError("Wake-word detector is closed.")
            self._feed_id += 1
            self._last_detection = None
            self.last_candidate = None
            np = self._np
            audio = np.asarray(samples, dtype=np.float32)
            if audio.ndim != 1 or not np.isfinite(audio).all() or (np.abs(audio) > 1).any():
                raise ValueError("Wake-word audio must be finite mono floats between -1 and 1.")
            started = time.monotonic()
            if audio.size:
                if self._diagnostic_start is None:
                    self._diagnostic_start = started
                self._diagnostic_samples += audio.size
                self._diagnostic_energy += float(np.dot(audio, audio))
                self._diagnostic_peak = max(self._diagnostic_peak, float(np.max(np.abs(audio))))
                self._diagnostic_clipped += int(np.count_nonzero(np.abs(audio) >= .999))
                self._diagnostic_min_chunk = min(self._diagnostic_min_chunk, audio.size / self.sample_rate)
            # Decode as audio arrives rather than queuing a whole large caller chunk.
            for offset in range(0, len(audio), 1600):
                part = np.ascontiguousarray(audio[offset:offset + 1600])
                self._retain_audio(part)
                self._stream_samples += len(part)
                self._stream.accept_waveform(self.sample_rate, part)
                while self._spotter.is_ready(self._stream):
                    self._spotter.decode_stream(self._stream)
                    result, _, _ = self._result(self._stream)
                    if result:
                        detected = result.strip() == self._label
                        remaining = len(audio) - offset - len(part)
                        candidate = None
                        if detected:
                            retained = (np.concatenate(tuple(self._history)) if self._history
                                        else part.copy())
                            candidate = WakeCandidate(self._epoch, self._feed_id, self.threshold,
                                                      self.boost, retained, remaining,
                                                      self._stream_samples - len(retained))
                        event = self._align_detection(remaining) if detected and align else None
                        self._reset_stream()
                        self._last_detection = event
                        self.last_candidate = candidate
                        return detected
            if self._diagnostic_start is not None:
                ended = time.monotonic()
                self._diagnostic_max_feed = max(self._diagnostic_max_feed, ended - started)
                elapsed = ended - self._diagnostic_start
                interval = 10 if self._diagnostic_windows < 6 else 60
                if elapsed >= interval:
                    log.info("Wake microphone: %.2fs audio over %.2fs wall, coverage %.3f, "
                             "RMS %.4f, peak %.3f, clipped %.4f, max inference %.1fms, min chunk %.1fms",
                             self._diagnostic_samples / self.sample_rate, elapsed,
                             self._diagnostic_samples / self.sample_rate / elapsed,
                             math.sqrt(self._diagnostic_energy / self._diagnostic_samples),
                             self._diagnostic_peak, self._diagnostic_clipped / self._diagnostic_samples,
                             self._diagnostic_max_feed * 1000,
                             self._diagnostic_min_chunk * 1000)
                    self._diagnostic_windows += 1
                    self._diagnostic_start = ended
                    self._diagnostic_samples = 0
                    self._diagnostic_energy = self._diagnostic_peak = self._diagnostic_max_feed = 0.0
                    self._diagnostic_clipped = 0
                    self._diagnostic_min_chunk = math.inf
            return False

    def close(self) -> None:
        with self._lock:
            self._last_detection = None
            self.last_candidate = None
            self._history.clear()
            self._history_samples = 0
            self._stream = None
            self._spotter = None
