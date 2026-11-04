# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0 (the "License").

"""Optional Piper speech synthesis, isolated in a cancellable child process."""

from __future__ import annotations

import asyncio
import contextlib
from collections import deque
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
import json
import logging
import math
from pathlib import Path
import struct
import sys
import time
from typing import AsyncIterator, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

log = logging.getLogger(__name__)
# Bump when PCM16 conversion, resampling, or chunk post-processing changes.
PCM_PROCESSING_VERSION = 1


@dataclass(frozen=True)
class SpeechRestart:
    """Drain the previous voice before repeating this sentence from its start."""


class PreparedSpeech:
    """Produce independently of playback, with at most 2 MiB of queued PCM.

    When a sample rate is supplied, prime an 80 ms buffer and re-prime after
    starvation. The target grows to the measured p99 uncovered producer wait,
    bounded at 500 ms. Priming adds at most 500 ms after the first available
    chunk; it never treats slow synthesis as a failed connection.
    """

    def __init__(self, stream, sample_rate: int | None = None, *,
                 min_buffer_s: float = .08, max_buffer_s: float = .5,
                 progress_at: Callable[[], float] | None = None):
        if sample_rate is not None and (type(sample_rate) is not int or sample_rate <= 0):
            raise ValueError("Prepared speech sample rate must be a positive integer")
        if (not all(math.isfinite(value) for value in (min_buffer_s, max_buffer_s))
                or not 0 <= min_buffer_s <= max_buffer_s or max_buffer_s <= 0):
            raise ValueError("Prepared speech buffer bounds must be finite and ordered")
        self._progress_at = progress_at
        self._minimum_s, self._maximum_s = min_buffer_s, max_buffer_s
        self._stream = stream
        self._rate = sample_rate
        self._chunks = asyncio.Queue(maxsize=128)
        self._available = asyncio.Event()
        self._ready = asyncio.Event()
        self._finished = False
        self._priming = True
        self._queued_samples = 0
        self._gaps: deque[float] = deque(maxlen=512)
        self._target_s = min_buffer_s
        self._gap_p99 = 0.0
        self._prime_wait_s = 0.0
        self._starvations = 0
        self._consumed_audio = False
        self._task = asyncio.create_task(self._produce())

    @property
    def progress_at(self) -> float:
        return self._progress_at() if self._progress_at is not None else 0.0

    @property
    def metrics(self):
        return {"jitter_target_ms": round(self._target_s * 1000, 3),
                "jitter_wait_ms": round(self._prime_wait_s * 1000, 3),
                "producer_starvations": self._starvations,
                "producer_gap_ms_p99": round(self._gap_p99 * 1000, 3),
                "producer_gap_window_count": len(self._gaps)}

    async def _produce(self):
        previous_audio_s = 0.0
        try:
            while True:
                started = time.monotonic()
                try:
                    chunk = await self._stream.__anext__()
                except StopAsyncIteration:
                    break
                if isinstance(chunk, SpeechRestart):
                    await self._chunks.put(chunk)
                    self._available.set()
                    self._ready.set()
                    previous_audio_s = 0.0
                    continue
                if self._rate is not None:
                    gap = max(0.0, time.monotonic() - started - previous_audio_s)
                    if previous_audio_s:
                        self._gaps.append(gap)
                        ordered = sorted(self._gaps)
                        self._gap_p99 = ordered[max(0, math.ceil(.99 * len(ordered)) - 1)]
                        self._target_s = min(self._maximum_s,
                                             max(self._minimum_s, self._gap_p99 + .04))
                    previous_audio_s = len(chunk) / self._rate
                for offset in range(0, len(chunk), 4096):
                    part = chunk[offset:offset + 4096].copy()
                    await self._chunks.put(part)
                    self._queued_samples += len(part)
                    self._available.set()
                    if self._rate is None or self._queued_samples >= self._target_s * self._rate:
                        self._ready.set()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self._chunks.put(exc)
        finally:
            await self._stream.aclose()
            self._finished = True
            self._available.set()
            self._ready.set()
        await self._chunks.put(None)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._consumed_audio and self._priming and not self._available.is_set():
            self._starvations += 1
        await self._available.wait()
        if self._priming and self._rate is not None:
            started = time.monotonic()
            try:
                await asyncio.wait_for(self._ready.wait(), self._maximum_s)
            except asyncio.TimeoutError:
                pass
            self._prime_wait_s += time.monotonic() - started
            self._priming = False
        chunk = await self._chunks.get()
        if chunk is None:
            raise StopAsyncIteration
        if isinstance(chunk, Exception):
            raise chunk
        if not isinstance(chunk, SpeechRestart):
            self._queued_samples -= len(chunk)
            self._consumed_audio = True
        if self._chunks.empty() and not self._finished:
            self._available.clear()
            self._ready.clear()
            self._priming = True
        return chunk

    def cancel(self):
        self._task.cancel()
        while not self._chunks.empty():
            self._chunks.get_nowait()
        self._queued_samples = 0
        self._chunks.put_nowait(None)
        self._available.set()
        self._ready.set()

    async def aclose(self):
        self.cancel()
        await asyncio.gather(self._task, return_exceptions=True)


class PiperSpeech:
    """Speak through an installed Piper ONNX voice and its JSON sidecar.

    One child process retains the loaded model. Conversation text goes through
    stdin; errors report the exit status without exposing Piper's stderr, which
    may contain input text. Close active streams before closing the backend.
    The caller supplies the playback and conversation timeout.
    """

    def __init__(self, model_path: Path, *, cache_dir: Path | None = None):
        self.model_path = Path(model_path).expanduser().resolve()
        if not self.model_path.is_file():
            raise ValueError(f"Piper voice model is missing: {self.model_path}")
        config_path = Path(str(self.model_path) + ".json")
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError(f"Piper voice configuration is missing or invalid: {config_path}") from exc
        audio = config.get("audio") if isinstance(config, dict) else None
        rate = audio.get("sample_rate") if isinstance(audio, dict) else None
        if isinstance(rate, bool) or not isinstance(rate, int) or not 8000 <= rate <= 192_000:
            raise ValueError("Piper voice configuration needs a valid audio.sample_rate")
        self.sample_rate = rate
        self._config = config
        self._cache_dir = cache_dir
        self._disk_cache = None
        self._process = None
        self._lock = asyncio.Lock()
        self._warming_lock = asyncio.Lock()
        self._warming = None
        self._preempt_warming = asyncio.Event()
        self._foreground = 0
        self._idle = asyncio.Event()
        self._idle.set()
        # Fixed recordings live as long as this voice, across Muse reconnects.
        self.phrase_cache: dict[int, dict[str, tuple[np.ndarray, ...]]] = {}

    def _load_phrases(self, texts: tuple[str, ...], output_rate: int) -> dict:
        from reachy_muse.phrase_cache import PhraseCache

        if self._cache_dir is None:
            return {}
        if self._disk_cache is None:
            # Piper uses the sidecar's inference/phoneme settings and its default
            # synthesis arguments. Library upgrades can change those defaults.
            libraries = {}
            for package in ("piper-tts", "onnxruntime", "av"):
                try:
                    libraries[package] = version(package)
                except PackageNotFoundError:
                    libraries[package] = None
            try:
                self._disk_cache = PhraseCache(
                    self._cache_dir, self.model_path,
                    synthesis={"config": self._config, "libraries": libraries},
                    processing_version=PCM_PROCESSING_VERSION)
            except OSError:
                log.warning("Reachy fixed speech cache model could not be hashed; synthesizing instead")
                return {}
        return {text: audio for text in texts
                if (audio := self._disk_cache.load(text, output_rate)) is not None}

    async def load_phrases(self, texts: tuple[str, ...], output_rate: int) -> dict:
        return await asyncio.to_thread(self._load_phrases, texts, output_rate)

    async def save_phrase(self, text: str, output_rate: int, chunks) -> None:
        if self._disk_cache is not None:
            await asyncio.to_thread(self._disk_cache.save, text, output_rate, chunks)

    async def warm_phrase(self, text: str, output_rate: int) -> tuple:
        """Retry interrupted cache work after live speech releases the worker."""
        async def record():
            stream = self.stream(text, output_rate)
            try:
                return tuple([chunk async for chunk in stream])
            finally:
                await stream.aclose()

        async with self._warming_lock:
            while True:
                await self._idle.wait()
                self._preempt_warming.clear()
                self._warming = asyncio.create_task(record())
                interrupted = asyncio.create_task(self._preempt_warming.wait())
                try:
                    done, _ = await asyncio.wait({self._warming, interrupted}, timeout=60,
                                                 return_when=asyncio.FIRST_COMPLETED)
                    if self._warming in done:
                        return self._warming.result()
                    if interrupted not in done:
                        raise TimeoutError("Piper fixed phrase synthesis timed out")
                finally:
                    self._warming.cancel()
                    interrupted.cancel()
                    cleanup = asyncio.gather(self._warming, interrupted, return_exceptions=True)
                    cancelled = False
                    while not cleanup.done():
                        try:
                            await asyncio.shield(cleanup)
                        except asyncio.CancelledError:
                            cancelled = True
                    self._warming = None
                    if cancelled:
                        raise asyncio.CancelledError

    @contextlib.asynccontextmanager
    async def _synthesis_slot(self):
        foreground = asyncio.current_task() is not self._warming
        if foreground:
            self._foreground += 1
            self._idle.clear()
            self._preempt_warming.set()
        try:
            async with self._lock:
                yield
        finally:
            if foreground:
                self._foreground -= 1
                if not self._foreground:
                    self._idle.set()

    async def start(self) -> None:
        """Load the model once and wait until its worker is ready."""
        async with self._lock:
            try:
                await self._ensure_worker()
            except BaseException:
                await self._stop_worker()
                raise

    async def close(self) -> None:
        """Kill and reap the worker after the conversation has stopped."""
        async with self._lock:
            await self._stop_worker()

    def prepare(self, text: str, output_rate: int) -> PreparedSpeech:
        """Prepare one upcoming sentence while the speaker plays the previous one."""
        return PreparedSpeech(self.stream(text, output_rate), output_rate)

    async def _ensure_worker(self) -> None:
        if self._process is not None and self._process.returncode is None:
            return
        await self._stop_worker()
        spawn = asyncio.create_task(asyncio.create_subprocess_exec(
            sys.executable, "-m", "reachy_muse.local_speech", "--model", str(self.model_path),
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        ))
        try:
            self._process = await asyncio.shield(spawn)
        except asyncio.CancelledError:
            self._process = await spawn
            raise
        if await self._packet_size(self._process) != 0:
            raise ValueError("Piper worker returned an invalid ready handshake")

    async def _stop_worker(self) -> None:
        process, self._process = self._process, None
        if process is not None:
            process.stdin.close()
            if process.returncode is None:
                with contextlib.suppress(ProcessLookupError):
                    process.kill()
            # Drain stdout after killing the worker. A full asyncio pipe can
            # otherwise keep wait() blocked even after the child has exited.
            await process.communicate()

    @staticmethod
    async def _packet_size(process) -> int:
        try:
            header = await process.stdout.readexactly(4)
        except asyncio.IncompleteReadError:
            status = await process.wait()
            raise RuntimeError(f"Piper worker ended with exit status {status}; check the model and installation") from None
        size = struct.unpack("<I", header)[0]
        if size > 65536:
            raise ValueError("Piper audio packet exceeds 64 KiB")
        return size

    async def stream(self, text: str, output_rate: int) -> AsyncIterator[np.ndarray]:
        """Yield contiguous mono float32 chunks resampled to the speaker rate."""
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Piper speech requires nonempty text")
        if isinstance(output_rate, bool) or not isinstance(output_rate, int) or output_rate <= 0:
            raise ValueError("output_rate must be a positive integer")
        try:
            import av
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("Local speech requires the Muse Reachy audio dependencies") from exc
        async with self._synthesis_slot():
            succeeded = False
            try:
                await self._ensure_worker()
                process = self._process
                process.stdin.write((json.dumps({"text": text}, ensure_ascii=False) + "\n").encode("utf-8"))
                await process.stdin.drain()
                resampler = av.AudioResampler(format="fltp", layout="mono", rate=output_rate)
                pending = bytearray()
                samples_emitted = 0
                while True:
                    size = await self._packet_size(process)
                    if not size:
                        break
                    try:
                        pending.extend(await process.stdout.readexactly(size))
                    except asyncio.IncompleteReadError:
                        raise ValueError("Piper returned an incomplete audio packet") from None
                    complete = len(pending) // 2 * 2
                    if not complete:
                        continue
                    pcm = bytes(pending[:complete])
                    del pending[:complete]
                    samples = np.frombuffer(pcm, dtype="<i2").reshape(1, -1)
                    frame = av.AudioFrame.from_ndarray(samples, format="s16", layout="mono")
                    frame.sample_rate = self.sample_rate
                    for output in resampler.resample(frame):
                        samples = np.ascontiguousarray(output.to_ndarray().reshape(-1))
                        samples_emitted += len(samples)
                        yield samples
                if pending:
                    raise ValueError("Piper returned an incomplete PCM16 sample")
                for output in resampler.resample(None):
                    samples = np.ascontiguousarray(output.to_ndarray().reshape(-1))
                    samples_emitted += len(samples)
                    yield samples
                if not samples_emitted:
                    raise RuntimeError("Piper returned no speech audio")
                succeeded = True
            finally:
                if not succeeded:
                    await self._stop_worker()


def _worker(model_path: Path) -> None:
    import onnxruntime
    from piper import PiperVoice
    from piper.config import PiperConfig

    config = PiperConfig.from_dict(json.loads(Path(str(model_path) + ".json").read_text(encoding="utf-8")))
    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = 2
    voice = PiperVoice(config=config, session=onnxruntime.InferenceSession(
        str(model_path), sess_options=options, providers=["CPUExecutionProvider"],
    ))
    output = sys.stdout.buffer
    output.write(struct.pack("<I", 0))
    output.flush()
    for line in sys.stdin:
        request = json.loads(line)
        text = request["text"]
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Piper worker requires nonempty text")
        for chunk in voice.synthesize(text):
            audio = chunk.audio_int16_bytes
            for offset in range(0, len(audio), 4096):
                packet = audio[offset:offset + 4096]
                output.write(struct.pack("<I", len(packet)))
                output.write(packet)
                output.flush()
        output.write(struct.pack("<I", 0))
        output.flush()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Muse local speech worker")
    parser.add_argument("--model", required=True, type=Path)
    _worker(parser.parse_args().model)
