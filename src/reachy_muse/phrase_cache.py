# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0 (the "License").

"""Disposable, content-addressed recordings of public fixed phrases."""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
from pathlib import Path
import tempfile

log = logging.getLogger(__name__)
_MAGIC = b"RMPCM\x00\x01"
_HEADER_SIZE = len(_MAGIC) + 64
_MAX_PCM_BYTES = 32 * 1024 * 1024


class PhraseCache:
    """Store mono little-endian float32 PCM; all methods run off the audio loop."""

    def __init__(self, directory: Path, model: Path, *, synthesis: dict, processing_version: int):
        self.directory = directory
        digest = hashlib.sha256()
        with model.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        self._voice = {
            "model_sha256": digest.hexdigest(),
            "synthesis": synthesis,
            "processing_version": processing_version,
        }

    def _key(self, text: str, rate: int) -> bytes:
        identity = dict(self._voice, text=text, sample_rate=rate)
        return hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":"),
                                         ensure_ascii=False, allow_nan=False).encode("utf-8")).digest()

    def load(self, text: str, rate: int):
        import numpy as np

        key = self._key(text, rate)
        try:
            with (self.directory / (key.hex() + ".pcm")).open("rb") as source:
                data = source.read(_HEADER_SIZE + _MAX_PCM_BYTES + 1)
        except FileNotFoundError:
            return None
        except OSError:
            log.warning("Reachy fixed speech cache could not be read; synthesizing instead")
            return None
        pcm = data[_HEADER_SIZE:]
        if (data[:len(_MAGIC)] != _MAGIC or data[len(_MAGIC):len(_MAGIC) + 32] != key
                or not pcm or len(pcm) > _MAX_PCM_BYTES or len(pcm) % 4
                or data[len(_MAGIC) + 32:_HEADER_SIZE] != hashlib.sha256(pcm).digest()):
            log.warning("Reachy ignored a corrupt fixed speech recording")
            return None
        samples = np.frombuffer(pcm, dtype="<f4")
        if not np.isfinite(samples).all():
            log.warning("Reachy ignored nonfinite fixed speech audio")
            return None
        return tuple(samples[offset:offset + 4096] for offset in range(0, len(samples), 4096))

    def save(self, text: str, rate: int, chunks) -> None:
        import numpy as np

        pcm = b"".join(np.asarray(chunk, dtype="<f4").tobytes() for chunk in chunks)
        if not pcm or len(pcm) > _MAX_PCM_BYTES:
            return
        key = self._key(text, rate)
        temporary = None
        try:
            self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=self.directory, prefix=".phrase-", suffix=".tmp",
                                             delete=False) as target:
                temporary = Path(target.name)
                target.write(_MAGIC + key + hashlib.sha256(pcm).digest() + pcm)
                target.flush()
                os.fsync(target.fileno())
            os.replace(temporary, self.directory / (key.hex() + ".pcm"))
        except OSError:
            log.warning("Reachy fixed speech cache could not be saved; retaining audio in memory")
        finally:
            if temporary is not None:
                with contextlib.suppress(OSError):
                    temporary.unlink(missing_ok=True)
