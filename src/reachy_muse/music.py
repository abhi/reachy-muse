# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0 (the "License").

"""One replaceable music or room-dance session, independent of speech admission."""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field
import logging
import math
from pathlib import Path
import queue
import random
import threading
import time

import numpy as np

from reachy_muse.motion.dance import BeatTracker, DanceTimeline
from reachy_muse.motion.dance_signature import DanceSignature
from reachy_muse.music_analysis import AudioAnalyzer
from reachy_muse.music_structure import SongPlan, StructureTracker
from reachy_muse.music_metadata import TrackMetadata
from reachy_muse.music_audio import AudioBlock, MusicMix, MusicOutput, decode

log = logging.getLogger(__name__)
DUCK_GAIN = 10 ** (-18 / 20)
FORMATS = frozenset({".mp3", ".ogg", ".flac", ".wav"})


def select_track(folder: Path, query: str) -> Path:
    root = folder.expanduser().resolve()
    matches = []
    for count, path in enumerate(root.rglob("*")):
        if count >= 10000:
            break
        if path.suffix.lower() in FORMATS and path.is_file():
            resolved = path.resolve()
            if (resolved.is_relative_to(root)
                    and (query.casefold() == "shuffle" or query.casefold() in path.name.casefold())):
                matches.append(resolved)
    if not matches:
        raise ValueError("No matching music file")
    return random.choice(matches)


@dataclass
class _Session:
    source: str | Path | None
    title: str | None
    dance: bool
    volume: float
    started: float
    deadline: float
    cancelled: threading.Event = field(default_factory=threading.Event)
    blocks: queue.Queue = field(default_factory=lambda: queue.Queue(maxsize=25))
    rooms: deque = field(default_factory=lambda: deque(maxlen=25))
    eof: bool = False
    failed: bool = False
    block: AudioBlock | None = None
    position: int = 0
    consumed: int = 0
    tail: float = 0.
    gain: float = 0.
    gain_target: float = -1.
    gain_remaining: int = 0
    last_beat: float = 0.
    room_gap: bool = False
    primed: bool = False
    plan: SongPlan = SongPlan()
    metadata: TrackMetadata = TrackMetadata()
    signatures: dict = field(default_factory=dict)
    requested: set = field(default_factory=set)


class MusicController:
    def __init__(self, output: MusicOutput, input_rate: int, *, folder=Path("~/Music"),
                 echo_cancelled=False, motion=True, decoder=decode):
        self.output, self.input_rate = output, input_rate
        self.folder, self.motion = folder, motion
        self._cap = .5 if echo_cancelled else .2
        self._decode = decoder
        self._clock = output.clock
        self._lock = output.lock
        self._session = None
        self._holds = set()
        self.dance = DanceTimeline()
        self._closed = threading.Event()
        self._changed = threading.Event()
        self._worker = self._pump = self._analysis_worker = None
        output.music = self._mix

    @property
    def active(self):
        return self._session is not None

    @property
    def playing(self):
        session = self._session
        return session is not None and session.source is not None

    @property
    def dancing(self):
        with self._lock:
            return (self._session is not None and self._session.dance and not self._holds
                    and (self._session.source is None or self._clock() < self._session.tail + .06)
                    and self._clock() >= self.output.speech_until)

    def status(self):
        with self._lock:
            session = self._session
            if session is None:
                return {"playing": False, "elapsed": 0., "dancing": False}
            elapsed = (max(0., session.consumed / self.output.rate
                           - max(0., session.tail + .06 - self._clock()))
                       if session.source is not None else self._clock() - session.started)
            label = ({"title": session.title} if session.title is not None else
                     {"source": session.source} if session.source is not None else {})
            return {"playing": session.source is not None, "elapsed": round(elapsed, 2),
                    "dancing": self.dancing, **label, **session.metadata.payload()}

    def play(self, params, *, allowed=lambda: True):
        source = params.get("url") or select_track(self.folder, params["query"])
        with self._lock:
            if not allowed():
                raise ValueError("Music request expired")
            return self._start(source, source.stem if isinstance(source, Path) else None,
                               params.get("dance", True), min(params.get("volume", .35), self._cap), math.inf)

    def listen(self, seconds=60.):
        if not self.motion:
            raise ValueError("Motion unavailable")
        return self._start(None, None, True, 0., self._clock() + seconds)

    def _start(self, source, title, dance, volume, deadline):
        with self._lock:
            if self._closed.is_set():
                raise ValueError("Music unavailable")
            self.stop()
            now = self._clock()
            self._session = _Session(source, title, dance and self.motion, volume, now, deadline,
                                     last_beat=now)
            if self._worker is None:
                self._worker = threading.Thread(target=self._work, name="reachy-music-decode", daemon=True)
                self._pump = threading.Thread(target=self._run_output, name="reachy-music-output", daemon=True)
                self._analysis_worker = threading.Thread(target=self._analyze_files, name="reachy-music-analysis", daemon=True)
                self._analysis_worker.start()
                self._worker.start()
                self._pump.start()
            self._changed.set()
            return self.status()

    def hold(self, reason: str, active: bool):
        with self._lock:
            if active:
                self._holds.add(reason)
            else:
                self._holds.discard(reason)

    def stop(self, *, speech=False):
        with self._lock:
            session, self._session = self._session, None
            if session is not None:
                session.cancelled.set()
                self.dance.clear()
                self.output.clear(speech=speech)
                log.info("Music stopped after %.2f s", max(0., self._clock() - session.started))
            self._changed.set()

    def close(self):
        self.stop(speech=True)
        self._closed.set()
        self._changed.set()
        for worker in (self._worker, self._pump, self._analysis_worker):
            if worker is not None:
                worker.join(timeout=6)
                if worker.is_alive():
                    raise RuntimeError("Music worker did not stop")

    def tap(self, pcm, captured_at, *, gap=False):
        if not self._lock.acquire(blocking=False):
            session = self._session
            if session is not None:
                session.room_gap = True
            return
        try:
            session = self._session
            if session is None or session.source is not None:
                return
            if len(session.rooms) == session.rooms.maxlen:
                session.rooms.clear()
                session.room_gap = True
            session.rooms.append((pcm.copy(), captured_at, gap or session.room_gap))
            session.room_gap = False
        finally:
            self._lock.release()
        self._changed.set()

    def _work(self):
        handled = None
        while not self._closed.is_set():
            with self._lock:
                session = self._session
            if session is None or session is handled:
                self._changed.wait(.05)
                self._changed.clear()
                continue
            handled = session
            if session.source is None:
                self._room(session)
                continue
            try:
                for block in self._decode(session.source, self.output.rate, session.cancelled):
                    if isinstance(session.source, str) and block.plan.sections:
                        with self._lock:
                            session.plan = block.plan
                    while not session.cancelled.is_set():
                        try:
                            session.blocks.put(block, timeout=.05)
                            break
                        except queue.Full:
                            continue
                    if session.cancelled.is_set():
                        break
            except Exception:
                with self._lock:
                    session.failed = True
                log.warning("Music decode failed")
            finally:
                with self._lock:
                    session.eof = True

    async def signatures(self, source):
        while True:
            await asyncio.sleep(.1)
            if source is None or not getattr(source, "library_available", False):
                continue
            with self._lock:
                session = self._session
                if session is None or not session.dance or self._holds or len(session.requested) >= 8:
                    continue
                seconds = (session.consumed / self.output.rate if session.source is not None
                           else self._clock() - session.started)
                peak = next((s for s in session.plan.sections
                             if (s.chorus or s.drop) and s.family not in session.requested
                             and s.end > seconds), None)
                if peak is None:
                    continue
                session.requested.add(peak.family)
            try:
                motion = await source.generate("dance_" + peak.character, 4 * peak.period)
                with self._lock:
                    if self._session is session and not session.cancelled.is_set():
                        ready = (session.consumed / self.output.rate if session.source is not None
                                 else self._clock() - session.started)
                        signature = DanceSignature.from_motion(motion, ready)
                        if signature is not None:
                            session.signatures[peak.family] = signature
            except asyncio.CancelledError:
                raise
            except Exception:
                log.warning("Music signature unavailable")

    def _analyze_files(self):
        handled = None
        while not self._closed.wait(.02):
            with self._lock:
                session = self._session
            if (session is None or session is handled or not isinstance(session.source, Path)
                    or not session.primed):
                continue
            handled = session
            structure = StructureTracker(16000, rolling=False)
            cpu_start, wall_start = time.thread_time(), time.monotonic()
            published = 0
            try:
                for block in self._decode(session.source, 16000, session.cancelled):
                    if session.cancelled.is_set():
                        break
                    structure.feed(block.features)
                    if block.offset / 16000 >= published + 4:
                        published = block.offset / 16000
                        with self._lock:
                            if self._session is session:
                                session.plan = structure.plan()
                        delay = (time.thread_time() - cpu_start) / .15 - (time.monotonic() - wall_start)
                        if session.cancelled.wait(max(0., delay)):
                            break
                if not session.cancelled.is_set():
                    plan = structure.plan(complete=True)
                    with self._lock:
                        if self._session is session:
                            session.plan = plan
            except Exception:
                log.warning("Music analysis failed")

    def _room(self, session):
        tracker = BeatTracker(self.input_rate)
        analyzer = AudioAnalyzer(self.input_rate)
        structure = StructureTracker(self.input_rate)
        offset = 0
        while not self._closed.is_set() and not session.cancelled.is_set():
            with self._lock:
                frame = session.rooms.popleft() if session.rooms else None
            if frame is None:
                self._changed.wait(.02)
                self._changed.clear()
                continue
            pcm, at, gap = frame
            if gap:
                tracker = BeatTracker(self.input_rate)
                analyzer = AudioAnalyzer(self.input_rate)
                structure = StructureTracker(self.input_rate)
                offset = 0
            beats = tracker.feed(pcm)
            features = analyzer.feed(pcm)
            structure.feed(features)
            start = at - len(pcm) / self.input_rate
            with self._lock:
                if self._session is session:
                    if gap:
                        self.dance.clear()
                    session.plan = structure.plan()
                    self.dance.offer_features(features, offset, start, self.input_rate, session.plan, session.signatures)
                    if beats:
                        session.last_beat = at
            offset += len(pcm)

    def _mix(self, count, start, speech):
        session = self._session
        if session is None or session.source is None:
            return None
        if not session.primed:
            if session.blocks.qsize() < 12 and not session.eof:
                return None
            session.primed = True
        mixed = np.zeros(count, dtype=np.float32)
        used = 0
        scheduled = []
        while used < count:
            if session.block is None:
                try:
                    session.block = session.blocks.get_nowait()
                    session.position = 0
                except queue.Empty:
                    break
            block, position = session.block, session.position
            size = min(count - used, len(block.pcm) - position)
            mixed[used:used + size] = block.pcm[position:position + size]
            beats = tuple(b for b in block.beats if block.offset + position <= b.sample < block.offset + position + size)
            features = tuple(f for f in block.features if block.offset + position <= f.sample < block.offset + position + size)
            scheduled.append((beats, features, block.offset + position, start + used / self.output.rate + .06,
                              session.plan if session.plan.at((block.offset + position) / self.output.rate) is not None else block.plan, block.metadata))
            session.position += size
            used += size
            if session.position == len(block.pcm):
                session.block = None
        if not used:
            return None
        target = session.volume * (DUCK_GAIN if self._holds or speech or start < self.output.speech_until else 1.)
        if target != session.gain_target:
            session.gain_target = target
            session.gain_remaining = max(1, round(self.output.rate * (.04 if target < session.gain else .4)))
        ramp = min(count, session.gain_remaining)
        gains = np.full(count, target, dtype=np.float32)
        if ramp:
            step = (target - session.gain) / session.gain_remaining
            gains[:ramp] = session.gain + step * np.arange(1, ramp + 1)
            session.gain += step * ramp
            session.gain_remaining -= ramp
        def accepted():
            session.tail = start + used / self.output.rate
            session.consumed += used
            for beats, features, offset, at, plan, metadata in scheduled:
                session.metadata = metadata
                if features:
                    self.dance.offer_features(features, offset, at, self.output.rate, plan, session.signatures)
                self.dance.offer(beats, offset, at, self.output.rate)
        return MusicMix(mixed * gains, accepted)

    def tick(self):
        with self._lock:
            session = self._session
            if session is not None:
                now = self._clock()
                expired = session.source is None and (now >= session.deadline or now - session.last_beat >= 8)
                ended = (session.source is not None and session.eof and session.block is None
                         and session.blocks.empty() and now >= session.tail + .06)
                if expired or ended or session.failed:
                    self.stop()
            self.output.pump()

    def _run_output(self):
        while not self._closed.wait(.01):
            try:
                self.tick()
            except Exception:
                log.warning("Music output failed")
                with self._lock:
                    session, self._session = self._session, None
                    if session is not None:
                        session.cancelled.set()
                    self.dance.clear()
