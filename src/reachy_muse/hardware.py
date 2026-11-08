# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Reachy Mini audio and motion, with one thread owning the target poses.

The public methods are synchronous. An asyncio caller should offload start,
close, read_audio, clear_audio, and run_command. Speech sway comes from the
audio handed to play_audio and is added on the motion thread, or, with
REACHY_MUSE_SWAY=sdk, from the SDK's wobbler, which adds offsets on the daemon.
Neither replaces this driver's primary poses.
"""

from __future__ import annotations

import http.client
import json
import logging
import math
import queue
import socket
import threading
import time
import urllib.request
from concurrent.futures import Future, TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Collection, NamedTuple, Sequence

from reachy_muse.contract.commands import POSE_LIMITS, parse_command
from reachy_muse.motion import MotionStyle
from reachy_muse.presence import Presence
from reachy_muse.speaker_direction import DoAReading, SpeakerYaw, speaker_yaw

if TYPE_CHECKING:
    import numpy as np

    from reachy_muse.arrival import WatchPose
    from reachy_muse.motion.layers import ClipTimeline
    from reachy_muse.motion.sentence import SentenceMotion

log = logging.getLogger(__name__)
EMOTIONS_DATASET = "pollen-robotics/reachy-mini-emotions-library"
STATES = frozenset({"idle", "listening", "thinking", "speaking", "error", "sleeping"})
VOLUME_TIMEOUT_S = 1.0
_DOA_TIMEOUT_S = .1
_DOA_RESPONSE_BYTES = 4096
BUILTIN_EXPRESSIONS = frozenset({
    "neutral", "happy", "sad", "surprised", "curious", "nod", "shake",
    "listening", "thinking", "error",
})
EXPRESSION_POSES = {
    "neutral": ({}, [-10, 10]), "happy": ({"z": 7, "roll": -8}, [-35, 35]),
    "sad": ({"z": -5, "pitch": 10}, [-55, 55]),
    "surprised": ({"z": 10, "pitch": -7}, [-5, 5]),
    "curious": ({"roll": 12, "pitch": -4}, [-20, 5]),
    "listening": ({"pitch": -3, "z": 3}, [-12, 12]),
    "thinking": ({"roll": 12, "pitch": -5, "z": 5}, [-45, 45]),
    "error": ({"pitch": 8, "z": -3}, [-35, 35]),
}
# Recorded Pollen emotions shown when a spoken line carries the expression. Each is
# the shortest, calmest fitting clip in EMOTIONS_DATASET; docs/reachy-mini.md lists
# their measured length and speed. Other expressions keep their static pose.
EXPRESSION_CLIPS = {"happy": "success1", "surprised": "surprised1", "curious": "inquiring2"}
SPOKEN_CLIP_INTENSITY = 0.6
# Leaving the sleep pose: the head lifts and the antennas unfold over this long. Pollen's
# wake-mini-up clip lasts 15.7 s and swings the body 62 degrees, too long to hold a request for.
WAKE_UP_S = 1.0
SPEAKER_TURN_S = .65
_MOTION_PERIOD_S = 0.025
GESTURE_DURATION_S = 1.6
PROGRESS_ACCENT_S = 1.4
_GESTURE_KEYFRAMES = ((0., 0.), (.25, 1.), (.45, 1.), (.95, -1.),
                      (1.15, -1.), (GESTURE_DURATION_S, 0.))
HEAD_GESTURES = frozenset({"nod", "shake", "tilt"})
CAMERA_STILL_STEP = 0.02
CAMERA_AXES = [0, 1, 2, 3, 4, 5, 8]
CAMERA_STILL_S = 0.5
WATCH_TURN_S = 4.0


class _WatchTurn(NamedTuple):
    angles: dict
    share_from: float
    share_to: float
    since: float


def _ease(progress: float) -> float:
    progress = max(0.0, min(1.0, progress))
    return progress ** 3 * (10 - 15 * progress + 6 * progress ** 2)


def _blend_factor(reference: float, elapsed_s: float) -> float:
    """Keep the same smoothing rate when target writes take variable time."""
    elapsed_s = max(0.0, min(0.1, elapsed_s))
    return 1 - (1 - reference) ** (elapsed_s / _MOTION_PERIOD_S)


class ReachyHardwareError(RuntimeError):
    """The robot connection or media backend needs a fresh controller."""


@dataclass
class _Command:
    name: str
    params: dict
    deadline: float
    done: threading.Event = field(default_factory=threading.Event)
    cancelled: threading.Event = field(default_factory=threading.Event)
    result: dict | None = None


@dataclass
class _Clip:
    """A recorded move shown over the resting pose; a command clip answers its request when it fades out."""

    move: object
    timeline: ClipTimeline
    intensity: float
    request: _Command | None = None
    result: dict | None = None
    during: str | None = None  # the state the clip belongs to; it fades out when the state changes
    wake: bool = False


@dataclass(frozen=True)
class _Travel:
    """An eased move between two pose vectors, shaped like a recorded move so it plays as a clip."""

    start: np.ndarray
    end: np.ndarray
    duration: float
    head: Callable
    yaw_duration: float | None = None

    def evaluate(self, elapsed: float):
        values = self.start + (self.end - self.start) * _ease(elapsed / self.duration)
        if self.yaw_duration is not None:
            values[[5, 8]] = self.start[[5, 8]] + (self.end[[5, 8]] - self.start[[5, 8]]) * _ease(
                elapsed / self.yaw_duration)
        return (self.head(**dict(zip(("x", "y", "z", "roll", "pitch", "yaw"), values[:6]))),
                values[6:8] * (math.pi / 180), math.radians(values[8]))


@dataclass(frozen=True)
class _Frames:
    """Generated poses at ``fps``, shaped like a recorded move so they play as a clip.

    ``frames`` is (T, 9) in the order and units of POSE_LIMITS' axes; a looping one wraps around.
    """

    frames: np.ndarray
    fps: int
    loop: bool
    head: Callable

    @property
    def duration(self) -> float:
        return len(self.frames) / self.fps

    def evaluate(self, elapsed: float):
        position = elapsed * self.fps
        count = len(self.frames)
        if self.loop:
            position %= count
        index = min(int(position), count - 1)
        following = (index + 1) % count if self.loop else min(index + 1, count - 1)
        fraction = min(position - index, 1.0)
        values = self.frames[index] * (1 - fraction) + self.frames[following] * fraction
        return (self.head(**dict(zip(("x", "y", "z", "roll", "pitch", "yaw"), values[:6]))),
                values[6:8] * (math.pi / 180), math.radians(values[8]))


@dataclass(frozen=True)
class _PendingMotion:
    move: _Frames
    states: frozenset
    start_by: float


@dataclass(frozen=True)
class _SpeakerTurn:
    target: SpeakerYaw
    started: float
    head_from: float
    body_from: float

    def at(self, now: float) -> SpeakerYaw:
        weight = _ease((now - self.started) / SPEAKER_TURN_S)
        return SpeakerYaw(self.head_from + (self.target.head - self.head_from) * weight,
                          self.body_from + (self.target.body - self.body_from) * weight)


class ReachyController:
    """Own the robot for a standalone Muse session.

    Inject factories only for tests. No SDK, numpy, or move datasets are loaded
    until start. The emotion library loads in the background at start when
    expression clips are on, and otherwise for the first explicit named move.
    Set sleep_on_close=False when a Reachy app host owns the sleep lifecycle.
    """

    def __init__(
        self, host: str = "reachy-mini.local", connection_mode: str = "auto",
        media_backend: str = "default", *, port: int = 8000,
        enable_motion: bool = True, sleep_on_close: bool = True,
        library_expressions: bool = False,
        antenna_mode: str = "both",
        face_follow_model: Path | None = None,
        face_follow_idle: bool = False,
        face_follower_factory: Callable | None = None,
        arrival_model: Path | None = None,
        arrival_watcher_factory: Callable | None = None,
        wave_wake: bool = False,
        wave_watcher_factory: Callable | None = None,
        robot_factory: Callable | None = None,
        pose_factory: Callable | None = None, pose_interpolator: Callable | None = None,
        moves_factory: Callable | None = None,
        speaker_eq_gains: tuple[float, ...] | None = None,
        style: MotionStyle = MotionStyle(),
    ) -> None:
        if antenna_mode not in ("both", "left", "right", "none"):
            raise ValueError("antenna_mode must be both, left, right, or none")
        self.antenna_mode = antenna_mode
        self._inactive_antennas = tuple(
            index for index, side in enumerate(("right", "left"))
            if antenna_mode not in ("both", side)
        )
        self._frozen_antennas = None
        self._robot_factory = robot_factory
        self._pose_factory = pose_factory
        self._interpolate_pose = pose_interpolator
        self._moves_factory = moves_factory
        self._face_follow_model = face_follow_model
        self._face_follow_idle = face_follow_idle
        self._face_follower_factory = face_follower_factory
        self._face_follower = None
        self._face_failure_logged = False
        self._face_tracking_on = True
        self._face_tracking_status = None
        self._head_angles = (0.0, 0.0)
        self._arrival_model = arrival_model
        self._arrival_watcher_factory = arrival_watcher_factory
        self._arrival_watcher = None
        self._wave_enabled = wave_wake
        self._wave_watching = False
        self._wave_back_after = self._wave_ignore_until = -math.inf
        self._wave_pending: int | None = None
        self.wave_direction: float | None = None
        self._wave_watcher_factory = wave_watcher_factory
        self._wave_watcher = None
        from reachy_muse.camera_gaze import GazeSettler
        self._wave_gaze = GazeSettler()
        self._wave_gaze_status = None
        self._wave_lock = threading.Lock()
        self._camera_lock = threading.Lock()
        from reachy_muse.vision_budget import VisionBudget
        self.vision_budget = VisionBudget()
        self._arrival_warned = False
        self._arrival_blind_warned = False
        self._arrival_frames_at = 0.0
        self._presence = Presence()
        self._presence_detector = None
        self._presence_lock = threading.Lock()
        self._watching = False
        self._watch = _WatchTurn({}, 0.0, 0.0, -math.inf)
        self._still_since = None
        self._connect_options = dict(host=host, port=port, connection_mode=connection_mode,
                                     media_backend=media_backend)
        self._doa_lock = threading.Lock()
        self._doa_requests = queue.Queue()
        self._doa_thread = None
        self._doa_connection = None
        self._doa_socket = None
        self._sleep_on_close = sleep_on_close
        self._enable_motion = enable_motion
        self.library_expressions = library_expressions and enable_motion and style.expression_clips
        self._initial_motor_mode = None
        self._robot = None
        self._speaker_eq = None
        self._speaker_eq_gains = speaker_eq_gains
        self._np = None
        self._sample_rate = 0
        self._output_sample_rate = 0
        self.echo_cancelled_input = False
        self._motion_thread = None
        self._state_lock = threading.Lock()
        self._audio_lock = threading.Lock()
        from reachy_muse.motion.sentence_timeline import SentenceTimeline
        from reachy_muse.motion.sentence_delivery import SentenceDelivery
        self._sentences = SentenceTimeline()
        self._sentence_delivery = SentenceDelivery()
        self._audio_end = -math.inf
        self._sentence_terminal = None
        self._sentence_expression = None
        self._sentence_paused = False
        self._sentence_active = False
        self._sentence_serial = 0
        self._sentence_epoch = 0
        self._motion_epoch = 0
        self._state = "idle"
        self._state_started = time.monotonic()
        self._progress_started = None
        self._reply_expression = None
        self._stop = threading.Event()
        self._started = threading.Event()
        self._failure = None
        self._shutdown_failure = None
        self._commands: queue.Queue[_Command] = queue.Queue(maxsize=8)
        self._moves = None
        self._moves_load = None
        self._last_pose = None
        self._last_antennas = None
        self._last_body_yaw = 0.0
        self._style = style
        self._layers = None
        self._rest = None
        self._clip = None
        self._clip_line = None
        self._clip_source = None
        self._last_written_vector = None
        self._generated = None
        self._pending_motion = None
        self._stop_generated = False
        self._deferred = None
        self._speaker_pending: tuple[SpeakerYaw, float] | None = None
        self._speaker_look: SpeakerYaw | None = None
        self._speaker_turn: _SpeakerTurn | None = None
        self._speaker_gaze = 0.0
        self._command_active = False
        self._gesture_name = None
        self._gesture_started = 0.0
        self._gesture_epoch = None
        self.wobbling_available = False

    @property
    def motion_enabled(self) -> bool:
        return self._enable_motion

    @property
    def face_tracking_enabled(self) -> bool:
        return (self._face_follower is not None
                and getattr(self._face_follower, "failure", None) is None)

    @property
    def face_seen_at(self) -> float | None:
        return getattr(self._face_follower, "last_seen_at", None)

    def allow_face_tracking(self, enabled: bool) -> None:
        with self._state_lock:
            self._face_tracking_on = enabled

    def change_volume(self, step: int) -> int | None:
        """The daemon plays a short sound at the new level."""
        base = f"http://{self._connect_options['host']}:{self._connect_options['port']}/api/volume"
        try:
            with urllib.request.urlopen(f"{base}/current", timeout=VOLUME_TIMEOUT_S) as response:
                current = json.load(response)["volume"]
            level = max(0, min(100, int(current) + step))
            request = urllib.request.Request(f"{base}/set", data=json.dumps({"volume": level}).encode(),
                                             headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(request, timeout=VOLUME_TIMEOUT_S) as response:
                return int(json.load(response)["volume"])
        except (OSError, ValueError, KeyError, TypeError) as exc:
            log.warning("Reachy could not change its speaker volume: %s", type(exc).__name__)
            return None

    def read_doa(self) -> DoAReading | None:
        """Read world geometry only; missing arrays and older daemons are optional."""
        started = time.monotonic()
        if self._robot is None or self._stop.is_set() or not self._doa_lock.acquire(blocking=False):
            return None
        result = Future()
        if self._doa_thread is None:
            self._doa_thread = threading.Thread(target=self._doa_worker, name="muse-reachy-doa", daemon=True)
            self._doa_thread.start()
        self._doa_requests.put((started, result))
        try:
            return result.result(max(0.0, _DOA_TIMEOUT_S - (time.monotonic() - started)))
        except FutureTimeout:
            self._interrupt_doa()
            return None

    def _interrupt_doa(self) -> None:
        sock = self._doa_socket
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

    def _close_doa(self) -> None:
        if self._doa_thread is not None:
            self._doa_requests.put(None)
            self._interrupt_doa()
            self._doa_thread.join(timeout=.2)

    def _doa_worker(self) -> None:
        try:
            while (request := self._doa_requests.get()) is not None:
                started, result = request
                try:
                    reading = self._read_doa(started)
                except Exception as exc:
                    result.set_exception(exc)
                else:
                    result.set_result(reading)
                finally:
                    self._doa_socket = None
                    self._doa_lock.release()
        finally:
            if self._doa_connection is not None:
                self._doa_connection.close()
                self._doa_connection = None

    def _read_doa(self, started: float) -> DoAReading | None:
        if time.monotonic() - started >= _DOA_TIMEOUT_S:
            return None
        if self._doa_connection is None:
            self._doa_connection = http.client.HTTPConnection(
                self._connect_options["host"], self._connect_options["port"], timeout=_DOA_TIMEOUT_S)
        connection = self._doa_connection
        try:
            pose = self._robot.get_current_head_pose().copy()
            if connection.sock is None:
                connection.connect()
            if time.monotonic() - started >= _DOA_TIMEOUT_S:
                raise TimeoutError()
            self._doa_socket = connection.sock
            connection.request("GET", "/api/state/doa")
            with connection.getresponse() as response:
                payload = response.read(_DOA_RESPONSE_BYTES + 1)
                if (response.status != 200 or len(payload) > _DOA_RESPONSE_BYTES
                        or time.monotonic() - started >= _DOA_TIMEOUT_S):
                    raise ValueError("Invalid direction response")
            info = json.loads(payload)
            if not isinstance(info, dict):
                return None
            angle, speech = info.get("angle"), info.get("speech_detected")
            if (isinstance(angle, bool) or not isinstance(angle, (int, float))
                    or not math.isfinite(angle) or not 0 <= angle <= math.pi or not isinstance(speech, bool)):
                return None
            direction = pose[:3, :3] @ [math.sin(angle), math.cos(angle), 0.0]
            return DoAReading(math.pi / 2 - math.atan2(direction[1], direction[0]), speech)
        except (OSError, ValueError, TypeError, AssertionError, http.client.HTTPException):
            connection.close()
            self._doa_connection = None
            return None

    def look_at_speaker(self, angle: float) -> None:
        """Queue wake attention for the motion owner. An occupied motion is never interrupted."""
        target = speaker_yaw(angle)
        if not self._enable_motion:
            log.info("Wake direction angle=%.3f decision=still", angle)
            return
        with self._state_lock:
            if (self._command_active or self._clip is not None or self._generated is not None
                    or self._pending_motion is not None or self._deferred is not None):
                log.info("Wake direction yaw=%.1f body_yaw=%.1f decision=busy", target.head, target.body)
                return
            self._speaker_pending = (target, time.monotonic() + 1.0)

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def output_sample_rate(self) -> int:
        return self._output_sample_rate

    def start(self) -> None:
        if self._robot is not None:
            self._require_running()
            return
        import numpy as np
        self._np = np
        if self._robot_factory is None:
            from reachy_mini import ReachyMini
            self._robot_factory = ReachyMini
        if self._pose_factory is None:
            from reachy_mini.utils import create_head_pose
            self._pose_factory = create_head_pose
        if self._interpolate_pose is None:
            from reachy_mini.utils.interpolation import linear_pose_interpolation
            self._interpolate_pose = linear_pose_interpolation
        self._stop.clear()
        self._started.clear()
        self._failure = None
        self._shutdown_failure = None
        self._initial_motor_mode = None
        self._last_pose = None
        self._last_antennas = None
        self._frozen_antennas = None
        self._gesture_name = None
        self._gesture_epoch = None
        self._clip = None
        self._clip_line = None
        self._clip_source = None
        self._last_written_vector = None
        self._sentences.cancel()
        self._audio_end = -math.inf
        self._sentence_terminal = None
        self._sentence_active = self._sentence_paused = False
        self._generated = None
        self._pending_motion = None
        self._stop_generated = False
        self._deferred = None
        self._speaker_pending = self._speaker_look = self._speaker_turn = None
        self._speaker_gaze = 0.0
        self._command_active = False
        from reachy_muse.motion.layers import MotionLayers, Ramp
        self._layers = MotionLayers(self._style)
        self._generated_gain = Ramp(1.0)
        try:
            self._robot = self._robot_factory(**self._connect_options)
            import reachy_muse.speaker_eq
            self._speaker_eq = reachy_muse.speaker_eq.install_speaker_eq(
                self._robot.media.audio, gains=self._speaker_eq_gains or reachy_muse.speaker_eq.SPEAKER_EQ_GAINS)
            self._sample_rate = self._robot.media.get_input_audio_samplerate()
            self._output_sample_rate = self._robot.media.get_output_audio_samplerate()
            if min(self._sample_rate, self._output_sample_rate) <= 0:
                raise RuntimeError("Reachy Mini audio is unavailable; check the daemon media backend")
            self._robot.media.start_recording()
            # SDK 1.9's GStreamer queue defaults to 200 FIFO buffers. A live
            # conversation needs recent audio even while synthesis uses CPU.
            capture_sink = getattr(self._robot.media.audio, "_appsink_audio", None)
            if capture_sink is not None:
                capture_sink.set_property("max-buffers", 2)
                capture_sink.set_property("drop", True)
            self._robot.media.start_playing()
            from reachy_muse.duplex_audio import verified_echo_cancelled_input
            self.echo_cancelled_input = verified_echo_cancelled_input(self._robot.media.audio)
            log.info("Reachy microphone echo cancellation: %s",
                     "verified hardware duplex" if self.echo_cancelled_input else "playback gating")
            if self._speaker_eq is not None:
                speaker_rate = self._speaker_eq.verify_active()
                log.info("Reachy speaker equalization enabled at %d Hz", speaker_rate)
            if not self._enable_motion:
                return
            self._motion_thread = threading.Thread(target=self._motion_loop,
                                                   name="muse-reachy-motion", daemon=True)
            self._motion_thread.start()
            if self._style.expression_clips:
                self._start_loading_moves()
            if not self._started.wait(10):
                raise RuntimeError("Reachy Mini motion startup timed out")
            self._require_running()
            self._start_face_follow()
        except BaseException:
            self.close()
            raise

    def _require_running(self) -> None:
        if self._failure is not None:
            raise ReachyHardwareError(f"Reachy Mini motion failed: {self._failure}") from self._failure
        if self._robot is None or self._stop.is_set():
            raise ReachyHardwareError("Reachy Mini is not running")

    def read_audio(self) -> np.ndarray | None:
        """Read mono float32 at sample_rate; SDK 1.9 waits up to 20 ms."""
        self._require_running()
        try:
            sample = self._robot.media.get_audio_sample()
            if sample is None:
                return None
            audio = self._np.asarray(sample, dtype=self._np.float32)
            if audio.ndim == 2:
                audio = audio.mean(axis=1)
            if audio.ndim != 1 or not self._np.isfinite(audio).all():
                raise ValueError("Reachy Mini returned invalid microphone audio")
            return self._np.ascontiguousarray(audio, dtype=self._np.float32)
        except Exception as exc:
            raise ReachyHardwareError(f"Reachy Mini microphone failed: {exc}") from exc

    def discard_microphone(self) -> None:
        """Drop queued input without the SDK's 20 ms wait for fresh audio."""
        self._require_running()
        try:
            sink = self._robot.media.audio._appsink_audio
            while sink.try_pull_sample(0) is not None:
                pass
        except Exception as exc:
            raise ReachyHardwareError(f"Reachy Mini microphone flush failed: {exc}") from exc

    def play_audio(self, samples: np.ndarray, *, sentence: SentenceMotion | None = None,
                   accents: Sequence[tuple[int, str]] = ()) -> None:
        """Queue mono float32 samples at output_sample_rate without waiting.

        ``accents`` are (sample index into ``samples``, terminal) pairs: a sentence-end accent
        shape played when the speaker reaches that sample.
        """
        self._require_running()
        audio = self._np.asarray(samples, dtype=self._np.float32)
        if audio.ndim != 1 or not self._np.isfinite(audio).all():
            raise ValueError("Speaker audio must be a finite mono array")
        if audio.size:
            audio = self._np.ascontiguousarray(self._np.clip(audio, -1, 1))
            with self._audio_lock:
                try:
                    self._robot.media.push_audio_sample(audio)
                except Exception as exc:
                    raise ReachyHardwareError(f"Reachy Mini speaker failed: {exc}") from exc
                at = time.monotonic()
                from reachy_muse.motion.sentence_delivery import NEUTRAL_LIMITS, neutral_motion
                with self._state_lock:
                    if self._gesture_name == "wave":
                        self._gesture_name = None
                    epoch = self._sentences.epoch
                if self._sentence_epoch != epoch:
                    self._sentence_delivery.clear()
                sentence = self._sentence_delivery.accept(audio, self._output_sample_rate, sentence)
                planned = sentence.motion if sentence is not None else None
                if planned is None and sentence is not None and sentence.variant is not None:
                    planned = neutral_motion(sentence.variant)
                move = None
                if self._style.expression_clips and planned is not None:
                    from reachy_muse.motion.layers import clamp_to_limits
                    poses = self._np.asarray(planned.frames, dtype=float)
                    if (poses.ndim == 2 and poses.shape[1] == 9 and len(poses)
                            and self._np.isfinite(poses).all() and planned.fps > 0):
                        if sentence.variant is not None:
                            limits = self._np.asarray(NEUTRAL_LIMITS[sentence.variant])
                            poses = self._np.clip(poses, -limits, limits)
                        move = _Frames(clamp_to_limits(poses), planned.fps, False, self._head)
                from reachy_muse.motion.layers import SWAY_LATENCY_S
                from reachy_muse.motion.sentence_timeline import SentenceEvent
                start = max(at, self._audio_end)
                self._audio_end = start + audio.size / self._output_sample_rate
                with self._state_lock:
                    if sentence is not None and epoch == self._sentences.epoch:
                        self._sentence_paused = False
                        self._sentence_active = True
                        if not sentence.revised:
                            self._sentence_serial += 1
                        self._sentence_epoch = epoch
                        self._sentence_terminal = sentence.terminal
                        self._sentence_expression = sentence.expression.value
                        self._state = "speaking"
                        self._reply_expression = None
                        self._state_started = start + SWAY_LATENCY_S
                        self._clip_line = self._state_started
                        self._progress_started = None
                        self._sentences.offer(SentenceEvent(self._state_started, self._sentence_epoch,
                                                            "start", (sentence.expression.value, self._sentence_serial), move))
                        log.info("Reachy sentence motion %s variant=%s revised=%s", sentence.expression.value,
                                 sentence.variant, sentence.revised)
                if accents:
                    for index, terminal in accents:
                        self._sentences.offer(SentenceEvent(start + index / self._output_sample_rate + SWAY_LATENCY_S,
                                                            epoch, "end", (terminal, -1)))
                self._layers.feed_speech(audio, self._output_sample_rate, at)

    def finish_sentence(self) -> None:
        """Schedule a sentence-end accent at the accepted audio tail."""
        from reachy_muse.motion.layers import SWAY_LATENCY_S
        from reachy_muse.motion.sentence_timeline import SentenceEvent
        with self._audio_lock, self._state_lock:
            self._sentence_delivery.clear()
            if self._sentence_terminal is not None:
                self._sentences.offer(SentenceEvent(self._audio_end + SWAY_LATENCY_S, self._sentence_epoch,
                                                    "end", (self._sentence_terminal, self._sentence_serial)))
                self._sentence_terminal = None

    def clear_audio(self) -> None:
        """Flush queued speech without stopping the shared microphone pipeline."""
        self._require_running()
        with self._audio_lock:
            try:
                audio_backend = self._robot.media.audio
                clear_player = getattr(audio_backend, "clear_player", None)
                if clear_player is None:
                    raise RuntimeError("Reachy Mini audio backend cannot flush playback; use SDK 1.9 or newer")
                clear_player()
            except Exception as exc:
                raise ReachyHardwareError(f"Reachy Mini playback flush failed: {exc}") from exc
            now = time.monotonic()
            self._layers.interrupt_speech(now)
            self._audio_end = -math.inf
            self._sentence_delivery.clear()
            self._sentence_terminal = None
            with self._state_lock:
                self._sentence_active = False
                self._sentences.cancel()
                if self._gesture_epoch is not None:
                    self._gesture_name = None
                    self._gesture_epoch = None

    def set_state(self, state: str, *, expression: str | None = None) -> None:
        if state not in STATES:
            raise ValueError(f"Unknown conversation state: {state}")
        if expression is not None and expression not in BUILTIN_EXPRESSIONS:
            raise ValueError(f"Unknown reply expression: {expression}")
        if expression in HEAD_GESTURES:
            self.gesture(expression)
            expression = None
        with self._state_lock:
            if self._sentence_active:
                if state == "listening" and self._sentence_terminal is not None:
                    self._sentence_paused = True
                elif state == "speaking" and expression is None:
                    self._sentence_paused = False
            if self._starts_new_pose(state, expression):
                self._state_started = time.monotonic()
            self._state = state
            if state == "speaking" and self._gesture_name == "wave":
                self._gesture_name = None
            if state == "speaking" and expression is None and self._sentence_active:
                self._reply_expression = self._sentence_expression
            else:
                self._reply_expression = expression
            if state == "speaking" and expression is None and self._sentence_active:
                self._clip_line = self._state_started
            if state != "thinking":
                self._progress_started = None

    def _starts_new_pose(self, state: str, expression: str | None) -> bool:
        """Every spoken line and every changed pose restarts the motion clock; a thinking lean does not."""
        return (state != self._state or state == "speaking"
                or state != "thinking" and expression != self._reply_expression)

    def gesture(self, name: str) -> None:
        """Play an offset gesture independently of the resting pose and its clock."""
        if name not in HEAD_GESTURES and name != "wave":
            raise ValueError(f"Unknown head gesture: {name}")
        with self._state_lock:
            now = time.monotonic()
            if name == "wave":
                if (not self._enable_motion or self._state not in ("idle", "listening", "sleeping")
                        or now < max(self._wave_back_after, self._audio_end) or self._command_active):
                    return
                self._wave_back_after = now + GESTURE_DURATION_S + 3
                self._wave_ignore_until = now + GESTURE_DURATION_S + .5
                self._wave_gaze.invalidate()
            self._gesture_name = name
            self._gesture_started = now
            self._gesture_epoch = None

    def cue_progress(self) -> None:
        """Accent a spoken progress update without changing the thinking rhythm."""
        with self._state_lock:
            if self._state == "thinking":
                self._progress_started = time.monotonic()

    def play_motion(self, frames, fps: int, *, during: Collection[str], loop: bool = False,
                    start_within_s: float = 1.0) -> None:
        """Show generated poses over the resting pose, starting in one of the ``during`` states.

        ``frames`` holds one pose per row in the order and units of POSE_LIMITS' axes; every
        value is clamped to POSE_LIMITS here, whatever its source. The motion starts once the
        state is one of ``during`` and no other clip or command is showing, unless
        ``start_within_s`` passes first. It crossfades in and out like a recorded clip, a looping
        one repeats until stopped, and it fades out as soon as the state it started in changes.
        """
        import numpy as np
        from reachy_muse.motion.layers import CROSSFADE_S, clamp_to_limits, seamless_loop
        poses = np.asarray(frames, dtype=float)
        if poses.ndim != 2 or poses.shape[1] != 9 or not len(poses) or not np.isfinite(poses).all() or fps <= 0:
            raise ValueError("generated motion needs finite frames of nine values at a positive rate")
        poses = clamp_to_limits(poses)
        if not self._enable_motion:
            return
        if loop:
            poses = seamless_loop(poses, round(CROSSFADE_S * fps))
        pending = _PendingMotion(_Frames(poses, fps, loop, self._head), frozenset(during),
                                 time.monotonic() + start_within_s)
        with self._state_lock:
            self._pending_motion = pending

    def stop_motion(self) -> None:
        """Drop generated motion that has not started, and fade out the one showing."""
        with self._state_lock:
            self._pending_motion = None
            self._stop_generated = True

    def run_command(self, name: str, params: dict, timeout_ms: int | None = None) -> dict:
        """Wait for a bounded motion command; all motor targets stay on one thread."""
        try:
            if name not in ("reachy.expression", "reachy.move"):
                raise ValueError(f"unsupported command: {name}")
            self._require_running()
            if not self._enable_motion:
                raise RuntimeError("Reachy Mini motion is disabled for this session")
            command = parse_command(name, params, antenna_mode=self.antenna_mode, timeout_ms=timeout_ms)
            request = _Command(command.name, command.params, time.monotonic() + command.timeout_s)
            self._commands.put_nowait(request)
            if not request.done.wait(command.timeout_s):
                request.cancelled.set()
                return {"ok": False, "error": "Reachy Mini command timed out"}
            return request.result
        except (ValueError, RuntimeError, queue.Full) as exc:
            return {"ok": False, "error": str(exc) or "Reachy Mini motion queue is full"}

    def start_watching(self, pose: WatchPose | None) -> None:
        requested = {"yaw": pose.yaw_deg if pose else 0.0, "pitch": pose.pitch_deg if pose else 0.0}
        angles = {axis: max(POSE_LIMITS[axis][0], min(POSE_LIMITS[axis][1], value))
                  for axis, value in requested.items()}
        if angles != requested:
            log.info("Watch pose limited to yaw %.0f, pitch %.0f degrees", angles["yaw"], angles["pitch"])
        with self._state_lock:
            if not self._watching:
                self._arrival_frames_at = time.monotonic()
                self._arrival_blind_warned = False
            self._watching = True
            self._wave_pending = None
            self.wave_direction = None
        self.set_wave_watching(False)
        self._turn_watch(angles, 1.0)
        failed = self._arrival_watcher
        if failed is not None and failed.failure is not None:
            self._arrival_watcher = None
            self._arrival_warned = False
            failed.close()
        watcher = self._arrival_watcher or self._start_arrival_watcher()
        if watcher is not None:
            watcher.set_watching(True)

    def stop_watching(self) -> None:
        with self._state_lock:
            self._watching = False
        self._turn_watch(None, 0.0)
        if self._arrival_watcher is not None:
            self._arrival_watcher.set_watching(False)

    def _turn_watch(self, angles: dict | None, share: float) -> None:
        now = time.monotonic()
        current_angles, current = self._watch_share(now)
        if angles is None or angles == current_angles:
            angles = current_angles
        else:
            current = 0.0
        self._watch = _WatchTurn(angles, current, share, now)

    def _watch_share(self, now: float) -> tuple[dict, float]:
        turn = self._watch
        progress = _ease((now - turn.since) / WATCH_TURN_S)
        return turn.angles, turn.share_from + (turn.share_to - turn.share_from) * progress

    def _watched(self, coordinates: dict) -> dict:
        angles, share = self._watch_share(time.monotonic())
        return {**coordinates, **{axis: coordinates.get(axis, 0.0) + (value - coordinates.get(axis, 0.0)) * share
                                  for axis, value in angles.items()}}

    def arrival(self) -> float | None:
        watcher = self._arrival_watcher
        if watcher is None:
            return None
        failure = watcher.failure
        if failure is not None:
            self._warn_arrival_unavailable(type(failure).__name__)
            return None
        return watcher.arrival()

    def presence(self, timeout_s: float = 5.0) -> dict:
        """Reuse the watcher or inspect one frame; never move the robot."""
        deadline = time.monotonic() + timeout_s
        if timeout_s > 0 and self._presence_lock.acquire(timeout=timeout_s):
            result = Future()

            def inspect():
                try:
                    if time.monotonic() >= deadline:
                        raise FutureTimeout()
                    result.set_result(self._check_presence())
                except Exception as exc:
                    result.set_exception(exc)
                finally:
                    self._presence_lock.release()

            try:
                threading.Thread(target=inspect, name="muse-reachy-presence", daemon=True).start()
            except Exception:
                self._presence_lock.release()
                raise
            try:
                return result.result(max(0.0, deadline - time.monotonic()))
            except FutureTimeout:
                pass
        return Presence(last_seen_at=self._presence.last_seen_at).result(time.monotonic())

    def _check_presence(self) -> dict:
        from reachy_muse.arrival import YoloxPersonDetector, default_person_model, person_in_view

        observed = self._arrival_watcher.presence() if self._arrival_watcher is not None else Presence()
        last_seen = max((at for at in (self._presence.last_seen_at, observed.last_seen_at)
                         if at is not None), default=None)
        if self._watching:
            self._presence = Presence(observed.present, observed.observed_at, last_seen)
        else:
            observed_at = time.monotonic()
            present = None
            try:
                if self._robot is not None and not self._stop.is_set():
                    frame = self._read_camera_frame(background=False)
                    try:
                        if frame is not None:
                            if self._presence_detector is None:
                                self._presence_detector = YoloxPersonDetector(
                                    self._arrival_model or default_person_model())
                            present = person_in_view(self._presence_detector.detect(frame))
                    finally:
                        del frame
            except Exception:
                pass
            self._presence = Presence(last_seen_at=last_seen).observe(present, observed_at)
        return self._presence.result(time.monotonic())

    def log_arrival_diagnostics(self) -> None:
        from reachy_muse.arrival import ArrivalStats

        watcher = self._arrival_watcher
        stats = watcher.take_stats() if watcher is not None else ArrivalStats()
        if self._watching:
            now = time.monotonic()
            if stats.frames:
                self._arrival_frames_at = now
            elif (not self._arrival_blind_warned and now - self._arrival_frames_at >= 60
                  and (watcher is None or watcher.arrival() is None)):
                self._arrival_blind_warned = True
                log.warning("Arrival watching blind: no camera frames for at least 60s during morning window; "
                            "briefing cannot detect an arrival")
        yaw, pitch = self._read_head_angles()
        with self._state_lock:
            asleep = self._state == "sleeping"
        log.info("Arrival stats frames=%d frame_errors=%d brightness=%.1f motion_triggers=%d "
                 "detector_runs=%d max_person_confidence=%.3f pitch=%.1f yaw=%.1f asleep=%d still=%d",
                 stats.frames, stats.frame_errors, stats.brightness, stats.motion_triggers,
                 stats.detector_runs, stats.max_person_confidence,
                 math.degrees(pitch), math.degrees(yaw), asleep, self._camera_still())
        if self._watching and (watcher is None or watcher.failure is not None):
            angles = self._watch.angles
            from reachy_muse.arrival import WatchPose
            log.info("Arrival watcher restarting: unavailable")
            self.start_watching(WatchPose(angles["yaw"], angles["pitch"]))

    def _start_arrival_watcher(self):
        if self._arrival_model is None or not self._enable_motion or self._robot is None:
            self._warn_arrival_unavailable("no person model or camera")
            return None
        factory = self._arrival_watcher_factory
        if factory is None:
            from reachy_muse.arrival import ArrivalWatcher
            factory = ArrivalWatcher
        try:
            watcher = factory(lambda: self._read_camera_frame(background=False),
                              self._camera_still, self._arrival_model)
            watcher.start()
        except Exception as exc:
            self._warn_arrival_unavailable(type(exc).__name__)
            return None
        self._arrival_watcher = watcher
        log.info("Arrival watching ready; camera frames remain on Reachy")
        return watcher

    def _warn_arrival_unavailable(self, reason: str) -> None:
        if not self._arrival_warned:
            self._arrival_warned = True
            log.warning("Arrival detection is unavailable: %s; voice remains available", reason)

    def _camera_still(self) -> bool:
        still_since = self._still_since
        return (self._failure is None and still_since is not None
                and time.monotonic() - still_since >= CAMERA_STILL_S)

    def _read_camera_frame(self, *, background=True):
        """All local vision consumers share this reader; no frame cache survives it."""
        with self._camera_lock:
            if (self._robot is None or self._stop.is_set()
                    or background and not math.isfinite(self.vision_budget.delay())):
                return None
            return self._robot.media.get_frame()

    def _wave_still(self) -> bool:
        return (not self._watching and not self._stop.is_set()
                and self._state in ("idle", "listening", "sleeping") and self._failure is None
                and time.monotonic() >= max(self._wave_ignore_until, self._audio_end)
                and self._wave_gaze.gaze.settled
                and 0 <= time.monotonic() - self._wave_gaze.gaze.observed_at <= .1)

    def _read_wave_gaze(self):
        return self._wave_gaze.gaze

    def _update_wave_gaze(self, now, written, *, eligible):
        if not self._wave_enabled:
            return
        try:
            pose = self._robot.get_current_head_pose()
            joints, _ = self._robot.get_current_joint_positions()
            observed = [*self._pose_parameters(pose).values(), math.degrees(joints[0])]
            self._wave_gaze.update(now, written[CAMERA_AXES], observed, pose[:3, :3],
                                   eligible=eligible and not self._watching)
        except AssertionError:
            self._wave_gaze.invalidate()
        if not self._wave_watching:
            self._wave_gaze_status = None
            return
        status = "follow" if self._face_tracking_status == (True, "idle follow") else "moving"
        if self._wave_still():
            status = "wave watching"
        if status != self._wave_gaze_status:
            if status == "wave watching":
                log.info("Wave gaze: settled")
            log.info("Wave gaze: %s", status)
            self._wave_gaze_status = status

    def set_wave_watching(self, active: bool) -> None:
        """Called off the audio loop: disabling bounds the wait for in-flight inference."""
        with self._wave_lock:
            was_active = self._wave_watching
            active = (active and self._wave_enabled and not self._watching
                      and not self._stop.is_set() and self._robot is not None)
            if active and self._wave_watcher is None:
                from reachy_muse.wave_wake import WaveWatcher
                factory = self._wave_watcher_factory or WaveWatcher
                camera = getattr(self._robot.media, "camera", None)
                self._wave_watcher = factory(self._read_camera_frame, self._wave_still,
                                             self._arrival_model, read_gaze=self._read_wave_gaze,
                                             camera_matrix=getattr(camera, "K", None),
                                             budget=self.vision_budget)
                self._wave_watcher.start()
            if self._wave_watcher is not None:
                self._wave_watcher.set_active(active)
            self._wave_watching = active
            watcher = self._wave_watcher
        if watcher is not None and was_active and not active:
            watcher.drain()

    def take_wave(self) -> bool:
        with self._state_lock:
            generation = self._wave_gaze.gaze.generation
            detected = (not self._watching and self._wave_watcher is not None
                        and self._wave_watcher.take_wave())
            self._wave_pending = generation if detected else None
            self.wave_direction = self._waver_direction() if detected else None
            return detected

    def _waver_direction(self) -> float | None:
        """Convert the detected person's pixel center into an absolute bearing."""
        position = getattr(self._wave_watcher, "wave_position", None)
        if position is None:
            return None
        try:
            camera = self._robot.media.camera
            # A coarse pinhole bearing is enough to acquire the face for precise tracking.
            ray = self._np.linalg.solve(camera.K, [*position, 1.0])
            pose = self._robot.get_current_head_pose()
            direction = pose[:3, :3] @ [ray[2], -ray[0], -ray[1]]
            if not self._np.isfinite(direction).all():
                return None
            return math.pi / 2 - math.atan2(direction[1], direction[0])
        except (AttributeError, AssertionError, ValueError, self._np.linalg.LinAlgError):
            return None

    def claim_wave(self) -> bool:
        with self._state_lock:
            pending, self._wave_pending = self._wave_pending, None
            claimed = (pending is not None and pending == self._wave_gaze.gaze.generation
                       and self._wave_still())
            if not claimed:
                self.wave_direction = None
            return claimed

    def close(self) -> None:
        self._stop.set()
        self._close_doa()
        with self._wave_lock:
            if self._wave_watcher is not None:
                self._wave_watcher.close()
                self._wave_watcher = None
        if self._face_follower is not None:
            self._set_face_tracking(False, "shutdown")
        if self._motion_thread is not None:
            self._motion_thread.join(timeout=10)
            if self._motion_thread.is_alive():
                raise RuntimeError("Reachy Mini motion worker did not stop; media remains open")
            self._motion_thread = None
        if self._face_follower is not None:
            self._face_follower.close()
            self._face_follower = None
        if self._arrival_watcher is not None:
            self._arrival_watcher.close()
            self._arrival_watcher = None
        if self._robot is not None:
            robot, self._robot = self._robot, None
            with self._audio_lock, self._camera_lock:
                robot.__exit__(None, None, None)
        self._sample_rate = 0
        self._output_sample_rate = 0
        self.echo_cancelled_input = False
        if self._shutdown_failure is not None:
            failure, self._shutdown_failure = self._shutdown_failure, None
            raise ReachyHardwareError("Reachy Mini could not stop speech motion during shutdown") from failure

    def _head(self, **kwargs):
        return self._pose_factory(degrees=True, mm=True, **kwargs)

    def _start_face_follow(self) -> None:
        if self._face_follow_model is None or not self._enable_motion:
            return
        factory = self._face_follower_factory
        if factory is None:
            from reachy_muse.local_face_tracking import FaceFollower
            factory = FaceFollower
        self._face_follower = factory(
            self._read_camera_frame, self._face_follow_model,
            read_head_angles=self._read_head_angles,
            camera_matrix=self._robot.media.camera.K,
            distortion=self._robot.media.camera.D,
            budget=self.vision_budget,
        )
        self._face_follower.start()
        self._face_failure_logged = False
        self._face_tracking_status = None
        log.info("Local face tracking ready; camera frames remain on Reachy")

    def _read_head_angles(self) -> tuple[float, float]:
        # Center face coordinates on the wake gaze so a face there does not pull back to front.
        look = self._speaker_look
        origin = math.radians(look.head) if look is not None else 0.0
        try:
            pose = self._robot.get_current_head_pose()
        except AssertionError:
            # The SDK caches daemon poses; before its first frame use our
            # primary writer's last command rather than waiting on the network.
            return (self._head_angles[0] - origin, self._head_angles[1])
        coordinates = self._pose_parameters(pose)
        return (math.radians(coordinates["yaw"]) - origin, math.radians(coordinates["pitch"]))

    def _update_face_tracking(self, state: str, *, commanded: bool = False) -> None:
        follower = self._face_follower
        if follower is None:
            return
        failure = getattr(follower, "failure", None)
        if failure is not None and not self._face_failure_logged:
            log.warning("Local face tracking stopped: %s; voice remains available", type(failure).__name__)
            self._face_failure_logged = True
        if self._stop.is_set():
            reason = "shutdown"
        elif failure is not None:
            reason = "camera failure"
        elif not self._enable_motion:
            reason = "motion disabled"
        elif not self._face_tracking_on:
            reason = "disabled by user"
        elif state == "sleeping":
            reason = "sleeping"
        elif commanded:
            reason = "command"
        elif state in ("listening", "thinking", "speaking"):
            reason = "conversation"
        elif self._watching:
            reason = "arrival watching"
        elif state == "idle" and self._face_follow_idle:
            reason = "idle follow"
        else:
            reason = state
        self._set_face_tracking(reason in ("conversation", "idle follow"), reason)

    def _set_face_tracking(self, active: bool, reason: str) -> None:
        if reason == "idle follow":
            self._face_follower.set_active(active, idle=True)
        else:
            self._face_follower.set_active(active)
        status = (active, reason)
        if status != self._face_tracking_status:
            self._face_tracking_status = status
            log.info("Local face tracking %s: %s", "on" if active else "off", reason)

    def _face_pose(self, pose):
        follower = self._face_follower
        look = self._speaker_look
        target = follower.target() if follower is not None and self._speaker_turn is None else None
        if target is None:
            if look is None:
                return pose
            coordinates = self._pose_parameters(pose)
            coordinates["yaw"] = self._speaker_gaze
            return self._head(**coordinates)
        from reachy_muse.local_face_tracking import PITCH_LIMIT, YAW_LIMIT
        origin = look.head if look is not None else 0.0
        coordinates = self._pose_parameters(pose)
        coordinates["yaw"] = origin + math.degrees(target.yaw)
        coordinates["pitch"] = math.degrees(target.pitch)
        for name, limit in (("yaw", YAW_LIMIT), ("pitch", PITCH_LIMIT)):
            low, high = POSE_LIMITS[name]
            center = origin if name == "yaw" else 0.0
            low, high = max(low, center - math.degrees(limit)), min(high, center + math.degrees(limit))
            coordinates[name] = max(low, min(high, coordinates[name]))
        if look is not None:
            self._speaker_gaze = coordinates["yaw"]
        return self._head(**coordinates)

    def _clear_speaker(self) -> None:
        with self._state_lock:
            self._speaker_pending = None
        self._speaker_look = self._speaker_turn = None

    def _update_speaker(self, now: float, state: str, *, occupied: bool) -> None:
        with self._state_lock:
            pending = self._speaker_pending
            if pending is not None and (state != "sleeping" or now > pending[1]):
                self._speaker_pending = None
            else:
                pending = None
        if pending is not None:
            target, deadline = pending
            decision = "busy" if occupied else "stale" if now > deadline else "turn"
            log.info("Wake direction yaw=%.1f body_yaw=%.1f decision=%s", target.head, target.body, decision)
            if decision == "turn":
                self._speaker_look = target
                self._speaker_gaze = target.head
                self._speaker_turn = _SpeakerTurn(target, now, self._pose_parameters(self._last_pose)["yaw"],
                                                  math.degrees(self._last_body_yaw))

    def _write(self, pose, antennas, body_yaw: float) -> None:
        antennas = self._np.asarray(antennas, dtype=float).copy()
        for index in self._inactive_antennas:
            antennas[index] = self._frozen_antennas[index]
        self._robot.set_target(head=pose, antennas=antennas, body_yaw=float(body_yaw))
        self._last_pose = pose.copy()
        coordinates = self._pose_parameters(pose)
        self._head_angles = (math.radians(coordinates["yaw"]),
                             math.radians(coordinates["pitch"]))
        self._last_antennas = self._np.asarray(antennas, dtype=float).copy()
        self._last_body_yaw = float(body_yaw)

    def _pose_vector(self, pose, antennas, body_yaw: float):
        """One pose as values in the order and units of POSE_LIMITS."""
        return self._np.array([*self._pose_parameters(pose).values(),
                               *self._np.degrees(antennas), math.degrees(body_yaw)])

    def _write_vector(self, values) -> None:
        """Write a pose vector after the final POSE_LIMITS clamp."""
        from reachy_muse.motion.layers import AXES, clamp_to_limits
        values = clamp_to_limits(values)
        self._write(self._head(**dict(zip(AXES[:6], values[:6]))),
                    self._np.radians(values[6:8]), math.radians(values[8]))

    def _cue_clip(self, now: float, state: str, expression: str | None, line: float, *,
                  replace: bool = False, epoch: int | None = None) -> None:
        """Give each spoken line one chance, at its start, to play its expression's clip.

        A spoken clip fades out as soon as Reachy stops speaking.
        """
        clip = self._clip
        if (self._speaker_pending is not None or self._speaker_turn is not None
                or (clip is not None and clip.wake)):
            return
        if clip is not None and clip.during is not None and state != clip.during:
            clip.timeline.stop(now)
        if state != "speaking" or line == self._clip_line and not replace:
            return
        self._clip_line = line
        name = EXPRESSION_CLIPS.get(expression)
        moves = self._ready_moves() if self._style.expression_clips and name else None
        if clip is not None and (not replace or clip.request is not None):
            return
        if moves is None or name not in moves.list_moves():
            if clip is not None:
                clip.timeline.stop(now)
            return
        source = (self._last_written_vector.copy()
                  if clip is not None and self._last_written_vector is not None else None)
        from reachy_muse.motion.layers import ClipTimeline
        move = moves.get(name)
        incoming = _Clip(move, ClipTimeline(line if replace else now, float(move.duration)),
                         SPOKEN_CLIP_INTENSITY, during="speaking")
        with self._state_lock:
            if epoch is None or epoch == self._sentences.epoch:
                self._clip_source = source
                self._clip = incoming

    def _cue_generated(self, now: float, state: str) -> None:
        with self._state_lock:
            stop, self._stop_generated = self._stop_generated, False
            pending = self._pending_motion
            if pending is not None and now > pending.start_by:
                self._pending_motion = pending = None
            free = (self._clip is None and self._generated is None and self._deferred is None
                    and self._speaker_pending is None and self._speaker_turn is None)
            if pending is not None and free and state in pending.states:
                self._pending_motion = None
            else:
                pending = None
        generated = self._generated
        if generated is not None and (stop or state != generated.during):
            generated.timeline.stop(now)
        if pending is not None:
            from reachy_muse.motion.layers import ClipTimeline
            length = math.inf if pending.move.loop else pending.move.duration
            self._generated = _Clip(pending.move, ClipTimeline(now, length), 1.0, during=state)

    def _show_generated(self, now: float, pose, *, yielding: bool):
        """Crossfade generated motion over a resting pose vector; drop it once it has faded out.

        While ``yielding``, for a nod, tilt or progress accent on the resting pose, it eases out of the
        way over the crossfade and comes back afterwards.
        """
        if self._generated_gain.target != float(not yielding):
            self._generated_gain.to(float(not yielding), now)
        generated = self._generated
        if generated is not None and generated.timeline.finished(now):
            self._generated = generated = None
        return pose if generated is None else self._crossfade(generated, now, pose, self._generated_gain.at(now))

    def _show_clip(self, now: float, pose):
        """Crossfade the playing clip over a resting pose vector; drop the clip once it has faded out."""
        clip = self._clip
        if clip is None:
            return pose
        request = clip.request
        if request is not None and (request.cancelled.is_set() or now >= request.deadline):
            clip.timeline.stop(now)
        if clip.timeline.finished(now):
            self._clip = None
            self._clip_source = None
            if request is not None:
                request.result = clip.result
                request.done.set()
            return pose
        if self._clip_source is not None:
            source_time = min(now, clip.timeline.end)
            if source_time >= clip.timeline.started + clip.timeline.fade_in_s:
                self._clip_source = None
            else:
                from reachy_muse.motion.layers import crossfade_weight
                release = crossfade_weight(now - clip.timeline.end, clip.timeline.fade_s)
                source = self._crossfade(clip, source_time, self._clip_source)
                return source * (1 - release) + pose * release
        return self._crossfade(clip, now, pose)

    def _crossfade(self, clip: _Clip, now: float, pose, gain: float = 1.0):
        weight = clip.timeline.weight(now) * gain
        elapsed = min(clip.timeline.clip_time(now), clip.timeline.duration - 0.001)
        return pose * (1 - weight) + self._recorded_pose(clip.move.evaluate(elapsed), clip.intensity) * weight

    def _motion_loop(self) -> None:
        wobbling_active = False
        previous_state = None
        previous_written = None
        gaze = self._np.zeros(9)
        gesture_offset = self._np.zeros(9)
        try:
            if hasattr(self._robot, "client"):
                status = self._robot.client.get_status()
                backend = getattr(status, "backend_status", None)
                mode = getattr(backend, "motor_control_mode", None)
                mode = getattr(mode, "value", mode)
                if mode not in ("enabled", "disabled", "gravity_compensation"):
                    raise ReachyHardwareError("Could not read Reachy Mini's initial motor control mode")
                self._initial_motor_mode = mode
            else:
                self._initial_motor_mode = "disabled"
            self._robot.enable_motors()
            self._last_pose = self._robot.get_current_head_pose().copy()
            joints, antennas = self._robot.get_current_joint_positions()
            self._last_antennas = self._np.asarray(antennas, dtype=float).copy()
            if self._last_antennas.shape != (2,) or not self._np.isfinite(self._last_antennas).all():
                raise ReachyHardwareError("Reachy Mini returned invalid antenna positions")
            self._frozen_antennas = self._last_antennas.copy()
            self._last_body_yaw = float(joints[0])
            self.wobbling_available = all(
                callable(getattr(self._robot, name, None))
                for name in ("enable_wobbling", "disable_wobbling")
            )
            if self.wobbling_available:
                try:
                    self._robot.disable_wobbling()
                except Exception as exc:
                    log.warning("Reachy speech motion is unavailable: %s", exc)
                    self.wobbling_available = False
            wake = _Command("wake", {}, time.monotonic() + 5)
            self._travel(wake, self._head(), self._np.deg2rad([-10, 10]), 0, 1.5)
            self._started.set()
            previous_update = time.monotonic() - _MOTION_PERIOD_S
            while not self._stop.is_set():
                with self._state_lock:
                    state = self._state
                    expression = self._reply_expression
                    now = time.monotonic()
                    state_started = self._state_started
                    elapsed = now - state_started
                    progress_age = None if self._progress_started is None else now - self._progress_started
                    gesture = self._gesture_name
                    gesture_age = now - self._gesture_started
                    gesture_epoch = self._gesture_epoch
                gesture_active = (gesture is not None and gesture_age < GESTURE_DURATION_S
                                  and (gesture_epoch is None or gesture_epoch == self._sentences.epoch))
                wants_wobbling = self._style.sway == "sdk" and state == "speaking" and not gesture_active
                if self.wobbling_available and wants_wobbling != wobbling_active:
                    try:
                        if wants_wobbling:
                            self._robot.enable_wobbling()
                        else:
                            self._robot.disable_wobbling()
                        wobbling_active = wants_wobbling
                    except Exception as exc:
                        if not wants_wobbling:
                            raise ReachyHardwareError("Reachy Mini could not stop speech motion") from exc
                        self._robot.disable_wobbling()
                        log.warning("Reachy speech motion is unavailable: %s", exc)
                        self.wobbling_available = False
                # A command clip keeps the queue waiting, as every other command does.
                commanded = self._clip is not None and self._clip.request is not None
                request, self._deferred = self._deferred, None
                if request is None and not commanded:
                    try:
                        request = self._commands.get_nowait()
                    except queue.Empty:
                        pass
                if request is not None or commanded:
                    with self._state_lock:
                        if request is not None:
                            self._command_active = True
                        self._speaker_pending = self._speaker_look = self._speaker_turn = None
                        self._sentences.cancel()
                        self._sentence_active = False
                        self._sentence_terminal = None
                if request is not None and (self._clip is not None or self._generated is not None):
                    # A command starts from the pose last written, so showing clips fade out first.
                    for clip in (self._clip, self._generated):
                        if clip is not None:
                            clip.timeline.stop(now)
                    request, self._deferred = None, request
                if request is not None:
                    self._still_since = None
                    self._wave_gaze.invalidate()
                    if self._face_follower is not None:
                        self._update_face_tracking(state, commanded=True)
                    # Commands own absolute poses, including the gaze we were showing.
                    rest = self._pose_vector(*self._rest) + gaze + gesture_offset
                    self._rest = (self._head(**dict(zip(POSE_LIMITS, rest[:6]))),
                                  self._rest[1], self._rest[2])
                    gaze.fill(0)
                    gesture_offset.fill(0)
                    try:
                        self._check_command(request)
                        result = {"ok": True, "payload": self._execute_motion(request)}
                    except Exception as exc:
                        result = {"ok": False, "error": str(exc)}
                    finally:
                        with self._state_lock:
                            self._command_active = False
                            self._sentences.cancel()
                            self._sentence_active = False
                            self._sentence_terminal = None
                            self._gesture_name = None
                            self._gesture_epoch = None
                        if self._clip is not None and self._clip.request is request:
                            self._clip.result = result
                        else:
                            request.result = result
                            request.done.set()
                        previous_update = time.monotonic()
                        self._stop.wait(_MOTION_PERIOD_S)
                else:
                    if ((state == "sleeping" and previous_state != "sleeping")
                            or (self._watching and state in ("idle", "sleeping", "error"))):
                        self._clear_speaker()
                    self._update_speaker(now, state, occupied=(
                        self._clip is not None or self._generated is not None
                        or self._deferred is not None or commanded or gesture_active and gesture != "wave"))
                    self._update_face_tracking(state, commanded=(
                        commanded or self._deferred is not None or self._speaker_turn is not None))
                    if self._motion_epoch != self._sentences.epoch:
                        self._motion_epoch = self._sentences.epoch
                        self._clip_line = state_started
                        if self._clip is not None and self._clip.request is None and self._clip.during == "speaking":
                            self._clip.timeline.stop(now)
                    for event in self._sentences.due(now):
                        if commanded or self._deferred is not None:
                            continue
                        if event.kind == "start":
                            with self._state_lock:
                                if event.epoch != self._sentences.epoch:
                                    continue
                                expression, serial = event.value
                                move = event.move
                                if self._sentence_paused or self._state != "speaking":
                                    continue
                                self._reply_expression = expression
                                state = "speaking"
                                self._state_started = state_started = event.at
                                self._progress_started = None
                            elapsed = max(0.0, now - event.at)
                            if expression in HEAD_GESTURES:
                                with self._state_lock:
                                    if event.epoch != self._sentences.epoch:
                                        continue
                                    self._gesture_name = expression
                                    self._gesture_started = event.at
                                    self._gesture_epoch = gesture_epoch = event.epoch
                                gesture = expression
                                gesture_age = elapsed
                                gesture_active = gesture_age < GESTURE_DURATION_S
                            if move is None:
                                self._cue_clip(now, state, expression, state_started, replace=True, epoch=event.epoch)
                            else:
                                from reachy_muse.motion.layers import ClipTimeline
                                with self._state_lock:
                                    if (event.epoch == self._sentences.epoch and self._speaker_pending is None
                                            and self._speaker_turn is None
                                            and (self._clip is None or (self._clip.request is None and not self._clip.wake))):
                                        self._clip_source = self._last_written_vector.copy() if self._last_written_vector is not None else None
                                        self._clip = _Clip(move, ClipTimeline(event.at, move.duration),
                                                           SPOKEN_CLIP_INTENSITY, during="speaking")
                        else:
                            with self._state_lock:
                                if event.epoch != self._sentences.epoch:
                                    continue
                                terminal, serial = event.value
                                if serial == self._sentence_serial:
                                    self._sentence_active = False
                                paused = self._sentence_paused
                            if not paused:
                                self._sentences.set_accent(event.at, terminal, event.epoch, now)
                    if not commanded and self._deferred is None:
                        self._cue_clip(now, state, expression, state_started)
                    pose, antennas = self._state_pose(state, elapsed, expression, progress_age=progress_age)
                    followed = self._face_pose(pose)
                    tracking = followed is not pose
                    gaze_target = self._np.zeros(9)
                    if tracking:
                        coordinates = self._pose_parameters(followed)
                        gaze_target[4:6] = coordinates["pitch"], coordinates["yaw"]
                        coordinates["pitch"] = coordinates["yaw"] = 0.0
                        pose = self._head(**coordinates)
                    gesture_active = (gesture_active and not commanded
                                      and (gesture_epoch is None or gesture_epoch == self._sentences.epoch))
                    gesture_target = self._np.zeros(9)
                    if gesture_active:
                        if gesture == "wave":
                            from reachy_muse.motion.wave_back import wave_back_offset
                            gesture_target[:] = wave_back_offset(gesture_age)
                        else:
                            gesture_pose = self._gesture_pose(self._head(), gesture, gesture_age)
                            gesture_target[:6] = list(self._pose_parameters(gesture_pose).values())
                    if previous_state == "sleeping" and state != "sleeping" and self._clip is None:
                        self._wake_up(now, pose, antennas)
                    previous_state = state
                    self._cue_generated(now, state)
                    quick_cue = ((state == "listening" and expression == "happy" and elapsed < 1.0)
                                 or (gesture_active and state != "speaking"))
                    if quick_cue:
                        head_blend, antenna_blend = 0.18, 0.3
                    elif state == "thinking":
                        head_blend, antenna_blend = 0.12, 0.3
                    elif state == "speaking":
                        head_blend, antenna_blend = 0.16, 0.22
                    else:
                        head_blend, antenna_blend = 0.08, 0.08
                    dt = now - previous_update
                    previous_update = now
                    head_blend = _blend_factor(head_blend, dt)
                    antenna_blend = _blend_factor(antenna_blend, dt)
                    gaze += (gaze_target - gaze) * head_blend
                    if gesture == "wave":
                        gesture_offset[:] = gesture_target
                    else:
                        gesture_offset += (gesture_target - gesture_offset) * head_blend
                    rest_pose, rest_antennas, rest_body_yaw = self._rest
                    body_yaw = math.radians(self._speaker_look.body) if self._speaker_look is not None else 0.0
                    self._rest = (self._interpolate_pose(rest_pose, pose, head_blend),
                                  rest_antennas * (1 - antenna_blend) + antennas * antenna_blend,
                                  rest_body_yaw + (body_yaw - rest_body_yaw) * _blend_factor(0.08, dt))
                    if self._speaker_turn is not None:
                        direction = self._speaker_turn.at(now)
                        coordinates = self._pose_parameters(self._rest[0])
                        # Wake yaw uses the same additive gaze layer as face tracking.
                        gaze[5] = direction.head
                        coordinates["yaw"] = 0.0
                        self._rest = (self._head(**coordinates), self._rest[1], math.radians(direction.body))
                        if now >= self._speaker_turn.started + SPEAKER_TURN_S:
                            self._speaker_turn = None
                    resting = self._pose_vector(*self._rest) + self._layers.breathing_offset(
                        now, active=state in ("idle", "listening") and not gesture_active
                        and not (self._watching or self._wave_watching or self._wave_pending is not None))
                    accenting = progress_age is not None and 0 <= progress_age < PROGRESS_ACCENT_S
                    generated = self._show_generated(now, resting, yielding=gesture_active or accenting)
                    composed = self._show_clip(now, generated)
                    self._last_written_vector = self._np.asarray(composed).copy()
                    # Clips and their replacement sources contain expression offsets only.
                    # Add gaze once, after crossfades, so a full-strength clip cannot erase it.
                    written = (composed + gaze + gesture_offset
                               + self._layers.sway_offset(now, active=state == "speaking")
                               + self._np.asarray(self._sentences.offset(now)))
                    resting_camera = ((state in ("idle", "listening", "sleeping") or (state == "error" and self._watching))
                                      and not gesture_active and not commanded
                                      and self._clip is None and self._generated is None and self._deferred is None)
                    # Invalidate before writing a moving target, even between camera samples.
                    self._update_wave_gaze(now, written, eligible=resting_camera)
                    self._write_vector(written)
                    settled = (resting_camera
                               and previous_written is not None
                               and self._np.abs(written - previous_written)[CAMERA_AXES].max()
                               < CAMERA_STILL_STEP)
                    if not settled:
                        self._still_since = None
                    elif self._still_since is None:
                        self._still_since = now
                    previous_written = written
                    self._stop.wait(_MOTION_PERIOD_S)
        except Exception as exc:
            self._failure = exc
            log.exception("Reachy Mini motion worker failed")
        finally:
            self._started.set()
            self._still_since = None
            self._wave_gaze.invalidate()
            unfinished = [self._deferred, self._clip and self._clip.request]
            self._clip = self._deferred = self._generated = self._pending_motion = None
            while True:
                try:
                    unfinished.append(self._commands.get_nowait())
                except queue.Empty:
                    break
            for pending in filter(None, unfinished):
                pending.result = {"ok": False, "error": "Reachy Mini stopped"}
                pending.done.set()
            try:
                if wobbling_active:
                    self._robot.disable_wobbling()
            except Exception as exc:
                self._shutdown_failure = exc
                log.exception("Could not stop Reachy Mini speech motion")
            finally:
                self.wobbling_available = False
            try:
                if (self._sleep_on_close and self._last_pose is not None
                        and self._initial_motor_mode == "disabled"):
                    if self.antenna_mode == "both":
                        self._robot.goto_sleep()
                    elif self._frozen_antennas is not None:
                        antennas = self._np.deg2rad([-10, 10])
                        for index in self._inactive_antennas:
                            antennas[index] = self._frozen_antennas[index]
                        self._robot.goto_target(head=self._head(pitch=10, z=-10),
                                                antennas=antennas, body_yaw=0.0, duration=1.0)
                    self._robot.disable_motors()
                elif self._initial_motor_mode == "gravity_compensation":
                    self._robot.enable_gravity_compensation()
            except Exception:
                log.exception("Could not park Reachy Mini")

    def _wake_up(self, now: float, pose, antennas) -> None:
        """Lift the head and unfold the antennas from wherever the resting pose is, then hand back to it.

        The travel starts at the resting pose, so it shows at once; it fades out into breathing.
        """
        from reachy_muse.motion.layers import ClipTimeline
        turn = self._speaker_turn
        body_yaw = math.radians(turn.target.body) if turn is not None else 0.0
        travel = _Travel(self._pose_vector(*self._rest), self._pose_vector(pose, antennas, body_yaw),
                         WAKE_UP_S, self._head, SPEAKER_TURN_S if turn is not None else None)
        self._clip = _Clip(travel, ClipTimeline(now, WAKE_UP_S, fade_in_s=0.0), 1.0, wake=turn is not None)

    def _gesture_pose(self, pose, name: str, age: float):
        if not 0 <= age < GESTURE_DURATION_S:
            return pose
        coordinates = self._pose_parameters(pose)
        if name == "tilt":
            coordinates["roll"] += 10 * math.sin(math.pi * age / GESTURE_DURATION_S) ** 2
        else:
            wave = 0.0
            for (start, first), (end, last) in zip(_GESTURE_KEYFRAMES, _GESTURE_KEYFRAMES[1:]):
                if age <= end:
                    wave = first + (last - first) * _ease((age - start) / (end - start))
                    break
            coordinates["pitch" if name == "nod" else "yaw"] += (8 if name == "nod" else 10) * wave
        for axis, limit in (("yaw", 20), ("pitch", 12), ("roll", 20)):
            coordinates[axis] = max(-limit, min(limit, coordinates[axis]))
        return self._head(**coordinates)

    def _state_pose(self, state: str, now: float, expression: str | None = None,
                    *, progress_age: float | None = None):
        if state == "listening":
            coordinates, antennas = EXPRESSION_POSES.get(expression, EXPRESSION_POSES["listening"])
            if expression == "happy":
                greeting = _ease(now / .2) * (1 - _ease((now - .8) / .6))
                wiggle = math.sin(math.pi * min(1, max(0, now))) * math.sin(4 * math.pi * now)
                coordinates = {"pitch": -3, "z": 3 + 4 * greeting,
                               "roll": -8 * greeting, "yaw": 7 * wiggle}
                antennas = [-12 - 23 * greeting + 18 * wiggle,
                            12 + 23 * greeting - 18 * wiggle]
            return self._head(**coordinates), self._np.deg2rad(antennas)
        if state == "thinking":
            beat = max(0, now) % 3.6
            if beat < 0.45:
                energy = (1 - math.cos(math.pi * beat / 0.45)) / 2
            elif beat < 1.65:
                energy = 1.0
            elif beat < 2.25:
                energy = (1 + math.cos(math.pi * (beat - 1.65) / 0.6)) / 2
            else:
                energy = 0.0
            amplitude = 30 * energy
            flourish = 0.0
            if progress_age is not None and 0 <= progress_age < PROGRESS_ACCENT_S:
                accent = math.sin(math.pi * progress_age / PROGRESS_ACCENT_S) ** 2
                amplitude += (45 - amplitude) * accent
                flourish = 5 * accent * math.sin(progress_age * 2 * math.pi * 3.2)
            right = amplitude * math.sin(now * 2 * math.pi * 1.6) + flourish
            left = amplitude * math.sin((now - .14) * 2 * math.pi * 1.6) + flourish
            # A long wait gets a brief glance, not a continuous head loop.
            glance = 0.0
            direction = 1
            if now >= 9:
                glance_age = (now - 9) % 30
                direction = -1 if int((now - 9) / 30) % 2 else 1
                if glance_age < 1.8:
                    glance = math.sin(math.pi * glance_age / 1.8) ** 2
            lean = 10 if expression == "curious" else 0
            pose = self._head(pitch=-3 - lean / 2, z=4 + 2 * glance,
                              yaw=direction * 6 * glance, roll=lean + direction * 6 * glance)
            return pose, self._np.deg2rad([-10 + right, 10 - left])
        if state == "speaking":
            coordinates, antennas = EXPRESSION_POSES.get(expression, EXPRESSION_POSES["neutral"])
            strength = _ease(now / .35)
            strength *= 1 - .35 * _ease((now - .8) / .8)
            coordinates = {key: value * strength for key, value in coordinates.items()}
            neutral = self._np.asarray([-10, 10], dtype=float)
            antennas = neutral + (self._np.asarray(antennas) - neutral) * strength
            return self._head(**coordinates), self._np.deg2rad(antennas)
        if state == "error":
            return self._head(**self._watched({"pitch": 8, "z": -3})), self._np.deg2rad([-35, 35])
        if state == "sleeping":
            return self._head(**self._watched({"pitch": 15, "z": -10})), self._np.deg2rad([-80, 80])
        return self._head(**self._watched({})), self._np.deg2rad([-10, 10])

    def _check_command(self, request: _Command) -> None:
        if self._stop.is_set() or request.cancelled.is_set():
            raise RuntimeError("Reachy Mini command cancelled")
        if time.monotonic() >= request.deadline:
            raise RuntimeError("Reachy Mini command timed out")

    def _travel(self, request: _Command, pose, antennas, body_yaw: float, duration: float) -> None:
        start = time.monotonic()
        with self._state_lock:
            self._sentences.cancel(fade=False)
        self._layers.suspend(start)
        start_pose = self._last_pose.copy()
        start_antennas = self._last_antennas.copy()
        start_body = self._last_body_yaw
        while True:
            self._check_command(request)
            progress = min(1.0, (time.monotonic() - start) / duration)
            eased = _ease(progress)
            self._write(self._interpolate_pose(start_pose, pose, eased),
                        start_antennas * (1 - eased) + antennas * eased,
                        start_body * (1 - eased) + body_yaw * eased)
            self._rest = (self._last_pose, self._last_antennas, self._last_body_yaw)
            if progress >= 1:
                return
            self._stop.wait(0.025)

    def _hold(self, request: _Command, duration: float) -> None:
        end = time.monotonic() + duration
        while time.monotonic() < end:
            self._check_command(request)
            self._stop.wait(min(0.025, max(0, end - time.monotonic())))

    def _pose_parameters(self, pose) -> dict:
        rotation = pose[:3, :3]
        pitch = math.asin(max(-1, min(1, -float(rotation[2, 0]))))
        return dict(x=float(pose[0, 3]) * 1000, y=float(pose[1, 3]) * 1000,
                    z=float(pose[2, 3]) * 1000,
                    roll=math.degrees(math.atan2(rotation[2, 1], rotation[2, 2])),
                    pitch=math.degrees(pitch),
                    yaw=math.degrees(math.atan2(rotation[1, 0], rotation[0, 0])))

    def _execute_motion(self, request: _Command) -> dict:
        params = request.params
        if request.name == "reachy.move":
            coordinates = self._pose_parameters(self._last_pose)
            coordinates.update({key: params[key] for key in coordinates if key in params})
            for key, value in coordinates.items():
                low, high = POSE_LIMITS[key]
                coordinates[key] = max(low, min(high, value))
            antennas = self._last_antennas.copy()
            for index, key in enumerate(("right_antenna", "left_antenna")):
                if key in params:
                    antennas[index] = math.radians(params[key])
            body_yaw = math.radians(params["body_yaw"]) if "body_yaw" in params else self._last_body_yaw
            self._travel(request, self._head(**coordinates), antennas, body_yaw, params["duration"])
            self._hold(request, params["hold"])
            return {"coordinates": coordinates, "duration": params["duration"]}
        expression = params["name"]
        if expression in BUILTIN_EXPRESSIONS:
            self._builtin_expression(request)
            return {"name": expression, "source": "builtin", "duration": params["duration"]}
        self._load_moves(request)
        self._check_command(request)
        if expression not in self._moves.list_moves():
            raise ValueError(f"Unknown recorded expression {expression!r}; available: {', '.join(self._moves.list_moves())}")
        move = self._moves.get(expression)
        duration = float(move.duration)
        if not math.isfinite(duration) or not 0 < duration <= 15:
            raise ValueError("Recorded expressions must last at most 15 seconds")
        from reachy_muse.motion.layers import ClipTimeline
        self._clip = _Clip(move, ClipTimeline(time.monotonic(), duration), params["intensity"], request)
        return {"name": expression, "source": "recorded", "dataset": EMOTIONS_DATASET,
                "duration": duration, "intensity": params["intensity"], "bounded": True}

    def _start_loading_moves(self) -> dict:
        """Load the emotion library on a background thread, unless a load is running or succeeded."""
        loading = self._moves_load
        if loading is not None and not (loading["done"].is_set() and loading["error"] is not None):
            return loading
        loading = {"done": threading.Event(), "value": None, "error": None}
        self._moves_load = loading
        factory = self._moves_factory

        def load() -> None:
            try:
                nonlocal factory
                if factory is None:
                    from reachy_mini.motion.recorded_move import RecordedMoves
                    factory = RecordedMoves
                loading["value"] = factory(EMOTIONS_DATASET)
            except Exception as exc:
                loading["error"] = exc
                log.warning("Official Reachy emotion library is unavailable: %s; "
                            "spoken expressions use static poses", exc)
            finally:
                loading["done"].set()

        threading.Thread(target=load, name="muse-reachy-emotions", daemon=True).start()
        return loading

    def _ready_moves(self):
        """The emotion library if it has loaded, without waiting for it."""
        loading = self._moves_load
        if (self._moves is None and loading is not None and loading["done"].is_set()
                and loading["error"] is None):
            self._moves = loading["value"]
        return self._moves

    def _load_moves(self, request: _Command) -> None:
        if self._moves is not None:
            return
        loading = self._start_loading_moves()
        while not loading["done"].wait(0.025):
            self._check_command(request)
        self._check_command(request)
        if loading["error"] is not None:
            raise RuntimeError(f"Official Reachy emotion library is unavailable: {loading['error']}")
        self._moves = loading["value"]

    def _recorded_pose(self, values, intensity: float):
        """A recorded frame scaled by intensity and clamped to POSE_LIMITS, as a pose vector."""
        from reachy_muse.motion.layers import clamp_to_limits
        head, antennas, body_yaw = values
        pose = self._pose_vector(self._last_pose if head is None else head,
                                 self._last_antennas if antennas is None else antennas,
                                 self._last_body_yaw if body_yaw is None else float(body_yaw))
        return clamp_to_limits(pose * intensity)

    def _builtin_expression(self, request: _Command) -> None:
        params = request.params
        intensity = params["intensity"]
        duration = params["duration"]
        if params["name"] in ("nod", "shake"):
            axis = "pitch" if params["name"] == "nod" else "yaw"
            for angle in (-9, 9, -6, 0):
                self._travel(request, self._head(**{axis: angle * intensity}),
                             self._np.deg2rad([-10, 10]), 0, duration / 4)
        else:
            coordinates, antennas = EXPRESSION_POSES[params["name"]]
            target = self._head(**{key: value * intensity for key, value in coordinates.items()})
            self._travel(request, target, self._np.deg2rad(antennas) * intensity, 0, duration / 2)
            self._hold(request, duration / 4)
            self._travel(request, self._head(), self._np.deg2rad([-10, 10]), 0, duration / 4)
