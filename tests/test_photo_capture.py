"""One photo through Reachy's camera reader: a still head, a frame that is never kept, and a JPEG that fits."""

import gc
import math
import threading
import time
from types import SimpleNamespace
import weakref

import pytest

np = pytest.importorskip("numpy")

from reachy_muse.hardware import PHOTO_FRAME_S, PHOTO_SETTLE_S, PHOTO_STILL_S, ReachyController
from reachy_muse.motion import MotionStyle
from reachy_muse.photo import JPEG_MAGIC, PHOTO_MAX_BYTES, fit_jpeg, opencv_jpeg
from test_hardware import FakeRobot, start_controller


class Encoder:
    """A stand-in JPEG whose size depends only on its width and quality."""

    def __init__(self, sizes=(), default=1000):
        self.sizes = dict(sizes)
        self.default = default
        self.calls = []

    def __call__(self, frame, width, height, quality):
        self.calls.append((width, height, quality))
        return JPEG_MAGIC + bytes(self.sizes.get((width, quality), self.default) - len(JPEG_MAGIC))


def frame(height=720, width=1280):
    return np.zeros((height, width, 3), dtype=np.uint8)


def test_the_first_size_and_quality_that_fits_is_sent():
    encoder = Encoder({(1280, 80): 400_000, (1280, 70): 200_000, (1280, 60): PHOTO_MAX_BYTES + 1,
                       (960, 70): 160_000, (960, 55): PHOTO_MAX_BYTES})
    jpeg, width, height, quality = fit_jpeg(frame(), encoder, deadline=math.inf)
    assert (len(jpeg), width, height, quality) == (153_600, 960, 540, 55)
    assert encoder.calls == [(1280, 720, 80), (1280, 720, 70), (1280, 720, 60), (960, 540, 70), (960, 540, 55)]


def test_a_photo_that_never_fits_tries_every_size_and_quality_once():
    encoder = Encoder(default=PHOTO_MAX_BYTES + 1)
    assert fit_jpeg(frame(), encoder, deadline=math.inf) is None
    assert encoder.calls == [(1280, 720, 80), (1280, 720, 70), (1280, 720, 60), (960, 540, 70), (960, 540, 55),
                             (640, 360, 60)]


def test_a_small_camera_frame_is_never_enlarged():
    encoder = Encoder(default=PHOTO_MAX_BYTES + 1)
    assert fit_jpeg(frame(480, 640), encoder, deadline=math.inf) is None
    assert encoder.calls == [(640, 480, 80), (640, 480, 70), (640, 480, 60), (640, 480, 55)]


def test_encoding_stops_once_the_deadline_passes():
    with pytest.raises(TimeoutError):
        fit_jpeg(frame(), Encoder(), deadline=time.monotonic())

    class Slow(Encoder):
        def __call__(self, *args):
            time.sleep(.06)
            return super().__call__(*args)

    slow = Slow(default=PHOTO_MAX_BYTES + 1)
    with pytest.raises(TimeoutError):
        fit_jpeg(frame(), slow, deadline=time.monotonic() + .05)
    assert slow.calls == [(1280, 720, 80)]


def test_an_encoder_that_makes_no_jpeg_is_refused():
    with pytest.raises(ValueError):
        fit_jpeg(frame(), lambda *args: b"\x89PNG\r\n", deadline=math.inf)


def test_opencv_makes_a_jpeg_of_the_size_asked_for():
    cv2 = pytest.importorskip("cv2")
    scene = np.zeros((720, 1280, 3), dtype=np.uint8)
    scene[:, :, 0] = np.linspace(0, 255, 1280, dtype=np.uint8)
    cv2.rectangle(scene, (400, 200), (880, 520), (40, 180, 220), -1)
    cv2.putText(scene, "K7Q2", (480, 400), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 8)
    jpeg, width, height, quality = fit_jpeg(scene, opencv_jpeg, deadline=math.inf)
    assert jpeg.startswith(JPEG_MAGIC) and len(jpeg) <= PHOTO_MAX_BYTES
    assert (width, height, quality) == (1280, 720, 80)
    smaller = opencv_jpeg(scene, 640, 360, 60)
    assert cv2.imdecode(np.frombuffer(smaller, np.uint8), cv2.IMREAD_COLOR).shape == (360, 640, 3)


def camera(*frames, encoder=None):
    """A still controller whose camera returns ``frames`` in turn: None for no frame, an exception to raise."""
    frames = list(frames)
    reads = []

    def get_frame():
        reads.append(time.monotonic())
        item = frames.pop(0) if frames else None
        if isinstance(item, Exception):
            raise item
        return item

    controller = ReachyController(enable_motion=False, photo_encoder=encoder or Encoder())
    controller._robot = SimpleNamespace(media=SimpleNamespace(get_frame=get_frame))
    controller.vision_budget = SimpleNamespace(delay=lambda: math.inf)
    return controller, reads


def test_a_photo_reads_one_frame_while_background_vision_is_paused_and_keeps_none():
    image = frame()
    kept = weakref.ref(image)
    controller, reads = camera(image)
    del image
    assert controller._read_camera_frame() is None
    photo = controller.capture_photo()
    gc.collect()
    assert (photo.failure, photo.width, photo.height, photo.quality, len(photo.jpeg)) == (None, 1280, 720, 80, 1000)
    assert photo.capture_ms < 100 and photo.encode_ms < 100
    assert len(reads) == 1
    assert kept() is None


@pytest.mark.parametrize("frames, failure, reads", [
    ((), "no_frame", "many"),
    ((None, None, RuntimeError("camera unplugged")), "camera_failed", 3),
])
def test_a_camera_without_a_frame_fails_within_its_limit(frames, failure, reads):
    controller, read_at = camera(*frames)
    started = time.monotonic()
    photo = controller.capture_photo()
    assert (photo.failure, photo.jpeg) == (failure, None)
    assert time.monotonic() - started < PHOTO_FRAME_S + .3
    assert len(read_at) > 10 if reads == "many" else len(read_at) == reads


def test_a_camera_held_by_another_reader_is_busy_after_its_limit():
    controller, reads = camera(frame())
    with controller._camera_lock:
        started = time.monotonic()
        photo = controller.capture_photo()
        waited = time.monotonic() - started
    assert (photo.failure, reads) == ("camera_busy", [])
    assert PHOTO_FRAME_S - .05 <= waited < PHOTO_FRAME_S + .3


def test_a_stopping_robot_takes_no_photo():
    controller, reads = camera(frame())
    controller._stop.set()
    assert (controller.capture_photo().failure, reads) == ("camera_failed", [])


def without_opencv(frame, width, height, quality):
    raise ImportError("No module named 'cv2'")


@pytest.mark.parametrize("encoder, failure", [
    (Encoder(default=PHOTO_MAX_BYTES + 1), "too_large"),
    (without_opencv, "camera_failed"),
])
def test_a_photo_that_cannot_be_encoded_to_fit_is_not_sent(encoder, failure):
    controller, reads = camera(frame(), encoder=encoder)
    photo = controller.capture_photo()
    assert (photo.failure, photo.jpeg, len(reads)) == (failure, None, 1)


def test_one_photo_at_a_time():
    reading, release = threading.Event(), threading.Event()

    def slow_frame():
        reading.set()
        release.wait(2)
        return frame()

    controller, _ = camera()
    controller._robot.media.get_frame = slow_frame
    results = []
    first = threading.Thread(target=lambda: results.append(controller.capture_photo()))
    first.start()
    try:
        assert reading.wait(2)
        assert controller.capture_photo().failure == "camera_busy"
    finally:
        release.set()
        first.join(2)
    assert results[0].failure is None


def moving(controller):
    """A looping generated motion that sways the head forever."""
    poses = np.zeros((50, 9))
    poses[:, 5] = 10 * np.sin(np.linspace(0, 2 * math.pi, 50, endpoint=False))
    poses[:, 6:8] = -10, 10
    controller.set_state("thinking")
    controller.play_motion(poses, 25, during={"thinking"}, loop=True)


def camera_poses(writes):
    """The distinct camera poses among the robot's written targets: head pose and body yaw, not antennas."""
    return {tuple(np.round(np.r_[pose[:3, :].ravel(), body_yaw], 9)) for _, pose, _, body_yaw in writes}


def written_since(robot, since):
    return [write for write in list(robot.frames) if write[0] >= since]


def test_a_thinking_head_is_held_still_at_once_for_the_photo_and_then_moves_on():
    robot = FakeRobot()
    seen = []

    def get_frame():
        seen.append((time.monotonic(), len(robot.frames)))
        return frame()

    robot.media.get_frame = get_frame
    controller = start_controller(robot, photo_encoder=Encoder(), style=MotionStyle(breathing=False))
    try:
        moving(controller)
        time.sleep(1)
        assert len(camera_poses(written_since(robot, time.monotonic() - PHOTO_STILL_S))) > 1
        started = time.monotonic()
        photo = controller.capture_photo()
        read_at, written = seen[0]
        before_read = robot.frames[written - 3:written]
        time.sleep(.6)
        after = camera_poses(written_since(robot, time.monotonic() - .3))
    finally:
        controller.close()
    assert photo.failure is None
    assert read_at - started < PHOTO_SETTLE_S / 2
    assert len(camera_poses(before_read)) == 1
    assert len(after) > 1
    assert controller._failure is None


def test_the_curious_thinking_lean_a_turn_starts_is_held_rather_than_waited_out():
    robot = FakeRobot()
    seen = []

    def get_frame():
        seen.append((time.monotonic(), len(robot.frames)))
        return frame()

    robot.media.get_frame = get_frame
    controller = start_controller(robot, photo_encoder=Encoder(),
                                  style=MotionStyle(breathing=False, expression_clips=False))
    try:
        controller.set_state("listening")
        time.sleep(.5)
        controller.set_state("thinking", expression="curious")
        time.sleep(.05)
        started = time.monotonic()
        photo = controller.capture_photo()
        read_at, written = seen[0]
        before_read = robot.frames[written - 3:written]
    finally:
        controller.close()
    assert photo.failure is None
    assert read_at - started < PHOTO_SETTLE_S / 2
    assert len(camera_poses(before_read)) == 1


def test_a_photo_waits_until_the_head_has_turned_back_from_another_person():
    robot = FakeRobot()
    seen = []

    def get_frame():
        seen.append(round(math.degrees(controller.head_yaw)))
        return frame()

    robot.media.get_frame = get_frame
    controller = start_controller(robot, photo_encoder=Encoder(),
                                  style=MotionStyle(breathing=False, expression_clips=False))

    def until(condition):
        deadline = time.monotonic() + 3
        while not condition() and time.monotonic() < deadline:
            time.sleep(.01)

    try:
        controller.set_state("listening")
        assert controller.look_at_speaker(math.pi / 3)
        until(lambda: round(math.degrees(controller.head_yaw)) == 30 and controller._speaker_turn is None)
        assert controller.look_at_person(math.radians(-25), 0.0)
        until(lambda: round(math.degrees(controller.head_yaw)) == -25 and controller._speaker_turn is None)
        assert round(math.degrees(controller.head_yaw)) == -25
        controller.release_person_gaze(turn_back=True)
        photo = controller.capture_photo()
    finally:
        controller.close()
    assert photo.failure is None
    assert seen == [30]


def test_a_command_from_muse_waits_until_the_frame_is_read():
    robot = FakeRobot()
    reading, release = threading.Event(), threading.Event()

    def get_frame():
        reading.set()
        release.wait(2)
        return frame()

    robot.media.get_frame = get_frame
    controller = start_controller(robot, photo_encoder=Encoder(), style=MotionStyle(breathing=False))
    moved = []
    try:
        time.sleep(.3)
        photo = threading.Thread(target=controller.capture_photo)
        photo.start()
        assert reading.wait(2)
        command = threading.Thread(target=lambda: moved.append(
            controller.run_command("reachy.move", {"yaw": 20, "duration": .8, "hold": 0}, 5000)))
        command.start()
        time.sleep(.4)
        during_read = (list(moved), camera_poses(written_since(robot, time.monotonic() - .3)))
        released_at = time.monotonic()
        release.set()
        photo.join(2)
        command.join(3)
        time.sleep(.3)
    finally:
        release.set()
        controller.close()
    assert during_read[0] == [] and len(during_read[1]) == 1
    assert moved[0]["ok"] is True
    yaws = [controller._pose_parameters(pose)["yaw"] for _, pose, _, _ in written_since(robot, released_at)]
    assert round(max(yaws)) == 20
    assert max(abs(later - earlier) for earlier, later in zip(yaws, yaws[1:])) < 3


def test_a_photo_holds_the_pose_a_muse_move_left_when_the_move_ends_during_the_read():
    robot = FakeRobot()
    moved, read_yaws = threading.Event(), []

    def get_frame():
        moved.wait(3)
        time.sleep(.15)
        read_yaws.append(round(math.degrees(controller.head_yaw)))
        return frame()

    robot.media.get_frame = get_frame
    controller = start_controller(robot, photo_encoder=Encoder(), style=MotionStyle(breathing=False))
    results = []
    try:
        command = threading.Thread(target=lambda: (results.append(controller.run_command(
            "reachy.move", {"yaw": 20, "duration": .6, "hold": 1.0}, 5000)), moved.set()))
        command.start()
        time.sleep(.1)
        results.append(controller.capture_photo())
        command.join(3)
    finally:
        moved.set()
        controller.close()
    assert results[0]["ok"] is True and results[1].failure is None
    assert read_yaws == [20]
