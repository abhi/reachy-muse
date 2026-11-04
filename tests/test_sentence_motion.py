import numpy as np
import pytest

from reachy_muse.contract.expressions import Expression
from reachy_muse.hardware import ReachyController
from reachy_muse.motion import MotionStyle
from reachy_muse.motion.layers import MotionLayers
from reachy_muse.motion.sentence import SentenceMotion
from reachy_muse.motion.sentence_timeline import SentenceEvent, SentenceTimeline


class Speaker:
    def __init__(self):
        self.audio = self
        self.fail = False

    def push_audio_sample(self, audio):
        if self.fail:
            raise RuntimeError('push failed')

    def clear_player(self):
        pass


def controller():
    hardware = ReachyController()
    hardware._np = np
    hardware._output_sample_rate = 100
    hardware._layers = MotionLayers(MotionStyle(sway='sdk'))
    hardware._robot = type('Robot', (), {'media': Speaker()})()
    hardware._require_running = lambda: None
    return hardware


def test_accepted_audio_sets_each_sentence_start_and_tail(monkeypatch):
    monkeypatch.setattr('reachy_muse.hardware.time.monotonic', lambda: 10.0)
    hardware = controller()
    hardware.play_audio(np.ones(100), sentence=SentenceMotion(Expression.HAPPY, 'question'))
    hardware.finish_sentence()
    hardware.play_audio(np.ones(50), sentence=SentenceMotion(Expression.HAPPY, 'exclamation'))
    hardware.finish_sentence()
    events = hardware._sentences.due(20)
    assert [(e.kind, round(e.at, 2), e.value) for e in events] == [
        ('start', 10.06, ('happy', 1)), ('end', 11.06, ('question', 1)),
        ('start', 11.06, ('happy', 2)), ('end', 11.56, ('exclamation', 2))]


def test_empty_failed_and_cleared_audio_never_cue(monkeypatch):
    monkeypatch.setattr('reachy_muse.hardware.time.monotonic', lambda: 10.0)
    hardware = controller()
    plan = SentenceMotion(Expression.HAPPY)
    hardware.play_audio(np.array([]), sentence=plan)
    hardware._robot.media.fail = True
    with pytest.raises(Exception):
        hardware.play_audio(np.ones(10), sentence=plan)
    assert hardware._sentences.due(20) == []
    hardware._robot.media.fail = False
    hardware.play_audio(np.ones(10), sentence=plan)
    hardware.finish_sentence()
    hardware.clear_audio()
    assert hardware._sentences.due(20) == []


def test_accent_is_bounded_and_retargeted_without_jump():
    timeline = SentenceTimeline()
    timeline.set_accent(0, 'question', 0, 0)
    before = timeline.offset(.25)
    assert before[3] == 4 and before[6:8] == [-8, 8]
    timeline.set_accent(.25, 'exclamation', 0, .25)
    assert timeline.offset(.25) == before
    assert max(abs(value) for value in timeline.offset(.35)) <= 10
    before_cancel = timeline.offset(.35)
    timeline.cancel()
    assert timeline.offset(.36) == before_cancel
    assert timeline.offset(.57) == [0] * 9


def test_stale_events_are_invalidated():
    timeline = SentenceTimeline()
    timeline.offer(SentenceEvent(3, 0, 'start', ('happy', 1)))
    assert timeline.due(1) == []
    timeline.cancel()
    assert timeline.due(4) == []


def test_audio_onsets_drive_controller_and_clear_cannot_restart_clip():
    import time
    from test_hardware import FakeRobot, start_controller, move_library

    robot = FakeRobot()
    hardware = start_controller(robot, moves_factory=move_library())
    try:
        time.sleep(.2)
        sentence = SentenceMotion(Expression.HAPPY, 'question')
        hardware.play_audio(np.ones(1600), sentence=sentence)
        hardware.finish_sentence()
        time.sleep(.2)
        assert hardware._state == 'speaking'
        assert hardware._reply_expression == 'happy'
        assert hardware._clip is not None
        first = hardware._clip
        hardware.play_audio(np.ones(1600), sentence=sentence)
        hardware.finish_sentence()
        time.sleep(.2)
        assert hardware._clip is not first
        hardware.clear_audio()
        time.sleep(.08)
        assert hardware._clip is None or hardware._clip.timeline.end < time.monotonic()
        time.sleep(.5)
        assert hardware._clip is None
    finally:
        hardware.close()


def test_preempted_sentence_cannot_schedule_end_in_new_epoch(monkeypatch):
    monkeypatch.setattr('reachy_muse.hardware.time.monotonic', lambda: 10.0)
    hardware = controller()
    hardware.play_audio(np.ones(100), sentence=SentenceMotion(Expression.HAPPY, 'question'))
    hardware._sentences.cancel()
    hardware.finish_sentence()
    assert hardware._sentences.due(20) == []


def test_pause_and_resume_keep_expression_without_replaying_clip():
    import time
    from test_hardware import FakeRobot, move_library, start_controller

    requests = []
    robot = FakeRobot()
    hardware = start_controller(robot, moves_factory=move_library(requests))
    try:
        time.sleep(.2)
        hardware.play_audio(np.ones(16000 * 2), sentence=SentenceMotion(Expression.HAPPY, 'question'))
        time.sleep(.2)
        assert requests == ['success1']
        hardware.set_state('listening')
        time.sleep(.1)
        hardware.set_state('speaking')
        time.sleep(.2)
        assert hardware._reply_expression == 'happy'
        assert requests == ['success1']
    finally:
        hardware.close()


def test_sentence_motion_preserves_inactive_antenna():
    import time
    from test_hardware import FakeRobot, OfflineLibrary, start_controller

    robot = FakeRobot()
    hardware = start_controller(robot, antenna_mode='left', moves_factory=OfflineLibrary)
    try:
        hardware.play_audio(np.ones(1600), sentence=SentenceMotion(Expression.CURIOUS, 'question'))
        hardware.finish_sentence()
        time.sleep(.5)
        assert robot.frames
        np.testing.assert_allclose([frame[2][0] for frame in robot.frames], np.deg2rad(-10))
    finally:
        hardware.close()


def test_command_preemption_discards_sentence_tail():
    import time
    from test_hardware import FakeRobot, OfflineLibrary, start_controller

    hardware = start_controller(FakeRobot(), moves_factory=OfflineLibrary)
    try:
        hardware.play_audio(np.ones(16000), sentence=SentenceMotion(Expression.NEUTRAL, 'question'))
        time.sleep(.15)
        old_epoch = hardware._sentences.epoch
        assert hardware.run_command('reachy.move', {'yaw': 5, 'duration': .2, 'hold': 0})['ok']
        assert hardware._sentences.epoch > old_epoch
        hardware.finish_sentence()
        time.sleep(1.0)
        assert hardware._sentences.accent is None
        assert hardware._sentences.due(time.monotonic() + 10) == []
    finally:
        hardware.close()


def test_command_completion_does_not_restore_an_active_sentence_accent():
    import time
    from test_hardware import FakeRobot, OfflineLibrary, start_controller

    robot = FakeRobot()
    hardware = start_controller(robot, moves_factory=OfflineLibrary,
                                style=MotionStyle(sway='sdk', breathing=False))
    try:
        hardware.play_audio(np.ones(160), sentence=SentenceMotion(Expression.NEUTRAL, 'question'))
        hardware.finish_sentence()
        deadline = time.monotonic() + 1
        while hardware._pose_parameters(robot.pose)['roll'] < 3.5:
            assert time.monotonic() < deadline
            time.sleep(.002)
        assert hardware.run_command('reachy.move', {
            'roll': 0, 'yaw': 20, 'right_antenna': -10, 'left_antenna': 10,
            'duration': .2, 'hold': 0})['ok']
        completed = len(robot.frames)
        time.sleep(.15)
        frames = robot.frames[completed:]
        assert len(frames) >= 3
        np.testing.assert_allclose([hardware._pose_parameters(head)['roll']
                                    for _, head, _, _ in frames], 0, atol=.01)
        np.testing.assert_allclose([np.degrees(antennas) - [-10, 10]
                                    for _, _, antennas, _ in frames], 0, atol=.01)
    finally:
        hardware.close()


def test_sentence_started_during_command_cannot_play_after_completion():
    import threading
    import time
    from test_hardware import FakeRobot, move_library, start_controller

    executing = threading.Event()
    release = threading.Event()

    class BlockingRobot(FakeRobot):
        def set_target(self, head, antennas, body_yaw):
            super().set_target(head, antennas, body_yaw)
            if head[1, 0] > 0 and not executing.is_set():
                executing.set()
                assert release.wait(1)

    robot = BlockingRobot()
    requested = []
    hardware = start_controller(robot, moves_factory=move_library(requested),
                                style=MotionStyle(sway='sdk', breathing=False))
    result = []
    worker = threading.Thread(target=lambda: result.append(hardware.run_command(
        'reachy.move', {'yaw': 20, 'duration': .6, 'hold': 0})))
    try:
        worker.start()
        assert executing.wait(1)
        hardware.play_audio(np.ones(160), sentence=SentenceMotion(Expression.HAPPY, 'question'))
        hardware.finish_sentence()
        release.set()
        worker.join(2)
        assert not worker.is_alive()
        assert result[0]['ok'], result
        time.sleep(.15)
        completed = next(index for index, (_, head, _, _) in enumerate(robot.frames)
                         if hardware._pose_parameters(head)['yaw'] == 20)
        frames = robot.frames[completed:]
        assert len(frames) >= 4
        yaws = [hardware._pose_parameters(head)['yaw'] for _, head, _, _ in frames]
        intervals = np.diff([at for at, _, _, _ in frames])
        assert np.max(np.abs(np.diff(yaws)) / intervals) < 160, yaws
        assert requested == []
        assert hardware._sentences.offset(time.monotonic()) == [0] * 9
    finally:
        release.set()
        worker.join(2)
        hardware.close()


def test_continuing_sentences_cannot_keep_a_command_waiting_for_clip_fades():
    import threading
    import time
    from test_hardware import FakeRobot, move_library, start_controller

    robot = FakeRobot()
    hardware = start_controller(robot, moves_factory=move_library())
    stopped = threading.Event()

    def speak():
        while not stopped.is_set():
            hardware.play_audio(np.ones(640), sentence=SentenceMotion(Expression.HAPPY, "exclamation"))
            hardware.finish_sentence()
            stopped.wait(.04)

    speaker = threading.Thread(target=speak)
    try:
        time.sleep(.2)
        speaker.start()
        time.sleep(.15)
        result = hardware.run_command("reachy.move", {"yaw": 5, "duration": .2, "hold": 0}, timeout_ms=1500)
        assert result["ok"] is True, result
        assert any(abs(hardware._pose_parameters(head)["yaw"] - 5) < .1
                   for _, head, _, _ in robot.frames)
    finally:
        stopped.set()
        speaker.join(2)
        hardware.close()


def test_combined_sentences_sway_and_crossfades_stay_within_motion_limits():
    from bench.expressive_motion_tick import run

    _, peak, step = run(True)
    assert np.all(peak <= [15, 15, 20, 20, 20, 35, 90, 90, 35])
    assert peak[3] > 5 and peak[6] > 15
    assert np.all(step[:6] < [2, 2, 2, 4, 4, 4])
    assert np.all(step[6:] < [3, 3, 1])


def test_delayed_motion_tick_cannot_restore_speaking_after_audio_drains():
    import threading
    import time
    from test_hardware import FakeRobot, OfflineLibrary, start_controller

    blocked, release = threading.Event(), threading.Event()

    class SlowRobot(FakeRobot):
        stall = False

        def set_target(self, **kwargs):
            super().set_target(**kwargs)
            if self.stall:
                blocked.set()
                release.wait(2)

    robot = SlowRobot()
    hardware = start_controller(robot, moves_factory=OfflineLibrary)
    try:
        robot.stall = True
        assert blocked.wait(1)
        hardware.play_audio(np.ones(160), sentence=SentenceMotion(Expression.CURIOUS, "question"))
        hardware.finish_sentence()
        time.sleep(.1)
        hardware.set_state("listening")
        robot.stall = False
        release.set()
        time.sleep(.1)
        assert hardware._state == "listening"
        assert hardware._sentences.offset(time.monotonic())[3] > 0
    finally:
        release.set()
        hardware.close()
