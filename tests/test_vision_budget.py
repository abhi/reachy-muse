# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0 (the "License").

"""Audio loss reduces optional vision work, with slow recovery and no log spam."""

import math

from reachy_muse.vision_budget import VisionBudget


def test_coverage_throttles_then_suspends_and_restores_in_steps(caplog):
    budget = VisionBudget(clock=lambda: 0)
    budget.capture_active(True)
    budget.observe("capture", 0, now=0)
    for now, duration in [(2, .52), (4, .52), (6, 2), (24, 18), (26, 2), (46, 20)]:
        budget.observe("capture", duration, now=now)
        if now == 2:
            assert budget.delay() == 3
        if now == 4:
            assert budget.delay() == math.inf
        if now == 24:
            assert budget.delay() == math.inf
        if now == 26:
            assert budget.delay() == 3
    assert budget.delay() == 1
    assert [r.message for r in caplog.records] == [
        "Vision budget: throttled; capture audio coverage 0.260",
        "Vision budget: suspended; capture audio coverage 0.260",
        "Vision budget: resumed at reduced rate; capture audio coverage 1.000",
        "Vision budget: normal; capture audio coverage 1.000",
    ]


def test_marginal_coverage_breaks_recovery_and_threshold_is_strict():
    budget = VisionBudget(clock=lambda: 0)
    budget.capture_active(True)
    budget.observe("capture", 0, now=0)
    budget.observe("capture", 1.9, now=2)
    assert budget.delay() == 1
    budget.observe("capture", 1.89, now=4)
    assert budget.delay() == 3
    budget.observe("capture", 2, now=6)
    budget.observe("capture", 19.4, now=26)
    assert budget.delay() == 3
    budget.observe("capture", 2, now=28)
    budget.observe("capture", 18, now=46)
    assert budget.delay() == 3
    budget.observe("capture", 2, now=48)
    assert budget.delay() == 1


def test_wake_processing_loss_also_suspends_vision():
    budget = VisionBudget(clock=lambda: 0)
    budget.capture_active(True)
    budget.observe("wake", 0, now=0)
    budget.observe("wake", .52, now=2)
    assert budget.delay() == 3
    budget.observe("wake", .52, now=4)
    assert budget.delay() == math.inf


def test_audio_priority_and_stalled_capture_prevent_camera_work():
    now = [0.0]
    budget = VisionBudget(clock=lambda: now[0])
    budget.capture_active(True)
    budget.observe("capture", .02)
    budget.audio_busy(True)
    assert budget.delay() == math.inf
    budget.audio_busy(False)
    assert budget.delay() == 1
    now[0] = .51
    assert budget.delay() == math.inf
    budget.observe("capture", .02)
    assert budget.delay() == 1


def test_stopped_capture_clears_busy_and_coverage_until_a_fresh_start():
    now = [0.0]
    budget = VisionBudget(clock=lambda: now[0])
    budget.capture_active(True)
    budget.observe("capture", 0)
    for at in (2, 4):
        now[0] = at
        budget.observe("capture", .5)
    budget.audio_busy(True)
    assert budget.level == 2
    assert budget.delay() == math.inf
    budget.capture_active(False)
    now[0] = 60
    budget.audio_busy(True)
    budget.observe("capture", .02)
    assert budget.delay() == 1
    assert budget.level == 0
    budget.capture_active(True)
    budget.observe("capture", .02)
    now[0] = 62
    budget.observe("capture", 2)
    assert budget.delay() == 1
    now[0] += .51
    assert budget.delay() == math.inf


def test_persistent_loss_resumes_throttled_vision_after_one_minute(caplog):
    now = [0.0]
    budget = VisionBudget(clock=lambda: now[0])
    budget.capture_active(True)
    budget.observe("capture", 0)
    budget.observe("wake", 0)
    delays = []
    for at in range(2, 182, 2):
        now[0] = at
        budget.observe("capture", 2)
        budget.observe("wake", 1)
        if at in (4, 62, 64, 180):
            delays.append(budget.delay())
    assert delays == [math.inf, math.inf, 3.0, 3.0]
    assert [r.message for r in caplog.records] == [
        "Vision budget: throttled; wake audio coverage 0.500",
        "Vision budget: suspended; wake audio coverage 0.500",
        "Vision budget: resumed at reduced rate; audio loss persists after 60s suspended",
    ]
    for at in range(182, 206, 2):
        now[0] = at
        budget.observe("capture", 2)
        budget.observe("wake", 2)
    assert budget.delay() == 1.0
    now[0] = 206
    budget.observe("wake", 1)
    now[0] = 208
    budget.observe("wake", 1)
    assert budget.delay() == math.inf


def test_suspension_expires_even_without_new_coverage_windows(caplog):
    now = [0.0]
    budget = VisionBudget(clock=lambda: now[0])
    budget.capture_active(True)
    budget.observe("wake", 0)
    for at in (2, 4):
        now[0] = at
        budget.observe("wake", 1)
    assert budget.level == 2
    now[0] = 64
    budget.observe("capture", .02)
    assert budget.delay() == 3.0
    assert budget.level == 1
    assert caplog.text.count("resumed at reduced rate") == 1
