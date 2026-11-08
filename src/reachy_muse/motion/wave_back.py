# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0 (the "License").

"""A short greeting added to the current gaze by the existing motion owner."""

import math

DURATION_S = 1.6


def wave_back_offset(age: float) -> tuple[float, ...]:
    if not 0 < age < DURATION_S:
        return (0.0,) * 9
    phase = age / DURATION_S
    envelope = math.sin(math.pi * phase) ** 2
    swing = 24 * envelope * math.sin(6 * math.pi * phase)
    bob = envelope * (1 - math.cos(6 * math.pi * phase)) / 2
    return (0, 0, 3 * bob, 5 * envelope, -3 * bob, 0, swing, swing, 0)
