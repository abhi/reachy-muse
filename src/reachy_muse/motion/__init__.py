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

"""Expressive motion layers for Reachy Mini. Importing this package does not load numpy."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping

SWAY_MODES = ("animation", "sdk")


@dataclass(frozen=True)
class MotionStyle:
    """Which expressive layers the motion loop runs; ``from_env`` reads the operator's flags."""

    sway: str = "animation"
    breathing: bool = True
    expression_clips: bool = True

    def __post_init__(self) -> None:
        if self.sway not in SWAY_MODES:
            raise ValueError(f"REACHY_MUSE_SWAY must be animation or sdk, not {self.sway!r}")

    @classmethod
    def from_env(cls, environ: Mapping[str, str] = os.environ) -> MotionStyle:
        return cls(sway=environ.get("REACHY_MUSE_SWAY", "animation"),
                   breathing=_switch(environ, "REACHY_MUSE_BREATHING"),
                   expression_clips=_switch(environ, "REACHY_MUSE_EXPRESSION_CLIPS"))


def _switch(environ: Mapping[str, str], name: str) -> bool:
    value = environ.get(name, "1")
    if value not in ("0", "1"):
        raise ValueError(f"{name} must be 1 or 0, not {value!r}")
    return value == "1"
