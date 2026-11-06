"""Offline direction mapping check. Run with PYTHONPATH=src python bench/speaker_direction.py."""

import math

from reachy_muse.speaker_direction import speaker_yaw


def main():
    cases = [
        (0, 35, 15), (30, 35, 15), (60, 30, 10), (80, 10, 0),
        (90, 0, 0), (100, -10, 0), (120, -30, -10), (150, -35, -15), (180, -35, -15),
    ]
    print("array_deg  world_head_deg  body_deg  neck_deg")
    for array, expected_head, expected_body in cases:
        target = speaker_yaw(math.radians(array))
        assert math.isclose(target.head, expected_head, abs_tol=1e-9), (array, target)
        assert math.isclose(target.body, expected_body, abs_tol=1e-9), (array, target)
        print(f"{array:9.0f} {target.head:15.1f} {target.body:9.1f} {target.head - target.body:9.1f}")
    print("9 mappings passed; head yaw is absolute, not head plus body.")


if __name__ == "__main__":
    main()
