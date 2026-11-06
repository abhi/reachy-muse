"""Offline wave comparison. Synthetic pixels/boxes; optional CPU-only YOLOX timing."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import platform
import time

import numpy as np

from reachy_muse.arrival import MotionGate, PersonBox, YoloxPersonDetector, default_person_model
from reachy_muse.wave_wake import WaveDetector, WaveWatcher

FPS = 8
BOX = PersonBox(55 / 160, 18 / 120, 58 / 160, 94 / 120, .95)


def clip(kind, seed=0, *, seconds=4, fps=FPS):
    """A fixed box is an oracle annotation, not a claim that YOLOX detects cartoons."""
    rng = np.random.default_rng(seed)
    background = rng.integers(20, 55, (120, 160), dtype=np.uint8)
    hand = rng.integers(150, 245, (10, 8), dtype=np.uint8)
    phase = seed * .7
    frequency = (1.0, 1.4, 1.8)[seed % 3]
    amplitude = (10, 13, 16)[seed % 3]
    for index in range(round(seconds * fps)):
        now = index / fps
        image = background.copy()
        image[45:109, 70:98] = 110
        image[22:44, 77:92] = 145
        motion_time = min(now, 3) if kind in ("wave", "replay") else now
        x = round(114 + amplitude * math.sin(motion_time * frequency * 2 * math.pi + phase))
        y = 28
        if kind == "vertical":
            x, y = 114, round(31 + 13 * math.sin(now * 8 + phase))
        elif kind == "low_hand":
            y = 76
        elif kind == "single_sweep":
            x = round(100 + min(28, now * 12))
        elif kind == "one_reversal":
            x = round(100 + 25 * math.sin(min(math.pi, now * math.pi / 3)))
        elif kind == "still":
            x = 114
        if kind in ("wave", "vertical", "low_hand", "single_sweep", "one_reversal", "still", "replay"):
            image[y:y + 10, x:x + 8] = hand
        elif kind == "walk":
            person = image[18:112, 55:113].copy()
            image = background.copy()
            left = round(35 + now * 15)
            width = min(58, 160 - left)
            if width > 0:
                image[18:112, left:left + width] = person[:, :width]
        elif kind == "rock":
            person = image[18:112, 55:113].copy()
            image = background.copy()
            left = 55 + round(9 * math.sin(now * 8))
            image[18:112, left:left + 58] = person
        elif kind == "pet":
            image[90:105, x:x + 12] = 200
        elif kind == "curtain":
            image[10:100, 8:30] = rng.integers(80, 200, (90, 22))
        elif kind == "tv":
            image[18:112, 52:142] = rng.integers(0, 255, (94, 90))
        elif kind == "pan":
            image = np.roll(image, round(10 * math.sin(now * 6)), axis=1)
        elif kind == "light":
            image = np.clip(image.astype(float) + 30 * math.sin(now * 8), 0, 255).astype(np.uint8)
        elif kind == "flashes":
            if index % 2:
                image[28:38, x:x + 8] = hand
        elif kind == "no_person":
            image = background.copy()
            image[28:38, x:x + 8] = hand
        boxes = [] if kind in ("no_person", "pet") else [BOX]
        if seed % 2:
            image = image[:, ::-1]
            boxes = [PersonBox(1 - b.x - b.width, b.y, b.width, b.height, b.score) for b in boxes]
        yield now, np.repeat(image[:, :, None], 3, axis=2), boxes


class FlowWave(WaveDetector):
    """Comparison candidate: sparse LK positions, with the same admission policy."""

    def _position(self, previous, gray, changed, bounds):
        import cv2

        if super()._position(previous, gray, changed, bounds) is None:
            return None
        x1, y1, x2, y2 = bounds
        before = previous[y1:y2, x1:x2].astype(np.uint8)
        after = gray[y1:y2, x1:x2].astype(np.uint8)
        points = cv2.goodFeaturesToTrack(before, 40, .03, 3, blockSize=3)
        if points is None:
            return None
        tracked, valid, error = cv2.calcOpticalFlowPyrLK(
            before, after, points, None, winSize=(15, 15), maxLevel=2)
        movement = (tracked - points).reshape(-1, 2)
        keep = ((valid.ravel() == 1) & (error.ravel() < 18)
                & (np.abs(movement[:, 0]) > .7)
                & (np.abs(movement[:, 0]) > 1.8 * np.abs(movement[:, 1])))
        if keep.sum() < 2:
            return None
        chosen = tracked.reshape(-1, 2)[keep]
        return float(np.median(chosen[:, 0]) + x1), float(np.median(chosen[:, 1]) + y1)


NEGATIVES = ("vertical", "low_hand", "single_sweep", "one_reversal", "still", "walk", "rock",
             "pet", "curtain", "tv", "pan", "light", "flashes", "no_person")


def evaluate(factory, seeds):
    counts = Counter(TP=0, FP=0, TN=0, FN=0)
    failures = Counter()
    costs = []
    positive_costs = []
    for kind in ("wave", *NEGATIVES):
        for seed in seeds:
            detector = factory()
            detected = False
            for now, frame, boxes in clip(kind, seed):
                started = time.process_time()
                detected |= detector.observe(frame, boxes, now)
                cost = (time.process_time() - started) * 1000
                costs.append(cost)
                if kind == "wave":
                    positive_costs.append(cost)
            key = ("TP" if detected else "FN") if kind == "wave" else ("FP" if detected else "TN")
            counts[key] += 1
            if key in ("FP", "FN"):
                failures[kind] += 1
    return {**counts, "precision": counts["TP"] / max(1, counts["TP"] + counts["FP"]),
            "recall": counts["TP"] / max(1, counts["TP"] + counts["FN"]),
            "cpu_ms_median": round(float(np.median(costs)), 3),
            "cpu_ms_p95": round(float(np.percentile(costs, 95)), 3),
            "wave_cpu_ms_p95": round(float(np.percentile(positive_costs, 95)), 3),
            "failures": dict(failures)}


def scheduled(kind, inference_s, seed=0, *, seconds=3):
    """Drive the actual worker iteration with a virtual camera and CPU delay."""
    gesture = list(clip(kind, seed, seconds=seconds))
    frames = [next(clip("still", seed))] * (2 * FPS) + gesture + [gesture[-1]] * (6 * FPS - len(gesture))
    now = [0.0]
    reads, runs, cues = [], [], []
    seen_boxes = []

    def read():
        reads.append(round(now[0], 3))
        _, frame, boxes = frames[min(len(frames) - 1, int(now[0] * FPS))]
        seen_boxes[:] = boxes
        return frame

    class People:
        def detect(self, frame):
            runs.append(round(now[0], 3))
            now[0] += inference_s
            return list(seen_boxes)

    watcher = WaveWatcher(read, lambda: True, clock=lambda: now[0])
    watcher.set_active(True)
    detector = People()
    while now[0] < 8:
        started = now[0]
        period = watcher._sample(detector)
        if watcher.take_wave():
            cues.append(round(now[0], 3))
        now[0] = max(now[0], started + period)
    watcher.close()
    return {"reads": len(reads), "person_runs_at": runs, "cues_at": cues}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--person-timing", action="store_true")
    args = parser.parse_args()
    import cv2
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    report = {"architecture": platform.machine(), "opencv": cv2.__version__, "numpy": np.__version__}
    for label, factory in (("difference", WaveDetector), ("flow", FlowWave)):
        report[label] = {"development": evaluate(factory, range(6)),
                         "held_out": evaluate(factory, range(1000, 1040))}
    detector = WaveDetector()
    report["wave_on_screen_counterexample"] = any(
        detector.observe(frame, boxes, now) for now, frame, boxes in clip("replay", 21))
    report["schedule"] = {f"{kind}_{delay}s": scheduled(kind, delay)
                          for kind in ("wave", "curtain", "still") for delay in (0, .32)}
    report["scheduled_wave_recall"] = {
        str(delay): sum(bool(scheduled("wave", delay, seed)["cues_at"]) for seed in range(1000, 1040)) / 40
        for delay in (0, .32)}
    full_frame = cv2.resize(next(clip("wave"))[1], (1920, 1080))
    gate, decision = MotionGate(), WaveDetector()
    report["1080p_cpu_ms"] = {}
    for name, observe in (("gate", lambda i: gate.update(full_frame, i / FPS)),
                          ("gesture", lambda i: decision.observe(full_frame, [BOX], i / FPS))):
        costs = []
        for index in range(50):
            started = time.process_time()
            observe(index)
            costs.append((time.process_time() - started) * 1000)
        report["1080p_cpu_ms"][name] = {"median": round(float(np.median(costs)), 3),
                                        "p95": round(float(np.percentile(costs, 95)), 3)}
    if args.person_timing:
        person = YoloxPersonDetector(default_person_model())
        person._net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        person._net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        frame = next(clip("wave"))[1]
        person.detect(frame)
        costs = []
        for _ in range(30):
            started = time.process_time()
            person.detect(frame)
            costs.append((time.process_time() - started) * 1000)
        report["person_cpu_ms"] = {"median": round(float(np.median(costs)), 3),
                                    "p95": round(float(np.percentile(costs, 95)), 3)}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
