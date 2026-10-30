"""Measure arrival detection cost and recall on a pedestrian video."""

from __future__ import annotations

import argparse
import statistics
import time

import cv2
import numpy as np

from reachy_muse.arrival import (
    MIN_BOX_HEIGHT, WATCH_HZ, ArrivalDetector, MotionGate, YoloxPersonDetector, default_person_model,
)


def read_video(path: str) -> tuple[list, float]:
    capture = cv2.VideoCapture(path)
    fps = capture.get(cv2.CAP_PROP_FPS) or 10.0
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    if not frames:
        raise SystemExit(f"no frames in {path}")
    return frames, fps


def timed(call, items) -> list[float]:
    result = []
    for item in items:
        started = time.perf_counter()
        call(item)
        result.append((time.perf_counter() - started) * 1000)
    return result


def summary(label: str, samples: list[float]) -> str:
    ordered = sorted(samples)
    p90 = ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))]
    return f"| {label} | {statistics.median(samples):.2f} | {p90:.2f} | {len(samples)} |"


def gate_rows(frames) -> list[str]:
    rows = []
    for name, size in (("1080p", (1920, 1080)), ("720p", (1280, 720))):
        resized = [cv2.resize(frame, size) for frame in frames[:200]]
        gate = MotionGate()
        clock = iter(range(10 ** 6))
        rows.append(summary(f"motion gate, {name} input", timed(lambda f: gate.update(f, next(clock) / WATCH_HZ),
                                                                resized)))
    return rows


def detector_rows(frames, detector) -> tuple[list[str], list[int], list[int]]:
    sample = frames[::4]
    counts, tall = [], []

    def detect(frame):
        boxes = detector.detect(frame)
        counts.append(len(boxes))
        tall.append(sum(box.height >= MIN_BOX_HEIGHT for box in boxes))

    rows = [summary("YOLOX-nano cv2.dnn, 768x576 frame (letterbox + forward + decode)",
                    timed(detect, sample))]
    big = [cv2.resize(frame, (1920, 1080)) for frame in sample[:60]]
    rows.append(summary("YOLOX-nano cv2.dnn, 1080p frame", timed(detector.detect, big)))
    return rows, counts, tall


def forward_rows(frames) -> list[str]:
    blobs = []
    for frame in frames[::8]:
        image = np.full((416, 416, 3), 114, dtype=np.uint8)
        scale = min(416 / frame.shape[1], 416 / frame.shape[0])
        size = (round(frame.shape[1] * scale), round(frame.shape[0] * scale))
        image[:size[1], :size[0]] = cv2.resize(frame, size)
        blobs.append(cv2.dnn.blobFromImage(image))
    net = cv2.dnn.readNetFromONNX(str(default_person_model()))

    def forward(blob):
        net.setInput(blob)
        net.forward()

    forward(blobs[0])
    rows = [summary("YOLOX-nano forward only, cv2.dnn", timed(forward, blobs))]
    try:
        import onnxruntime
    except ImportError:
        return rows
    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = onnxruntime.InferenceSession(str(default_person_model()), options,
                                           providers=["CPUExecutionProvider"])
    feed = session.get_inputs()[0].name
    session.run(None, {feed: blobs[0]})
    rows.append(summary(f"YOLOX-nano forward only, onnxruntime {onnxruntime.__version__} CPU",
                        timed(lambda blob: session.run(None, {feed: blob}), blobs)))
    return rows


class Counting:
    def __init__(self, detector):
        self.detector, self.calls = detector, 0

    def detect(self, frame):
        self.calls += 1
        return self.detector.detect(frame)


def first_arrival(frames, detector, step: float):
    decision = ArrivalDetector(detector)
    for index, frame in enumerate(frames):
        if decision.observe(frame, index * step):
            return index * step
    return None


def condition_rows(frames, fps, detector):
    stride = max(1, round(fps / WATCH_HZ))
    step = stride / fps
    still = [frames[0]] * round(60 / step)
    dark = (frames[0] * 0.01).astype(np.uint8)
    prefix = [dark] * round(2 / step)
    scenes = [("static, 60 s", still), ("walking", frames[::stride])]
    for exposure in (0.2, 0.5, 1.0):
        lit = (frames[0] * exposure).astype(np.uint8)
        walking = [(frame * exposure).astype(np.uint8) for frame in frames[::stride]]
        scenes.extend([(f"light on, static, exposure {exposure}", prefix + [lit] * len(still)),
                       (f"light on, walking, exposure {exposure}", prefix + walking)])
    for name, scene in scenes:
        counted = Counting(detector)
        arrived = first_arrival(scene, counted, step)
        result = "none" if arrived is None else f"{arrived:.2f} s"
        yield f"| {name} | {result} | {counted.calls} |"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("video")
    parser.add_argument("--conditions-only", action="store_true")
    args = parser.parse_args()
    cv2.setNumThreads(1)
    frames, fps = read_video(args.video)
    print(f"{len(frames)} frames, {frames[0].shape[1]}x{frames[0].shape[0]} at {fps:g} fps, "
          f"OpenCV {cv2.__version__}, numpy {np.__version__}")
    detector = YoloxPersonDetector(default_person_model())
    if args.conditions_only:
        print("| scene | arrival | detector runs |\n|---|---|---|")
        print("\n".join(condition_rows(frames, fps, detector)))
        return
    print("| step | median ms | p90 ms | samples |\n|---|---|---|---|")
    for row in gate_rows(frames):
        print(row)
    rows, counts, tall = detector_rows(frames, detector)
    for row in rows + forward_rows(frames):
        print(row)
    print(f"\npersons per frame on {len(counts)} frames: mean {statistics.mean(counts):.2f}, "
          f"min {min(counts)}, max {max(counts)}; at least {MIN_BOX_HEIGHT:.0%} of the frame tall: "
          f"mean {statistics.mean(tall):.2f}, min {min(tall)}")
    stride = max(1, round(fps / WATCH_HZ))
    watched = frames[::stride]
    step = stride / fps
    walkers = Counting(detector)
    arrived = first_arrival(watched, walkers, step)
    print(f"walkers, one frame in {stride} ({1 / step:.2f} fps): arrival at "
          f"{'never' if arrived is None else f'{arrived:.2f} s'} after {walkers.calls} detector runs")
    still = Counting(detector)
    static = [frames[0]] * round(60 * WATCH_HZ)
    arrived = first_arrival(static, still, 1 / WATCH_HZ)
    print(f"one frame repeated for 60 s: arrival {'never' if arrived is None else f'at {arrived:.2f} s'}, "
          f"{still.calls} detector runs")


if __name__ == "__main__":
    main()
