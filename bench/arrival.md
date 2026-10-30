# Arrival detection

`bench/arrival.py` measures what it costs Reachy to watch for someone walking in, and how readily the person detector finds people.
Measured on 2026-10-10 on an Apple M3 Ultra, single threaded, with OpenCV 4.14.0, numpy 2.5.4 and onnxruntime 1.31.0.
The input is OpenCV's `vtest.avi` sample: 795 frames of pedestrians, 768x576 at 10 fps, filmed from above.
The video is not in the repository; pass its path.

    python bench/arrival.py path/to/vtest.avi

## Arrival reliability check, 2026-10-30

Run `python bench/arrival.py path/to/vtest.avi --conditions-only` to repeat these
checks with the packaged model. The input is the public
[OpenCV vtest sample](https://github.com/opencv/opencv/blob/4.x/samples/data/vtest.avi).
This comparison uses OpenCV 5.0.0 and numpy 2.5.3 on the local Mac, with `d3c15cf`
as the baseline. No robot camera frames were collected.

| Scene | Baseline arrival | Updated arrival |
| --- | --- | --- |
| Walking | 1.80 s | 1.80 s |
| Light on, walking, 20% exposure | 3.90 s | 3.60 s |
| Light on, walking, 50% exposure | 3.90 s | 3.60 s |
| Light on, walking, 100% exposure | 3.90 s | 3.60 s |
| One frame repeated for 60 s | None | None |
| Light on, static frame, 20%, 50%, or 100% exposure | None in all 3 | None in all 3 |

The light-on scenes begin with about 2 seconds at 1% exposure. Arrival times
include that prefix. The four negative scenes produce zero false arrivals in
both versions. This is a fixture check, not an estimate of false positives in a
real room. The video contains people throughout, so its walking results do not
measure recall against labeled arrivals.

The existing synthetic tests and added dim-light cases also reject all eight
negative scenes: a poster beside a moving curtain, light on beside a poster,
a camera turn, a single detection flicker, and four stationary person-box
scenes at different brightness levels. The synthetic light-on walker improves
from 4.0 to 3.33 seconds. A camera turn without the stillness reset remains an
intentional false-positive counterexample in the tests.

The confidence threshold remains 0.4. A light-on event now opens an 8-second
burst with a 0.25-second minimum detector interval, bounded by the 3 Hz camera
sampling rate. Dim frames in that burst receive at most 3x brightness gain.
The recorded static light-on scenes run the detector 27 times instead of 5
at the bench's 3.33 Hz input rate. At the watcher's 3 Hz rate, the synthetic
light-only fixture runs it 24 times. This increases short-lived CPU work after
a light switch. Robot throughput still needs an operator check.

## Results

| Step | Median ms | p90 ms | Samples |
|---|---|---|---|
| Motion gate, 1080p input | 1.26 | 1.70 | 200 |
| Motion gate, 720p input | 0.79 | 0.90 | 200 |
| YOLOX-nano with cv2.dnn, 768x576 frame (letterbox, forward, decode) | 25.24 | 25.92 | 199 |
| YOLOX-nano with cv2.dnn, 1080p frame | 25.75 | 26.35 | 60 |
| YOLOX-nano forward pass only, cv2.dnn | 25.08 | 25.60 | 100 |
| YOLOX-nano forward pass only, onnxruntime CPU | 20.10 | 20.60 | 100 |

The motion gate includes the gray conversion, the downscale to 160 pixels wide and the background update.
The detector cost does not depend on the camera resolution, because every frame is letterboxed to 416x416.
onnxruntime is about 20% faster than cv2.dnn, but it is not installed on the robot and cv2.dnn already runs the face tracker there.
So the robot uses cv2.dnn.

## Detection

The video has no ground truth labels, so this measures detections per frame, not true recall.
On 199 frames (every fourth frame), YOLOX-nano reports a mean of 6.77 people per frame, with a minimum of 3 and a maximum of 10.
A mean of 5.69 of them are at least 8% of the frame tall, the minimum size that counts toward an arrival.
The smaller ones are distant walkers in this overhead scene.
In a room, a person 5 m from the camera is about 35% of the frame tall.

## The whole decision

The bench runs `ArrivalDetector` with the real model over the same scenes.

| Scene | Result |
|---|---|
| vtest, every third frame (3.3 fps), walkers from the first frame | Arrival at 1.80 s after 2 detector runs |
| The first vtest frame repeated for 60 s at 3 fps | No arrival and no detector runs |

The 1.80 s comes from the 1 s settle at the start, then two detections 0.6 s apart.

## Estimate for Reachy Mini's CM4

This is an estimate, not a measurement on the robot.
It assumes the CM4's Cortex-A72 at 1.5 GHz is 10 to 15 times slower than one M3 Ultra performance core for this work.
That ratio combines a clock ratio of about 2.7 (1.5 GHz against about 4 GHz) with the A72's narrower NEON throughput and much lower memory bandwidth.
It also assumes `get_frame` returns an already decoded BGR frame, so its cost is not counted.

| Item | M3 Ultra, measured | CM4, estimated |
|---|---|---|
| Motion gate per frame, 1080p | 1.3 ms | 13 to 19 ms |
| YOLOX-nano detection | 25 ms | 250 to 380 ms |

The watcher reads 3 frames per second while it watches and Reachy is still.
Motion or a lighting change opens a 3 s burst.
In that burst the detector runs at most every 0.5 s, which at 3 fps means every second frame: about 5 runs per burst.

Average CPU over a 70-minute window, as a share of one of the CM4's four cores, at the upper estimate:

| Scenario | Gate | Detector | Average |
|---|---|---|---|
| Quiet room, 20 bursts in the window | 5.7% (3 x 19 ms per s) | 0.9% (20 x 5 x 380 ms over 4200 s) | about 7% of one core, 1.7% of the CM4 |
| Something moving in view the whole window, such as a curtain in a draught | 5.7% | 57% (1.5 runs per s x 380 ms) | about 63% of one core, 16% of the CM4 |

The watcher stops reading frames at the first arrival, so the briefing itself never pays these costs.
If the worst case shows up on the robot, the first lever is a longer `DETECT_PERIOD_S`. The second is a lower `WATCH_HZ`.
