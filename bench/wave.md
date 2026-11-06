# Wave wake

## Design and measured choice

`--wave-wake` adds a local visual invocation, with or without an acoustic wake
model. The conversation consumes `hardware.take_wave()` only while waiting for
invocation. It drains the wave worker, discards queued microphone audio, and uses
the existing `WakeHeard` cue and sleep-exit motion. Fresh speech enters the usual
conversation window without a wake-prefix requirement.

`WaveDetector.observe(frame, boxes, now) -> bool` owns image evidence.
`WaveWatcher.set_active(bool)` owns cadence, person inference, cooldown and
pending cues. Disabling waits for in-flight work and releases image history.
The hardware supplies one locked camera reader to wave, arrival, face following
and presence. Morning arrival watching has priority and suspends wave detection.
The conversation also suspends wave work during calls, output and pending turns.
No wave worker exists unless enabled. Images stay in RAM on the robot; diagnostics
contain frame/error/inference/cue counts and the maximum oscillation score.

Two independent design candidates proposed sparse optical flow and the trajectory
of a changing region. Both required a stable person and torso, raised lateral
motion, camera-motion rejection and explicit suspension before calls. The review
preferred the smaller region detector if measurements supported it. We retained
the flow candidate's stationary torso check and the region candidate's bounded
retry duty cycle. A new pose model was unnecessary given the measured comparison;
no pose-model performance claim is made.

The first region prototype accepted blinking patches. Requiring both covered and
uncovered pixels rejected that fixture. After that change, thresholds were frozen
before evaluating seeds 1000-1039. Seeds 0-5 were development fixtures; seeds
20-39 and 100-139 were exploratory and are excluded from final results. Limiting
the action to three seconds exposed clipping at the outer edge of the arm region.
The final region extends 45% of the person width outside the box on either side.

The selected detector downsizes to 160 pixels wide, tracks each raised side region
separately, excludes the head and lower arm, requires three horizontal reversals
over 1-3 seconds, and rejects torso movement, broad scene change, light jumps,
ambiguous people, box discontinuities and frame gaps over 0.6 seconds. Balanced
positive/negative image differences reject appearing/disappearing patches.
Shoulder height is approximated from the full person box. This deliberately favors
clear raised waves over distant, occluded or truncated people.

## Reproduce

Use an environment with the robot's numpy and OpenCV dependencies, then run:

```
PYTHONPATH=src python bench/wave.py --person-timing
```

This generates clips in memory, compares both detectors and times the bundled
YOLOX-nano using the OpenCV CPU backend with one thread and OpenCL disabled.
It does not use a camera, network service or GPU. No clips are saved.

## Offline results

Measured locally on an arm64 Mac on 2026-11-06, OpenCV 5.0.0 and numpy 2.5.3. These are
gesture-level results on synthetic 160x120 clips at 8 Hz, with oracle person
boxes. They measure motion logic, not YOLOX accuracy or field precision/recall.
The evaluation includes 40 positive clips, with three seconds of waving on either
side, and 560 negatives. Negatives cover
walking, rocking, pets, curtains, TV-like noise, pans, light changes, flashing
patches, no person, stillness, low hands, vertical movement and isolated sweeps.

| Detector | TP | FP | TN | FN | Precision | Recall | Median CPU ms/frame | p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Balanced region trajectory | 40 | 0 | 560 | 0 | 100% | 100% | 0.170 | 0.257 |
| Sparse Lucas-Kanade candidate | 39 | 0 | 560 | 1 | 100% | 97.5% | 0.169 | 0.270 |

Most negative frames exit before tracking, so aggregate timings understate active
tracking cost. Wave-frame p95 was 0.422 ms for regions and 0.299 ms for flow in
this run; timings vary with host load. Both costs are small beside person inference.
The region method has a smaller implementation and slightly higher fixture recall.
Neither these fixture counts nor zero false positives establish a field error rate.

A screen replaying the positive wave produces a cue. A monocular RGB camera
cannot reliably distinguish that replay from the same view of a real person.
Ordinary non-wave TV motion is rejected, but wave-on-screen rejection is not solved.
Keep screens showing people outside the camera view when enabling this feature.
Moving objects directly beside a person's raised-arm region are another residual
association risk. Real seated/standing waves and room negatives still need a
consented, labeled on-device evaluation before relying on this cue.

## CPU budget

Idle sampling averages 2 Hz with alternating 0.4/0.6-second intervals. A fixed
0.5-second interval completely missed a periodic fixture by sampling the same
phase repeatedly. Motion permits a three-second
8 Hz burst, followed by two seconds without reads, even under continuous motion.
YOLOX runs at most once per 1.25 seconds within a burst. A cue requires two person
observations. After a cue, the 15-second cooldown survives disable/re-enable and
reconnects. Cues expire after 0.75 seconds. Nothing is queued for later wake.

The measured YOLOX CPU median is 20.892 ms, p95 21.442 ms, on synthetic input.
Following the explicitly approximate 10-15x CM4 multiplier in `bench/arrival.md`,
inference is estimated at 209-322 ms. This is not a robot measurement; camera
decode is excluded. At 1080p, the gate measured 0.840 ms median/0.963 ms p95 and
the gesture stage, including another downsample, 0.898/1.005 ms. At the upper 15x
estimate, idle costs about 2.9% of one CM4 core. At most three inferences per
five-second burst/rest cycle cost roughly 13-19% of one core. Full-frame image
work adds at most about 14% averaged over those cycles. That conservative total
is about 34% of one core, or 8.5% of the four-core CM4. No inference starts while
a conversation is active.
Disabling drains one already-started inference before opening a call, which can
add approximately 0.3 seconds to wake admission on the CM4.

The benchmark also drives the actual worker iteration with a virtual clock,
oracle boxes and a 320 ms inference delay. All 40 final waves are detected both
with and without that delay. For the seed-zero three-second wave beginning at
2.0 s, the cue occurs at 4.150 s without delay and 4.165 s with delay. The delayed
case reads 17 frames and runs person inference twice over eight seconds. A still
room reads 16 frames and runs no inference. The curtain fixture runs inference
three times, emits no cue, then rests. These scheduling checks do not simulate
camera decoding, operating-system contention or real person detection errors.

The three-second bursts and two-second rests trade missed waves for bounded CPU.
An operator may need to repeat a wave during a rest or while morning watching
owns the camera. Hardware throughput and real-camera accuracy remain unmeasured
because this work did not access the robot.
