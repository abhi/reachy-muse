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

With `--face-follow-idle`, the face worker stays active while wave wake is enabled.
The motion loop checks both commanded and observed head poses at 40 Hz. Wave
frames are eligible after angular speed stays below 1 degree/second for 300 ms;
translation must also stay below 0.8 mm/second. Body turns, commands, gestures,
clips and conversation motion prevent admission. A motion-loop sample older than
100 ms is not eligible. Sleep still disables face following but permits waves
after the sleep pose settles. Arrival watching retains priority over both workers.
Idle breathing remains suspended while wave watching is enabled.

Each moving tick changes a gaze generation. The wave worker checks it before and
after capture, after person inference, and when consuming a cue. Movement between
camera samples therefore discards partial waves, person confirmations and pending
cues even if the head has already settled again. The final conversation claim
also checks the generation after draining camera work; breathing stays suspended
through that handoff. The 15-second cooldown survives
these resets. Content-free transitions read `Wave gaze: follow`,
`Wave gaze: settled`, and `Wave gaze: wave watching`.

For eligible frames, camera calibration and the observed head rotation project the
image into the first frame's orientation. The motion gate, gesture detector and
person inference all use that image, so cached person boxes keep the same
coordinates during small residual rotations. Images are reduced to at most 416
pixels wide before projection, matching person inference. A cue's position maps
back to native camera pixels for wake attention. Torso and broad-scene rejection
remain active; the rotation projection does not compensate translation, parallax
or lens distortion. No additional image history is retained.

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
It also reruns the region detector with simulated head rotations and compensation.
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
| Balanced region trajectory, static camera | 40 | 0 | 560 | 0 | 100% | 100% | 0.160 | 0.219 |
| Balanced region trajectory, compensated head motion | 40 | 0 | 560 | 0 | 100% | 100% | 0.298 | 0.378 |
| Sparse Lucas-Kanade candidate, static camera | 39 | 0 | 560 | 1 | 100% | 97.5% | 0.157 | 0.266 |

The head-motion run applies yaw `0.8 sin(0.7t)` degrees and pitch
`0.4 sin(0.5t)` degrees to every frame. Combined angular speed remains below
0.61 degree/second, within the settled bound. The same 600 clips and oracle boxes
are used without changing gesture thresholds. Precision and recall both differ
from the static baseline by 0 percentage points. Compensation cost is included;
synthetic camera-image generation is excluded. Both static and moving-camera
worker schedules detect 40/40 waves with zero or 320 ms person-inference delay.
Unit and integration tests separately exercise faster motion, a wave entirely
during motion, a new wave after settling, motion during capture/inference and
between samples, arrival priority, and waking from sleep using synthetic pixels.

Most negative frames exit before tracking, so aggregate timings understate active
tracking cost. Wave-frame p95 was 0.206 ms for static regions, 0.398 ms with head
compensation and 0.285 ms for flow; timings vary with host load.
These costs are small beside person inference.
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
YOLOX runs at most once per 1.25 seconds within a burst, or five times the last
inference cost if longer. This deadline survives gaze resets. A cue requires two person
observations. After a cue, the 15-second cooldown survives disable/re-enable and
reconnects. Cues expire after 0.75 seconds. Nothing is queued for later wake.

The measured YOLOX CPU median is 20.431 ms, p95 21.650 ms, on synthetic input.
Following the explicitly approximate 10-15x CM4 multiplier in `bench/arrival.md`,
inference is estimated at 204-325 ms. This is not a robot measurement; camera
decode is excluded. At 1080p, the gate measured 0.821 ms median/0.880 ms p95 and
the gesture stage, including another downsample, 0.792/0.879 ms. Reducing and
projecting a 1080p input measured 0.408/0.476 ms. Linear resize matches YOLOX's
existing interpolation and avoids an expensive area resize at this ratio.
At the upper 15x estimate, projected idle sampling costs about 4.1% of one core.
At most three person inferences per five-second burst/rest cycle cost roughly
12-20% of one core. Using the full-resolution gate/gesture costs as an upper
estimate after projection adds about 16% averaged over those cycles. That total
is about 36% of one core, or 9% of the four-core CM4, excluding face inference.
Combined face/wave worker throughput remains unmeasured on the robot.
No wave inference starts while a conversation is active.
Disabling drains one already-started inference before opening a call, which can
add approximately 0.3 seconds to wake admission on the CM4.

The benchmark also drives the actual worker iteration with a virtual clock,
oracle boxes and a 320 ms inference delay. All 40 final waves are detected both
with and without that delay. For the seed-zero three-second wave beginning at
2.0 s, the cue occurs at 4.150 s without delay and 4.415 s with delay. The delayed
case reads 19 frames and runs person inference twice over eight seconds. A still
room reads 16 frames and runs no inference. The curtain fixture runs inference
twice, emits no cue, then rests. These scheduling checks do not simulate
camera decoding, operating-system contention or real person detection errors.

The three-second bursts and two-second rests trade missed waves for bounded CPU.
An operator may need to repeat a wave during a rest or while morning watching
owns the camera. Hardware throughput and real-camera accuracy remain unmeasured
because this work did not access the robot.
