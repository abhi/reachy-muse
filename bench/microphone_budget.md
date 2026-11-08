# Microphone priority

The robot report on main `3c198c1` measured roughly 0.26 wake-audio coverage with
idle face following and wave wake enabled, and 0.99–1.01 after those two features
were disabled. This change was developed offline. No robot or service was used.

## Code trace

`VoiceConversation._capture_microphone` formerly submitted each SDK read to the
asyncio default executor, awaited its result on the event loop, then submitted
the next read. That was independent of keyword inference as an asyncio task,
but it was not an independent recorder. Executor congestion or event-loop delays
left the SDK undrained. A 250 ms stall caused a continuity reset; smaller repeated
stalls could lose coverage without crossing that threshold.

`muse-face-detector` and `muse-wave` are dedicated Python threads. They share the
camera lock, CPU, memory bandwidth and GIL with capture and motion. Neither calls
inference on the microphone thread. YuNet and YOLOX both already set OpenCV to one
thread. YOLOX uses OpenCV DNN, not ONNX Runtime. Wake inference already uses one
thread. Adding another ONNX thread setting would not fix this path.

The cached Reachy SDK camera path pulls a GStreamer sample with a 20 ms timeout,
then copies BGR bytes with `extract_dup`. The audio path has a separate appsink.
Camera reads do not hold the audio lock. Motion remains on one 40 Hz worker;
SDK head/joint feedback reads cached values, and target writes go to the daemon.
No model or camera is reconstructed when the gaze log changes. The transitions
reflect face movement and the 300 ms settling requirement, not a busy wait.

One concrete budget defect was in `WaveWatcher._reset`: motion discarded the
person-inference timestamp along with image evidence. Thus the advertised rate
limit did not survive gaze resets. The new inference deadline survives resets
and accounts for measured inference cost. Face work also leaves at least four
times its measured cost idle, with no catch-up iterations.

There was no feedback from audio coverage to either worker. Both could continue
working indefinitely while capture lost most audio. Coverage now throttles vision
below 0.95, suspends on another bad window, and restores one level after 20 seconds
at or above 0.98. Capture and wake-consumer coverage are measured separately over
two-second windows. Stale capture blocks new camera work. User speech, replies,
turn processing and wake verification also suspend new work. An already-running
native operation finishes; there is no unsafe thread cancellation.

Capture now has a continuous dedicated reader. Its event-loop handoff is bounded
by the capture buffer's capacity, coalesces notifications, and reports overflow
as a discontinuity. Epoch checks still discard pre-invocation and in-flight audio.
Wake thresholds, candidate verification and echo rules are unchanged.

## Measurements

Run with the project's Python and SDK dependencies:

```
PYTHONPATH=src:. python -m bench.microphone_budget --seconds 20 --executor-load --reader-revision 3c198c1
PYTHONPATH=src:. python -m bench.microphone_budget --seconds 20 --executor-load
PYTHONPATH=src:. python -m bench.microphone_budget --seconds 20 --native-models
```

The benchmark runs the actual face and wave workers on synthetic frames. Camera
reads cost 20 ms, face work costs 50 ms, and person work costs 320 ms. Model stand-ins
execute Python arithmetic to include GIL competition. Audio arrives in 20 ms chunks
through a simulated one-chunk device queue. The executor-load case adds repeated
60 ms jobs to a one-worker default executor. This is a controlled starvation test,
not a claim about the robot's executor size or its exact workload.

Measured on the development Mac:

| Reader, constrained executor | Coverage | p99 read interval | Maximum interval | Continuity gaps |
| --- | ---: | ---: | ---: | ---: |
| `3c198c1` capture loop | 0.3099 | 80.01 ms | 109.01 ms | 0 |
| Dedicated reader | 0.9988 | 28.13 ms | 43.33 ms | 0 |

The old-reader run uses the new worker budgets for a controlled reader comparison.
It throttled then suspended vision and still could not drain the microphone. The
new-reader run kept vision active: 40 face inferences, three person inferences and
60 camera reads in 20 seconds.

With `--native-models` and OpenCV installed, the stand-ins repeat the bundled
YuNet and YOLOX calls for the same cost targets. That run measured 1.0000 coverage,
24.67 ms p99, 31.05 ms maximum, and no continuity gaps, with 40 face and three
person work batches in 20 seconds.

A separate local GStreamer 1080p buffer test measured `extract_dup` at 0.234 ms
median and 1.216 ms maximum; mapping measured 0.144 ms median and 0.528 ms maximum.
Neither caused a long GIL stall on this machine. Those measurements do not rule
out different Pi bindings, camera conversion, native scheduling or daemon load.
The precise native cause of the reported Pi stalls remains unmeasured. This patch
removes the demonstrated scheduling dependency and adds protection against both
capture and inference throughput loss; it cannot certify Pi coverage offline.

After adding wave-back and quiet rearming, the same 20-second runs measured
0.9979 coverage under constrained-executor load and 0.9999 with native models.
Both had zero continuity gaps. The p99/maximum reader intervals were
29.88/39.40 ms and 29.10/29.97 ms respectively. Each run made 88 camera reads,
40 face batches and two person batches. Quiet rearming adds only motion-gate
reads, without repeated person inference. These final checks overlapped the
remaining local test workers; they are not isolated host benchmarks.

The requested full pytest command passed with 3,542 tests and seven skips in
404.33 seconds. A separate OpenCV-enabled run passed all 129 vision, motion and
attention tests, including the still-mode wave-back and self-motion checks.
