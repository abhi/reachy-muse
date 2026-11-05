# Offline motion library measurements

Measured on 2026-11-04 with synthetic, pose-bounded 25 FPS trajectories. This run
used the real library builder, snapshot reader, `animate` implementation, protocol
codec and companion WebSocket server on an ephemeral loopback port. It did not
load a planner, generator or speech model, or connect to a robot.

| Measurement | p50 | p95 | Maximum |
|---|---:|---:|---:|
| In-memory `animate` | 0.127 ms | 0.196 ms | 0.266 ms |
| Local WebSocket request and decoded response | 0.383 ms | 0.546 ms | 0.889 ms |

Each path made 1,000 requests across all 13 intent keys. These measurements are
below the 5 ms target. Network latency to a robot is outside this measurement.

The temporary library held 156 motions, 12 per intent, in 158 files including its
manifest and lock. File contents totaled **513,638 bytes**, about 502 KiB. Filesystem
allocation totaled **720,896 bytes**, 704 KiB. These are synthetic-library sizes.
Real trajectories, recipes and model fingerprint metadata will change the result.
The temporary library was removed after the run.

## Expected real build duration

[The existing generated-motion benchmark](generated_motion.md#time-to-motion)
measured cold request p50 times of 0.78 to 1.04 seconds per motion, with a maximum
of 1.10 seconds. At 156 motions that implies **122 to 162 seconds** of generation,
or about 172 seconds at the measured maximum. Its recorded startup took 9 to 14
seconds for the combined engines. Allow roughly **2 to 3 minutes**, plus local
weight hashing and startup overhead, with weights already prepared and the GPU
quiet. New expression prompts and low process priority may change that estimate.

This is an estimate from existing measurements. No real build time was measured
for this change. A complete resumed build hashes and validates files but does not
load MotionEngine's models.

## Reproduce without models

Use a Python environment with the project's test dependencies:

```sh
PYTHONPATH=src python bench/motion_library.py --requests 1000 --variants 12
```

The script writes only to a temporary directory and prints JSON. Its
`SyntheticEngine` cannot call the planner or generator.
