# Generated motion on the companion Mac

`reachy-companion serve --motion` loads [reachy-motion-generator](https://github.com/pham-tuan-binh/reachy-motion-generator)
(Apache-2.0, commit `96497d6`). A fine-tuned Qwen3.5 planner writes a short recipe for an intent such as
"nodding yes", a 21.8M-parameter flow-matching generator turns it into 25 Hz frames for 9 channels, and the
SDK's inverse kinematics pulls every frame back to a reachable pose. `MotionEngine` in
`src/reachy_muse/companion_server/motion.py` then clamps every frame into `POSE_LIMITS`.

This page records what that costs on the Mac Studio (M3 Ultra, 96 GB, macOS 26.4) and which configuration
the engine uses.

## Recommendation

Generated motion is too slow to start with a spoken line. Use it for moments that last longer than the
wait: a greeting, thinking while Muse works, a celebration, an apology.

- The fastest planner that nods and shakes on command is the 4B. It needs 0.73 to 0.88 s per intent
  at p50 and 0.9 to 1.1 s at p95. A narrator line takes about 0.5 s and its first audio about 0.15 s,
  so motion that waits for its planner arrives after the line has started, and often after it has ended.
- The 0.8B is fast enough at 4 bits (0.27 s p50) but its motion is wrong for the intents we need. For
  "shaking head no" it writes a pitch snap with no yaw, and for "nodding yes" a single dip. It passes
  7 to 12 of upstream's 16 probes, against 14 to 15 for the 4B.
- The engine uses the 4B planner at 8 bits on MLX, with the generator on torch MPS. At 8 bits it scores
  the same probes as bf16 (15 of 16) at 0.88 s p50 instead of 1.22 s, with a 6.9 GB peak footprint
  instead of 10.6 GB. The 4-bit 4B is 0.14 s faster and 2.3 GB smaller but drops one more probe and
  changes every recipe relative to bf16, so it is the fallback if memory gets tight.
- To use motion with a spoken line, start generating when the request arrives rather than when the line
  is ready. A line about Muse's answer is known seconds before it is spoken, so 0.9 s of planning fits
  in that wait.

## Where the time goes

The planner is more than 95% of the time. The generator takes 22 ms on MPS (190 ms on CPU) and reach projection
takes 2 ms. Planner time follows recipe length: 55 to 85 tokens of JSON per answer.

All rows below ran with the live companion server (launchd `com.abhi.reachy-companion`, 8.7 GB footprint)
resident. Each timed row is 8 intents × 3 repeats after one warm-up request, decoded greedily as upstream
serves it. Rows marked `contended` decode a second MLX model (Qwen3-4B-Instruct-2507-4bit, the narrator's
model) in a loop in another process during the run. Rows marked `formatted` use the same 8 intents in the
trained prompt shape ("nodding yes. You agree with what was just said."). Times are milliseconds.

| config | n | planner p50 / p95 | generator p50 / p95 | reach p50 / p95 | total p50 / p95 | tokens p50 | motion s p50 | peak GB | load s |
|---|---|---|---|---|---|---|---|---|---|
| hf-cpu-0.8b | 2 | 743045 / 768288 | 190 / 198 | 2 / 2 | 743238 / 768488 | 56 | 3.6 | 0.9 | 4 |
| hf-mps-0.8b | 24 | 1519 / 1699 | 23 / 50 | 2 / 4 | 1547 / 1738 | 84 | 4.6 | 3.7 | 4 |
| hf-mps-4b | 24 | 3450 / 3764 | 20 / 24 | 2 / 6 | 3473 / 3787 | 60 | 4.0 | 11.8 | 11 |
| mlx-bf16-0.8b | 24 | 399 / 497 | 22 / 58 | 2 / 3 | 424 / 526 | 78 | 4.6 | 3.7 | 3 |
| mlx-4bit-0.8b | 24 | 244 / 685 | 22 / 27 | 2 / 3 | 267 / 711 | 69 | 4.1 | 2.7 | 3 |
| mlx-4bit-0.8b-gencpu | 24 | 245 / 687 | 175 / 192 | 2 / 3 | 420 / 876 | 69 | 4.1 | 2.6 | 4 |
| mlx-4bit-0.8b-contended | 8 | 441 / 1018 | 38 / 63 | 2 / 4 | 479 / 1084 | 69 | 4.1 | 2.7 | 3 |
| mlx-bf16-4b | 24 | 1197 / 1419 | 22 / 26 | 2 / 6 | 1222 / 1442 | 58 | 4.0 | 10.6 | 4 |
| **mlx-8bit-4b** | 24 | 852 / 982 | 22 / 27 | 2 / 3 | 876 / 1089 | 58 | 4.0 | 6.9 | 4 |
| mlx-4bit-4b | 24 | 712 / 861 | 22 / 39 | 2 / 4 | 734 / 893 | 66 | 3.7 | 4.6 | 4 |
| mlx-4bit-4b-contended | 8 | 886 / 983 | 30 / 35 | 2 / 3 | 920 / 1014 | 66 | 3.7 | 4.9 | 3 |

- `hf` is upstream's transformers path (`pipeline.py --planner`), bf16. On MPS and CPU, transformers runs
  Qwen3.5's linear-attention layers through its pure-PyTorch fallback, which is why the 0.8B takes 1.5 s
  on MPS and 12 minutes per intent on CPU. The CPU row timed 2 intents and is there only to rule CPU out.
- `mlx` is the planner converted with `mlx_lm.convert` (mlx-lm 0.32.0 has `qwen3_5`; its `sanitize` drops
  the MTP head). The generator stays on torch.
- `peak GB` is the process's lifetime peak physical footprint (`proc_pid_rusage`), which counts Metal
  allocations. `load s` is the time to load cached weights. The first `--motion` start also downloads the
  4B bundle and converts it to 8 bits, which took under a minute here with the weights already on disk.
- The 0.8B p95 of 0.69 s is one intent ("shaking head no") whose first answer failed validation and took a
  second sample.

## Contention with the live server

The live server sat idle through almost every run: its CPU read 0.0 to 0.1% before and after each one.
It served the robot once, during the third repeat of `mlx-8bit-4b`: its footprint rose from 8.7 GB to
12 GB, and that repeat's planner times were 5 to 15% slower (965 ms to 1113 ms for "surprised by good
news").

The `contended` rows are the stress case: another process decoding the narrator's 4B model nonstop. The
4B planner slowed 24% at p50 (712 ms to 886 ms) and the generator 22 ms to 30 ms. The 0.8B planner
slowed 81%. A real narrator decodes in short bursts, so the cost during a conversation sits between the
two. The reverse, motion slowing the narrator, was not measured. Both share the GPU, so expect a similar
margin, and keep motion requests away from the moments when the narrator must answer within its 0.8 s
acknowledgement budget.

## Quality

| config | probes (16, greedy) | OOD core | skill | frames outside POSE_LIMITS | unreachable | clamped frames | nod pitch swings | shake yaw swings |
|---|---|---|---|---|---|---|---|---|
| hf-mps-0.8b | 0.62 | 0.62 | 0.62 | 0 | 0 | 297 / 947 | 3 | 0 |
| mlx-bf16-0.8b | 0.75 | 0.88 | 0.62 | 0 | 0 | 156 / 947 | 3 | 0 |
| mlx-4bit-0.8b | 0.44 | 0.38 | 0.50 | 0 | 0 | 351 / 865 | 3 | 4 |
| hf-mps-4b | 0.94 | 1.00 | 0.88 | 0 | 0 | 93 / 777 | 7 | 8 |
| mlx-bf16-4b | 0.94 | 1.00 | 0.88 | 0 | 0 | 93 / 789 | 7 | 8 |
| **mlx-8bit-4b** | 0.94 | 1.00 | 0.88 | 0 | 0 | 92 / 777 | 7 | 8 |
| mlx-4bit-4b | 0.88 | 0.88 | 0.88 | 0 | 0 | 143 / 795 | 6 | 5 |
| mlx-8bit-4b-formatted | | | | 0 | 0 | 118 / 810 | 2 | 7 |

- **Probes.** Upstream's 16 behavioral probes (`planner/distill/probes.py`) check each recipe physically:
  does the sneeze release move the head down, does "nodding yes" oscillate pitch at least twice. Upstream
  samples each prompt 12 times; this bench takes one greedy recipe, so one probe is 0.06 of the score.
  The 4B misses "yawning widely" in every precision. Upstream reports 0.91 for the 4B and 0.66 for the 0.8B.
- **Nod and shake.** These count moves of at least 5° of pitch ("nodding yes") or 8° of yaw ("shaking head
  no") between turning points of the final, clamped trajectory, so 4 is two full nods. The 4B nods 3 to 4
  times and shakes 2 to 4 times. The 0.8B dips once and, for "shaking head no", does not turn its head.
- **Limits.** No frame of any run leaves `POSE_LIMITS`, and every final frame passes the SDK's IK check.
  That is by construction, so the useful number is what the clamp cut. For the 4B at 8 bits, 92 of 777
  frames touch a bound, mostly the antennas at ±90° (upstream's ears droop to 165° for sad and sleepy) and
  pitch at ±20° (upstream allows ±25°). Sad intents lose some droop on this robot.
- **Prompt shape.** The trained shape is "word. one sentence of context.", but it did not help these
  intents. "nodding yes. You agree with what was just said." gave one dip; bare "nodding yes" gave a
  sustained nod. Phrase intents as the short gesture or feeling.
- **Fidelity.** The MLX bf16 4B wrote the same recipe as transformers bf16 for 6 of 8 intents and 12 of
  16 probes; the rest differ in a number or two, which is bf16 rounding in a different kernel. The 8-bit
  4B matched bf16 for 5 of 8 and 10 of 16. The 4-bit models matched none.

Every run's recipes, timings and quality statistics are in `bench/motion_runs/`. The quality records
include pitch/yaw swing counts and per-channel counts of frames at a limit, calculated from the rounded
trajectories. `bench/motion_table.py` reads these summaries; raw frame arrays are omitted.

New runs save model directory basenames or Hugging Face repo IDs. To keep raw trajectories locally,
pass `--raw-out bench/out/raw.json`; `bench/out/` is ignored by Git. To compact an older report, run
`python bench/motion_report.py path/to/report.json`.

## Reproduce

Install the extra in your own environment, then download and convert the planners:

```sh
uv venv -p 3.13 .venv-motion && VIRTUAL_ENV=.venv-motion uv pip install -e ".[companion-motion]"
hf download binhpham/reachy-mini-motion-planner-0.8b --revision 3561b10c2483c30307a2a2c4ab7d4a213206706d --local-dir $MOTION/models/0.8b
hf download binhpham/reachy-mini-motion-planner-4b --revision 1b60d8f16b85473ceb20e3b62e464780c532b519 --local-dir $MOTION/models/4b
mlx_lm.convert --hf-path $MOTION/models/4b/planner --mlx-path $MOTION/mlx/4b-8bit -q --q-bits 8 --q-group-size 64
```

Convert the other `mlx/{0.8b,4b}-{bf16,4bit}` models the same way (`--dtype bfloat16` without `-q` for
bf16). Then run every configuration and print the tables:

```sh
MOTION=$MOTION PY=.venv-motion/bin/python bench/motion_all.sh
python bench/motion_table.py
```

The full matrix takes about 30 minutes, most of it the CPU row. To check the engine itself, run
`pytest -m real tests/companion_server/test_motion.py`; it downloads and converts the 4B into
`~/Library/Application Support/reachy-companion/motion` on first use.
