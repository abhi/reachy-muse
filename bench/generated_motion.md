# Generated motion beside the speech models

`reachy-companion serve --motion` offers the `animate` op, and the robot plays the result for a greeting,
while Muse works, and at the end of a happy or sad reply. This page records what that costs on the Mac
Studio (M3 Ultra, 96 GB, macOS 26.4): how soon motion starts for each moment, how much memory the
companion holds, and whether motion slows hearing, speech, narration and classify.

`bench/motion.md` chose the models: the 4B planner at 8 bits on MLX and the generator on torch MPS.

## Summary

- **Prefetched motion starts at once.** Once the robot has connected, every moment has a motion ready within
  3.7 s, and a cue hands it to the motion thread in under 0.1 ms. The robot starts it on its next 25 ms motion
  tick and fades it in over 0.4 s.
- **A cold cue waits about a second.** With nothing ready, a request takes 0.78 to 1.04 s at p50 and at most
  1.1 s. That is fine for thinking, which Muse holds for 3 to 10 s, and for a greeting, which may wait 3 s.
  It is too slow for a closing that must be ready when speech ends, so closings rely on the prefetch and are
  skipped after 1 s.
- **Motion slowed every speech path until speech went first.** With two motion requests always outstanding,
  condensing a reply took 1.27 s at p50 instead of 0.66 s, and the final transcript's p95 rose from 0.64 s to
  3.6 s. The companion now pauses the motion planner before each token while a speech op runs (`SpeechFirst`
  in `engines.py`). Under the same load, every speech path then stays within a few milliseconds of running
  alone.
- **Memory.** With hearing, speech, the narrator and motion loaded, the companion's footprint is 15.5 GB,
  peaking at 17.3 GB during the run. Load takes 9 to 14 s with the weights on disk.

## Time to motion

From the moment the robot cues a motion to the moment it hands frames to its motion thread. "Cold" is one
`animate` round trip per intent over a fresh connection: the wait when no motion was ready. "Prefetched" is
the robot's `MomentDirector` with a motion already made, timed from `cue` (or, for a closing, from the end of
speech) to `play_motion`. 10 runs; cold runs ask every intent in the moment's table once per run.

| Moment | Cold p50 / p95 / max (s) | Prefetched p50 / p95 (ms) | Motion length p50 (s) |
|---|---|---|---|
| thinking | 0.78 / 0.94 / 1.10 | 0.04 / 0.05 | 4.0 |
| greeting | 1.04 / 1.06 / 1.06 | 0.04 / 0.09 | 2.5 |
| celebration | 0.95 / 0.95 / 0.95 | 0.04 / 0.05 | 2.8 |
| apology | 0.87 / 0.87 / 0.87 | 0.03 / 0.04 | 2.5 |

The robot then adds up to 25 ms for the next motion tick. A thinking cue fires at dispatch and never delays
it: the request reaches Muse first and the motion arrives on its own task.

All four moments were ready 3.7 s after the director started on a fresh connection. They are made one at a
time, thinking first, then greeting, celebration and apology.

## Contention with the speech paths

Each speech path ran 10 times on its own connection: `narrate` acknowledge and lines, `classify`, the first
audio of a `speak`, and `hear_final`, the time from the end of speech in `tests/companion_server/data/reminder.wav`
to the companion's endpoint. "With motion" repeats them while a second connection keeps two `animate`
requests outstanding at all times, the most the robot ever has. A real robot asks for motion in short bursts
(four requests at connect, then one per moment), so this is the worst case.

Without `SpeechFirst` (`generated_motion_runs/without-speech-first.json`), times in seconds:

| Path | Alone p50 / p95 | With motion p50 / p95 |
|---|---|---|
| acknowledge | 0.369 / 0.692 | 0.477 / 0.719 |
| lines | 0.664 / 0.829 | 1.270 / 1.367 |
| classify | 0.058 / 0.060 | 0.090 / 0.383 |
| speak first audio | 0.066 / 0.071 | 0.108 / 0.371 |
| hear final | 0.627 / 0.635 | 1.122 / 3.641 |

The narrator's lines doubled and the final transcript's p95 grew by 3 s. The robot waits 4 s for lines and
0.9 s for an acknowledgement, so this load would have pushed narrations past their deadlines, as the earlier
GPU spike did.

With `SpeechFirst` (`generated_motion_runs/with-speech-first.json`), times in seconds:

| Path | Alone p50 / p95 | With motion p50 / p95 |
|---|---|---|
| acknowledge | 0.381 / 0.598 | 0.377 / 0.645 |
| lines | 0.668 / 0.683 | 0.670 / 0.675 |
| classify | 0.062 / 0.064 | 0.063 / 0.064 |
| speak first audio | 0.071 / 0.075 | 0.072 / 0.078 |
| hear final | 0.643 / 0.674 | 0.634 / 0.663 |

Motion now waits instead. Under the nonstop speech load above, one motion took 33 s at p50, against 2.3 s
without `SpeechFirst`, because the planner gets the GPU only in the gaps. In a conversation the gaps are
long: the acknowledgement narration takes about 0.4 s, and nothing else runs while Muse works. The
planner waits at most 2 s per token, so a long hear stream slows motion but never stops it.

The torch generator step (about 22 ms) does not wait. It runs once per motion, after the planner.

## Memory

`vmmap --summary` on the bench companion, with the live launchd companion untouched on its own port:

| When | Physical footprint | Peak |
|---|---|---|
| All models loaded and warmed | 15.5 GB | 15.5 GB |
| After the run | 12.5 GB | 17.3 GB |

The motion models add about 6.9 GB at peak to the companion (`bench/motion.md`). The Mac has 96 GB.

## Reproduce

The bench starts its own companion on port 8799 with a temporary pairing, so a companion serving the robot
keeps running. It needs the `companion-server` and `companion-motion` extras and the motion models in
`~/Library/Application Support/reachy-companion/motion`.

```sh
.venv/bin/python bench/generated_motion.py --runs 10 --out bench/generated_motion_runs/with-speech-first.json
```

`--only motion` or `--only contention` runs one half. The run takes about 13 minutes, most of it the
contention half. The "without" numbers came from the same script run against the source before `SpeechFirst`
was added.

espeak-ng, which Kokoro uses, cannot read its data from a path longer than about 160 characters. Run the bench
from a virtual environment with a short path.
