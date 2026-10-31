# Evaluate phase 1 arbitration

The [independent audit](arbitrator.md) adds 60 cases on the `audit` split and
reports current results for all 324 cases. The 264-case numbers below are the
original baseline. The development and holdout assignments remain frozen.

Run these commands from the repository root with its test dependencies installed.
They use synthetic inputs and fakes. They do not open sockets or load GPU models.

```sh
python -m bench.arbitration_eval --engine rules
python -m bench.arbitration_eval --engine recorded
python -m bench.arbitration_eval --engine rules --split holdout
python -m bench.arbitration_replay --engine recorded
python -m pytest tests/test_arbitration*.py
```

Read `decisions` for the validated policy score. Read
`proposals_before_validation` to see rejected model proposals too. The recorded
engine uses handwritten fake outputs, including a wrong social reply, an unknown
target, malformed output, and a late result. Its score does not measure Qwen.
Neither runner prints transcripts, generated lines, or raw model output.

The [corpus](arbitration_cases.json) transcribes all 240 design rows. Shared
defaults specify admission, recognition, capability, and output state; each row
overrides them where needed. The split has 160 development and 80 holdout cases,
with paraphrase families kept together. Treat those assignments as frozen.
Another 24 cases cover the six reviewed two-target examples and adversarial
suffixes, Unicode, quoted commands, and disfluencies. These extra cases are
reported separately by `--split adversarial`.
Use `--split audit` for the 60 later cases.

Compare a run with this phase 1 baseline:

| Metric | Rules | Rules with fake proposals |
|---|---:|---:|
| Exact decisions, all cases | 235/264, 89.02% | 235/264, 89.02% |
| Development | 141/160, 88.13% | 141/160, 88.13% |
| Frozen holdout | 74/80, 92.50% | 74/80, 92.50% |
| Additional adversarial cases | 20/24, 83.33% | 20/24, 83.33% |
| False local replies on fact or action cases | 0/199 | 0/199 |
| Local reply precision and recall | 40/40 each | 40/40 each |
| Exact Muse user-text portions | 147/147 | 147/147 |
| Unsupported device, cancel, or ignore effects | 0 | 0 |

The 29 misses are semantic follow-ups conservatively routed to Muse. Five
explicit presentation changes and two yes/no clarification answers are covered
by the rules. The full accuracy gate is not met. The fake model's proposals
score 241/264 before validation, including one unsafe local reply; validation
rejects it. Those fake proposals cannot widen admission in phase 1.

Inspect `per_class`, `confusion`, and `errors` before changing any rule. Accuracy
includes command, target, ignore reason, and allowed template, not just route.
An absent denominator reports `null`, not success. Rule confidence is a fixed
contract value in this phase, not an empirical model calibration.

Use `offline_cpu_ms` only for the Python evaluation path. It includes first-use
imports and fake-model processing but excludes simulated model time, recognition,
networking, playback, and GPU work. The runner reports p50, p95, p99, maximum,
and count for each selected route. It does not establish the 600 ms greeting
target. Replay arrival times are synthetic. In the original 264-case baseline,
all ten provisional decisions are revoked at final confirmation; three change
route after an added clause.

## Observe the existing conversation

In a separately authorized run, add `--arbitration-shadow` to the normal robot
command. Omit it to disable observation. This work does not change any installed
service configuration.

The observer runs the final rules after wake-prefix handling in both normal and
follow-up turns. It emits `route`, `evidence`, and `rules_ms` to a separate log
line. It returns no decision to the executor, retains no transcripts, and never
calls a model. An observer error logs only `invalid_snapshot`; the existing
conversation continues. Default behavior, protocol golden vectors, and
`tests/pins/*.json` remain unchanged.

Legacy routing does not retain ordered amendments or request-bound clarification
text. Live shadow snapshots mark target context as omitted rather than inferring
it from unrelated output. Target handling is therefore more conservative in
live observation than in the complete offline fixtures. Echo and noise also
need positive evidence; shadow observation does not manufacture it from lexical
overlap. Full snapshots and transcript revisions belong to later protocol work.

## Run the model comparison later

Arrange a quiet GPU window independently, then run:

```sh
python -m bench.arbitration_models --real-models --split holdout --rounds 2
```

Without `--real-models`, the script exits before importing MLX. With the flag,
it loads the existing Qwen3-4B-Instruct model and evaluates eligible synthetic
cases sequentially with a 32-token output limit. It reports raw model scores,
the rule-gated result, loading time, first-request time, and per-round latency.
It does not save prompts or raw generations. Closed listening windows are never
passed to the model. The script does not contact Muse or the robot, open ports,
change services, or control the live companion.

Model time includes generation and GPU synchronization. Late output is measured
and rejected after completion; this script does not attempt to interrupt a
running Metal call. It does not implement prefix caching, concurrent engine
stress, or audio-onset measurement. GPU ownership, both voice engines under
load, long recordings, and the hour-long stress run remain phase 2 work. Real
model and contention results have not been collected for this change.

## Continue implementation

Keep shadow mode separate from execution until phase 3 supplies negotiation,
execution claims, final deadlines, capability failover, and owned-chat retirement.
The existing executor still owns routing, request presentation, speech fallback,
and selected-chat behavior. No new per-request Muse wrapper is sent in phase 1.
The non-owned chat, cancellation, uncertain-send, and reconnect regressions in
the design remain required when those behaviors migrate. The replay checks
decision identity and state changes; it does not claim to test a future executor.
