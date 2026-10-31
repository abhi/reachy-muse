# Independent arbitrator audit

Audit date: 2026-10-31. Baseline: `8d15448` on `companion/arbitrator`.
The corrected rules score **295/324, 91.05%**, with **0/241 false local replies**
on fact or action cases. The release gate still fails. All 29 remaining misses
are semantic follow-ups routed to a new `MUSE` request instead of
`MUSE_FOLLOW_UP`. Follow-up accuracy is only 10/39, 25.64%.

The arbitrator remains a shadow observer. These results do not demonstrate
changed live routing, a working execution protocol, or Qwen accuracy. This
audit used synthetic utterances, fake model outputs, and numeric timing evidence.
It did not load GPU models, contact Muse or the robot, open ports, or change services.

The [corpus](arbitration_cases.json) adds E01-E60 on a separate `audit` split.
All 60 texts and expected decisions were written before the first audit run.
No labels changed after that run. The original 264 rows, including the 80-row
holdout, are unchanged. These cases are now regression coverage, not an unseen
test set for the corrected rules.

| New case family | IDs | Count | What it tests |
|---|---|---:|---|
| Mixed small talk and requests | E01-E10 | 10 | Greetings, gratitude, and feelings with factual or action suffixes |
| ASR errors | E11-E20 | 10 | Homophones, missing punctuation, fillers, whitespace, and a cut-off greeting |
| Partial transcripts | E21-E30 | 10 | Final suffixes revoke speculative speech, commands, and cancellation |
| Playback echoes | E31-E40 | 10 | Confirmed echoes versus independent speech and unmatched factual suffixes |
| Side conversations | E41-E50 | 10 | Closed windows, other-person admission, and admitted control cases |
| Morning briefing context | E51-E60 | 10 | Greetings during outages, completed output, requested versus background briefings, and clarification |

Every new row records its rationale, state, forbidden effects, and exact
Muse-bound text where applicable. Admission and echo evidence are supplied
inputs. The eval does not establish that live perception can produce them.

| Metric | Before fixes | After fixes |
|---|---:|---:|
| New audit decisions | 57/60, 95.00% | 60/60, 100.00% |
| All decisions | 292/324, 90.12% | 295/324, 91.05% |
| Original development | 141/160, 88.13% | 141/160, 88.13% |
| Frozen holdout | 74/80, 92.50% | 74/80, 92.50% |
| Earlier adversarial split | 20/24, 83.33% | 20/24, 83.33% |
| False `LOCAL_REPLY`, new fact or action cases | 0/42, 0% | 0/42, 0% |
| False `LOCAL_REPLY`, all fact or action cases | 0/241, 0% | 0/241, 0% |
| Local reply precision and recall, all cases | 48/48 each | 48/48 each |
| Unsupported effects, all cases | 1 | 0 |
| Exact Muse-bound transcript preservation | 178/178 | 178/178 |

Accuracy includes the command, target, ignore reason, and allowed reply template.
The false-local metric divides fact or action cases incorrectly answered locally
by all cases marked `fact_or_action`. Across all expected nonlocal decisions,
the corresponding rate is also zero, 0/276 overall and 0/52 in the new split.
Zero errors in this synthetic set is not a population error bound.

The following confusion matrix covers all 324 final decisions after the fixes.
Rows are expected routes and columns are selected routes. There are no invalid
final decisions or wrong arguments within a matching route.

| Expected | LOCAL_DEVICE | LOCAL_REPLY | MUSE | MUSE_FOLLOW_UP | MUSE_CANCEL | IGNORE |
|---|---:|---:|---:|---:|---:|---:|
| LOCAL_DEVICE | 41 | 0 | 0 | 0 | 0 | 0 |
| LOCAL_REPLY | 0 | 48 | 0 | 0 | 0 | 0 |
| MUSE | 0 | 0 | 139 | 0 | 0 | 0 |
| MUSE_FOLLOW_UP | 0 | 0 | 29 | 10 | 0 | 0 |
| MUSE_CANCEL | 0 | 0 | 0 | 0 | 19 | 0 |
| IGNORE | 0 | 0 | 0 | 0 | 0 | 38 |

For the new 60 cases, the matrix is diagonal with counts 5, 8, 28, 3, 2, and 14
in the same route order. Rules and rules with handwritten fake proposals produce
identical validated scores. The latter makes 168 model calls: 155 accepted,
12 invalid, and one late. Its one proposed false local reply is rejected.
These proposal timings are test fixtures, not measured inference times.

The audit found two decision bugs in `_detail()`:

- E60 accepted a yes/no response to a clarification offering alternatives.
  `test_yes_no_does_not_resolve_an_alternative_clarification` failed with
  `MuseFollowUp(...) == Muse()`. Alternative questions containing `or` now
  fall back to Muse with the exact transcript. This is a conservative lexical
  check, not a general parser for questions.
- E54 and E58 treated polite requests for a shorter briefing as new requests.
  `test_whole_polite_presentation_changes_amend_the_only_request` failed with
  `Muse(...) == MuseFollowUp('recipe-1')`. The five existing presentation phrases
  now accept bounded polite prefixes and a trailing `please`. They still require
  one complete, unfinished target. Negations, quotations, compound requests,
  and competing targets do not gain follow-up authorization.

The regression run failed in 12 parametrized cases before the production edit.
All 213 arbitration unit tests passed afterwards. The broader offline run passed
603 tests, including shadow-mode executor behavior, protocol pins, grounding,
privacy, and companion failover. `grounded()` and `ungrounded()` are unchanged.
All 23 provisional decisions in the combined replay are revoked at final
confirmation. Thirteen change action, including ten from the new cases.

The [rules run](arbitrator_runs/rules.json) measured the following Python selection
times. Units are milliseconds; percentiles use nearest rank. Each case runs once,
so first-use imports and regex compilation are included. Local reply maximum
was 8.162 ms. These measurements exclude recognition, inference, transport,
execution, and playback.

| Selected route | Cases | Python p50, ms | Python p95, ms |
|---|---:|---:|---:|
| LOCAL_DEVICE | 41 | 0.0083 | 0.0143 |
| LOCAL_REPLY | 48 | 0.1255 | 0.2336 |
| MUSE | 168 | 0.0190 | 0.0367 |
| MUSE_FOLLOW_UP | 10 | 0.0157 | 0.0218 |
| MUSE_CANCEL | 19 | 0.0114 | 0.4842 |
| IGNORE | 38 | 0.0046 | 0.0084 |

The [timing replay](arbitration_timing.py) advances a fake clock through hearing,
the actual `select()` function with a timed fake model, and the selected reply
path. It neither sleeps nor executes effects. Its two scenarios use the median
and p95 stage delays from
[the recorded companion benchmark](generated_motion_runs/with-speech-first.json):

| Fake stage | Median-stage scenario, ms | P95-stage scenario, ms |
|---|---:|---:|
| End of speech to final transcript | 643 | 674 |
| Classification | 62 | 64 |
| Speech request to first audio frame | 71 | 75 |

The classifier benchmark measured an older device-classification task. Its delay
is a proxy for this simulation, not a measurement of the arbitration prompt.
Speech timings come from the existing companion voice benchmark, not a new
Chatterbox comparison. The replay begins synthesis after the final decision.
It does not assume speculative PCM, a warm phrase cache, or concurrent inference.

The [journal timing fixture](arbitration_timing.json) contains only numeric spans,
source line numbers, route classes, and provenance. Its source is
`robot-turns-since-1026.txt`, SHA256
`2285b101edf0afd927559a157bade85825a0697ef56d76acfc6b8429e541d590`.
Lines 50, 78, and 84 supply three paired dispatch and first-answer delays.
Each Muse-bound case replays all three pairs without breaking their correlation.
The original slow hearing spans are replaced by the companion stage delays above.
The [latency baseline](latency_rsi.md) defines the distinction between answer
audio, openers, queue waits, and missing marks used here.

The next table uses timed fake proposals. Each cell gives the median-stage and
p95-stage scenarios in that order, in milliseconds. For remote answers, each
number is the median over the three journal samples. These are simulated
scenarios, not measured end-to-end percentiles. Python CPU time is excluded.

| Selected route | Cases | Speech end to decision, ms | Speech end to first answer, ms |
|---|---:|---:|---:|
| LOCAL_DEVICE | 41 | 643 / 674 | not modeled |
| LOCAL_REPLY | 48 | 643 / 674 | 714 / 749 |
| MUSE | 168 | 705 / 738 | 19,683 / 19,716 |
| MUSE_FOLLOW_UP | 10 | 643 / 674 | 19,621 / 19,652 |
| MUSE_CANCEL | 19 | 643 / 674 | not modeled |
| IGNORE | 38 | 643 / 674 | no audio |

Muse dispatch medians are 737 and 770 ms after speech end. First-text medians
are 18,346 and 18,379 ms. First-answer p95 values in the simulation are 28,946
and 28,979 ms, the largest of the three journal samples. Follow-ups borrow the
same Muse delays because no measured follow-up timeline exists. They do not
include waiting behind older work. Counts of 504 remote samples per scenario
are repetitions across 168 cases, not 504 independent observations.

All 168 fake model calls exceed the 50 ms final budget; 166 report `deadline`
and two fail parsing first. The replay retains the full 62 or 64 ms cost because
`select()` checks duration after the call returns. It does not pretend that a
timeout interrupted inference. With rules alone, Muse decision times are
643 and 674 ms, and median first-answer times are 19,621 and 19,652 ms.
Every other route has the same timing in both engines.

The simulated 714-749 ms local audio does not meet the design's 600 ms greeting
target. The journal has no local-chat or device-command measurements, and no
briefing playback marks. Ten separate briefing fetches have dispatch-to-first-text
times of 19,365 ms p50 and 32,474 ms p95 by nearest rank. They are background
research, not the response time for saying good morning. Hardware effects,
cancellation acknowledgement, new openers, outages, cold model loads, and GPU
contention remain unmeasured. The simulation cannot establish live reliability.

Raw results are [before fixes](arbitrator_runs/before.json),
[rules](arbitrator_runs/rules.json), [fake proposals](arbitrator_runs/recorded.json),
[new audit cases](arbitrator_runs/audit.json),
[revision traces](arbitrator_runs/replay.json),
[timed rules](arbitrator_runs/timing-rules.json), and
[timed fake proposals](arbitrator_runs/timing-recorded.json).
The saved revision report retains only cases with provisional revisions.

Reproduce the offline scores and timing reports from the repository root:

```sh
PYTHONPATH=src .venv/bin/python -m bench.arbitration_eval --engine rules
PYTHONPATH=src .venv/bin/python -m bench.arbitration_eval --engine recorded
PYTHONPATH=src .venv/bin/python -m bench.arbitration_eval --engine rules --split audit
PYTHONPATH=src .venv/bin/python -m bench.arbitration_replay --engine recorded
PYTHONPATH=src .venv/bin/python -m bench.arbitration_timing --engine rules
PYTHONPATH=src .venv/bin/python -m bench.arbitration_timing --engine recorded
```

To rebuild timing evidence directly from the supplied local journal, set `JOURNAL`
to its path and run:

```sh
PYTHONPATH=src .venv/bin/python -m bench.arbitration_timing --engine recorded \
	--journal "${JOURNAL:?Set JOURNAL to the local journal path}"
```

Reproduce the focused validation with:

```sh
PYTHONPATH=src .venv/bin/python -m pytest tests/test_arbitration*.py \
	tests/test_device_command.py tests/test_follow_up.py tests/test_capabilities.py \
	tests/test_local_chat.py tests/companion_server/test_grounding.py tests/test_protocol.py \
	tests/test_pins.py tests/test_failover.py tests/test_repo_hygiene.py tests/test_duplex_privacy.py -q
```

For the operator's later real-model run, first arrange a quiet GPU window.
The audit worktree's `.venv` has no MLX packages. The existing companion
environment has MLX 0.32.3 and mlx-lm 0.32.0, verified from package metadata
without importing them. Set `COMPANION_PYTHON` to the companion interpreter's path,
then run this command from the audit repository root to use this worktree's source:

```sh
PYTHONPATH=src "${COMPANION_PYTHON:?Set COMPANION_PYTHON to the companion interpreter}" \
	-m bench.arbitration_models --real-models \
	--model mlx-community/Qwen3-4B-Instruct-2507-4bit --split all --rounds 2 \
	> /tmp/reachy-arbitrator-real.json
```

This command runs all 324 cases sequentially and reports raw model accuracy,
validated decisions, transcript fidelity, and loading and inference times.
Closed windows and other ineligible inputs do not reach the model. The command
does not send to Muse or the robot or manage the companion service. It loads
MLX only with `--real-models`; it was not run during this audit. Two rounds
measure first-use versus subsequent requests, not concurrent GPU safety.
