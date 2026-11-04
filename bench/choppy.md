# Choppy speech incident replay

The deployed baseline is `ab63f78`. The supplied November 4 robot log starts at
03:20; the companion log also contains earlier days. This investigation used
those files read-only. No robot, companion process, service, model, or GPU was
started or contacted.

## What the logs establish

| Observation | Recorded evidence |
|---|---|
| A healthy companion was rejected for slow speech | Five `companion speech stopped (TimeoutError)` warnings at 05:42:44.053, 05:42:46.057, 05:43:11.077, 05:43:43.729, and 05:44:40.325. There was no link-down event during this window. |
| Speech and motion overlapped | The greeting motion finished at 05:42:47.254 after 5.105 s, including 5.063 s in its planner. This supports contention as an explanation; there is no per-chunk synthesis trace proving which GPU operation delayed each speech chunk. |
| The briefing contained long pauses | Logged sentence-end to next-start intervals include 15.279 s, 8.759 s, and 7.213 s. These exclude any unlogged starvation inside a sentence. |
| Hearing was quarantined before the local endpoint | Link-down and `did not end the turn` occurred at 06:43:48.194. The turn timeline places speech end at 06:43:47.297 and the local endpoint at 06:43:49.297. Quarantine therefore occurred 0.897 s after speech end. |
| The companion returned before the answer | Link-up was 06:44:05.224, after 17.030 s down. First reply text arrived at 06:44:06.131; first answer audio started at 06:44:10.968, 4.837 s later. The old route had already selected the local voice. |
| The second answer sentence was delayed | Sentence one started at 06:44:10.968 and ended at 06:44:15.210. Sentence two started at 06:44:28.360: **13.150 s of logged silence**, or 17.392 s between starts. Sentence three followed after another 0.870 s gap. |
| Input paused output | One pause/resume pair, 05:44:00.725 to 05:44:02.094, lasted 1.369 s. A two-character transcript was dispatched during it. The log does not contain the recording, so it cannot prove that the sound was echo. |
| Queue clearing happened at reply admission | `Cleared player queue` appears at 06:44:06.264 after reply text arrived. The other two clears were service shutdown/reconnect events. |
| Motion repeatedly killed the companion | All 19 watchdog exits in the complete companion log name motion. None occurs after 03:20 on November 4. Counting these as 19 exits during today's incident would be incorrect. |

The old fallback warning says `speaking it`, not `repeating the line`, on all
five failures. Preparation can fail while an earlier sentence is playing. These
warnings prove unwanted local routing, but do not reveal the precise audible
switch count or prove a word was cut in half. The logs also lack accepted-PCM
timestamps and speaker recordings. The replay below measures those properties
with controlled recorded PCM rather than claiming to reconstruct missing audio.

## CPU replay

`choppy.py` calls production `VoiceConversation._speak_lines`, `FailoverVoice`,
`LocalVoice`, `PreparedSpeech`, and `PCMPlayer`. It feeds recorded speech from
the existing reminder, correction, and negation WAV fixtures. The audio is
unchanged; the replay takes the first 3.0, 4.0, and 3.5 seconds, respectively.
The source yields 4096-byte-equivalent PCM16 packets, matching the Piper child.
It substitutes timed producers for inference and a measuring sink for hardware.
No ASR or TTS model is loaded. Every run asserts exact PCM equality and records
a SHA-256 digest, so dropped, repeated, reordered, or changed audio fails the run.

The fixtures preserve the failure triggers, not unrecorded model timings:

* `morning_slow_companion` keeps the link alive and delays the middle sentence's
  first audio by 2.2 s. This crosses the deployed 2 s cutoff. The logs establish
  that cutoff was crossed, but do not reveal the actual Kokoro delay.
* `hearing_recovers_before_reply` selects a route while the link is down and
  restores it before a reply begins, matching the 06:44 ordering. Local producer
  waits of 4.831, 13.150, and 0.870 s approximate the observed delay pattern.
  They are injected waits, not measured Piper inference times. The available
  companion produces each sentence in 0.2 s, an explicit fixture assumption.
* `cpu_fallback_pipeline` keeps the companion down and uses local producer waits
  of 0.2, 1.3, and 0.4 s. The second sentence can be prepared during the first.
  This isolates synthesis/playback overlap without assuming a CM4 speedup.
* `cpu_fallback_incident_stall` retains the 4.831, 13.150, and 0.870 s waits
  while the companion remains down. It deliberately exceeds what the available
  preceding audio can hide and measures the remaining starvation.

The sink computes starvation from accepted samples and their playback duration.
A reported gap exceeds 10 ms. This measures the software output timeline; it
does not measure DAC output or pauses already present in the source recordings.
Voice identities travel beside PCM through the production queues without
changing samples. Preparation and playback execute on the real asyncio loop.

Run from the repository root with the project environment:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python bench/choppy.py --output bench/choppy-after.json
```

For the baseline, extract `src` from `ab63f78` into a temporary directory and run
the **same current benchmark file**, putting that extracted `src` on
`PYTHONPATH`. The checked-in before/after JSON files contain the measurements.
They are single CPU replays, not a latency distribution or a CM4 benchmark.

| Replay | Gaps per reply, before / after | Worst gap, before / after | Voice switches, before / after | First audio, before / after |
|---|---:|---:|---:|---:|
| Slow companion during briefing | 2 / 0 | 5.556 / 0 s | 2 / 0 | 0.201 / 0.202 s |
| Hearing recovers before answer | 2 / 0 | 13.203 / 0 s | 0 / 0 | 4.834 / 0.202 s |
| Local fallback synthesis ahead | 2 / 0 | 1.353 / 0 s | 0 / 0 | 0.202 / 0.203 s |
| Local fallback with incident-sized stall | 2 / 1 | 13.202 / 10.153 s | 0 / 0 | 4.832 / 4.834 s |

The recovered answer changes from Piper throughout to companion throughout. A
preliminary replay before the final output cleanup had one 13 ms scheduler gap;
the table contains the final-code rerun. These single-run differences should
not be read as evidence about a millisecond-scale hardware latency guarantee.
The worst-case local producer still starves for 10.153 s, despite preparation
starting during the previous sentence. All eight
runs delivered the same 168,000 frames with SHA-256
`101ed3f02cb01748c530be7ef4d2f71bab5882efa6df96fe04e1a0f9ed621d85`,
and none cleared the output queue. The short behavior test pins 4,800 recorded
frames and their literal digest through the same production path.

## Measured timing bounds

The previous benchmarks store counts, maxima, medians, and p95 values. For fewer
than 100 observations, the nearest-rank empirical p99 is the maximum; that value
can be recovered exactly without inventing missing raw samples.

| Measurement | Samples | Empirical p99 | Source |
|---|---:|---:|---|
| Kokoro first audio under realistic load | 24 | 0.840 s | `voice_load_runs/kokoro-realistic.json` |
| Kokoro first audio under full load | 24 | 0.717 s | `voice_load_runs/kokoro-full.json` |
| Hearing final while Kokoro speaks | 3 | 2.768 s | `voice_load_runs/kokoro-realistic.json` |
| Hearing final with competing motion | 10 | 3.641 s | `generated_motion_runs/without-speech-first.json` |
| Completed motion since 03:20 today | 10 | 5.105 s | Supplied companion log |

Three hearing samples cannot establish a production population p99. None of
these reports measures today's audio-chunk gaps or pong round-trip distribution.
The 2 s speech/hearing limits were below observed healthy loaded operations.
They must not decide connection health. Transport heartbeat progress and actual
stream progress are the evidence for liveness.

The revised keepalive sends a ping every 5 s and allows 20 s without incoming
frames or a pong. Twenty seconds rounds three times today's 5.105 s empirical
motion p99 up to the next ten seconds. This is a conservative allowance for
host scheduling under measured load, **not a measured network RTT p99**. A dead
socket is detected after 20 s without incoming progress, subject to event-loop
scheduling. Ping transmission and pong receipt share that budget. No model inference timer alone
marks a connected companion down. Obtaining a production heartbeat/chunk p99
requires collecting those timings during a future permitted run.

Prepared audio starts with 80 ms of queued PCM and measures uncovered producer
waits. Its target follows the rolling empirical p99 plus 40 ms, capped at
500 ms; priming itself waits at most 500 ms after the first available chunk.
This cap controls added latency and memory. It is not a synthesis deadline.

## Deadline audit

`3c8c35c` on `perf/turn-2` added the 2 s speech-chunk timer and the 4/1/8 s
narration/classification/animation timers. It also raised keepalive to a 5 s
interval with a 10 s pong wait and raised the handshake timeout to 10 s.
`2cd9018` serialized speech before the preexisting 2 s failover gap timer.
That serialization prevented one queued request from timing another out, but
left both inference-speed cutoffs in place. The preexisting 2 s hearing first
response and endpoint timers likewise treated a healthy slow worker as dead.
The revision removes those application-health cutoffs, including the internal
companion voice wrapper's 1 s switch-to-Kokoro timer. Actual dead-link fallback
restarts the current sentence at its beginning and retains that fallback for
the rest of the reply. A new reply chooses an available companion again.

The remaining bounds have distinct owners:

| Bound | Purpose and failure scope |
|---|---|
| Handshake / pong / close | Transport establishment, missing peer progress, and cleanup. Inference duration does not expire an active stream. |
| Optional acknowledgement, progress, and local classification | Relevance budgets for optional work. Missing a useful-response window must not quarantine the companion or change an answer's voice. |
| Speech playback | Lack of output or companion progress, excluding deliberate user holds; not total reply duration. |
| Local batch ASR startup/transcription, 60 s | A separate local subprocess boundary with no partial-result protocol. It does not mark the companion down. |
| Closing motion, 1 s; motion output wait | Optional motion relevance/cleanup. These do not own audio delivery. |
| Explicit device-command timeouts | Caller-requested expiration. An expired command must not start later. |
| Introduction send/reply waits | Bounded external-request ownership and ambiguous-send cleanup. These do not describe companion liveness. |
| Wake windows, reply completion, recorder caps | Conversation policy, not connection-health evidence. |

## Limits

A bounded playback buffer cannot hide indefinitely slow synthesis. If a producer
needs 13 s before its next samples and only 3 s of previous audio is available,
one-sentence lookahead alone leaves at least 10 s uncovered. Eliminating that
case requires earlier PCM production or a longer initial wait. This replay
therefore does not claim a universal gap-free guarantee on CM4. The incident
improvement primarily comes from retaining the available companion, recovering
it at reply boundaries, and removing avoidable output/prefetch pauses.

## Validation

The full requested pytest command passed with **2,741 passed, 3 skipped, and
6 deselected** in 826.66 seconds. The repository's default selection excluded
real-model tests. Focused tests cover slow healthy streams, missing pongs and
blocked ping writes, sentence restart boundaries, PCM preservation, echo gating,
source-owned progress, and arrivals during the final playback drain.
