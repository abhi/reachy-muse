# Reachy latency experiments

Baseline code: `e8b454b`, branch `perf/rsi-1`. All experiments run offline.
No robot, Muse, launchd, deployment, or existing companion port is used.
The chat model, registration, prompts, endpoint policy, and spoken content are outside the changes.

The read-only input is `robot-journal-14d.txt`. Its SHA256 is
`e325b662e2b316c637242e3f1dddd451a4b6cbbaea982645457dc18b4d96da5f`.
The requested filter is October 12 onward, but the supplied timestamps cover only
October 20 10:50:39.328 through October 26 14:05:04.472. There are no October 12–19 samples.
The file contains appended time ranges, so file order is preserved when joining nearby log events.
There are 13 timeline records: three Muse-bound spoken requests, nine morning briefing fetches,
and one empty transcript. There are no measured device commands or local-chat answers.
"Muse question" below means the Muse route; the logs do not expose the user's words.

Reproduce the journal analysis from the repository root:

```sh
python3 tools/turn_timeline.py "$JOURNAL" --steps
python3 -m bench.latency_journal "$JOURNAL" --since 2026-10-12
```

Each cell gives `n: p50 / p95` in seconds, with linear interpolation between sorted samples.
Stages are differences between the named marks, never between unrelated marks that happen to exist.
An absent mark is missing data, not zero latency. `first_audio` records the playback start callback,
before the first hardware write; it does not measure sound at the speaker. Acknowledgements and
progress are excluded from first-answer audio. Briefing fetches stop their timeline before arrival
and playback, so their wake-to-audio and speech-end-to-audio values cannot be recovered.

| Stage | All measured turns | Device command | Local chat | Muse question | Morning briefing | Empty transcript |
|---|---|---|---|---|---|---|
| Wake to speech end | 1: 4.590 / 4.590 | missing | missing | 1: 4.590 / 4.590 | missing | missing |
| Speech end to endpoint | 4: 2.000 / 2.493 | missing | missing | 3: 2.000 / 2.522 | missing | 1: 2.000 / 2.000 |
| Endpoint to transcript | 4: 10.840 / 15.869 | missing | missing | 3: 12.395 / 16.073 | missing | 1: 9.285 / 9.285 |
| Transcript to dispatch | 3: 0.034 / 0.038 | missing | missing | 3: 0.034 / 0.038 | missing | missing |
| Dispatch to Muse acknowledgement | 12: 0.881 / 2.131 | missing | missing | 3: 0.898 / 1.929 | 9: 0.864 / 2.126 | missing |
| Acknowledgement to first status | 12: 0.132 / 0.229 | missing | missing | 3: 0.154 / 0.186 | 9: 0.132 / 0.231 | missing |
| First status to first text | 12: 10.853 / 42.188 | missing | missing | 3: 10.466 / 12.903 | 9: 11.241 / 44.187 | missing |
| First text to reply complete | 12: 0.920 / 1.730 | missing | missing | 3: 0.840 / 0.984 | 9: 1.023 / 1.776 | missing |
| Reply complete to narration | 3: 0.007 / 0.047 | missing | missing | 3: 0.007 / 0.047 | missing | missing |
| Narration to answer audio | 3: 1.968 / 23.871 | missing | missing | 3: 1.968 / 23.871 | missing | missing |
| Dispatch to acknowledgement audio | missing | missing | missing | missing | missing | missing |
| Transcript to local-chat audio | missing | missing | missing | missing | missing | missing |
| Wake to first-answer audio | 1: 17.198 / 17.198 | missing | missing | 1: 17.198 / 17.198 | missing | missing |
| Speech end to first-answer audio | 3: 30.716 / 54.683 | missing | missing | 3: 30.716 / 54.683 | missing | missing |

The three speech-end-to-answer samples are 30.716, 57.346, and 12.608 seconds.
The single wake-to-answer sample is 17.198 seconds. These sample sizes cannot establish a stable tail.
The empty transcript took 11.285 seconds from speech end and produced no answer.
Source lines are 393, 434, 486, 2436, 4373, 6334, 6387, 8317, 8359, 9415, 10240, 10275, and 10315.
The analysis script prints their dates and route classifications.

The latest startup has the following timeline. The systemd start message has no timestamp,
so elapsed time starts at the first timestamped application line, a lower bound on process startup.

| October 26 log time | Event | Since first application log, s | Since previous row, s |
|---|---|---:|---:|
| 13:28:51.400 | Morning briefing configuration | 0.000 | 0.000 |
| 13:28:51.890 | Silero ready | 0.490 | 0.490 |
| 13:28:57.064 | Timed wake handoff ready | 5.664 | 5.174 |
| 13:29:01.179 | Hardware connection mode selected | 9.779 | 4.115 |
| 13:29:04.291 | Face tracking ready | 12.891 | 3.112 |
| 13:29:04.530 | Companion link up | 13.130 | 0.239 |
| 13:29:28.394 | Microphone/speaker selected, local workers started | 36.994 | 23.864 |
| 13:29:29.271 | Muse registered | 37.871 | 0.877 |
| 13:29:35.273 | Dedicated side chat ready | 43.873 | 6.002 |
| 13:31:48.716 | Reachy ready | 177.316 | 133.443 |

Across 124 complete registration/ready pairs, registration-to-ready is
81.026 seconds p50, 122.172 seconds p95, and 10,413.732 seconds in total.
These pairs mostly describe reconnections, not full process starts. The initial ready line lacks
its preceding registration. A second complete application startup takes 170.246 seconds,
07:05:11.776 to 07:08:02.022. Across the two application starts, p50 is 173.781 seconds and
p95 is 176.963 seconds. Setup and warm-up policies changed during this journal, so these are historical observations.

The silent gap is in `VoiceConversation.run()` before microphone capture. It awaits
`warm_up()`, which awaits the main `LocalVoice.warm()` and then the progress voice.
Each warm-up synthesizes every fixed phrase through Piper and keeps its PCM in that
short-lived `LocalVoice`. `FailoverVoice.warm()` does the same local work even with the
companion connected. Only afterwards does `run()` start the subscriber and microphone.
`ReachyService.serve()` rebuilds the voices on each connection while the CLI retains the Piper workers.
This discards the recordings and repeats the synthesis after every reconnect.
No warm-up log existed in the baseline. The journal therefore bounds the whole gap but cannot split
its 133.443 seconds among synthesis, subscription acknowledgement, and microphone initialization.
The production-path offline replay below isolates the synthesis work without contacting any service.

The Reachy-controlled opportunities rank as follows. Totals are observed stage time and upper bounds,
not promises that all of the time is removable. Backend generation is excluded.

| Rank | Candidate | Observed total, s | Interpretation |
|---:|---|---:|---|
| 1 | Repeated fixed-phrase synthesis before ready | 10,413.732 registration-to-ready envelope | 124 gaps; isolate warm-up before attributing exact savings |
| 2 | Local transcription after endpoint | 38.165 | 28.880 across answered turns, 9.285 on the empty turn; current streaming/companion fast path is 0.003 s |
| 3 | Narration to playback | 28.443 | Includes 26.305 s on one turn, ordered earlier speech, interruption, TTS, and speaker ownership |
| 4 | Endpoint silence | 8.580 | 6.580 on answered turns; shortening it can cut off a user, so needs audio evidence |
| 5 | Dispatch preparation | 0.083 | Already small; measure acknowledgement and command work before changing it |
| 6 | Narration | 0.065 | All three observed replies bypassed condensation as short plain answers |

Dispatch-to-acknowledgement totals 13.412 seconds but mixes transport with Muse's acceptance.
First-status-to-first-text totals 183.064 seconds and reply completion adds 12.380 seconds.
Those are not assigned to Reachy. First-status delivery adds 1.679 seconds, also mixed ownership.

Experiment 1 hypothesis: the same fixed PCM can survive reconnection in the existing Piper object.
This should remove all repeated synthesis without changing phrase text, audio, ordering, or routing.
`bench.latency_replay` runs real `VoiceConversation.run()`, `warm_up()`, `backends_for()`,
subscription handling, and the ready log. Its fake Piper inference costs 10 ms per phrase.
It checks the SHA256 of all replayed PCM across the first connection and the reconnect.
Its session is an in-memory queue and its microphone returns no samples. It cannot send to Muse.
The replay clock begins at `VoiceConversation.run()`, after registration/setup. "Cold" means
an empty phrase cache. These measurements exclude CLI/hardware startup and actual model loading.

```sh
PYTHONPATH=src python -m bench.latency_replay --runs 7
```

Experiment 1 accepted. Store the fixed recordings on `PiperSpeech`, whose lifetime already
spans reconnects, keyed by output sample rate. Each conversation still gets a new narrator,
route, and turn state. Only completed recordings enter the cache; cancellation preserves
completed phrases and retries the incomplete phrase. Private replies never enter this cache.
Warm-up now logs the cached/missing counts and elapsed time, without phrase text.

| Replay metric, seven runs | Before | After |
|---|---:|---:|
| Reconnect to ready p50, ms | 567.643 | 0.599 |
| Reconnect to ready p95, ms | 572.069 | 0.805 |
| Reconnect synthesis calls | 49 | 0 |
| Cold start to ready p50, ms | 573.062 | 557.955 |
| Cold start synthesis calls | 49 | 49 |

The small cold-start timing difference is noise, not an accepted cold-start improvement.
The 49 phrases are 14 acknowledgements/wake cues and 35 progress phrases.
The PCM digest is identical before and after:
`604e5162d4d9936bd0d096bbf4c55348ad89947006ff966efd7578d2a298968f`.
Raw results: `latency_rsi_runs/warm-before.json` and `latency_rsi_runs/warm-after.json`.
Validation: 58 focused tests passed, including actual ready-log startup, PCM equality,
companion failover reuse, separate voice/rate caches, and interrupted warm-up recovery.
This cache is in memory. A process restart still performs the first synthesis.

Experiment 2 hypothesis: the main and progress voices have independent Piper workers,
so their cold warm-ups can overlap. Keep capture blocked until both phrase banks are complete.
Measure the same 49-phrase replay before and after, and verify that errors or cancellation
join both workers' warm-up tasks. This tests scheduling, not Raspberry Pi CPU throughput.

Experiment 2 accepted. Warm both phrase banks concurrently and join both tasks on success,
failure, or cancellation. All 49 phrases still finish before capture. The theoretical replay
critical path is 35 rather than 49 inference delays. Seven measured cold starts improved
from 548.562 to 393.762 ms p50 and 655.199 to 493.261 ms p95. The PCM digest is unchanged.
Raw results: `latency_rsi_runs/parallel-before.json` and `latency_rsi_runs/parallel-after.json`.
All 62 focused tests passed, including simultaneous start, waiting for the slower bank,
and cancelling either bank when its sibling fails. Actual cold-start savings depend on
CPU contention on the robot and must be measured there.

Experiment 3 hypothesis: subscribing while fixed speech warms removes an unnecessary serial
transport wait after setup. Registration and the setup challenge remain untouched. The
installed SDK clears `chat_subscribed` when the setup iterator closes, then sets it after the
next subscription's acknowledgement. Replay that acknowledgement with a fixed 120 ms delay.

```sh
PYTHONPATH=src python -m bench.latency_replay --runs 7 --subscription-delay-ms 120
```

Experiment 3 accepted. Start the existing subscription before warming speech; start capture
only after warm-up and the subscription acknowledgement. Seven cold-start replays improved
from 530.848 to 405.139 ms p50 and 634.970 to 503.582 ms p95 with the fixed 120 ms
subscription delay. Reconnects, which already reuse the cache, remain transport-bound
at 122.685 versus 121.740 ms p50. No extra request is made and all PCM remains identical.
Raw results: `latency_rsi_runs/subscribe-before.json` and `latency_rsi_runs/subscribe-after.json`.
All 64 focused tests passed. The new lifecycle tests verify subscription overlap, no capture
during warm-up, and closing the subscription on warm-up failure or cancellation.
The initial full-suite run also passed: 1,968 passed, two skipped, six real-model tests deselected.
The suite is repeated after the final production changes.

Experiment 4 hypothesis: an acknowledgement delays a Muse-bound question's dispatch and
first answer. Replay a 200 ms acknowledgement versus a zero-delay acknowledgement, with
the same 20 ms fake Muse response and 10 ms voice inference. Exercise `turn()`, `_dispatch()`,
the real subscription reader, narrator, and playback callback. Assert exactly one outgoing
request with the original 28-byte text on every turn. No production edit is made before measurement.

```sh
PYTHONPATH=src:tests python -m bench.latency_turn_replay dispatch --runs 7
```

Experiment 4 rejected. Dispatch p50 is 0.256 ms with the slow acknowledgement and 0.244 ms
with the immediate one. First-answer p50 is 33.086 versus 33.093 ms, so removing the 200 ms
opener delay produces no first-answer gain. Question acknowledgements already run in parallel
and are cancelled when the answer arrives. Small talk intentionally waits for classification
to avoid sending a locally answered turn to Muse. Raw results: `latency_rsi_runs/dispatch.json`.

Experiment 5 hypothesis: Reachy sends its full instructions on every spoken request, inflating
Muse's input. Replay the real `_dispatch()` through two steady marker requests, a switch to
plain replies, another plain request, and a switch back. Count payload bytes and requests,
without using a model or changing the prompt.

Experiment 5 rejected as a steady-turn optimization. The five requests carry 28, 28,
2,211, 28, and 2,231 bytes, respectively. There is exactly one send per turn; formatting
instructions only return when the reply style changes. Local-chat history is separately
capped at 300 bytes. No redundant steady-turn setup request exists to remove. A shorter
style-switch reminder might reduce prompt processing, but this replay cannot prove its
formatting and persona behavior or Muse TTFT. It is left for a live experiment.
Raw results: `latency_rsi_runs/prompt.json`; reproduce with the `prompt` experiment.

Experiment 6 hypothesis: reducing the companion/fallback silence threshold from 2.0 to
0.8 seconds saves 1.2 seconds without changing turn boundaries. Replay the repository's
recorded reminder, including its correction "not at nine", then insert a one-second
hesitation at its existing comma. Keep the existing 1.2-second short-utterance allowance.

Experiment 6 rejected for changed behavior. The unmodified recording endpoints move from
5.00 to 3.80 seconds, a 1.20-second gain. With the hesitation, the baseline yields one turn
ending at 6.00 seconds; the candidate yields two turns ending at 2.78 and 4.80 seconds.
It separates the correction from the request. This is recorded audio with an inserted pause,
using the supported WebRTC gate, not a claim about the live Silero/companion recognizer.
Raw results: `latency_rsi_runs/endpoint.json`; reproduce with the `endpoint` experiment.

Experiment 7 hypothesis: interrupting queued output can remove the post-narration wait
without losing speech. Queue a 600 ms synthetic briefing line, then a new answer. Compare
normal ordering with calling the existing hush operation before enqueueing the answer.
Count every prior-line sample delivered, as well as answer latency. The journal's largest
outlier overlaps a briefing and a 4.2-second fallback notice; it cannot be labelled pure TTS cost.

Experiment 7 rejected for lost speech. Interrupting the previous line reduces answer p50
from 651.229 to 40.634 ms, but delivers only 0.080 of its 0.600 seconds. Normal ordering
delivers all 0.600 seconds on every replay. That apparent gain discards 0.520 seconds of
already accepted speech. No queue priority or interruption policy was changed.
Raw results: `latency_rsi_runs/queue.json`; reproduce with `queue --runs 7`.

Experiment 8 hypothesis: starting batch transcription before the earlier turn drains
removes a queue-dependent recognition delay. The source already calls `ended.start()`
in `queue_turn()` at admission, before the request waits. Replay a 20 ms fake recognizer
behind a 60 ms earlier turn, then try an additional early start. Measure the remaining
wait at `endpoint()` and verify that recognition happens only once.

Experiment 8 rejected. Remaining recognition wait is 0.016 ms p50 in both variants.
Total queue-plus-recognition p50 is 61.466 versus 61.821 ms. Both perform exactly one
recognition. Scheduling it early is already implemented, so no further gain is established.
Results: `latency_rsi_runs/asr.json`; reproduce with `asr --runs 7`.

Changing recognizer models, decoding limits, or cropping audio is deferred. The journal contains
no user recordings for the slow recognition cases, and the local environment has no robot
recognizer loaded. Fake inference timing cannot establish word accuracy or a Pi inference gain.
The baseline already uses a short-context Whisper path for suitable turns and supports streaming
recognition. Removing uncertainty checks would trade correctness for speed.

Experiment 9 hypothesis: a two-sentence Muse answer waits on an avoidable companion narration
round trip. Replay `FailoverNarrator.lines()` with a 200 ms companion operation and with an
immediate one, keeping the exact two spoken sentences. Count companion calls.

Experiment 9 rejected. Both variants take 0.005 ms p50 and make zero companion requests.
The existing short-plain-reply path already removes this round trip. The observed 0.065 seconds
of narration across three answers provides little remaining opportunity.
Results: `latency_rsi_runs/narration.json`; reproduce with `narration --runs 7`.

Three changes were accepted, each in its own commit:

| Commit | Change | Offline p50 before/after, ms | Offline p95 before/after, ms |
|---|---|---:|---:|
| `1e67ac4` | Reuse fixed PCM on reconnect | 567.643 / 0.599 | 572.069 / 0.805 |
| `5fc5774` | Warm independent voices concurrently | 548.562 / 393.762 | 655.199 / 493.261 |
| `1e954d5` | Overlap subscription acknowledgement with warm-up | 530.848 / 405.139 | 634.970 / 503.582 |

The rows are sequential experiments with different workloads, not additive robot savings.
No accepted change claims a measured improvement in live wake-to-answer or speech-end-to-answer latency.
All rejected variants remain confined to the offline experiments.
For before/after reproduction, run the current replay scripts with `PYTHONPATH` pointing first
to the `src` directory of the desired checkout. Experiment 1 starts at `06fd273`, experiment 2
at `1e67ac4`, and experiment 3 at `5fc5774`. Keep the same interpreter, fake delays, and run count.
For turn experiments also include the current `tests` directory in `PYTHONPATH`.

The next experiments require the live robot and were not run:

1. Compare process startup and reconnect separately with the same build and voice models.
   Record both warm-up spans, synthesis counts, subscription start/acknowledgement, hearing start,
   and the ready event. Confirm zero synthesis calls on reconnect and measure cold parallel warm-up
   under normal camera/VAD load. Compare PCM and monitor CPU contention before accepting a live gain.
2. Collect at least 30 turns per requested kind, with wake, actual speech end, first synthesized
   sample, queue admission, speaker ownership, and actual output onset. Add answer-audio marks to
   device-command acknowledgements and briefing playback; their current timelines cannot supply
   those metrics. For briefings, keep prefetch time separate from arrival-to-playback time.
3. Reproduce the fallback case with the same recorded utterances and the robot's current recognizer.
   Separate recognizer queue time from inference. Compare existing streaming and short-context
   paths on exact transcript/correction preservation before changing endpoint or decoder settings.
4. Reproduce the October 23 briefing overlap and companion outage. Measure time in prior audio,
   fallback notice audio, companion first-chunk timeout, Piper inference, and hardware writes.
   Keep all speech and ordering while testing more preparation ahead of playback. The 26.305-second
   stage currently combines these causes, so a timeout reduction alone is not justified.
5. Measure a shorter format reminder only on an actual reply-style switch. Compare TTFT and
   exact speech-format/persona compliance while keeping the chat model and registration unchanged.
   Steady requests already omit the full instructions. This experiment would contact Muse and
   therefore remains outside this offline run.

The prescribed suite is invoked below with `REACHY_PYTHON` pointing to the existing project
interpreter. Only `UV_CACHE_DIR` was redirected to writable scratch because the default cache
was sandbox-restricted. The repository's default selection excludes its six real-model tests.

```sh
uv run -q --no-project --python "$REACHY_PYTHON" \
  --with pytest --with pytest-timeout --with 'av>=12' --with 'webrtcvad-wheels>=2.0.14' \
  env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/src" \
  python -m pytest -q -p no:cacheprovider --timeout 120
```

Final full-suite result at `4c8a5c0`: **1,981 passed, two skipped, six deselected in 685.73 seconds**.
The initial full run had 1,968 passes in 690.94 seconds. The 13 added tests cover cache reuse,
baseline PCM, rates, cancellation, concurrent warm-up, worker restart, and subscription lifetime.
The two worker-startup cancellation cases failed before the fix with a running worker after
warm-up returned. Both pass now. Cold and reconnect PCM are pinned to the pre-cache revision
`06fd273`; replacing cached phrases with silence now fails five audio tests.
A focused run of the latency and local-speech tests passed all 40 checks.
All six rejected-turn replay commands completed with their behavior assertions intact.
All 12 result files parse as JSON, and every warm-up replay has the same PCM digest.
The journal hash is unchanged. `git diff --check` passes. Only `conversation.py`,
`local_backends.py`, and `local_speech.py` changed in production source.
