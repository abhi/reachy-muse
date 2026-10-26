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
81.027 seconds p50, 122.172 seconds p95, and 10,413.732 seconds in total.
These pairs mostly describe reconnections, not full process starts. The initial ready line lacks
its preceding registration. A second complete application startup takes 170.246 seconds,
07:05:11.776 to 07:08:02.022. Across the two application starts, p50 is 173.781 seconds and
p95 is 176.962 seconds. Setup and warm-up policies changed during this journal, so these are historical observations.

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

```sh
PYTHONPATH=src python -m bench.latency_replay --runs 7
```

Results and subsequent hypotheses are recorded below as they are tested.
