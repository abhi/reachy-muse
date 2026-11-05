# November 5, 06:11: a turn that went silent

The deployed baseline is `a7c37d3`. The robot log for the turn is the only
production record. No companion log exists for November 5. This
investigation read the logs only. It did not start or contact the robot, the
companion, any service, Muse, or any model.

## What the log shows

The turn timeline line at 06:13:54.948 gives these values:

```
wake=0.000 speech_end=6.340 endpoint=43.620 transcript=43.644 dispatch=43.798
muse_ack=45.458 first_status=45.754 first_text=136.898 first_audio=137.566 forced=no
```

| Seconds after the user stopped | Recorded event |
|---:|---|
| 0 | Reachy's voice detector hears the last speech. |
| 0 to 37.0 | Nothing is logged. Reachy does not move or speak. |
| 37.28 | The turn ends. `forced=no` shows that the companion's own endpoint ended it, not Reachy's silence. |
| 37.46 | The request is dispatched, and the thinking motion starts. |
| 38.36 | "narration timed out", then "Reachy skips the acknowledgement; its companion's narration was late". |
| 39.12 | Muse accepts the request. |
| 47.14 | Reachy selects the waiting cue, 8 s after Muse accepted. |
| 50.76 | The cue becomes audible after 3.6 s of companion synthesis: 'The answer to "…" is taking a little longer.' |
| 131.23 | The first answer sentence plays. Muse took 91.4 s to answer. |
| 131.2 to 150 | All four sentences are logged "with expression neutral". |

The companion link stayed up for the whole turn. The log has no link-down
event, outage, or reconnect after 06:00:57.

## Root causes

1. **The robot stopped bounding the end of a turn.** In `f11b32a`
   (November 4, 07:28, "Use incoming frames and pongs for companion liveness"),
   three commits before `470fe5c`, the robot lost `ENDPOINT_WAIT_S`,
   `HEAR_RESPONSE_S`, `_overdue` and `hearing_stalled`. At `a7c37d3`, a stream
   that Reachy's voice detector finished goes into `_closing` and stays there
   (`companion_client.py:604`). `active` stays true for as long as that stream
   is open (`:573`). Only a link failure can end the wait (`:627`), and pongs
   keep the link alive. The same commit deleted
   `test_a_turn_the_companion_never_ends_is_finished_on_the_robot_within_the_deadline`.
   It added `test_a_slow_hearing_endpoint_keeps_the_reply_on_companion`, which
   asserted that no request is sent 2.2 s after the user stops. That commit had
   a real reason: the old timer took the whole link down. Taking the link down
   switched Muse's answer to Piper (`bench/choppy.md`, 06:43 on November 4).
   The fix removed the robot's upper bound along with the quarantine. The
   companion's FIFO hear task needed 37 s to work through the audio queued
   ahead of `finish`. That is why the companion's own endpoint arrived first.
2. **A late narration skipped the acknowledgement.** At `a7c37d3`,
   `failover.settle` returns `Hold(guess, None)` for `LATE`
   (`failover.py:320`). This has been the case since `e8b454b`. When the
   companion missed the 0.9 s deadline, Reachy said nothing.
3. **The only waiting cue came late and sounded like filler.**
   `PROGRESS_FIRST_S = 8.0` (`progress.py:17`) counts from Muse's
   acknowledgement. The wording, "is taking a little longer" (`:121`), dates
   from `2cd8053`. The cue was synthesized only when it became due, so the
   busy companion voice added 3.6 s.
4. **Every sentence was neutral.** The companion narrator is told to use
   "neutral otherwise" (`narrator_mlx.py:102`, from `e888a2a`). The local
   classifier treats any explicit hint as final (`motion/sentence.py:39`, from
   `045a098`), so the model's default hid every word cue.
5. **No motion before the endpoint.** Pause nods depend on transcript partials
   (`conversation.py:948`). The flag they test is the recorder's sustained
   speech state, which clears only at the silence endpoint
   (`voice_audio.py:222`). The 0.6 s pause nod therefore could not fire in a
   real turn. With no partials, as from this backlogged companion, Reachy did
   not move at all. Without a wake word, a finished turn also flashed
   listening, idle, then thinking (`conversation.py:1293`).

## Replay

`turn_smoothness.py` drives the production `ReachyService` in companion mode
against the in-process fake companion over loopback WebSocket. The fake
companion keeps answering pings while its hearing lags:

* The user speaks for 5.8 s. Reachy's silence endpoint is 2.0 s, the
  companion-mode default.
* The companion sends no partials. Its own endpoint arrives 37.28 s after the
  user stops.
* Until then, its acknowledgement narration takes 3 s, and each sentence it
  speaks takes 3.6 s to start.
* Its narrator labels every line neutral.
* Reachy's own recognizer takes 2.232 s. That is the Whisper time logged on
  November 4 at 06:43:51 after a companion failure.
* Muse acknowledges 1.66 s after dispatch, reports "working" 0.3 s later, and
  answers 93.1 s after dispatch.

The answer has three sentences that carry a feeling. The 06:11 answer text is
not in the log. Run the replay with each revision's `src` on `PYTHONPATH`:

```sh
PYTHONPATH=src python -m bench.turn_smoothness --output bench/turn_smoothness-after.json
PYTHONPATH=<a7c37d3 checkout>/src python -m bench.turn_smoothness --output bench/turn_smoothness-before.json
```

Times are seconds after the user's last voiced chunk reached the microphone.
Timeline stages come from Reachy's own turn timeline, which counts speech end
in audio samples. The fake microphone delivers audio about 1.6 % slower than
the wall clock, so the replay's 37.28 s companion delay reads as 36.70 s.

| Stage | 06:11 log (`a7c37d3`) | Replay, `a7c37d3` | Replay, this branch |
|---|---:|---:|---:|
| First motion | 37.28, thinking pose (not logged) | 37.30, idle, then thinking | **0.60, nod** |
| Thinking pose | 37.28 (not logged); thinking motion 37.46 | 37.30 | 2.04 |
| Turn ends | 37.28 | 36.70 | **2.00** |
| Request words ready | 37.30 | 36.70 | 5.73 (1.5 s grace, then Reachy's recognizer) |
| Dispatch | 37.46 | 36.70 | 5.73 |
| Muse accepts | 39.12 | 38.36 | 7.39 |
| Spoken acknowledgement | skipped | skipped | **6.77**, Reachy's recorded "Hmm, good question." |
| Waiting cue audible | 50.76 | 47.19 | **14.04**, 6.6 s after Muse accepted |
| Waiting cue | 'The answer to "…" is taking a little longer.' | same | 'Muse is still working on "…".' |
| First answer audio | 131.23 | 130.02 | **99.04** |
| Sentence expressions | neutral ×4 | neutral ×3 | happy, sad, curious |

Before the fix, the user waited 37 s for any motion and 47 to 51 s for any
sound. After it, Reachy nods 0.6 s after the user stops and moves to its
thinking pose at the 2.0 s endpoint. It says a fixed opener 1.0 s after
dispatch and the contextual waiting cue 6.6 s after Muse accepts the request.
The answer starts 31 s sooner. All of that gain is the endpoint. Muse's own
91 s is unchanged. The raw results, including every conversation log line,
are in `turn_smoothness-before.json` and `turn_smoothness-after.json`.

For comparison, the turn at 06:43 on November 4 ran before `f11b32a`. Its
endpoint came 2.0 s after speech, and Whisper took 2.232 s. The quarantine
took the link down and moved the answer to Piper. The fix keeps that bound and
leaves the link and the answer voice alone.

## Commits

| Commit | Change |
|---|---|
| `db889cb` | The earlier of the companion's endpoint and Reachy's silence endpoint ends the turn. The companion gets 1.5 s (`ENDPOINT_WAIT_S`) after Reachy's endpoint, and then Reachy transcribes the same audio. The link stays up, and the stream id is aborted. |
| `5041989` | A streaming-only recognizer hands the audio to Muse. It no longer raises `AttributeError` and ends the conversation. |
| `1f59065` | A late narration gets a fixed opener instead of silence. |
| `0475983` | The waiting cue is due 6 s after Muse accepts and names the request. |
| `0d8b89d` | The waiting cue is synthesized 3 s before it is due. |
| `bf1677f` | A 0.6 s pause nods once, even without partials. |
| `ee1c7b9` | A finished turn goes straight from listening to thinking. |
| `14e4af2` | A neutral hint lets the sentence's words choose the expression. |
| `ec9b856` | A companion sentence with no audio for 6 s (`SPEECH_STALL_S`) restarts in Piper. |
| `03de88e` | The guide and protocol describe both bounds. |
| `11df46f` | The opener that replaces a late narration plays from Reachy's startup recording. |
| `b129408` | Reachy's streaming recognizer transcribes a late companion's turn, so commands, cancels and echo filtering still apply. |
| `3a941a5` | A companion endpoint that lands in the same feed as Reachy's endpoint keeps its words. |
| `a337949` | The user's last pause nods even after an earlier pause nod. |
| `4ee6e6a` | A waiting cue never names a topic cut mid-phrase. |
| `e043caf` | Two later waiting lines follow, 15 s apart, while Muse reports nothing. |

## Every wait and silence on the turn path

| Step | Bound or feedback at `a7c37d3` | Now |
|---|---|---|
| Wake to request | 10 s wake window; happy listening pose | unchanged |
| User pauses | no motion without partials; the partial pause nod could not fire | nod at 0.6 s quiet, once per pause |
| Reachy's silence endpoint to the companion's words | **unbounded while pongs arrive** | 1.5 s, then Reachy's recognizer (60 s limit) |
| Turn end to thinking pose | idle flash without a wake word | direct |
| Acknowledgement narration | 0.9 s; late meant silence | 0.9 s; late plays a recorded opener |
| Acknowledgement and answer speech on the companion voice | **unbounded while pongs arrive** (pongs count as speech progress) | 6 s without audio, then Piper for the rest of that reply |
| First waiting cue | 8 s after Muse accepts, synthesized when due | 6 s, synthesized 3 s ahead; two later lines 15 s apart while Muse reports nothing |
| Muse's answer | 180 s `REPLY_TIMEOUT_S` | unchanged; outside this repository |
| Sentence lines narration | 4 s `LINES_DEADLINE_S` | unchanged |
| Library expression clip | 0.05 s | unchanged |

## Open

* **Why the companion's hearing fell 37 s behind.** There is no companion log
  for November 5. The offline motion library (`9c5ba61`, `164d07e`, deployed
  the evening before) is a plausible GPU competitor, but no evidence confirms
  it. Three late endpoints in a row now rest companion hearing for 60 s, so a
  companion that stays slow stops costing the 1.5 s grace.
* **Spoken feedback in the fallback path comes at 6.8 s.** It adds up the
  2.0 s silence endpoint, the 1.5 s endpoint grace, Reachy's 2.2 s
  recognizer, and the 1.0 s acknowledgement grace. Motion covers the first
  2 s. Shortening any of those four waits needs production audio and timing
  data.
* **Thinking motion from the companion.** `MomentDirector._take` waits for a
  companion animation without a deadline when none is prefetched
  (`motion/moments.py:140`). The static thinking pose still appears at once.
* **A recorded opener is in Piper's voice.** When narration is late, the
  opener plays from Reachy's recording, and the answer stays on the companion
  voice. Measured loaded first-audio times (1.47 s maximum) say this is rarer
  than the incident suggests, but no production distribution exists.
* The 1.5 s endpoint and 6 s speech bounds come from bench measurements
  (`generated_motion_runs`, `voice_load_runs`). They are not production
  percentiles.
