# Voice turn, 2026-11-03

Starting point: deployed `origin/main` at `6f827a2`, branch `perf/turn-2`.
All new measurements use fake engines, offline fixtures, and ephemeral loopback
WebSockets. They do not measure live GPU throughput, contact Muse, or control
the robot. Run the experiments with `PYTHONPATH=src:tests python -m bench.turn2`.

## Incident evidence

The supplied workout log records a companion ping timeout 2.358 seconds after
wake, a 4.839-second Whisper worker load, and a 2.000-second silence endpoint.
The question's first Muse text arrives at 39.752 seconds after speech end;
reply completion is 41.771 seconds and first answer audio is 45.092 seconds.
The companion reconnects before local recognition finishes. The aggregate
72-hour counts supplied with the incident are 25 companion drops, including 18
ping timeouts, and 11 handshake timeouts. Muse has a separate 36 drops and 50
HTTP 502 rejections. These aggregate counts are observations, not replay results.

## Hearing work off the event loop

`VoxtralHearing.open()` constructed `_Utterance` on asyncio's thread. That
constructor calls `create_streaming_session`, outside both the engine executor
and its watchdog. Feed and final inference already used the worker. Session
construction, PCM conversion, event extraction and release now use that same
single worker. A stream finished without audio returns an empty transcript.

`python -m bench.turn2 engine` drives the production server and hearing adapter
with a fake utterance that sleeps 300 ms in construction and another 300 ms in
feed. A real WebSocket exchanges pings every 10 ms while the calls run.

| Measurement | Before | After |
|---|---:|---:|
| Maximum gap between successful pongs | 313.12 ms | 17.77 ms |
| Pongs during the 300 ms constructor | 0 | 28 |
| Pongs during the 300 ms feed | 28 | 27 |
| Partial and endpoint | Should I work out today? | Should I work out today? |

The logs do not contain a native stack trace or an idle-versus-warm model
comparison. This fixes a reproduced scheduling defect; it does not prove which
native call stalled in the live incident. No periodic GPU warm-up is added
without evidence of an idle penalty. Existing startup inference warm-up remains.

## Transport tolerance and operation deadlines

Client and server keepalives now use a 5-second interval and 10-second pong
timeout. Opening and protocol handshakes allow 10 seconds. Hearing still has its
2-second progress and endpoint deadlines. Speech has a 2-second chunk deadline;
classification, narration and animation have 1, 4 and 8-second response limits.
The existing shorter caller deadlines still apply. An expired operation fails
that operation without declaring the socket dead. The companion's independent
10-second loop and 30-second worker watchdogs still supervise genuine hangs.

`python -m bench.turn2 jitter` pauses reads on a real loopback WebSocket just
before a scheduled client ping. Burst lengths are 0, 0.5, 1, 1.5, 2, 3, 4, 6
and 8 seconds. Network times are scaled by 0.05; no model work runs. Run against
the parent revision or set the replay's imported client constants to 0.5/1.25
for the old configuration. Each burst uses a fresh connection so reconnects
cannot hide a drop.

| Replay result | Before, 0.5/1.25 s | After, 5/10 s |
|---|---:|---:|
| Link drops across nine bursts | 6 | 0 |
| Smallest burst that drops the link | 1.5 s | none through 8 s |

This compares injected jitter, not a prediction of the next 72 hours. Server
keepalive uses the same new bounds. Tests also prove that a stalled operation
expires while the connection remains usable for a later speech request.

## Reconnect while idle

The link's existing background owner keeps reconnecting independently of turns.
Its retry delay now starts at 0.2 seconds and caps at 5 seconds, instead of
1 and 30 seconds. New turns can use a welcomed link after a 0.1-second grace
that rejects immediate post-welcome failures. The 10-second health window still
controls outage logging and backoff reset; it no longer withholds a working
companion from routing.

`python -m bench.turn2 reconnect` aborts a loopback connection while idle, waits
for production routing to become ready, and requests a fake narration. Recovery
fell from **11002.89 ms to 302.03 ms**. Both return the literal line
"Should I work out today?". The whole-turn replay also disconnects while the
user speaks and reconnects before their utterance ends. Retained local audio
produces exactly one unchanged Muse request and one answer; a recovered link
can voice that answer. Immediate-failure flapping still produces one outage log.

## Stream the first complete sentence

Plain companion replies now release their first complete, unformatted sentence
while Muse streams the remainder. Sentence boundaries retain the existing
abbreviation, decimal, quote and control-markup rules. Incomplete, formatted,
oversized and structured openings keep the final narration path. The final
reply must extend the committed prefix. An unheard mismatch replaces queued
speech; a mismatch after speech starts drops that message's remaining speech.
Audio already heard cannot be retracted. The final remainder still uses the
existing narrator, including condensation for long or formatted tails.

`python -m bench.turn2 answer` runs the actual conversation, reply tracker and
speaker queue. Muse's final message follows its first sentence by 2 seconds;
fake TTS takes 250 ms per line and emits 500 ms of audio. First sentence to
answer audio fell from **2252.71 ms to 252.08 ms**. Both versions speak exactly
"A light workout is a good option." and "Take a rest day if you feel worn out."
The post-change 1500 ms gap is waiting for the second sentence's final text in
this fixture. It is not synthesis overhead. The live 0.6-second target still
depends on actual TTS and transport performance, which this offline run does
not establish.

## Pipeline companion speech

Companion speech now implements the existing bounded preparation path. The
speaker owns one playing sentence and one prepared sentence; cancellation
closes both producers, and preparation retains the usual local voice fallback.
`python -m bench.turn2 pipeline` supplies both final sentences together, with
250 ms synthesis and 500 ms playback per sentence. The inter-sentence gap fell
from **301.95 ms to 50.74 ms**. First audio remained 252.38 versus 251.20 ms.
The remaining gap is the existing 50 ms output drain allowance.

## Avoid introducing an uninterrupted answer

A slow answer now names its original topic only if another accepted utterance
arrived while it was pending. The late-answer introduction still disambiguates
intervening talk. Waiting on a single question no longer adds a spoken "About"
line before the answer.

`python -m bench.turn2 late-answer` marks the pending request 32 seconds old,
without actually waiting or sending anything externally. With streaming and
lookahead already enabled, first sentence to answer audio fell from
**802.60 ms to 252.11 ms**. Only the two answer sentences are spoken. A separate
behavior test retains the contextual introduction after intervening talk.

## End the fallback capture sooner

The companion-mode VAD cap, also inherited by local fallback, is 1.2 seconds
instead of 2.0. Explicit silence overrides and the standalone robot's 0.8-second
default remain. A Smart Turn endpoint still ends capture immediately; a test
delivers one at 0.52 seconds of silence under a 2-second cap and receives the
complete unforced transcript on the next poll.

`python -m bench.turn2 endpoint` feeds the existing WAV fixtures through the
production recorder with WebRTC VAD. Times below are seconds from clip start.
Every case produces exactly one turn. The one-second correction pause is why
the companion fallback uses 1.2 rather than 0.8 seconds.

| Recording | Before endpoint | After endpoint | Retained WAV before / after |
|---|---:|---:|---:|
| Reminder, including "not at nine" | 5.00 | 4.20 | 4.88 / 4.08 |
| Correction, including "instead of seven" | 6.20 | 5.40 | 6.08 / 5.28 |
| Negation | 5.26 | 4.46 | 5.14 / 4.34 |
| Reminder with a one-second pause before correction | 6.00 | 5.20 | 5.88 / 5.08 |
| Reminder with a one-second pause after a short fragment | 6.00 | 5.20 | 5.88 / 5.08 |

All five lose only 0.8 seconds of trailing silence. These fixtures establish
capture continuity, not recognition accuracy on arbitrary pauses. Endpoint,
CLI, audio and wake-handoff checks: 187 passed.

## Warm Whisper before listening

`LocalHearing.start()` now starts any transcriber with a startup method, including
batch Whisper. Previously it called only streaming recognizers. The microphone
already awaits this method before accepting turns, and failover already calls
the local hearing startup. The worker's existing ready handshake includes model,
encoder/decoder, WAV conversion and VAD warm-up, and its process remains loaded
until shutdown or a failed/cancelled recognition.

`python -m bench.turn2 whisper` uses the production Whisper process owner and
packet protocol with a fake child that takes 300 ms to become ready and 20 ms
per transcript. Startup changes from 0 to **324.45 ms**; first fallback changes
from **341.62 ms to 24.36 ms**. The next transcript takes 23.25 ms in the same
process. Both texts remain "Should I work out today?". The incident's 4.839-second
load is moved before microphone readiness by this fix, not measured anew here.
Startup, transcription and related checks: 108 passed, 1 skipped.

## One contextual waiting line

The default waiting cue now occurs at 8 seconds after acknowledgement and names
the request's topic. There is no rotating filler or spoken generic working or
responding phase. After the first cue, further speech requires a new public
frame or a concrete backend activity, with the existing cadence, deduplication,
attribution and total cap. With no usable request text, waiting stays silent.
Thinking motion continues. Validated progress is spoken directly, so a narrator
cannot replace it with generic filler or delay it with another model call.

`python -m bench.turn2 progress` replays a 32-second wait with only Muse's generic
working phase. Before: two lines at 5 and 20 seconds, "Muse says it's working
on your request." and "Still on it." After: one line at 8 seconds, "Your workout
answer is taking a little longer." There is no further speech through 32
seconds. Behavior tests also retain actual new tool reports after that cue.
