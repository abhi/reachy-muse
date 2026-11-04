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
