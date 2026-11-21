# Companion hearing after the November 20 stall

The November 20 hearing stall kept the connection alive while its model worker
stopped progressing. The 30-second watchdog restart then added about 42 seconds
of model loading. The incident lost about 76 seconds of companion hearing.
Paging under memory pressure is the leading explanation, but the available logs
do not distinguish it from a GPU hang. See [the earlier stall report](companion_hang.md),
[concurrent GPU failures](voice_richness.md#open-gpu-hangs-under-concurrent-load),
and [the deadline audit](choppy.md#deadline-audit).

## Recovery without restarting healthy engines

The server checks the hearing worker every 100 ms. After three seconds without
progress, it ends streams that have supplied audio with `error{overloaded}`.
New `hear_open` requests get the same error until the worker progresses or
returns. This check is shared across connections through the process watchdog.
It leaves the socket, speech, narration, and animation available.

Three seconds allows more than six times Voxtral's configured 480 ms streaming
delay. It is an initial service budget, not a measured upper bound on inference.
Progress includes completed model stages and text deltas, so a long utterance,
silence, or several seconds of queued work do not themselves indicate a stall.
The soft error arrives within about 3.1 seconds of the last worker progress,
plus event-loop scheduling delay. A slow native stage with no observable
progress can trigger local hearing even if it eventually completes.

The hearing hard limit is 120 seconds without progress. This gives a cold call
four times the former limit to recover while Reachy uses local hearing, avoiding
another model reload merely because paging took longer than 30 seconds.
The event-loop limit remains 10 seconds, and other workers remain at 30 seconds.
Model loading still completes before the watchdog is armed. Cancellation of an
async waiter cannot hide a native call from the hard watchdog.

The robot already treats OVERLOADED as a per-stream failure. It now checks an
open stream's failure before consuming queued speech-end events, including in
command mode. Previously ended turns remain deliverable. Three consecutive
hearing failures rest companion hearing for 60 seconds, retain audio for local
hearing, and leave other companion capabilities available.

## Abandoned work

Protocol v1 already has `hear_close` with `how=abort`. Wake verification already
sends it when its waiter is cancelled, so no new message or version negotiation
is needed. A peer that sends no abort still gets normal finish, disconnect, and
stall handling; the server cannot infer an otherwise unreported abandonment.

Each model session has a cancellation event. Queued calls are cancelled and
model boundaries check the event before starting more work. An abandoned stream
cannot start Voxtral feeding or a Parakeet final pass after that check. A native
call already executing cannot be preempted. Its next model stage is skipped,
and array release stays on the hearing worker without delaying stream cleanup.

## Residency and evidence

After every engine load and warm-up finishes, startup clears free cached
allocations, sets MLX's wired limit to `max_recommended_working_set_size`, and
sets the free-cache limit to 256 MiB. The cache bound leaves room for reusable
scratch buffers without retaining the startup allocation peak indefinitely.
Both byte limits appear once in `MLX memory limits: wired=... cache=... bytes`.

The pinned `mlx_lm` context saves the previous wired limit, sets the recommended
limit, and restores the saved limit. After startup, both values are the
recommended limit, including across overlapping calls. No dependency patch or
narrator behavior change is needed. The test executes the installed context's
Python code against a fake `mx`, without importing its GPU runtime.

Hearing calls over one second log total time, queue time, and time in model
stages. The labels distinguish stream creation, Voxtral feed and decode, VAD,
turn detection, Parakeet features, and final decoding. The hard watchdog writes
all Python thread stacks and cumulative minor and major page-fault counts to
stderr before exit 70. The new diagnostics contain no audio, transcripts,
prompts, token text, or local variable values. Fault counts and stacks provide
evidence for the next incident; cumulative counts alone do not prove paging
caused it.

## Verify on the Mac after deployment

Use the deployed checkout and its test environment for these fake-engine tests:

```sh
PYTHONPATH=src python -m pytest -q -p no:cacheprovider \
  tests/companion_server/test_hearing_cancellation.py \
  tests/companion_server/test_hearing_deadline.py \
  tests/test_failover.py tests/test_companion_client.py \
  tests/companion_server/test_memory.py \
  tests/companion_server/test_hearing_timings.py \
  tests/companion_server/test_watchdog.py
```

- **Cancellation:** The cancellation tests send real protocol aborts through the
  companion client and server on ephemeral loopback ports. They assert that
  abandoned wake clips, streams, and disconnected sessions perform no remaining
  model stages, including Parakeet final decoding.
- **Fast fallback:** The deadline test holds a fake hearing worker, advances its
  clock to three seconds, and checks the exact OVERLOADED target. It checks
  refusal, continued animation, and admission on progress before the blocked
  call returns. The failover tests check local audio after three errors and a
  companion retry at the 60-second cooldown boundary. During a natural live
  stall, look for `hearing overloaded` and the robot's hearing-rest message.
  A recovering call below 120 seconds should keep the same companion PID.
- **Residency:** Confirm one startup memory-limit line after all engine-ready
  lines. Compare `vmmap -summary "$COMPANION_PID"` before and after a typical idle
  interval, then after normal narration and classification. Inspect resident
  and compressed memory and the next hearing call's stage timings. The fake
  memory test separately proves that per-call restoration preserves the limit.
- **Forensics:** The watchdog test uses a fake clock and injected exit to check
  the 120-second threshold, all-thread dump, page-fault counts, and ordering.
  The timing test checks exact content-free output. If a live hard stall occurs,
  retain the reason, thread dump, and fault-count line before the restart.
  Do not induce a live GPU stall to exercise this path.
