# Companion stall, 2026-10-30

The 07:03 incident took 57.952 seconds from wake to first answer audio. The robot
waited 7.63 seconds before abandoning companion hearing, then spent 20.7 seconds
in local recognition. It sent the greeting to Muse and spoke a 4.3-second outage
notice. Sources are the supplied `incident-0703-robot.txt` and
`incident-0703-companion.txt`, against deployed commit `d3c15cf`.

## Failure and recovery

The companion stopped logging after motion preparation at 06:51:12. It stayed
alive in process state U, so launchd's `KeepAlive` had no exit to restart. A motion
event logged at 07:04:57 reports 199.120 seconds of work, including 198.291 seconds
in the planner. Concurrent MLX submissions are the likely underlying GPU failure,
as reproduced in [voice_richness.md](voice_richness.md). These logs do not contain
a native stack trace that proves the precise blocking call. The GPU scheduler
remains a separate change.

The serving process now has a watchdog thread independent of asyncio and the
engine executors. After a missed bound it writes the reason to stderr and calls
`os._exit(70)`, bypassing executor joins and Python shutdown hooks. The deployed
launchd configuration can then restart it.

| Watch | Limit | Basis |
|---|---:|---|
| Event-loop heartbeat | 10 s, checked every 0.5 s | Inference runs on workers; the loop should keep scheduling while they run. |
| Worker without progress | 30 s, checked every 0.5 s | Above the incident's normal cold motion maximum of 10.848 s, the 3.641 s hearing p95 in `generated_motion.md`, and the 2.938 s loaded voice sentence p95 in `voice_richness.md`. |

The timer starts when work executes, so a queued call does not time out while
another call progresses. Model loads and downloads finish before arming. Tokens
and speech chunks reset the worker timer. Motion can therefore take the 33-second
median measured under constant speech load, and a morning briefing can run for
minutes. Cancelling an asyncio waiter does not clear the running native call's
timer. A Python watchdog needs the interpreter to schedule its thread; it cannot
guarantee recovery from a kernel or extension failure that prevents that.

The old robot keepalive allowed 2 seconds before a ping, 3 seconds for its pong,
and the WebSocket library's default close wait before publishing link failure.
That last wait accounts for why the link-down warning came much later than the
first hearing fallback. The new settings are 0.5 + 1.25 + 0.1 seconds, or at most
about 1.85 seconds plus scheduling delay. Hearing also has a two-second first
response deadline, a two-second audio gap without partials, and a two-second
deadline after `finish`. A live socket with a stalled hearing engine is kept out
of routing until its pending endpoint arrives or a new handshake succeeds.
The stalled turn finishes locally from the retained audio. Reconnect flapping
does not create a new outage until a session has stayed healthy for 10 seconds.

## Local recognition

The supplied log is not enough to attribute all 20.7 seconds to one cause:

| Factor | What the deployed code establishes |
|---|---|
| Model | The incident used Whisper tiny.en. The worker requests CPU int8. |
| Threads | Two inference threads and one worker, already before this change. Native BLAS thread counts were inherited from the environment. |
| Audio | The dispatched WAV is 2.7 seconds. The local endpoint transcribes that same WAV, including recording silence. No entire wake window is substituted. |
| Beam | Short turns try greedy decoding. Rejected results restart with a 30-second encoder, beam 3, and temperature retries with up to five hypotheses. |
| Cold start | CLI startup loads weights and VAD before listening, but previously ran no inference or audio conversion. A dead worker can also reload on a later request. |
| Contention | The log records a 0.708-second microphone processing stall during replay, but no CPU profile or worker-stage timings. Face detection and Silero already limit their own inference threads. |

The worker now warms WAV conversion, VAD, the encoder, and a bounded decoder call
before its ready handshake. Its environment caps BLAS at one thread while keeping
two inference threads. An uncertain short result retries beam 3 on the existing
short encoding before using the full decoder. The full decoder no longer predicts
timestamps, which this app never consumes. Confidence, repetition, token limits,
and the full-waveform fallback remain. No model is replaced or downsized, and no
speech is cropped in production.

Logs now distinguish worker startup, WAV duration and conversion, VAD, short
encoding and decoding, and full fallback. They contain no recognized words.
The upstream [faster-whisper 1.2.1 decoder](https://github.com/SYSTRAN/faster-whisper/blob/v1.2.1/faster_whisper/transcribe.py)
provides the 30-second padding and temperature retry behavior described above.

### CPU measurement and CM4 estimate

`fallback_asr.py` runs the production subprocess and packet protocol with
faster-whisper 1.2.1, CTranslate2 4.8.4, CPU int8, and two inference threads.
PyAV 16.1.0 is used because PyAV 19 removed the `metadata_errors` argument used by
this faster-whisper release. No GPU model, companion service, or robot is used.
Each of the three recorded phrases is recognized five times as its original
8.2-second WAV and five times in a short speech window. Only the benchmark
removes corpus padding, retaining all VAD speech regions plus 250 ms margins.

```sh
uv run --no-project --with faster-whisper==1.2.1 --with av==16.1.0 \
  env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  python bench/fallback_asr.py --model /path/to/whisper-tiny.en --runs 5
```

The M3 Ultra CPU results, including IPC, WAV conversion and VAD, are in
[fallback_asr.json](fallback_asr.json). Startup took 0.612 seconds.

| Phrase | Short WAV | Short p50 / p95 | Original 8.2 s WAV p50 / p95 |
|---|---:|---:|---:|
| Reminder | 3.124 s | 65 / 73 ms | 224 / 252 ms |
| Correction | 4.116 s | 71 / 78 ms | 237 / 247 ms |
| Negation | 3.412 s | 66 / 71 ms | 226 / 234 ms |

Before warming WAV conversion and VAD, the first short window took 282 ms in the
same CPU experiment. After warming, every short window was below 78 ms. Baseline
and changed decoders returned identical text on all six windows. The reminder
still has a tiny.en recognition error, recorded in the JSON; these three phrases
are a regression check, not a general accuracy evaluation.

For a CM4 estimate, apply the 10 to 15 times M3-to-Cortex-A72 factor used in
[arrival.md](arrival.md) to the measured short-window p95, then allow a further
factor of two for other robot work. That gives roughly 0.7 to 1.2 seconds before
contention and 1.4 to 2.3 seconds with that allowance. The 2.7-second incident WAV
fits the same 500-frame short encoder as the reminder and negation windows.
This is an engineering estimate, not a Pi measurement: Whisper's int8 kernels
may scale differently from vision, and a full decoder fallback can exceed three
seconds. The incident audio and stage timings were not available, so neither an
exact explanation of its 20.7 seconds nor a guaranteed sub-three-second result
can be established without a permitted CM4 measurement.

## Local replies and silent fallback

When companion narration is unavailable or late, cached grammar replies handle
greetings, thanks, and "how are you". They respect `maybe_chat` and
`--local-chat off`; a greeting that includes a calendar question still goes to
Muse. Companion mode caches seven extra phrases at startup. Repeated greetings
reuse that audio and send no Muse request.

Reachy never announces an outage or voice switch. The current request
continues in the robot's own voice, repeating an interrupted line if needed.
Later turns also use the local fallback without a notice. The fallback is logged
once per outage, including when the companion repeatedly reconnects and drops.
After a healthy session, another outage can produce a new log entry.

## Verification

Behavior tests cover actual subprocess exit 70 for a blocked loop, cancelled
native work, and a cancelled Chatterbox stream; long loads, queued work, idle
workers, and progressing narration; silent sockets and working pongs with stuck
hearing; recovery only after an endpoint; retained local audio; cached small talk
without Muse requests; silent fallback and outage log deduplication; and short beam
retries without a second encoding. All model-related suite tests use fakes.
The existing `real` marker remains excluded from the full suite.

The original stall-fix full suite passed: **2,005 passed, 2 skipped, 6 deselected** in
704.32 seconds. This adds 18 behavior cases to the deployed baseline. The
separate CPU benchmark completed 30 recognitions; all six baseline comparisons
returned unchanged text.

The silent fallback update passed the requested full suite: **2,041 passed,
3 skipped, 6 deselected** in 776.79 seconds.
