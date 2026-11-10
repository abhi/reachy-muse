# Music and dance

Design recorded before implementation, 2026-11-09. Base: `edbeb09`, branch
`robot/music-dance`. Work stays local. No robot, Muse, launchd, or ports
8765/8766 are involved.

## Existing ownership

`conversation._capture_microphone` drains the SDK on one dedicated reader.
Its bounded handoff feeds wake detection and authorized turn capture. A room
music tap belongs immediately after that read, before speech admission drops
samples. It must only copy into a bounded queue, never run FFTs or open a mic.

`ReachyController._motion_loop` writes all motor targets every 25 ms. Rest,
breathing, speech sway, sentence accents, recorded/generated clips and gaze
compose there, with `clamp_to_limits` at `_write_vector`. Dance is another
bounded offset sampled there, never another motion writer. Explicit commands,
gestures, clips, listening and speech take priority.

`play_audio` is a speech API. It schedules sentence motion and sway using
accepted sample lengths plus `SWAY_LATENCY_S = .06`. `PCMPlayer` paces 40 ms
chunks with 40 ms lead. Music must not pass through speech analysis. A shared
output clock must include both music-only and mixed writes.

`_has_output` and `_input_blocked` describe speech. Counting music there would
disable wake and notification delivery for whole songs. `Notifications.deliver`
waves before acquiring the shared speech owner. `VisionBudget` observes capture
and wake coverage and sheds vision load; music must not mark it permanently busy.

Read motion/*, hardware.py, speech_playback.py, voice_audio.py, the microphone
and lifecycle paths in conversation.py, vision_budget.py, contract/{guide,commands}.py,
notifications.py and duplex_audio.py. The specified virtualenv has no importable
`reachy_mini`. Instead, inspected official SDK v1.9.0
[AudioBase](https://github.com/pollen-robotics/reachy_mini/blob/v1.9.0/src/reachy_mini/media/audio_base.py),
[GStreamerAudio](https://github.com/pollen-robotics/reachy_mini/blob/v1.9.0/src/reachy_mini/media/audio_gstreamer.py),
and [MediaManager](https://github.com/pollen-robotics/reachy_mini/blob/v1.9.0/src/reachy_mini/media/media_manager.py).
Those sources append PCM contiguously through one appsrc. There is no public
second mixing lane or DAC cursor. `play_sound` uses a separate playbin and
cannot supply the required PCM clock. Robot SDK parity remains unverified.

## Usage and data

Muse calls `reachy.music.play` with exactly one `query` or `url`, `dance=true`
by default and optional volume. Query is a case-insensitive substring of files
under owner `--music-dir`, default `~/Music`, or `shuffle`. Formats are mp3,
ogg, flac and wav. URL must be http(s) audio, without service scraping or
credential lookup. `reachy.music.stop` stops both modes; `reachy.music.status`
returns playing, title or source, and elapsed seconds. `reachy.dance` listens
to room music for 60 seconds by default, at most 600, stopping after about
8 seconds without rhythmic activity. Playback and room listening replace one
another. No track title, query, URL, path or PCM enters logs.

Core shapes, with bodies to be implemented:

```python
Beat(sample: int, period: float, index: int, energy: float)
AudioBlock(offset: int, pcm: float32[N], beats: tuple[Beat, ...])
RoomFrame(pcm: float32[N], captured_at: float, gap: bool)
MusicSession(generation: int, source: str | Path | None, title: str | None,
             dance: bool, volume: float, deadline: float)

BeatTracker.feed(pcm) -> tuple[Beat, ...]
DanceTimeline.offer(beats, source_offset, audible_start, rate) -> None
DanceTimeline.sample(now, eligible) -> float64[9]
MusicController.play(params) -> status
MusicController.listen(seconds) -> status
MusicController.stop() -> None
MusicController.hold(reason, active) -> None
MusicController.tap(pcm, captured_at, gap=False) -> None
MusicController.status() -> dict
MusicOutput.push_speech(pcm) -> accepted_start
```

## Modules and clock

`motion/dance.py` owns streaming onset/tempo tracking and analytic safe dance
poses. Numpy short-frame onset detection retains only recent intervals. It
predicts beats after lock for live room audio. Playback analyzes decoded PCM
ahead of the speaker. Beat indices group into inferred four-beat bars, not
claims about actual musical meter. Three patterns change every four bars with
a blend. Head bob, small body yaw and downbeat antenna accents scale with energy.
Attack/release take about .4 s. On interruption, freeze beat-driven movement
and release the last offset toward rest; there are no new beat pulses in turns.

`music_audio.py` owns PyAV decoding and a serialized software mixer. Decode
resamples to output-rate mono float32, bounds lookahead to about one second,
and uses network open/read deadlines. Only the latest requested session survives
replacement. A stalled read cannot submit audio after its generation ends.
File selection is confined to the owner folder, including resolved symlinks.
URL decoding restricts protocols to network audio, not local files. PyAV
diagnostics are captured and discarded; errors crossing the boundary are fixed
strings. No downloaded track cache persists.

The mixer uses one lock and one accepted-output tail. A 20 ms pump fills gaps;
speech pushes synchronously mix exactly the next music samples under the same
lock. Both advance `tail = max(now, tail) + count/rate`. This avoids a second
speech queue and keeps speech-only calls synchronous. Hardware returns the
accepted tail to PCMPlayer, which uses it instead of inventing another clock.
Speech sentence events and sway use that same accepted start. Music samples
are mapped to wall time only on successful sink acceptance:

`audible_beat = accepted_start + (beat.sample - source_offset)/rate + .06`.

Underflow emits no phantom source progress. New PCM anchors at the actual next
accepted time; stale dance energy decays. Elapsed status follows submitted
samples with the outstanding output lead subtracted, not decoder progress.
Natural EOF drains accepted audio then releases dance. The sink delay is an
estimate; real acoustic and motor phase need robot measurement.

`music.py` owns sessions, bounded room handoff, the decoder worker, output pump
and stop/hold policy. Hardware owns this controller after audio startup and
closes it before media shutdown. It runs no capture API. Conversation supplies
the tap and foreground holds. A held session continues playing at
`10**(-18/20) = .12589254` gain but has no beat-driven motion. Default volume
is .35, maximum .5, with output clipped to normalized PCM. Unqualified AEC
paths cap music more conservatively at .2. These are digital gains, not SPL.

## Wake, speech and lifecycle

Wake inference continues during music. A separate music flag in WakeContext
requires stronger acoustic replay while playing, initially .35 versus the
existing .25 fallback; real local replay will check this choice. Keep phrase
verification. Vosk uses its native verified phrase with a stricter confidence
floor. Never set unknown-speech blocking just because music is playing.
On admitted wake, duck and pause dance before listening gestures. Hold through
the wake window, active capture/turn, queued speech, and realtime work. Resume
only after all are clear. Speech ownership also holds independently, and a
notification scope holds before its attention wave through delivery.

Hush stops music synchronously and invalidates decode/motion generations;
sleep and stop-listening already call hush. Stop and shutdown cancel both modes.
A music stop flushes combined output when speech is absent. During speech it
removes music immediately from future mixed blocks and drains only the bounded
accepted tail, preserving the reply. This physical tail cannot be recalled
independently with the SDK. Target lead is 20-40 ms plus the estimated 60 ms
sink path; tests must report that limit honestly. New play commands stamped
before a hush, expired commands, and old sessions cannot resurrect playback.

The Wireless SDK path routes ALSA through the XMOS loopback. This repo's
`verified_echo_cancelled_input` additionally reads XVF3800 routing, echo enable,
four microphones and far-end reference controls. Other backends are not
assumed echo-cancelled. Qualification proves configuration, not residual echo
quality. Gain limits, stronger wake threshold and prompt ducking protect wake;
lyrics saying the exact wake phrase and loud room music remain acoustic risks.

## Design comparison

Independent candidates A and B are recorded in `/tmp/reachy-dance-a.md` and
`/tmp/reachy-dance-b.md`. Both traced the one-reader, one-motor-writer invariants.
A places output ownership in hardware; B starts with controller-owned lanes
and evaluates speaker-lock handoff. Select A's hardware-owned shared clock,
adapt B's full lifecycle holds and captured-context preservation. Replace A's
queued speech writer with synchronous mixing under one output lock. This
hides the same timing complexity with fewer queues and preserves call behavior.
Reject speaker-lock serialization because it pauses rather than ducks music;
reject a second SDK player because its timeline and flushing are independent.
Both candidates used the same available model family, so review diversity is
architectural rather than model-based. No design checkpoint was requested.
Cross-review at `/tmp/reachy-dance-review.md` agreed with this choice and
identified lazy installation during a reply as a race. Construct the output
owner eagerly at hardware startup, but start workers only on demand. Its
speech-only path remains a synchronous, unchanged PCM push.

## Verification and remaining risks

Separate commits cover tracker/motion, playback/coexistence, then contract/CLI/docs.
Literal tests cover 90/120/150 BPM clicks, beat and pose timing, energy, limits,
pattern changes, silence and duration, mixing/duck ratios, stop/replacement,
wake/hush/sleep/notification, single reader and payload validation. Preserve
4096-byte guide and 8192-byte registration budgets, updating literal sizes.
Run the exact requested full pytest command. Run a reproducible local CPU
benchmark and synthetic music plus a stored wake clip through the real local
detector if its assets are available. Record numeric results below after runs.

Outstanding before implementation: beat ambiguity and tempo changes; decoder
stall/replacement races; speaker flush granularity; command arrival during a
turn; no-wake continuous-listening configurations; background camera movement;
SDK differences; Pi CPU contention and thermal behavior. No local benchmark
can verify four Pi cores under simultaneous wake, STT and vision. No local
test can verify speaker SPL, physical stop latency, motor comfort or acoustic
echo cancellation. Those remain explicit robot-only checks.

## Measurements

Pending implementation and local benchmark. Pi measurements are prohibited by
the task's no-robot constraint and will not be invented.
