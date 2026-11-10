# Music and dance

Design recorded before implementation in commit `3e3c2ed`, 2026-11-09, then
updated with implementation details and measurements. Base: `edbeb09`, branch
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

Core shapes:

```python
Beat(sample: int, period: float, index: int, energy: float)
AudioBlock(offset: int, pcm: float32[N], beats: tuple[Beat, ...])
RoomFrame(pcm: float32[N], captured_at: float, gap: bool)
_Session(source: str | Path | None, title: str | None, dance: bool,
         volume: float, started: float, deadline: float, cancelled: Event, ...)
MusicMix(pcm: float32[N], accepted: Callable[[], None])

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

`RoomFrame` is a tuple in the implementation. Session object identity and its
cancellation event replace a numeric generation. Each session holds at most
25 decoded blocks and 25 microphone frames. Queue overflow or a capture reset
resets room tempo estimates. Command serials separately prevent a delayed
folder lookup from starting after stop, hush, replacement, or disconnection.
`MusicMix.accepted` publishes beat times and consumed samples only after the
SDK accepts the PCM.

## Modules and clock

`motion/dance.py` owns streaming onset/tempo tracking and analytic safe dance
poses. Numpy RMS energy rises in 10 ms frames detect onsets, with a 240 ms
refractory period and a median of up to eight tempo intervals. The timeline
predicts poses between onsets using that tempo. Playback analyzes decoded PCM
ahead of the speaker. Beat indices group into inferred four-beat bars, not
claims about actual musical meter. Three patterns change every four bars with
a blend. Head bob, small body yaw and downbeat antenna accents scale with energy.
Attack/release take about .4 s. On interruption, freeze beat-driven movement
and release the last offset toward rest; there are no new beat pulses in turns.

`music_audio.py` owns PyAV decoding and a serialized software mixer. Decode
resamples to output-rate mono float32, bounds lookahead to about one second,
and uses network open/read deadlines. Only the latest requested session survives
replacement. A stalled read cannot submit audio after cancellation. The
decoder uses one codec thread, a three-second open timeout and a two-second
read timeout. Playback waits for 480 ms of decoded PCM or EOF before starting.
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
accepted time. Dancing releases when accepted music runs out, and stale onset
energy also decays. Elapsed status follows submitted
samples with the outstanding output lead subtracted, not decoder progress.
Natural EOF drains accepted audio then releases dance. The sink delay is an
estimate; real acoustic and motor phase need robot measurement.

`music.py` owns sessions, bounded room handoff, the decoder worker, output pump
and stop/hold policy. Hardware owns this controller after audio startup and
closes it before media shutdown. It runs no capture API. Conversation supplies
the tap and foreground holds. A held session continues playing at
`10**(-18/20) = .12589254` gain but has no beat-driven motion. Default volume
is .35, maximum .5, with output clipped to normalized PCM. Gain ramps take
40 ms to duck and 400 ms to start or restore. Unqualified AEC
paths cap music more conservatively at .2. These are digital gains, not SPL.

## Wake, speech and lifecycle

Wake inference continues during music. A separate music flag in WakeContext
requires stronger acoustic replay while playing, .35 versus the existing .25
fallback. Phrase verification remains in place. Vosk requires its native
verified phrase with confidence at least .9. Music does not set unknown-speech
blocking.
On admitted wake, duck and pause dance before listening gestures. Hold through
the wake window, active capture/turn, queued speech, and realtime work. Resume
only after all are clear. Speech ownership also holds independently, and a
notification scope holds before its attention wave through delivery.

Hush stops music synchronously and invalidates decode/motion generations;
sleep and stop-listening already call hush. Stop and shutdown cancel both modes.
A music stop flushes combined output when speech is absent. During speech it
removes music immediately from future mixed blocks and drains only the bounded
accepted tail, preserving the reply. This physical tail cannot be recalled
independently with the SDK. Music alone has 20-40 ms of output lead. Normal
speech pacing can leave up to 80 ms queued, plus the estimated 60 ms sink path.
Thus a music-only stop during speech may take about 140 ms to become inaudible.
Hush flushes both. Beat drive stops immediately and the last offset settles
over 400 ms. New play commands stamped
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
4096-byte guide and 8192-byte registration budgets. The guide is 3124 bytes;
registration is 6753 bytes including framing, or 7039 with the worst escaped
place label. Codec tests decode actual MP3, Ogg Opus, FLAC and WAV, plus WAV
over an ephemeral loopback HTTP server. Sink acceptance, stale commands,
nonblocking capture, shutdown and the actual motor loop run against local fakes.

Remaining risks include ambiguous beats, tempo changes, SDK differences, Pi
CPU contention and thermal behavior. Sustained music without sharp onsets can
produce no dance. Speech and other rhythmic sounds can produce onsets; the
tracker classifies neither music nor meter. Continuous conversation
holds dancing, including an open listening window. Dancing suppresses face
tracking and camera eligibility for wave wake, so acoustic wake is the way to
interrupt an active dance. No local benchmark
can verify four Pi cores under simultaneous wake, STT and vision. No local
test can verify speaker SPL, physical stop latency, motor comfort or acoustic
echo cancellation. Those remain explicit robot-only checks.

## Measurements

The reproducible benchmark is `bench/music.py`; `bench/music.json` contains
the numeric results. This run used arm64 macOS 26.4, Python 3.13.14, NumPy
2.5.3 and PyAV 19.0.2. CPU percentages divide process CPU seconds by represented
audio seconds, expressed as a fraction of one core. Synthetic PCM generation
and imports occur outside the timed sections. These offline costs exclude
realtime thread wakeups, the SDK sink and concurrent speech or vision work.
They are not Pi measurements.

| Work | Audio duration | CPU time | Wall time | One core |
| --- | ---: | ---: | ---: | ---: |
| 16 kHz tracker and 50 Hz dance sampling | 600 s | 0.3057 s | 0.3059 s | 0.0510% |
| 48 kHz tracker and 50 Hz dance sampling | 600 s | 0.3083 s | 0.3087 s | 0.0514% |
| WAV decode, 48 to 16 kHz resample and tracking | 60 s | 0.0473 s | 0.0473 s | 0.0788% |

Each tracker trial found 1200 onsets and sampled 30000 poses. The decode trial
produced exactly 960000 samples. The low measured cost supports keeping the
NumPy tracker; librosa is not imported or required by this feature. Pi CPU,
thermal limits and contention with wake, STT and vision remain unmeasured.

Wake replay used the real local sherpa-onnx 1.13.8 keyword model,
`sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01`, and an existing generated
2.085-second wake clip. Its SHA-256 is
`1f401bde31192d2c956f218ce0f54a05abbcce0663fe5c614090bd38cb5d3254`.
No speech verifier assets were present, so the production conservative replay
path was used. Music is a synthetic chord with 120 BPM bass pulses. SNR is
speech RMS relative to music RMS before clipping.

| Input | Accepted wakes | Replay floor | CPU time |
| --- | ---: | ---: | ---: |
| Clean wake clip | 1 | .25 | 0.0625 s |
| Wake plus music, 20 dB SNR | 1 | .35 | 0.0801 s |
| Wake plus music, 10 dB SNR | 1 | .35 | 0.0809 s |
| Wake plus music, 0 dB SNR | 0 | .35 | 0.0766 s |
| Music alone, 30 s | 0 | .35 | 0.4472 s |

At equal speech and music RMS the stronger threshold rejected the candidate.
This is a known wake sensitivity limit, not evidence of acoustic echo
suppression. One generated clip and synthetic accompaniment do not establish
false-accept or false-reject rates for real speakers, lyrics or rooms.

From the checkout, set `REACHY_PYTHON` to the Python interpreter in the
application environment, `WAKE_MODEL` to the model folder, and `WAKE_CLIP` to
the WAV clip. The benchmark command is:

```sh
uv run -q --no-project \
  --python "$REACHY_PYTHON" \
  --with 'av>=12' --with sherpa-onnx==1.13.8 --with sentencepiece==0.2.2 \
  env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD/src python -m bench.music \
  --wake-model "$WAKE_MODEL" --wake-clip "$WAKE_CLIP"
```

Omitting the two wake arguments runs the CPU trials without model assets.
The clip is not committed. No robot, microphone, speaker or external audio
service participates in this benchmark.
