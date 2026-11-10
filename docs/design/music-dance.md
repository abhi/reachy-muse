# Music and dance

Redesign recorded before implementation on 2026-11-09, after `f638b94`.
Base remains deployed `edbeb09`; branch `robot/music-dance`. All work and tests
stay local. No robot, Muse, launchd, deployment or ports 8765/8766.

## Ownership and caller behavior

Keep playback selection, the streaming BeatTracker, the shared MusicOutput
accepted sample clock, PCM mixing, wake/turn/notify/hush holds and the single
microphone reader and motor writer. The commands keep their existing shapes.
`reachy.music.status` adds title, artist, album and genre when known, plus music
music admission, energy and character. Missing identity stays unknown. Titles,
artists, queries, URLs, paths, fingerprints and API keys never enter logs.

`conversation._capture_microphone` only copies into the bounded room queue.
`ReachyController._motion_loop` samples one bounded dance offset every 25 ms,
composes existing layers, then calls `clamp_to_limits` at `_write_vector`.
Explicit motions, listening, speech and holds continue to take priority.

## Data and module boundaries

```python
AudioFeatures(sample: int, loudness: float, bass: float, snare: float,
              treble: float, harmony: float, chroma: tuple[float, ...],
              timbre: tuple[float, ...], flatness: float, tonal: float,
              period: float, phase: float, regularity: float,
              music: bool, character: str)
AudioBlock(offset: int, pcm: float32[N], beats: tuple[Beat, ...],
           features: tuple[AudioFeatures, ...], plan: SongPlan,
           metadata: TrackMetadata, epoch: int)
Section(start: float, end: float, family: int, occurrence: int,
        chorus: bool, energy: float, build: float, drop: bool,
        period: float, character: str, beat_origin: float)
SongPlan(sections: tuple[Section, ...], complete: bool)
TrackMetadata(title: str | None, artist: str | None,
              album: str | None, genre: str | None)

AudioAnalyzer.feed(pcm) -> tuple[AudioFeatures, ...]
StructureTracker.feed(features) -> None
StructureTracker.plan(complete=False) -> SongPlan
DanceTimeline.offer_features(features, offset, audible_start, rate, plan) -> None
DanceTimeline.sample(now, eligible) -> float64[9]
MusicController.signatures(source: MotionSource) -> async long-running task
```

`music_analysis.py` owns the FFT, band envelopes, chroma/timbre descriptors,
classification and character. `music_structure.py` owns compact self-similarity,
section families, build/drop evidence and section intensity. `motion/dance.py`
owns continuous feature-to-pose mapping, timed signatures and smoothing.
`music_audio.py` owns decode, tags and transport. `music_stream.py` handles ICY
framing; `music_http.py` bounds header, chunk and body reads with cancellation
and monotonic deadlines. `music.py` owns session cancellation, acceptance and worker lifetimes.
`music_identity.py` owns optional fingerprint collection and lookup.

The robot does all audio analysis and pose sampling. The Mac companion only
looks up occasional signature moves through existing `generate(intent, duration)`.
No PCM or song identity is sent to the companion for dance. Signature requests
contain a bounded character and tempo intent. The existing ANIMATE wire schema
and ordinary expression lookup remain backward compatible.

## Continuous motion

Analyze Hann-windowed audio every 40 ms, with a roughly 64 ms window. Features
carry the final source sample of their hop so fragmented decoder blocks cannot
lose an event that was discovered in the next block. Keep original BeatTracker
for tempo/phase; feature envelopes continue even without confident beats.
Bands are bass 35-180 Hz, middle 180-2500 Hz and high above 2500 Hz.
Positive band flux distinguishes middle onsets from sustained harmony.
Absolute loudness sets amplitude, so normalization cannot turn a quiet intro
into a loud dance. Silence produces zero target.

Bass energy drives alternating body yaw. Snare-like middle onsets drive z/pitch
bob. High-band flux drives opposed antenna twitches. Sustained tonal energy
drives slow roll/head yaw and small x/y lean. Mood is a coarse signal-derived
character: calm sway, upbeat bounce or head-nod groove. Character changes axis
gain and pace; it never overrides which audio band drives an axis. Fixed patterns
remain a low-amplitude fallback only when beat evidence is ambiguous and music
is admitted. A kick-only fixture must move body yaw, and a high-only fixture
must move antennas, without borrowing unrelated axes.

The final target is capped to [2, 2, 3, 5, 7, 4, 14, 14, 9] mm/degrees.
Three cascaded exponential stages preserve the convex position bounds across
retarget, stop and holds. Exact integration uses the elapsed monotonic interval.
For each axis, continuous jerk is bounded by `8 * cap / tau**3`, with tau=.12 s;
velocity and acceleration are bounded by `2 * cap / tau` and
`4 * cap / tau**2`. Tests exercise finite differences and analytic derivatives.
These are bounds on the additive dance layer. The final composed clamp and
other priority layers require physical verification for motor comfort.

## Song structure and timing

Playback still primes after 480 ms of decoded PCM or EOF. Analysis never gates
this threshold. A single separate cancellable worker scans local files as fast
as its duty budget permits, discards PCM and publishes compact section plans
while scanning and at EOF. Replacements coalesce; never create a scan thread
per command. Playback has its own bounded worker, so full analysis cannot block
its decoder. A scan may finish after playback starts, in which case only future
accepted frames use its result.

Aggregate one-second chroma and normalized spectral-band timbre. Compare nearby
aggregates for novelty and section boundaries, and compare section centroids
for recurring families. Bound rolling history to 120 s and family prototypes
to 64; compact long-file summaries rather than retaining PCM or every FFT.
A repeated energetic family is a chorus candidate. Full-file recurrence marks
its first occurrence too. Repeated sections keep the same signature/family,
with bounded growth per return. Rising energy across several seconds adds
tension; an abrupt energy rise after a quieter passage marks a drop. Drops get
the largest allowed gain. The complete plan fades the outro over its last four
seconds. These are musical heuristics, not authoritative verse/chorus labels.

Streams permit up to three seconds of decoded lookahead and keep a 120 s past summary,
with no second network decode. Live stations may provide only the initial 480 ms
lead, so a four-frame energy preview detects drops in about 160 ms. The companion
task prepares the current character and likely energetic character before a peak
is certain. A test paces the producer against a real 40 ms audio clock and proves
that the cached move and drop preview precede playback reaching that drop.
Room mode has only captured past audio and online repetition detection. Gaps
reset classification, timing, recurrence and identified song. Source epochs keep
pending identity/companion responses from restoring discarded state.

Only `MusicMix.accepted` publishes frames and metadata onto audible time.
`audible_feature = accepted_start + (sample - source_offset)/rate + .06`.
Underflow produces no source progress. Rejected sink writes terminate the music
session, including rejection during speech, rather than retrying consumed data.
A new plan cannot change already accepted features. Natural EOF drains accepted
samples and releases movement; stop and replacement discard future features.

## Companion signatures

Peak sections request a four-beat move before their accepted start.
A conversation-owned async task permits one outstanding dance request and a
small per-session cache, with at most eight request keys. It skips foreground holds. Library lookup accepts
canonical dance intents matched to character/tempo, with aliases to existing
expression variants when a dance-specific library has not been built.
The robot validates finite nine-axis poses, centers them as offsets, caps them,
and time-warps their normalized progress over an integer beat duration. Repeated
choruses reuse cached frames. The same audio bands continue modulating the move.

An absent, saturated or slow companion leaves band-driven motion running.
A late result is usable for a later peak only, never inserted mid-move.
The measured beat origin survives section boundaries. Section-relative
whole-beat counts repeat the slow sway shape without shifting fractional phase. ICY epochs retain the
current and next song's cached moves separately while PCM is buffered. ANIMATE
has no cancel message and canceled requests occupy a connection slot until the
remote answers; keep one request alive instead of repeatedly timing out/retrying.
Stop/replacement discards results by session identity and analysis epoch, and conversation shutdown
cancels the task.

## Music admission and optional identity

Use deterministic spectral evidence instead of adding an unmeasured ONNX model
and assets. YAMNet is a candidate for later evaluation, but this checkout has no
Pi timing or validated ONNX artifact for it. Reuse tonal concentration, chroma
stability, spectral flatness, sustained energy and multi-band tempo regularity.
Room admission requires sustained harmonic content and repeated rhythmic or
polyphonic evidence over several seconds. A periodic single tap is insufficient.
Close admission on speech/noise evidence, silence or stale capture. Only admitted
music refreshes the eight-second room timeout. A room request alone must not
suppress gaze or claim dancing. Playback speech/noise also produces no dance.
Admission starts after at least 60 analysis frames plus ten consecutive positive
frames, about 2.76 s. This conservative gate also applies to playback, so sparse
or percussion-only tracks can remain still. Isolated-stem mapping tests supply
admitted features; separate tests assert that those stems cannot open room
admission. Test generated music, isolated stems, single taps, noise and the repository's
recorded speech fixtures. A broad room/TV corpus remains a robot qualification
requirement; spectral heuristics can miss sparse music or reject vocals.

PyAV reads file container and stream tags. PyAV 16.1 and current main copy
container metadata at open, so live ICY titles need a bounded HTTP reader that
requests `Icy-MetaData: 1`, strips fragmented metadata blocks, and supplies audio
to PyAV. Test actual local ICY audio decoding and changes. Restrict redirects to
HTTP(S), preserve cancellation/read deadlines, and discard transport diagnostics.

`--room-song-id` defaults off. `--acoustid-api-key` or `ACOUSTID_API_KEY` supplies
the owner's key. Only an active requested room dance collects a bounded admitted
music window. Chromaprint runs locally via fpcalc, with PCM through stdin and
no temporary audio file. At most one lookup is outstanding; use HTTPS POST for
fingerprint, duration, client key and requested result fields. No PCM, transcript,
location or device identity leaves the robot. This is optional best-effort
recognition: AcoustID's recording fingerprints may not match short noisy excerpts.
Missing fpcalc/key, cancellation and service failure skip without affecting dance.

## Budget and verification plan

Assume four Pi cores, reserving at least 3.5 for wake, STT, vision and operating
system work. Target sustained live music work below 10% of one Pi core; full-file
background scanning may use another 15% of one core. Bound scan CPU by sleeping
against thread CPU time, with a startup yield until playback is primed. Measure
Mac process CPU/audio duration and first-480-ms decode latency; do not extrapolate
Mac speed into a claimed Pi result. Keep rolling state and full-file compact
plans below 32 MiB, and include full analysis plus classification in benchmarks.

Each requested part gets its own commit. Tests use literal expectations for
band isolation, quiet/drop scaling, different-song trajectories, A-B-A-B family
reuse, build/drop/outro, companion failure and beat retiming, metadata updates,
speech/tap rejection, stop/replacement and position/jerk bounds. Run the requested
full suite and review the resulting integration twice before finalizing.

## Design comparison

Two independent candidates and a cross-judge reviewed the existing code. Choose
A's end-anchored accepted features and motor-owned filter; add B's compact family
map, single companion request and explicit lifecycle limits. Both initially
mapped some instruments to the wrong axes, corrected above to the user's mapping.
Reject a full precomputed pose score because late analysis would need to revise
past movement. Reject full analysis in the bounded playback decoder because the
queue prevents it reaching later sections. Preserve the shared output architecture.

## References

- [PyAV metadata snapshot](https://github.com/PyAV-Org/PyAV/blob/v16.1.0/av/container/input.pyx)
- [FFmpeg ICY options](https://ffmpeg.org/ffmpeg-protocols.html#http)
- [YAMNet model](https://github.com/tensorflow/models/tree/master/research/audioset/yamnet)
- [AcoustID lookup protocol](https://acoustid.org/webservice)
- [Chromaprint stdin and format options](https://github.com/acoustid/chromaprint/blob/master/src/cmd/fpcalc.cpp)

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
through the three-stage filter. A static step releases about 65% in
400 ms and over 98% in one second. New play commands stamped
before a hush, expired commands, and old sessions cannot resurrect playback.

The Wireless SDK path routes ALSA through the XMOS loopback. This repo's
`verified_echo_cancelled_input` additionally reads XVF3800 routing, echo enable,
four microphones and far-end reference controls. Other backends are not
assumed echo-cancelled. Qualification proves configuration, not residual echo
quality. Gain limits, stronger wake threshold and prompt ducking protect wake;
lyrics saying the exact wake phrase and loud room music remain acoustic risks.

## Current measurements and review

`bench/music.py` and `bench/music.json` record this redesign on arm64 macOS 26.4,
Python 3.13.14, NumPy 2.5.3 and PyAV 19.0.2. CPU percentage divides process CPU
seconds by represented audio seconds; 100% means one core. Synthetic generation
and imports are outside the timed sections. SDK writes, scheduler wakeups,
concurrent STT/vision and physical actuators are excluded.

| Work | Audio | CPU | Wall | One core |
| --- | ---: | ---: | ---: | ---: |
| 16 kHz FFT, classifier, structure and 50 Hz motion | 600 s | 2.4378 s | 2.4393 s | 0.4063% |
| 48 kHz FFT, classifier, structure and 50 Hz motion | 600 s | 2.6981 s | 2.6997 s | 0.4497% |
| WAV decode, resample and complete analysis | 60 s | 0.2364 s | 0.2365 s | 0.3940% |

Each live trial processed 15,000 spectral frames, admitted 14,932, found 1,200
onsets and sampled 30,000 poses. The decode trial produced exactly 960,000
samples and had 480 ms of PCM ready in 12.496 ms. An earlier cold run
measured 302.101 ms for that prefix. These are two local observations, not a
startup percentile or a Pi guarantee. The full 60-second scan completed in
0.2365 s without CPU throttling. Production's separate file worker applies its
15%-of-one-core duty budget and starts only after playback primes.

The assumed Pi budget remains 10% of one core for live music and another 15%
for background scanning, leaving at least 3.5 of four cores for other work.
The Mac result has margin against that target, but cannot establish Pi timing,
thermal behavior or wake/STT/vision contention. No new model assets are required.

Two fresh review rounds covered both old plumbing and new analysis. Fixes include
failure during mixed speech canceling the session, timestamp-correct filter
integration, measured beat origins, live drop preview and advance signature
preparation, bounded slow HTTP headers/chunks, separate stream identity epochs,
room-gap invalidation and releasing fingerprint audio after work ends.
The exact requested full suite passed on the final implementation: 3,797 passed,
7 skipped in 343.64 s with six workers and the 180-second per-test timeout.
The guide is 3,364 UTF-8 bytes, below its 4,096-byte limit. Decoder and ICY
checks also pass on the robot's pinned PyAV 16.1.0, 26 focused tests.
Local tests cover the actual decoder, local HTTP/ICY, socket-pair drip behavior,
recorded speech, a paced live producer, the fake hardware motor loop, fingerprint
payload contents, and literal structure, motion and jerk expectations.

Real-robot checks remain acoustic/motor phase including filter delay, composite
joint comfort and limits, speaker loudness and physical stop latency, wake/AEC
with lyrics and overlapping speech, sustained Pi CPU/thermal behavior under
wake/STT/vision, room classifier accuracy on music/conversation/TV/tapping, and
optional Chromaprint identification of noisy room excerpts. No local result
qualifies those physical behaviors.

## Previous implementation measurements

The previous benchmark results are retained in `bench/music.previous.json`.
The following numbers describe the superseded beat-pattern implementation. This run used arm64 macOS 26.4, Python 3.13.14, NumPy
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
