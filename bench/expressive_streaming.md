# Expressive streamed speech

Streamed sentences use local expression rules and the existing recorded clips,
static poses, and PCM sway. Neither speech synthesis nor playback waits for a
motion model. Explicit narrator or Muse expressions still take precedence.

## Audio owns sentence boundaries

The playback owner prepares a `SentenceMotion` before requesting speech. It
passes that value with the first nonempty PCM chunk and calls `finish_sentence`
after the final chunk has been accepted:

```python
sentence = classify_sentence(text, hint)
hardware.play_audio(first_pcm, sentence=sentence)
hardware.play_audio(remaining_pcm)
hardware.finish_sentence()
```

`play_audio` submits PCM before publishing motion metadata. The controller
places the expression at the queued audio start plus the existing 60 ms sink
estimate. The end gesture uses the accepted PCM tail, so synthesis gaps and
microphone pauses cannot trigger an early sentence-end gesture. This is an
estimate of acoustic timing; the sink does not expose a DAC cursor.

The motion thread consumes bounded start and end events. Every start can replace
the previous sentence clip through `ClipTimeline`, including repeated emotions.
The transition begins from the previous composed pose, before additive sway and
accents. Missing clips use the expression pose. A pause retains the expression
without replaying the opening clip, and a delayed cue cannot restore speaking
after playback has ended. Clear and device-command events invalidate pending
sentence events. Commands retain priority while their preceding clip fades.

PCM still supplies sway and syllable accents. A sentence-end nod or question
tilt adds at most four degrees. Question and exclamation antenna accents add
eight and ten degrees respectively. The final pose clamp and inactive-antenna
rule apply after composition. There is no forced word alignment or inferred
word timestamp.

Thinking motion ends when answer audio starts. Closing selection waits for the
output queue to drain and uses the last sentence actually heard. A cancelled or
revised queued sentence cannot select the closing. The next turn cancels any
older closing still waiting for generation.

## Design choice

Two designs were compared: a motion-owned event scheduler and speech-owned cue
snapshots with a classifier worker. The implementation uses speech's accepted
PCM boundaries and motion's event scheduler. A synchronous, bounded lexicon
avoids a second worker queue and missed first-frame classifications. Rendering,
clip priority, and cancellation stay in the controller.

The worker design's useful constraint was to anchor cues after successful sink
submission. Its background classification and late refinement were omitted:
the measured local classifier fits well below a millisecond, and no companion
request is needed. The accepted tradeoffs are conservative English rules,
neutral fallback for unmatched text, estimated sink latency, and skipped motion
when a device command has priority.

## Offline results

Measured on the development Mac, with synthetic PCM and fake devices. The
baseline is `ae922da`, the `perf/turn-2` revision used to create this branch.
Both versions ran the same replay, 15 times per case. No robot, Muse request,
service, or model was used. This replay drives `ReplyReader` and `_speak`
directly. It measures the playback path, excluding turn dispatch, `_deliver`,
and the deferred output worker.

| First actual sink push | Baseline p50 | Expressive p50 | Baseline p95 | Expressive p95 |
| --- | ---: | ---: | ---: | ---: |
| Streamed PCM, 5 ms fake synthesis | 6.420 ms | 6.494 ms | 6.694 ms | 6.700 ms |
| Prepared PCM | 0.329 ms | 0.207 ms | 4.118 ms | 0.239 ms |

The streamed p95 difference was 0.006 ms. Independent wall-clock runs include
scheduler noise; the prepared baseline had a 12.435 ms maximum outlier. These
results show no material first-audio regression in this replay. They do not
measure live TTS, network delay, or acoustic output.

The existing full-turn `bench.latency_turn_replay dispatch --runs 15` also passed.
Its first-audio callback p95 changed from 35.340 to 35.275 ms with a 200 ms fake
opener, and from 35.188 to 34.904 ms with an immediate opener. That replay includes
dispatch and delivery, but uses legacy fake hardware whose callback precedes
sink submission. The table above measures the sink itself.

All three replay sentences received an expression in both playback modes:
neutral, curious, and happy. All 360 PCM frames, chunk boundaries, and their hash
matched the baseline. First audio preceded reply completion in every run.
Classifier CPU over 12,000 calls was 0.002 ms p50, 0.105 ms p95, and 0.177 ms
maximum. The corpus includes an input longer than the 2048-character scan limit.

The synthetic 40 Hz controller benchmark runs the real `_motion_loop` with a
fake recorded move, alternating and repeated expressions, and real PCM sway:

| Motion loop | Mean CPU per tick | Share of 25 ms | p99 CPU per tick |
| --- | ---: | ---: | ---: |
| Layered sway | 38.7 µs | 0.15% | 51.8 µs |
| Sentence expressions, accents, and sway | 50.9 µs | 0.20% | 90.2 µs |

All sampled targets stayed within `POSE_LIMITS`. Maximum adjacent head-angle
change was 3.08 degrees per 25 ms; maximum antenna change was 2.19 degrees.
The behavior test bounds these at four and three degrees respectively for this
fixture. Crossfades still take 0.4 seconds. These are target changes, not measured
motor speeds. CPU measurements exclude robot I/O, PCM feeding, and real scheduling
jitter. Robot CPU and physical motion have not been measured.

## Repeat the checks

Use a Python environment with the repository's test dependencies. To measure the
baseline, run the same replay file with `PYTHONPATH` pointing to the baseline's
`src` and `tests` directories. Save its JSON report, then compare the current code:

```sh
PYTHONPATH=src:tests python bench/expressive_streaming.py --runs 15 --baseline baseline.json
PYTHONPATH=src python bench/expressive_motion_tick.py
PYTHONPATH=src python -m pytest -q tests/test_expressive_playback.py tests/test_sentence_expression.py tests/test_sentence_motion.py tests/test_moments.py
```

The replay asserts identical PCM and reports expression coverage. Behavior tests
also cover failed and empty pushes, exact onset and tail schedules, repeated
expressions, pauses, revisions, clears, command priority, missing clip assets,
inactive antennas, delayed ticks, and closing supersession.
