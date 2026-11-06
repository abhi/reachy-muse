# Sentence expression variety

Compared with `main` at `3fd1129`, the local classifier agrees with 199 of 200 authored labels, up from 102. Factual speech keeps neutral meaning and gains three bounded gestures: attentive nod, curious tilt, and warm. Both chat and realtime use the shared speaker path. No robot, Muse connection, GPU model, deployment, or service was used for this evaluation.

The 200 sentences in [expression_sentences.json](expression_sentences.json) cover headlines, answers, advice, encouragement, questions, greetings, sign-offs, negation, mixed sentiment, and ambiguous direction words. They are a development and calibration set, not an independent test of real-world accuracy. Labels were written before evaluating the expanded classifier. The one remaining disagreement is `happy-30`: "The market rebounded after yesterday's selloff." The classifier conservatively returns neutral because it sees both sentiments.

| Semantic expression | Labels | Baseline | Updated |
| --- | ---: | ---: | ---: |
| happy | 35 | 13 | 34 |
| sad | 35 | 7 | 35 |
| curious | 25 | 23 | 25 |
| surprised | 15 | 7 | 15 |
| nod | 25 | 8 | 25 |
| thinking | 15 | 0 | 15 |
| neutral | 50 | 142 | 51 |

Neutral is a semantic label, not a requirement to stay still. The 51 updated neutral results become 17 attentive nods, 14 curious tilts, and 20 warm gestures. A separate run of 30 factual sentences gives ten of each, with no adjacent duplicate. If delivery repeatedly prefers one variant, the third sentence selects another. Clear lexical emotions may repeat; the existing motion library rotates its recorded variants. Neutral realtime sentences now request that library too. For neutral speech, missing or late library results use cached local gestures.

Local gestures last 1.2 seconds. The following bounds describe their input poses, before the existing 0.6 spoken-clip gain and crossfade. Their peaks are three degrees of pitch for attentive, five degrees of roll for curious tilt, and two millimetres of lift with three degrees of roll for warm. Antenna excursions stay within eight degrees; the body does not turn. Library motions retain their variant selection but are bounded per axis to the same family. Existing clipping, crossfades, command priority, and cancellation remain active.

Prosody reads the actual accepted output PCM at its output rate. It reuses `ProsodyTracker` pitch and energy measurements and adds a syllable-envelope peak rate estimate. This is not a measured words-per-minute rate. Analysis covers at most the first 0.8 seconds of a sentence and never waits for more audio. Quiet or slow delivery prefers attentive; rising or wide pitch prefers curious tilt; energetic, faster delivery prefers warm. Silence, DC, unvoiced noise, and short samples provide no vote. Prosody never changes a semantic expression, so it cannot turn bad news into happiness or trigger a celebration from loudness alone. Greeting and sign-off cues keep their lexical warmth.

All analysis and gesture selection in the shared controller occur after `push_audio_sample`. A later refinement preserves the sentence serial and happens at most once. A clear late transcript correction stops neutral prosody refinement. End, flush, and command cancellation discard pending refinement. INFO logs now include the selected variant and whether it is a revision.

| Synthetic output profile | Pitch range, semitones | Energy, dBFS | Envelope peaks/s | Semantic changes across 200 sentences |
| --- | ---: | ---: | ---: | ---: |
| quiet | 0.23 | -35.80 | 7.50 | 0 |
| animated | 0.23 | -16.07 | 7.50 | 0 |
| rising | 8.81 | -19.59 | 7.50 | 0 |

Synthetic PCM verifies signal handling and precedence. It does not establish perception or timing quality on a real robot.

CPU timing uses 20,000 calls on arm64 macOS with Python 3.13.14. The CM4 estimate multiplies host CPU time by ten. That is a conservative planning assumption, not a CM4 measurement or a hardware-derived conversion.

| CPU per sentence | p50 | p95 | Maximum |
| --- | ---: | ---: | ---: |
| Classifier | 0.009 ms | 0.012 ms | 0.103 ms |
| Classifier plus selection | 0.010 ms | 0.013 ms | 0.109 ms |
| CM4 estimate, classifier | 0.090 ms | 0.120 ms | 1.030 ms |
| Host, 2048-character scan bound | 0.267 ms | 0.316 ms | 0.417 ms |

The long-input maximum estimates to 4.17 ms on the same multiplier. Ordinary sentences stay well below the 5 ms budget. Prosody is separate from classifier timing and runs after the sink push; the three 0.8-second synthetic samples cost 1.30 to 2.50 ms of host CPU each.

The morning log contains ten briefing expression records, eight neutral and two sad, and three realtime replies with five neutral and one happy across six sentences. It contains neither spoken text nor PCM. Exact lexical agreement and real-audio prosody quality for those sentences therefore cannot be measured. The included log inventory records its SHA-256 and replays every observed semantic choice through the new selector. The briefing sequence becomes attentive, curious tilt, warm, attentive, curious tilt, sad, sad, warm, attentive, curious tilt. This is a gesture replay, not a claim to have recovered or relabeled the missing words. The separate happy greeting notice is excluded from the ten-sentence briefing.

First audio retains the same PCM, chunk order, and scheduling rules. Behavior tests observe the actual speaker push before any new analysis. The original streaming replay still pushes before reply completion and preserves all 360 PCM frames and their SHA-256. The controller replay adds a memory-only sink behind the real `ReachyController`, with 48 kHz output and a 24 kHz realtime input. It compares 15 warm runs per path with the baseline source tree and asserts identical output PCM hashes and frame counts.

| Controller replay | First push p50 before / after | First push p95 before / after | Audio ready to sink p95 before / after |
| --- | ---: | ---: | ---: |
| prepared | 0.352 / 0.427 ms | 0.562 / 0.598 ms | 0.120 / 0.120 ms |
| streamed | 6.081 / 6.085 ms | 6.558 / 6.615 ms | 0.368 / 0.335 ms |
| realtime | 0.518 / 0.393 ms | 0.819 / 0.472 ms | 0.819 / 0.472 ms |

| Existing replay | First push p50 before / after | First push p95 before / after |
| --- | ---: | ---: |
| Chat prepared | 0.224 / 0.232 ms | 0.246 / 0.319 ms |
| Chat streamed | 6.246 / 6.011 ms | 6.793 / 7.571 ms |
| Realtime | 0.700 / 0.900 ms | 2.300 / 7.000 ms |

These are wall-clock replays, not proof of identical latency distributions. Scheduler tails vary: an initial 8 kHz controller replay had a 21.18 ms streamed maximum versus 6.84 ms baseline, despite comparable medians. The existing realtime replay also has a higher p95 in this run. Those initial results are retained alongside the instrumented 48 kHz replay. No buffering interval or wait was added; feature computation follows the first push. A physical CM4 latency measurement remains unperformed.

Full suite: **3069 passed, 3 skipped in 284.73s (0:04:44)** using the requested command with six workers. With the project-pinned SDK at `4737636`, the classifier and sentence-delivery tests report **98 passed, 2 skipped**. The two realtime delivery cases skip when the SDK lacks voice call events, while chat coverage stays active. [Behavior tests](../tests/test_sentence_delivery.py) assert literal classifications, PCM-derived delivery, neutral rotation, lexical precedence, late corrections, motion bounds, cancellation, failed and empty pushes, and both chat and `TranscriptDelta` playback through the real controller. [Classifier regressions](../tests/test_sentence_expression.py) cover repeated adverse subjects and affirmative and negative information phrases.

Raw measurements are in [expression_variety_runs](expression_variety_runs). Reproduce with a Python environment containing the test dependencies and the supplied SDK checkout. Run timing commands sequentially. Use the same replay scripts with `PYTHONPATH` pointing to a clean baseline source tree for the before measurements.

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python bench/expression_variety.py --morning-log "$MORNING_LOG" --json coverage.json
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tests python bench/expression_playback.py --runs 15 --baseline playback-before.json --json playback.json
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tests python bench/expressive_streaming.py --runs 15 --baseline streaming-before.json
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python bench/realtime_expressive.py --json realtime.json
uv run -q --no-project --python "$REACHY_PYTHON" --with pytest --with pytest-timeout --with pytest-xdist --with 'av>=12' --with 'webrtcvad-wheels>=2.0.14' --with "$SDK_CHECKOUT/linux" env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m pytest -q -p no:cacheprovider --timeout 180 -n 6
```
