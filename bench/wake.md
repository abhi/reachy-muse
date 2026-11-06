# Wake reliability

This change makes wake failures measurable and reduces false accepts in local
replay. It does not establish perfect field recall, a 300 ms acknowledgement
bound, or immunity to real residual echo.

## Implementation

The selected candidate threshold is **0.10**, with keyword boost **1.0**, one
tokenization and one CPU inference thread. Candidates use a 1.5-second hearing
clip. Companion hearing has a 60 ms allowance; local streaming recognition
starts concurrently and shares a 180 ms total deadline. A negative transcript
rejects the candidate. Missing or late verification requires a separate 0.25
threshold replay. Losing hearing streams are aborted.

Every exported candidate receives a content-free decision with its threshold,
boost, verifier result, replay bound when available, inference/verification/
alignment timings, and an acceptance or rejection reason. The reasons include
`below_threshold`, `cooldown`, `speaking`, `echo`, `verification_rejected`, and
`capture_gap`. Vosk also logs rejected final hypotheses and available mean word
confidence. Microphone level windows survive recognition resets and include RMS,
peak and clipping. Detection continues during an open wake window, so duplicate
and playback rejections remain observable.

Stock sherpa 1.13.8 exposes keywords, tokens and timestamps, but neither native
posteriors nor rejected internal hypotheses. Its logs therefore use `score:null`
and `score_kind:unavailable`. A trigger floor or replay bound is not relabelled
as confidence. Hypotheses below 0.10 remain invisible. Literal posterior logging
would require a native binding change. See the pinned [result binding](https://github.com/k2-fsa/sherpa-onnx/blob/v1.13.8/sherpa-onnx/python/csrc/keyword-spotter.cc)
and [native decoder](https://github.com/k2-fsa/sherpa-onnx/blob/v1.13.8/sherpa-onnx/csrc/transducer-keyword-decoder.cc).

Playback evidence is captured with microphone samples and retained through
queueing and batching. Own playback containing the phrase rejects as echo;
unknown or unqualified playback rejects as speaking. Known unrelated playback
on qualified duplex may admit a verified human wake. Simultaneous human and
speaker occurrences of the same phrase are conservatively rejected. No speaker
identity claim follows from transcription or AEC qualification.

The existing epoch/feed-checked audio handoff remains authoritative. Bounded
alignment replay now retains the original encoder phase after its history rolls
over. Capture gaps during verification invalidate acceptance before opening a
call. No provisional candidate opens the microphone to Muse.

## Corpus and method

The [manifest](wake-corpus.jsonl) records 274 WAV hashes and relative paths,
covering 906.774 seconds. [Provenance](wake-provenance.json) records the TTS
model/voice hashes and seed. All inference used CPU providers.

- Six existing Kokoro voice identities: `af_heart`, `am_michael`, `bf_emma`,
  `bm_george`, `af_sky`, and `bm_lewis`, using the ONNX CPU export.
- Positive phrases at 0.85 and 1.15 speed, with clean audio and seeded 20, 10,
  and 0 dB noise. Punctuation variants use the same acoustic phrase.
- Confusables, ordinary speech, TV-like overlapping synthetic speech, silence,
  noise, and synthetic own-voice playback with known output context.
- Three existing recorded human negatives, 24.6 seconds total.
- One continuous 120-second idle followed by a wake attempt.

`af_heart` and `am_michael` supply 32 tuning positives. The first British voices
are development data, since they were inspected while debugging recognition.
Fresh `af_sky` and `bm_lewis` voices supply held-out positives. The 120-second
idle case is reported separately and included in aggregate totals. Augmented
copies of an utterance stay in the same split.

The baseline loads `wake_word.py` directly from **dd1ed17**. Both runs consume
the identical PCM in 256-sample chunks. The new run executes the production
`WakePolicy` and the real local streaming recognizer, with the companion absent.
No companion service, robot, Muse connection or GPU model was used. Detailed
candidate records remain in `bench/out/wake-results.json`; the compact tracked
[results](wake-results.json) include versions, hashes, all tuning settings and
grouped metrics.

## Results

| Configuration | Positive hits | Recall | False accepts / 571.118 negative seconds | Stress-set false accepts/hour |
|---|---:|---:|---:|---:|
| dd1ed17 | 55/97 | 56.70% | 46 | 289.96 |
| New policy, local verifier available | 56/97 | 57.73% | 0 | 0 observed |
| Companion and local verifier both unavailable | 56/97 | 57.73% | 13 | 81.95 |

These rates describe an enriched stress corpus, not expected household rates.
Zero events in 571.118 seconds has an approximate one-sided 95% Poisson upper
bound of **18.88/hour**, before accounting for correlated synthetic examples.
More negative exposure is needed to establish a low field false-trigger rate.

| Positive condition | Before | After |
|---|---:|---:|
| Clean | 22/24 | 22/24 |
| 20 dB SNR | 23/24 | 22/24 |
| 10 dB SNR | 10/24 | 12/24 |
| 0 dB SNR | 0/24 | 0/24 |
| 120-second idle case | 0/1 | 0/1 |

Held-out recall remained **17/33**, while held-out false accepts fell from
**16 to 0** over 252.590 negative seconds. Thus the evidence supports improved
false-accept control, but does not establish a general recall improvement.
All 56 accepted candidates in the verified run supplied a timed handoff.

The local recognizer frequently transcribes /heɪ/ as `HAY`. The verifier accepts
`HAY MUSE`, but continues to reject `PAY MUSE`, `HI MUSE`, `HEY NEWS`, `HEY MUSIC`
and `HEY MOOSE`. The KWS graph still registers the canonical phrase. Lowering
the threshold to 0.05, increasing boost to 1.5, or adding three BPE segmentations
did not improve the best tuning result of 22/32. A two-second verifier window
reduced tuning recall to 20/32. Higher boosts of 3.0 and 5.0 reduced tuning hits
to 14/32 and 7/32 at threshold 0.10; threshold 0.05 with boost 3.0 reached 17/32.
`hi muse` remains outside the wake vocabulary.

[Phase probes](wake-probes.json) exposed a remaining acoustic weakness. One
identical clean waveform triggers with 0.24, 0.56, 0.88 or 1.20 seconds of
prefix silence, but misses at other phases. Two staggered streams recovered no
extra tuning/idle positives. Four recovered one, while CPU time rose from
4.191 to 16.839 seconds over 347.688 seconds of audio. They were not enabled.
KWS homophone spellings `HEY MEWS` and `HAY MUSE` also added no raw candidates.

## Latency and CPU

Measured phrase-end-to-decision lag was **202 ms p50 / 397 ms p95** before and
**257 ms p50 / 465 ms p95** after. This is unpaced CPU replay: audio-time detection
lag plus the final processing call, relative to the rendered phrase end. It
excludes real capture backlog, network transport, robot scheduling, motion and
audible acknowledgement playback. It is not a robot acknowledgement measurement.

The T45/stride32 model has a 320 ms decode cadence and additional lookahead.
The existing alignment stage can spend up to 700 ms. A 180 ms verifier budget
therefore does not imply a 300 ms end-to-end bound. The 300 ms target remains
unmet, and the measured p95 became slower with verification.

Parent-process CPU over the 906.774-second corpus was 12.792 seconds before,
16.328 seconds with verification, and 17.949 seconds with acoustic fallback.
These are **1.41%, 1.80% and 1.98% of one host CPU core**. They exclude the local
recognizer child. A planning-only 10–15x host-to-CM4 multiplier estimates the
KWS/policy portion at **18–27% of one CM4 core** with verification and **20–30%**
with acoustic fallback. The multiplier is an assumption, not a measurement.

The supplied log has 7,277 microphone summaries and nine detections. Median
coverage is 1.000; median window-maximum feed cost is 97.0 ms and p95 is 107.9 ms.
Using those maxima as a pessimistic per-decode proxy gives 30–34% of one core
for one 320 ms stream. Four streams could exceed one serialized core's budget.
The existing [streaming CPU benchmark](realtime_input.md) measured 4.960 host CPU
seconds per 138 audio seconds. At 10–15x, a 1.5-second verification clip plus
0.66-second flush could cost roughly **0.78–1.16 CM4 CPU seconds**. It may miss
the 180 ms deadline and use conservative replay. Physical CM4 measurements are
still needed; no low-latency local-verifier claim is made.

## Reproduction and operator measurement

Obtain the [KWS archive](https://github.com/k2-fsa/sherpa-onnx/releases/download/kws-models/sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01.tar.bz2)
and the [Kokoro ONNX assets](https://github.com/thewh1teagle/kokoro-onnx/releases/tag/model-files-v1.0).
Set `WAKE_MODEL`, `VERIFY_MODEL`, `TTS_MODEL` and `TTS_VOICES` to local assets.
The verifier model uses the existing streaming-transcription file layout.

```sh
uv run --no-project --with sherpa-onnx==1.13.8 --with sentencepiece==0.2.2 \
  --with kokoro-onnx==0.6.1 --with soundfile==0.14.0 --with numpy==2.5.3 \
  env PYTHONPATH=src python bench/wake.py build \
  --tts-model "$TTS_MODEL" --voices "$TTS_VOICES" --output bench/out/wake

uv run --no-project --with sherpa-onnx==1.13.8 --with sentencepiece==0.2.2 \
  --with kokoro-onnx==0.6.1 --with soundfile==0.14.0 --with numpy==2.5.3 \
  env PYTHONPATH=src python bench/wake.py replay \
  --model "$WAKE_MODEL" --verify-model "$VERIFY_MODEL" --corpus bench/out/wake \
  --output bench/out/wake-results.json --summary bench/wake-results.json

uv run --no-project --with sherpa-onnx==1.13.8 --with sentencepiece==0.2.2 \
  --with soundfile==0.14.0 --with numpy==2.5.3 \
  env PYTHONPATH=src python bench/wake.py probe \
  --model "$WAKE_MODEL" --corpus bench/out/wake --output bench/wake-probes.json

muse-reachy wake-test --count 10 --wake-model "$WAKE_MODEL" \
  --verify-model "$VERIFY_MODEL" --report wake-trials.json
```

The operator command prompts for N wake attempts at 0.5, 1.5 and 3 metres, then
N other-phrase/silence trials. It drains prompt-time audio, leaves motion off,
and counts only captured negative audio in the hourly denominator. Capture gaps
invalidate trials. `--record-dir RECORDINGS` explicitly enables labelled WAV
capture; `bench/wake.py build --recordings RECORDINGS ...` imports valid trials.
Without that flag, the command saves no audio or transcript content.

Literal tests cover unavailable/late/negative verifiers, cancellation cleanup,
own-voice and unknown-output rejection, unrelated duplex speech, queued capture
context, independent trial cooldown, encoder-phase rollover, gap invalidation,
and unchanged same-breath handoff audio. The exact supplied full-suite command
passed with **3,121 passed, 3 skipped** in 281.64 seconds using six workers.
Operator hardware trials were not performed.

Remaining evidence gaps are recorded human wake positives across accents and
distances, actual head-motion noise, room reverberation/double-talk, real Reachy
and Muse speaker recordings, prolonged real-room negatives, and CM4 timing.
Synthetic motor noise and playback context are useful tests but do not
substitute for them.
