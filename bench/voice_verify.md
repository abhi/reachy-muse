# Speaker verification, CPU evaluation

This is an experimental additional wake check, disabled by default. The
synthetic results below do **not** support using it as an authentication
boundary. No robot, microphone, companion service, Muse endpoint, or GPU was
used. Robot CPU latency and real operator accuracy remain unmeasured.

## Model and license

The baseline is `wespeaker_en_voxceleb_resnet34.onnx`, 26,534,365 bytes,
256 dimensions, loaded through `sherpa-onnx==1.13.8` with `provider="cpu"`,
one thread. The SHA-256 is
`5ef208a9da1453335308a6b6f4e6dfbd7e183a38b604de0a57664f45d257fe94`.

Attribution: WeSpeaker, the WeNet community, ResNet34 trained on VoxCeleb.
The [upstream model license documentation](https://github.com/wenet-e2e/wespeaker/blob/master/docs/pretrained.md)
states CC BY 4.0 for VoxCeleb pretrained models. This permits commercial use
with attribution; it is not a non-commercial license. WeSpeaker code and
[Sherpa](https://github.com/k2-fsa/sherpa-onnx) use Apache 2.0.
No weights are bundled. The tested ONNX conversion is distributed in
[Sherpa's speaker model release](https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-recongition-models).
The runtime accepts compatible local ONNX files, so check the license and
calibrate separately before substituting weights. A model's code license does
not establish the license of its weights.

## Reproduction

Install the `voice-verify` extra. Download the model explicitly; enrollment and
runtime never download anything. The following paths are examples relative to
the working directory.

```sh
mkdir -p models
curl -fL -o models/speaker.onnx \
  https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/wespeaker_en_voxceleb_resnet34.onnx
shasum -a 256 models/speaker.onnx
PYTHONPATH=src python bench/voice_verify.py \
  --model models/speaker.onnx --audio-dir /tmp/voice-eval-audio \
  --output /tmp/voice-eval.json --synthesize
```

`--synthesize` uses installed macOS `say` voices and writes WAVs only to the
specified directory. Omit it to evaluate the same WAVs on another CPU. Output
contains scores, request retention, aggregate metrics, and hardware/model identifiers, with no
machine paths or embeddings. The script uses a fixed noise seed. OS voice
updates can change the corpus; preserve the WAVs privately for comparisons.

## Protocol and threshold

Three clean Samantha phrases form enrollment. Calibration uses two different
phrases from Samantha and impostors Daniel, Fred, and Kathy. Evaluation uses
two held-out Samantha phrases and four phrases from held-out impostor voices
Karen, Junior, and Ralph. Voices, phrases, and exact split rules are in
`voice_verify.py`. Wake trials use the first 1.5 seconds of the wake-prefixed
phrases. Request trials use separate `VOICE-INDEX-request.wav` files synthesized
from the words after "Hey Muse,". This avoids cutting off request words with an
estimated keyword timestamp. Wake-only phrase 3 has no request trial. Existing
corpora need another `--synthesize` run to create the request files.

Every complete request passes through the runtime `RequestGate` in 20 ms frames,
followed by one second of silence to flush its final segment. The report records
every segment score and counts input and retained nonzero samples. A complete
request retains all its nonzero samples. In noisy conditions this includes noise.
These trials isolate speaker verification; they do not measure keyword detection,
continuous wake-to-request prosody, or transcript accuracy.

Each trial has four conditions: clean; a distance approximation with 0.25 gain
and a 40 ms delayed reflection at 0.25 relative amplitude; and additive Gaussian
noise at 20 and 10 dB SNR. These are correlated synthetic augmentations, not
recordings at physical distances or independent human speakers. No confidence
interval from these counts would establish real-world performance.

Cosine similarity accepts at `score >= threshold`. Calibration includes every
wake and request segment. The threshold is the maximum calibration impostor
segment score plus 0.02, rounded upward to a hundredth. Recalibration with complete
requests still selects 0.74 for the pinned ResNet34 model. Evaluation scores do
not set this threshold. This margin rejects every calibration impostor segment
but also truncates genuine requests; it is not a demonstrated usable strict-mode
threshold.
Unknown models require an explicit calibrated `--voice-threshold` at enrollment.
The profile saves that threshold and the exact model hash. EER is the mean FAR
and FRR at the closest empirical crossing, without interpolation. Request EER,
FAR, and FRR use the minimum segment score, so acceptance requires all segments
to match. Separate retention counts reveal partial audio release and VAD loss.

## Measured results

Run on 2026-11-06, Apple M3 Ultra, arm64, CPU only, one inference thread.

| Path and split | Genuine trials | Impostor trials | EER | FAR at 0.74 | FRR at 0.74 |
|---|---:|---:|---:|---:|---:|
| Wake calibration | 8 | 24 | 35.42% | 0/24, 0% | 4/8, 50% |
| Wake evaluation | 8 | 48 | 25% | 7/48, 14.58% | 4/8, 50% |
| Request calibration | 4 | 12 | 0% | 0/12, 0% | 4/4, 100% |
| Request evaluation | 8 | 36 | 25% | 1/36, 2.78% | 6/8, 75% |

| Evaluation condition | Wake false accepts | Wake false rejects | Complete genuine requests | Impostor requests releasing any audio |
|---|---:|---:|---:|---:|
| Clean | 4/12 | 0/2 | 1/2 | 3/9 |
| Distance approximation | 3/12 | 0/2 | 1/2 | 2/9 |
| 20 dB SNR | 0/12 | 2/2 | 0/2 | 0/9 |
| 10 dB SNR | 0/12 | 2/2 | 0/2 | 0/9 |

Only 2/8 held-out genuine requests retain all their audio across the four
conditions. The clean Samantha-5 request retains all 13,141 nonzero samples.
Samantha-6 retains 19,198 of 24,737; its final segment scores 0.569919 and loses
5,539 nonzero samples. Prefix-only scoring missed this failure. Calibration
retains no complete genuine request. Five of 36 held-out impostor requests
release some audio, even though only one passes every segment. Request trials
run independently of wake acceptance so a rejected wake cannot hide these losses.

Warm preprocessing plus verification CPU latency across 219 decisions was 52.54 ms
p50, 77.95 ms p95, and 79.82 ms maximum. Model loading and warmup occur before
capture and are excluded. This meets the 150 ms inference target on this Mac;
it does not prove the target on the robot CPU or include the wake detector,
thread scheduling, or 48 kHz resampling cost. Request gating additionally holds
roughly one second of speech before release. It can delay interruptions too.

The earlier comparison with WeSpeaker CAM++ scored prefixes only. It does not
establish which model retains complete requests. The ResNet34 evaluation set was
used in that comparison, so these results are not an independent final qualification.

## Runtime behavior and next measurements

Wake audio uses the latest 1.5 seconds ending at the trusted keyword boundary;
untimed detectors use the latest buffered audio. Rejected wakes never open the
listening window or reach hearing. Strict rejection emits a quiet 160 ms local
descending chime, rate limited to once per five seconds. The cue uses the shared
speaker queue, playback pacing, echo suppression, and cancellation handling.

Requests and interruptions are checked repeatedly in bounded segments before
hearing or realtime forwarding. The gate retains 0.4 seconds of unforwarded
speech across cuts to preserve short final words, and clears it on capture
gaps or reset. A rejected segment becomes silence. Soft mode observes chat,
gates an available realtime call, and returns to observation on chat fallback.
Strict also gates chat. Off does not load verification dependencies or state.

The feature checks speaker similarity, not liveness. Replay, voice imitation,
overlapping speakers, or a speaker change within a segment can still pass.
Rejection of part of a request can leave an incomplete accepted fragment;
short isolated commands may have insufficient voice evidence. Use soft chat
observation to collect score-only measurements before trying strict mode.

Before relying on strict mode, run enrollment and held-out evaluation with the
actual operator, unrelated human impostors, room noise, and measured microphone
distances. Measure the robot's warm p95 wake cost with its normal concurrent
workload. These checks were not run because robot access was explicitly out of
scope. Logs added by this feature contain only score and decision.
