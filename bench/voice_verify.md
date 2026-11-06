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
contains scores, aggregate metrics, and hardware/model identifiers, with no
machine paths or embeddings. The script uses a fixed noise seed. OS voice
updates can change the corpus; preserve the WAVs privately for comparisons.

## Protocol and threshold

Three clean Samantha phrases form enrollment. Calibration uses two different
phrases from Samantha and impostors Daniel, Fred, and Kathy. Evaluation uses
two held-out Samantha phrases and four phrases from held-out impostor voices
Karen, Junior, and Ralph. Voices, phrases, and exact split rules are in
`voice_verify.py`. Every phrase begins with the wake phrase. Windows are the
first 1.5 seconds for approximate wake timing and the first second for request
segments. This does not benchmark the keyword detector's trigger timestamps.

Each trial has four conditions: clean; a distance approximation with 0.25 gain
and a 40 ms delayed reflection at 0.25 relative amplitude; and additive Gaussian
noise at 20 and 10 dB SNR. These are correlated synthetic augmentations, not
recordings at physical distances or independent human speakers. No confidence
interval from these counts would establish real-world performance.

Cosine similarity accepts at `score >= threshold`. The threshold is the maximum
calibration impostor score plus 0.02, rounded upward to a hundredth. It is 0.74
for the pinned ResNet34 model. Evaluation scores do not set this threshold.
Unknown models require an explicit calibrated `--voice-threshold` at enrollment.
The profile saves that threshold and the exact model hash. EER is the mean FAR
and FRR at the closest empirical crossing, without interpolation.

## Measured results

Run on 2026-11-06, Apple M3 Ultra, arm64, CPU only, one inference thread.

| Split | Genuine trials | Impostor trials | EER | FAR at 0.74 | FRR at 0.74 |
|---|---:|---:|---:|---:|---:|
| Calibration | 16 | 48 | 30.21% | 0/48, 0% | 8/16, 50% |
| Evaluation | 16 | 96 | 31.25% | 11/96, 11.46% | 8/16, 50% |

| Evaluation condition | False accepts / impostors | False rejects / genuine |
|---|---:|---:|
| Clean | 8/24 | 0/4 |
| Distance approximation | 3/24 | 0/4 |
| 20 dB SNR | 0/24 | 4/4 |
| 10 dB SNR | 0/24 | 4/4 |

Warm preprocessing plus embedding CPU latency across 224 windows was 51.78 ms
p50, 78.13 ms p95, and 91.36 ms maximum. Model loading and warmup occur before
capture and are excluded. This meets the 150 ms inference target on this Mac;
it does not prove the target on the robot CPU or include the wake detector,
thread scheduling, or 48 kHz resampling cost. Request gating additionally holds
roughly one second of speech before release. It can delay interruptions too.

An exploratory comparison with WeSpeaker CAM++ used the same corpus, CPU
provider, and one thread. Its calibrated threshold was 0.80, evaluation EER
56.25%, FAR 7/96, FRR 12/16, and latency 25.15 ms p95. We retained ResNet34 for
its lower EER. Evaluation was used for this model comparison, so it is not an
independent final qualification set. Neither result is a production security
claim. Similar synthetic voices and noise expose substantial failure cases.

## Runtime behavior and next measurements

Wake audio uses the latest 1.5 seconds ending at the trusted keyword boundary;
untimed detectors use the latest buffered audio. Rejected wakes never open the
listening window or reach hearing. Strict rejection emits a quiet 160 ms local
descending chime, rate limited to once per five seconds.

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
