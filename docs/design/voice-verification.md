# Local speaker verification

## Problem and traced boundary

`VoiceConversation._microphone` retains three seconds of wake pre-roll. A
successful detection currently changes the wake state, opens a realtime call,
and feeds that pre-roll to `record`. `record` can send audio to companion
hearing before a transcript exists. Realtime input also sends audio directly.
Verification therefore belongs before both wake admission and `record`.

## Usage and shape

```sh
muse-reachy enroll-voice --voice-model models/speaker.onnx
muse-reachy run --wake-model models/wake --voice-model models/speaker.onnx --voice-verify strict
muse-reachy enroll-voice --forget
```

The existing run mode still supplies its normal speech and connection options.
Enrollment records three short prompted phrases locally, retains only their
normalized mean embedding, and atomically replaces a mode-0600 profile. The
profile binds the embedding to a model SHA-256 and threshold. Forgetting works
without model dependencies, pairing, or hardware.

Caller-first sketch, before implementation:

```python
encoder = SpeakerEncoder(model_path)  # fixed CPU provider, one thread
profile = enroll(encoder, recordings, threshold)
profile.save(state_dir / "voice-profile.json")
verifier = VoiceVerifier(encoder, Profile.load(path), mode, realtime)
decision = verifier.verify(wake_audio, sample_rate)
gate = RequestGate(verifier, sample_rate)
accepted_audio = gate.feed(samples)
```

`Decision` contains only score, match, and allowed. The verifier owns audio
quality checks, embedding normalization, cosine scoring, and content-free
logging. `RequestGate` owns bounded local segmentation. Conversation owns wake
state, cue timing, and forwarding. Neither model nor embedding travels to the
companion. No automatic downloads or inference-provider fallback occur.

## Policy

- Off does not load the model or profile and preserves the current path.
- Soft logs decisions; chat passes audio through. Wakes that would open a
  usable realtime call and speech during a call require a match. Chat fallback
  returns to observation. Strict is required to gate that fallback too.
- Strict gates all wakes and requests. Rejection plays a brief, quiet local
  descending chime, at most once every five seconds. It uses no TTS service.
- Missing or mismatched enrollment fails startup. Insufficient speech,
  nonfinite output, and runtime inference failure reject under enforcement.
- Requests, including follow-ups and interruptions, pass through a local
  segment buffer before reaching hearing or realtime input. Verified
  interruptions may then stop playback. A wake does not grant other speakers
  access to the listening window.
- Capture gaps, blocked capture, and window closure discard held audio.

## Synthesis decision

Two independent design candidates and a separate boundary trace converged on
local capture admission. A separate cross-review found that full-utterance
replay could overflow the existing streaming queues. Implementation therefore
uses bounded segments and separate wake and request authorization.
The public interface hides model/storage/segmentation details while preserving
the existing conversation state machine.

Each segment is at most 1.5 seconds, with 0.4 seconds retained unforwarded
across cuts, including at least 0.3 seconds of voiced evidence, to preserve
short final words. Sparse speech exceeding that buffer is rejected.
Every release gets a fresh score.
Local VAD uses separate state from the downstream hearing recorder. Playback
receives admitted samples in 20 ms pieces at the hardware sample rate. Rejected
segments become silence. The wake check uses the last 1.5 seconds ending at the
keyword boundary when the detector supplies a trusted boundary.

We rejected a hearing-only wrapper because realtime input bypasses it, a
wake-scoped grant because it admits subsequent speakers, and a one-time prefix
check because it authorizes the unchecked remainder of the request. Companion
verification would add transport cost and export private enrollment material.

## Tradeoffs and validation

The wake target is under 150 ms of warm preprocessing and CPU inference on
already captured audio. Requests incur roughly 1 second of look-ahead before
forwarding; enforced realtime therefore delays speech. Short
commands may lack enough speaker evidence and require a longer repetition.
Speaker similarity is not liveness detection: recordings, imitations, and mixed
speakers within one segment remain limitations. The synthetic benchmark is
experimental and does not establish an authentication-grade error rate.

Evaluate held-out synthetic voices and noise/distance approximations separately
from calibration, report EER/FAR/FRR and CPU latency in `bench/voice_verify.md`,
and leave robot measurements explicitly unmeasured. Behavior tests use literal
decisions and audio payloads. Run the requested full parallel test suite. No
robot, deployment, companion service, Muse request, or GPU is needed.
