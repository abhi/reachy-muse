# Wake reliability design

The runner owns microphone continuity, the wake window, and the trusted audio
handoff. The detector owns native KWS and token alignment. A shared `WakePolicy`
applies the same candidate decisions in production, operator trials, and replay.

```python
policy = WakePolicy(verifier=WakeVerifier(local=recognizer, companion=get_link))
accepted = await policy.feed(detector, samples_16khz, context=WakeContext(
    armed=True, blocked=False, speaking=False, playback=(),
))
# Only accepted candidates enter the existing epoch/feed-checked handoff.
```

`WakeContext` contains authorization and playback facts, never a second copy of
conversation state. `WakeCandidate` contains bounded in-memory audio, the native
trigger threshold and boost, and a nullable score. `WakeDecision` contains only
numeric diagnostics and fixed reason codes. No audio or text enters event logs.

The native stream uses a permissive candidate floor. Each exported candidate
receives a decision, including cooldown, speaking, echo, below-threshold, and
verifier rejection. An explicit verifier rejection is final. Missing or late
verification requires a conservative acoustic replay; it cannot authorize a
weak candidate. Replay confidence bounds describe the replay, not the live
posterior. The pinned sherpa binding does not expose that posterior or native
hypotheses below the candidate floor.

Companion and local verification share one deadline. Local work starts early so
a stalled companion cannot consume its allowance. Every private hearing stream
is aborted on cancellation. Audio capture continues through the existing
producer while the consumer awaits a bounded decision. A queued capture gap
invalidates acceptance before the runner opens a call. The existing timed
handoff remains authoritative; no provisional wake authorizes request audio.

Playback containing the wake phrase rejects as echo. Unknown or unqualified
playback rejects as speaking. Qualified duplex with known unrelated playback
can accept a verified user wake. This deliberately rejects a simultaneous user
wake overlapping playback of the same phrase. Transcript checks cannot prove
speaker identity, and hardware AEC qualification is not a residual-echo test.

Two designs were compared before implementation. The independent judge scored
runner-owned observations 21/25 and a new coordinator 20/25. The former is the
base because it preserves handoff ownership. Shared decision logic and early
local verification came from the coordinator design. A new asynchronous pending
handoff lifecycle was rejected: bounded consumer waits use the existing capture
queue and keep epoch/feed identity intact. Always-on ASR was rejected because it
adds continuous CM4 work before candidate-only verification has been measured.

Capture snapshots carry playback evidence through the queue and batch merge.
Processing-time playback state can expire while old speaker audio is waiting.
Unknown output remains blocked even when a batch also has known unrelated text.
Independent operator trials reset cooldown; production gap resets retain it.

Replay found encoder-phase sensitivity on one identical clean recording.
Bounded candidate replay now trims its start to an original encoder boundary,
so alignment does not silently change phase. Two staggered continuous streams
added no tuning positives; four added one while consuming four times the CPU.
They remain excluded from the live detector. Three BPE segmentations and the
homophone spellings MEWS/HAY in the KWS graph also added no tuning candidates.
The verifier accepts HAY MUSE because the local recognizer uses HAY for /heɪ/.
It continues to reject PAY MUSE, HI MUSE, HEY NEWS, HEY MUSIC, and HEY MOOSE.

The model has a 320 ms decode stride and additional lookahead. A universal
300 ms acknowledgement bound is therefore not achievable with this artifact.
Report detection, alignment, and verification latency separately. Preserve the
tested privacy boundary while measuring the cost of alignment.

Verification proceeds through literal behavior tests, CPU-only replay with
seeded augmentation and held-out voices, the existing handoff tests, and the
complete parallel suite. Field recall, actual moving-head noise, real residual
echo, and robot CPU measurements remain separate from synthetic evidence.
