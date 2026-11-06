# Realtime call input, November 5, 2026

The first live call on `17cbde4` started Muse's answer 3.76 seconds after the
user stopped speaking, then Reachy interrupted it 0.95 seconds later. The
supplied log contains no microphone recording, so self-echo is a likely cause,
not an acoustically verified diagnosis. The code allowed 300 ms of VAD alone
to interrupt when streaming words were unavailable.

That call also waited for the companion's final endpoint, then ran Whisper on
the robot. The log reports 10.9 seconds for recognition of 2.780 seconds of
audio, including 7.853 seconds in the full decode, during Muse's playback.

## Input behavior

Calls now use streaming recognition for device commands only. Local hearing
cannot start a batch transcriber in this mode. Companion hearing snapshots its
latest partial and aborts the stream at the local endpoint instead of asking
for a final. A local streaming final gets at most 250 ms before cancellation.
Recognition failures produce no retry or greeting cues during a healthy call.

While Muse plays, its transcript deltas are available to the echo filter even
before sentence alignment or a final transcript arrives. The gate requires
300 ms of speech and 120 ms of words distinct from the reply. One-word
hypotheses cannot interrupt except for stop commands. Echoed stop commands are
still rejected. Until an interruption is confirmed, the remote VAD receives
silence. Confirmation releases at most two seconds of buffered audio.
Unconfirmed playback-time utterances are discarded before the turn queue.

This chooses recognized words as evidence; microphone energy alone cannot
distinguish a loud echo from the user. If recognition is unavailable, the gate
keeps playback-time audio out. It resumes ordinary microphone streaming between
replies. A real user who repeats Muse's exact words can also be suppressed.
ASR errors and actual room acoustics still need a later authorized robot test.

The behavioral tests reproduced `batch_calls: 1 != 0`, `barge_ins: 1 != 0`, and
13 voiced echo frames reaching Muse before the fix. They now check literal
outcomes for echo, noise, real interruption, sleep, volume, face following,
quoted stop commands, companion partials, and stalled streaming recognition.
They use scripted partials to test routing separately from ASR accuracy.

## Offline CPU replay

Run from the repository with local streaming model assets:

```sh
uv run --no-project --with sherpa-onnx==1.13.8 --with faster-whisper==1.2.1 \
  --with 'av>=12' --with 'webrtcvad-wheels>=2.0.14' \
  env PYTHONPATH=src python bench/realtime_input.py --vad silero \
  --streaming-model /path/to/zipformer-en-2023-06-26 --json bench/realtime_input.json
```

The replay uses the three checked-in speech recordings, each with one second
of trailing silence, repeated five times. It feeds 16 ms microphone chunks
through the real recorder, Silero VAD, echo gate and 24 kHz resampler. A separate
pass runs the real single-thread Sherpa model and its endpoint padding over the
same audio. The script neither connects to Muse nor accesses hardware. Model
loading is excluded from steady-state CPU time.

Results from [the saved report](realtime_input.json), on a Darwin arm64 host:

| Work over 138 seconds of audio | CPU seconds | Fraction of one host core | Feed p95 | Maximum feed |
|---|---:|---:|---:|---:|
| Silero, recording, echo gate, resampling | 0.618 | 0.45% | 0.119 ms | 11.321 ms |
| Sherpa native streaming inference | 4.960 | 3.59% | 0.061 ms | 23.908 ms |

Combined compute is about 4.0% of one host core. With companion partials, the
robot does not run the Sherpa part. The longest local model final was 22.8 ms
on this host. The benchmark excludes subprocess IPC, executor scheduling,
camera processing, motion, and speaker work. It measures the robot's input code
on the Mac, not CM4 utilization. The old log reports elapsed recognition time,
so it cannot support a CPU speedup ratio against this measurement. Removing
all batch-ASR calls eliminates the logged full-Whisper work from the call path;
the regression test verifies zero batch calls.
