# Voice richness and loudness

This bench asked two questions about companion speech on Reachy Mini Wireless:

1. Does a newer open-weight TTS model sound richer than Kokoro-82M and still start speaking fast enough?
2. Why is the voice hard to hear on the robot, and what fixes it?

Short answers: on the numbers, no model clearly beat Kokoro `af_heart` within the latency budget. By ear, the user picked Chatterbox Turbo, which is now the default voice, streamed (see [Chatterbox Turbo as the voice](#chatterbox-turbo-as-the-voice)). The voice was quiet because the chain left about 12 dB unused: 9 dB at the robot's volume setting and about 3.4 dB in the companion's gain stage. The gain stage now recovers the 3.4 dB.

## Method

Every model spoke 26 lines on the Mac Studio (M3 Ultra) through mlx-audio 0.5.8. Six of the lines are listening lines: a greeting, an acknowledgement, a progress line, an answer with numbers, an apology and a three-sentence answer. The other 20 are typical Reachy answers with names, times, amounts and short acknowledgements. Each line was split into sentences and synthesized one sentence per call, the way `KokoroVoice` speaks.

- **First chunk.** Time to the first streamed audio of the first sentence.
- **Whole sentence.** Time until the whole first sentence exists. The companion levels each sentence as a whole, so this is the time to first audio today.
- **Under load.** The same run while Qwen3-4B 4-bit (the narrator and motion-planner size) decoded back to back in another process, at about 155 tokens per second. The live companion stayed up and idle.
- **WER.** Each WAV was transcribed by Parakeet TDT 0.6B v2 and compared after spelling out numbers, times and amounts.
- **MOS.** UTMOS22 (strong) predicted naturalness, from 1 to 5. It is a proxy for a listening test, not a substitute.
- **Memory.** Peak RSS of the process and the MLX peak, in GB.

## Models

The 500 ms budget applies to the whole-sentence column for any engine that does not stream into the gain stage.

| Model, voice | License | First chunk p50/p95 ms | Whole sentence p50/p95 ms | Under load p95 ms (chunk / whole) | RTF | RSS / MLX GB | WER | MOS |
|---|---|---|---|---|---|---|---|---|
| Kokoro-82M `af_heart` | Apache-2.0 | 112 / 166 | 123 / 181 | 279 / 293 | 0.04 | 0.8 / 2.5 | 0.8% | 4.50 |
| Kokoro-82M `am_fenrir` | Apache-2.0 | 108 / 167 | 119 / 180 | | 0.04 | 0.9 / 2.5 | 0.4% | 4.44 |
| Kokoro-82M `bf_emma` | Apache-2.0 | 120 / 165 | 130 / 180 | | 0.04 | 0.8 / 2.5 | 0.8% | 4.35 |
| Kokoro-82M `am_michael` | Apache-2.0 | 118 / 355 | 128 / 371 | | 0.04 | 0.8 / 2.6 | 0.8% | 4.26 |
| Kokoro-82M `af_bella` | Apache-2.0 | 116 / 166 | 128 / 181 | | 0.04 | 0.8 / 2.5 | 0.4% | 4.25 |
| Qwen3-TTS 0.6B CustomVoice 8-bit, `aiden` | Apache-2.0 | 103 / 211 | 848 / 1651 | 157 / 2018 | 0.28 | 2.2 / 2.6 | 1.1% | 4.42 |
| Qwen3-TTS 0.6B CustomVoice 8-bit, `ryan` | Apache-2.0 | 108 / 125 | 1320 / 3612 | | 0.27 | 2.5 / 2.6 | 3.0% | 3.80 |
| Kyutai Pocket TTS, `alba` | CC-BY-4.0 (credit Kyutai) | 30 / 37 | 179 / 333 | 46 / 395 | 0.07 | 0.6 / 1.0 | 0.4% | 4.29 |
| Kyutai Pocket TTS, `marius` | CC-BY-4.0 (credit Kyutai) | 30 / 34 | 165 / 277 | | 0.07 | 0.4 / 1.0 | 6.4% | 2.93 |
| Soprano 1.1 80M | Apache-2.0 | 145 / 281 | 146 / 282 | 339 / 341 | 0.06 | 0.8 / 0.4 | 4.2% | 4.15 |
| Chatterbox Turbo 8-bit | MIT | 204 / 291 | 1489 / 2591 | 269 / 2938 | 0.53 | 2.0 / 2.6 | 0.0% | 3.70 |
| Marvis 250M 8-bit, `conversational_a` | Apache-2.0 | 352 / 383 | 1515 / 2090 | 407 / 2665 | 0.52 | 2.2 / 2.5 | 0.8% | 3.99 |
| VibeVoice Realtime 0.5B 8-bit, `en-Emma_woman` | MIT, model card limits use to research | 59 / 76 | 718 / 1284 | 95 / 1425 | 0.22 | 2.3 / 1.5 | 7.9% | 4.00 |
| Orpheus 3B 4-bit, `tara`, streaming | Apache-2.0 card on a Llama 3.2 base | 257 / 272 | 1902 / 2896 | | 0.36 | 2.5 / 2.5 | 26.0% | 2.96 |
| Orpheus 3B 4-bit, `tara`, not streaming | same | 1627 / 3112 | 1632 / 3124 | | 0.44 | 2.4 / 5.1 | 1.9% | 4.03 |

Notes:

- Sesame CSM-1B was not measured. Its default voice prompts download from the gated `sesame/csm-1b` repo, and this Mac has no Hugging Face login. Marvis uses the same architecture.
- mlx-audio's Orpheus streaming path repeats audio ("good good", "Lis to Lisbon"), which explains its 26% WER.
- Breeze TTS 2, Voxtral TTS, Fish S1-mini and S2 Pro, Higgs Audio v3, Spark-TTS and Echo TTS were left out because their weights are non-commercial or research-only. VoxCPM2 and Dia were left out because they cannot stream and are 1.6 to 2B parameters.
- Pocket TTS samples its output, so the same line differs from take to take. Kokoro is deterministic.

Only Kokoro, Pocket TTS and Soprano fit the budget without streaming. Pocket and Soprano score below Kokoro `af_heart` on MOS, and Soprano misreads times ("three dot dot teen"). Qwen3-TTS `aiden` comes closest on MOS and is the most expressive by reputation. Its first chunk arrives in 211 ms p95, but the companion would have to level and limit audio chunk by chunk instead of sentence by sentence, because the whole first sentence takes 1.65 s at p95. No model clearly beats Kokoro, so Kokoro `af_heart` stays the default. The listening page lets you judge richness by ear, since MOS cannot measure it.

## Why the voice was quiet

The robot plays everything at 16 kHz: `/proc/asound/card0/stream0` lists 16000 Hz as the only playback rate, and `/etc/asound.conf` runs the `reachymini_audio_sink` dmix at 16 kHz. Nothing above 8 kHz reaches the speaker, whatever the model.

`bench/speaker_chain.py` runs the companion's gain stage and the robot's real GStreamer EQ bin offline, then applies the ALSA mixer. Means over the six listening lines in Kokoro `af_heart`, in dBFS over voiced frames:

| Stage | Before: RMS | Before: 300 Hz to 8 kHz | Before: 1 to 4 kHz | After: RMS | After: 300 Hz to 8 kHz | After: 1 to 4 kHz |
|---|---|---|---|---|---|---|
| Kokoro output | -25.1 | -28.1 | -35.1 | -25.1 | -28.1 | -35.1 |
| Companion PCM | -17.9 | -21.1 | -28.0 | -13.8 | -17.5 | -24.9 |
| After the robot EQ | -18.7 | -20.3 | -25.2 | -14.9 | -16.9 | -22.1 |
| At the speaker, volume 85 | -27.7 | -29.3 | -34.2 | -23.9 | -25.9 | -31.1 |

Where the level went:

- **Volume 85 is -9 dB.** `amixer -c 0 scontents` on the robot shows `PCM` at 51 of 60 steps, `[-9.00dB]`. Volume 100 is 0 dB. This is the largest loss, and it is a setting, not code.
- **The companion aimed low.** It raised each sentence to -20 dBFS RMS, counting pauses. Peaks then sat near -2.5 dBFS, so a soft knee could not go higher without a look-ahead limiter.
- **The robot EQ adds peaks, not level.** The companion profile cuts 2 to 6.6 dB under 500 Hz and lifts 2 to 3 dB from 950 Hz to 3.8 kHz. That lifts peaks by about 2.5 dB, up to the 0.9 hard limit of its `audiodynamic` element. The overall RMS falls by about 1 dB.
- **No resampling loss.** The companion already speaks at the 16 kHz the robot asks for. The robot resamples to 48 kHz for its EQ and back to 16 kHz.

## What changed

`Loudness` now raises the voiced frames of each sentence to -12 dBFS RMS. A limiter with a 2 ms look-ahead holds peaks under -4 dBFS, which leaves room for the robot EQ's 2.5 dB lift under its 0.9 limit. The share of samples that reach the robot limiter stays near zero: 0.004%, against 0.003% before. Speech at the speaker is 3.8 dB louder (RMS) and 3.4 dB louder in the 300 Hz to 8 kHz band. MOS after the chain is 4.42 over all 26 lines, against 4.47 before.

Shaping was tried and rejected. On Kokoro `af_heart` over all 26 lines through the robot EQ:

| Gain stage | 300 Hz to 8 kHz dBFS | MOS |
|---|---|---|
| Before: -20 dBFS, soft knee | -20.5 | 4.47 |
| -12 dBFS, look-ahead limiter (shipped) | -17.0 | 4.42 |
| plus a 150 Hz high-pass and a 3 dB lift at 2.5 kHz | -17.0 | 4.34 |
| plus 3:1 compression | -16.2 | 4.23 |
| 200 Hz high-pass, 5 dB lift at 2.5 kHz, 3:1 compression | -15.5 | 4.10 |

The high-pass and the presence lift added almost no level. Compression bought 1 to 1.5 dB at a clear cost in predicted naturalness. MOS cannot judge a small driver that would not play the low end anyway, so a listening test on the robot could still favor the presence lift.

## Recommendations

1. Set the robot volume to 100 for +9 dB: `curl -X POST localhost:8000/api/volume/set -H 'Content-Type: application/json' -d '{"volume": 100}'` on the robot, or say "louder" twice (each step is 10). Listen for distortion at full volume before keeping it, since this bench played nothing on the robot.
2. Deploy the new gain stage to the Mac companion. The robot needs no change.

## Chatterbox Turbo as the voice

The listening page's Chatterbox row used mlx-audio's own streaming, with Chatterbox's built-in voice from `conds.safetensors` and no reference audio. The companion keeps that voice and clones no one. The weights are MIT. The MLX port does not add the PerTh watermark that Resemble's PyTorch package adds.

### Streaming without seams

mlx-audio's `generate(stream=True)` decodes all the speech tokens so far at every step and appends the samples past the previous end. Its decoder looks ahead at later tokens and draws fresh noise at every call, so each seam is a jump. Over 16 sentences decoded from the same tokens (UTMOS22):

| Decode | UTMOS | Decode time |
|---|---|---|
| One decode of the whole sentence | 4.25 | 9.3 s |
| mlx-audio's streaming splice | 3.70 | 36.1 s |
| Every 10 tokens, 80 ms held back, 40 ms crossfade | 4.01 | 24.2 s |
| The same, with noise drawn by position | 4.05 | 22.6 s |
| Every 20 tokens, 240 ms held back, 60 ms crossfade, noise by position (shipped) | 4.21 | 9.9 s |

The shipped decoder (`voice_chatterbox.py`) re-decodes every 20 speech tokens (0.8 s of speech). It sends what the next decode will not change and crossfades each seam. It draws the noise from per-sentence buffers so that a longer decode extends a shorter one's noise. Drawing those buffers once per sentence, not once per decode, matters: per decode, MLX's cache grew to 37 GB over 20 lines, against 5 GB.

### Level

Streamed speech cannot be leveled per sentence. At startup the voice speaks three fixed sentences and sets one gain that brings their voiced level to -12 dBFS before the limiter. Chatterbox's lines vary about 1 dB around that, against 0.4 dB for Kokoro. Its peaks sit higher, so after the -4 dBFS limiter it lands about 1.5 dB under Kokoro. The calibration moves about 1 dB between restarts because the model samples.

### Expressions

Turbo ignores `exaggeration` and `cfg_weight`: mlx-audio says so and warns. Its tokenizer has paralinguistic tags. On 4 lines, 2 takes each, event tags (`[chuckle]`, `[sigh]`, `[gasp]`, `[laugh]`) add an audible sound of 0.3 to 0.5 s and widen the pitch range, with WER at the untagged level (0 to 3.3% with fillers such as "hm" ignored; untagged 1.7%). Style tags (`[happy]`, `[surprised]`, `[angry]`, `[sarcastic]`, `[dramatic]`) changed neither length nor pitch measurably. `[laugh]` once turned "three" into "dream". So a narrated line's expression adds `[chuckle]` for happy, `[sigh]` for sad and `[gasp]` for surprised to its first sentence. Every other expression adds nothing. The robot's speak request carries no expression, so the companion remembers the expression of each line it narrated (the last 64) and finds it by the line's text.

### Latency and load

`bench/voice_load.py` starts its own companion with `--motion` and acts as the robot at 16 kHz. Each line is timed from `speak_start` to each speech frame received over loopback. That time is the companion's time from text to first frame, plus under a millisecond. Under load, a second connection hears `reminder.wav` in real time, over and over. A third asks for narrated lines every 4 s ("realistic") or back to back ("full"). A fourth keeps two `animate` requests outstanding. 8 lines, 3 runs, so 24 lines per cell. Runs are in `bench/voice_load_runs/`.

| | Kokoro | Chatterbox Turbo |
|---|---|---|
| First audio alone, p50 / p95 | 0.12 / 0.27 s | 0.50 / 0.59 s |
| First audio, realistic load | 0.19 / 0.39 s | 0.63 / 0.90 s |
| First audio, full load | 0.28 / 0.57 s | 0.50 / 1.29 s |
| RTF alone, p50 / p95 | 0.04 / 0.05 | 0.43 / 0.46 |
| RTF, realistic load | 0.08 / 0.10 | 0.60 / 0.75, max 0.78 |
| Lines with a gap, realistic load | 0 of 24 | 4 of 24, worst 0.34 s |
| Footprint loaded / after the run / peak | 12.6 / 12.2 / 17.4 GB | 13.8 / 25.2 / 25.3 GB |
| Load time with `--motion` | 12 to 13 s | 15 to 17 s |

Chatterbox's first audio includes a 150 ms hold. Without the hold, 5 of 24 lines alone had a gap of up to 0.23 s while the robot waited for the second decode. With it, none did. RTF stays under 0.8, so a longer buffer was not needed.

The other speech paths, alone and while another connection speaks without pause (realistic run, p50 / p95, s):

| Path | Alone | While Kokoro speaks | While Chatterbox speaks |
|---|---|---|---|
| acknowledge | 0.49 / 0.50 | 0.62 / 0.63 | 0.69 / 0.70 |
| lines | 0.84 / 0.86 | 1.17 / 1.19 | 1.02 / 1.10 |
| classify | 0.08 / 0.08 | 0.11 / 0.15 | 0.09 / 0.10 |
| hear final | 0.66 / 0.66 | 2.32 / 2.77 | 0.71 / 0.72 |

Chatterbox slows narration no more than Kokoro does, and slows the final transcript less. This bench did not measure why Kokoro slows the final transcript so much. Narration alone is slower than in `bench/generated_motion.md` (lines 0.84 s against 0.67 s) because of the command buffer cap below.

### Fallback

Kokoro `af_heart` loads beside Chatterbox. If Chatterbox fails to load, Kokoro speaks every line. If a sentence's first audio takes more than 1 s, or Chatterbox fails mid-line, Kokoro speaks the rest of the line, and the companion logs a warning. A failure mid-sentence repeats that sentence's start in Kokoro's voice. Underruns are logged, not acted on. One worker runs Chatterbox, so a second speak stream that overlaps a first waits behind it, misses the deadline and goes to Kokoro. The robot speaks one line at a time, so this happens only when streams overlap.

### Open: GPU hangs under concurrent load

With several MLX models evaluating at once on their own threads, Metal can report "Caused GPU Hang Error". The thread that sees it raises, and other threads' work is discarded. Sometimes the process aborts (`libc++abi: terminating`). This is not specific to Chatterbox: the full-load bench hung with Kokoro too. A three-thread repro (`bench/gpu_hang.py`: Qwen3-4B, Kokoro and Chatterbox in loops for 3 min) failed in 4 of 6 runs with default command buffers and 2 of 3 with 24 operations per buffer. It passed all 9 runs with 16 or fewer. A lock that gave the GPU to one thread at a time passed too, but starved the narrator. `serve` now sets 16 and loads its models one at a time, because warming them together also hung. With both, the companion's bench ran 4 times with Chatterbox and 2 times with Kokoro, without an abort. Kokoro's runs were clean. 3 of the 4 Chatterbox runs logged a hang or discarded work: the line failed over to Kokoro and the hearing or narration in flight failed. Before both changes, runs with either voice aborted. The fix that removes the cause is to give the GPU to one engine step at a time across hearing, speech, narration and motion, extending `SpeechFirst` into a scheduler. That is its own change.
