# Run the companion server on a Mac

A Mac on the same network can hear, speak and narrate for a Reachy Mini robot.
The `reachy-companion` command is the server that runs on the Mac.

The robot stays in charge of the conversation. It sends microphone audio and text to the companion, and the companion returns transcripts, speech audio and short spoken lines.
The robot and the server import the same codec, `src/reachy_muse/protocol.py`, so both sides use the same rules. The protocol spec is [reachy-companion-protocol.md](reachy-companion-protocol.md).

Hearing, speech and narration run real models on the Mac. Each engine is a Protocol in `src/reachy_muse/companion_server/engines.py`.

- Hearing uses Voxtral Mini 4B Realtime for the running transcript. Silero VAD and Smart Turn v3 decide when the user has finished. Parakeet TDT v2 then transcribes the whole utterance again for the final text.
- Speech uses Chatterbox Turbo, `mlx-community/chatterbox-turbo-8bit`, whose weights are MIT-licensed. It speaks in the one voice its weights ship with and never clones a voice. It streams: every 20 speech tokens, 0.8 s of speech, it decodes the sentence so far and sends the audio that the next decode will not change, with a 60 ms crossfade at each seam. The companion holds a line's first audio for 150 ms, which covers the wait for the second decode, so the first audio reaches the robot about 0.5 s after it asks (0.6 s at p95), against 0.1 s for Kokoro. The next sentence is ready before the robot finishes the last one. A narrated line's expression adds a sound to its first sentence: happy adds `[chuckle]`, sad `[sigh]` and surprised `[gasp]`. When the server starts, it measures the voice on three fixed sentences and sets one gain that brings the voiced level to -12 dBFS RMS. A limiter with a 2 ms look-ahead then holds peaks under -4 dBFS, so the robot's speaker EQ, which lifts peaks about 2.5 dB, stays under its own limiter. The MLX port does not add the PerTh watermark that Resemble's PyTorch package adds.
- Kokoro-82M `af_heart` stands in for Chatterbox. It speaks every line if Chatterbox fails to load. It speaks the rest of a line when a sentence's first audio takes longer than 1 s or Chatterbox fails mid-line. Kokoro speaks one sentence at a time and levels each sentence on its own.
- `serve` loads the models one at a time and caps MLX at 16 operations per Metal command buffer (`MLX_MAX_OPS_PER_BUFFER`, unless already set). With hearing, the narrator and Chatterbox evaluating at once on their own threads, Metal otherwise reported GPU hangs and aborted the process. The cap costs each model about 18% of its throughput. It makes hangs rarer, not impossible: see [bench/voice_richness.md](../bench/voice_richness.md).
- `serve --voice ENGINE[:VOICE]` picks the voice: `chatterbox-turbo`, the default, or `kokoro[:VOICE]`, such as `kokoro:bm_george`. `--loudness DBFS` sets the level for both. [bench/voice_richness.md](../bench/voice_richness.md) has the measurements.
- Narration uses Qwen3-4B-Instruct-2507, 4-bit, through mlx-lm. It writes the acknowledgement, the progress lines and the condensed answer. See [Narration](#narration). The same model answers `classify`: it names the device command in a short utterance the robot's own grammar did not match, such as "could you be a bit quieter", or none.

A speech benchmark on the Mac, mac-speech-bench-2, chose the speech models. `evals/narrator` chose the narrator.

## What the Mac needs

- A Mac with Apple Silicon. The models run on the GPU through MLX. Smart Turn and the VAD run on the CPU.
- About 15 GB of free memory. The speech models hold about 5 to 9 GB in steady use and peaked at 12 GB in the latency benchmark. The narrator adds 2.3 GB of weights and peaked at 3 GB on the eval set. MLX keeps freed GPU buffers for reuse, and capping that cache at 1 GB made the final text ten times slower.
- About 10 GB of disk for the models.

## Install

Use Python 3.10 or later:

```sh
uv venv --python 3.13
uv pip install -e '.[companion-server,test]'
```

To develop against a local checkout of the Muse Gadget SDK instead of the commit that `pyproject.toml` pins, put the checkout next to this repository:

```sh
uv pip install -e '.[companion-server,test]' --override <(echo "musegadget @ file://$PWD/../muse-gadget-sdk/linux")
uv pip install -e ../muse-gadget-sdk/linux
```

The `companion-server` extra installs the speech models' packages. Without it, only `serve --engines stub` works.

## Pair

```sh
reachy-companion pair
```

This creates a self-signed certificate, its key and a 32-byte token in `~/.reachy-companion`. The directory is readable only by you. The command prints one line:

```
wss://studio.local:8765  pin=sha256:4f…  token=rc1_…
```

Running `pair` again prints the same line. To make a new certificate and token, run `reachy-companion pair --force`. After that, add the companion to every robot again.

Use `--host` when the robot reaches the Mac by another name or address.

## Serve

```sh
reachy-companion serve
```

The server loads and warms the speech and narrator models, one at a time, before it listens. With `--motion` this took 15 to 17 s on the Mac Studio.
The first run also downloads the models, about 9.2 GB, from Hugging Face. Voxtral is 2.9 GB, Parakeet 2.3 GB, Qwen3-4B 2.3 GB, Chatterbox Turbo 1.3 GB and Kokoro 0.3 GB. They go to the Hugging Face cache, `~/.cache/huggingface/hub`. Set `HF_HOME` to put them somewhere else.

After loading, a watchdog thread monitors the asyncio heartbeat and each active
model worker. It logs to stderr and exits with status 70 after 10 seconds without
an event-loop tick or 30 seconds without engine progress. launchd's `KeepAlive`
then restarts the process. Tokens and audio chunks reset the engine timer, so a
long briefing or a motion planner yielding to speech can continue. Cancellation
does not hide native work still running on a model thread. Downloads and model
warm-up have no watchdog deadline. This recovers runtime stalls; serializing GPU
work remains the separate fix for concurrent MLX hangs. See
[the incident report](../bench/companion_hang.md) for bounds and tests.

The server listens on port 8765 at the path `/v1`, with TLS and the paired certificate. Stop it with Ctrl-C.

These options change the engines:

- `--engines stub` serves the stub engines instead: a fixed script, a 440 Hz tone and an echo. The tests use it.
- `--delay-ms 240` makes Voxtral's partial text arrive sooner. The default is 480 ms.
- `--narrator-model mlx-community/Llama-3.1-8B-Instruct-4bit` narrates with another MLX instruction model. The eval set below scores any model you name.
- `--motion` also generates full-body motion for the robot's greeting, thinking and closing moments. See [Generated motion](#generated-motion).
- `--no-final-pass` skips Parakeet and uses Voxtral's own final text. This saves about 1 GB of memory and 2.3 GB of download, but the final text arrives about 130 ms later.

`reachy-companion serve --insecure` serves plain `ws://` without TLS, and listens only on 127.0.0.1 unless you pass `--bind`. Use it only for tests.

## Generated motion

`serve --motion` offers the protocol's `animate` op. It runs
[reachy-motion-generator](https://github.com/pham-tuan-binh/reachy-motion-generator) (Apache-2.0): a
fine-tuned Qwen3.5 4B planner, 8-bit on MLX, writes a short recipe for an intent such as "thinking hard", and a
flow-matching generator on torch MPS turns it into 25 Hz frames for the head, antennas and body.
[bench/motion.md](../bench/motion.md) chose this configuration, and
[bench/generated_motion.md](../bench/generated_motion.md) measures it beside the speech models.

Install the extra into the same environment:

```sh
uv pip install -e '.[companion-server,companion-motion]'
```

The planner and generator live in `~/Library/Application Support/reachy-companion/motion`, outside the
checkout. Pass `--motion-dir DIR` to use another directory. The directory holds two entries:

- `binhpham--reachy-mini-motion-planner-4b@1b60d8f16b85/generator.pt`, the generator, 83 MB.
- `binhpham--reachy-mini-motion-planner-4b@1b60d8f16b85-mlx-8bit/`, the planner converted to 8-bit MLX, 4.2 GB.

When both exist, `serve` loads them without the network. Otherwise the first `serve --motion` downloads the
bundle, about 9 GB with the full-precision planner, and converts the planner, which takes about a minute. After
that, the bundle's `planner/` folder can be deleted.

Motion is decoration. A request the planner cannot answer gets a result with no frames, and the robot keeps its
fixed expressions. The robot keeps at most two `animate` requests outstanding, apart from the narrations.

## Add the companion to the robot

On the robot, paste the values that `pair` printed:

```sh
muse-reachy companion add wss://studio.local:8765 --pin sha256:4f… --token rc1_…
```

The robot then trusts only this certificate and sends the token when it connects. 

## Test

```sh
.venv/bin/python -m pytest tests/companion_server
```

The conformance tests run every golden vector through the codec, then drive the real server with a WebSocket client. They use the stub engines and take about a second.

```sh
.venv/bin/python -m pytest tests/companion_server -m real -s
```

These slow tests load the real models. One checks that Chatterbox starts a line within 1 s without clipping. One sends three recorded requests to the server in real time and checks the transcripts for the words that matter, such as "eight" and "not". Then it checks that speech comes back. Another sends three eval cases to the narrator over the WebSocket and scores its lines.

## Measure latency

```sh
.venv/bin/python bench/latency.py
```

The script starts `serve` and acts as the robot. It reports how long the final text takes after the user stops, how often partial text arrives, and how long the first speech frame takes. Pass `--corpus` to use other WAVs, and any `serve` option such as `--no-final-pass`.

## Compare voices

```sh
.venv/bin/python bench/voices.py --list                # the voices this Kokoro has
.venv/bin/python bench/voices.py                       # six test sentences in six voices, into bench/out/voices
.venv/bin/python bench/score_voices.py bench/out/voices/manifest.json --recordings DIR
```

`voices.py` renders each sentence the way `serve` speaks it: 48 kHz, at the `--loudness` level. The sentences hold names, numbers and a negation, and `manifest.json` lists the words in each that must survive. Play every file on the robot and record it from the robot's microphone under the same path, such as `DIR/bm_george/03.wav`. `score_voices.py` transcribes the recordings with Parakeet and reports, per voice, the word error rate, how many critical words were heard intact, and the share of energy between 1 and 4 kHz. Without `--recordings` it scores the rendered files, which gives the baseline before the speaker and the room. Pick the winner with `serve --voice NAME`.

## Narration

The robot asks for three kinds of line:

- `acknowledge` restates the request right after the user stops talking. It never claims a result.
  A greeting, a thank-you, other small talk, or a request under three words gets no line. A fixed rule decides this before the model runs, because Qwen3-4B restated "how are you" as "Okay, checking how you are." even when the prompt told it to stay silent.
- `progress` says what is happening, based only on Muse's public status label.
- `lines` condenses Muse's reply into one to three sentences, each with an expression from the robot's vocabulary, `reachy_muse.contract.expressions.Expression`.

The model answers in JSON. The narrator parses that answer and checks every line before it sends it:

- at most one line for `acknowledge` and `progress`, at most three for `lines`, each at most 240 bytes, with no markup;
- an expression from the contract's `Expression`;
- no result claimed in an acknowledgement, and no repeat of a progress line already said;
- no fact that the request, status or reply does not state. `src/reachy_muse/companion_server/grounding.py` checks numbers, number words, ordinals, decimals, times, amounts, date words and names. It also runs the robot's `grounded()` from `reachy_muse.capabilities`, so the robot accepts every line it passes.

A rejected answer gets one retry that names the problems, if the op's deadline leaves time. After that, `acknowledge` and `progress` return no lines, so the robot says its own, and `lines` returns the reply's first three sentences without markup.

### Model choice

`evals/narrator` holds 67 cases and 30 held-out cases written without sight of the prompts. Each case lists facts the narrator must not invent. Four models were scored on an M3 Ultra, each alone in its own process. A case passes when its final lines are grounded, avoid every forbidden fact, fit the format and carry valid expressions. "Raw" scores the model's first answer before the check. Latency is the time for a whole narration, retries included.

| Model | License | Pass (67 + 30) | Raw pass | Raw ungrounded | Fallbacks | Ack p50/p95 | Progress p50/p95 | Lines p50/p95 | Weights | MLX peak |
|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-4B-Instruct-2507 4-bit | Apache-2.0 | 67 + 30 | 67 + 30 | 0 + 0 | 0 + 0 | 0.33/0.36 s | 0.32/0.33 s | 0.45/0.74 s | 2.3 GB | 3.0 GB |
| Phi-4-mini-instruct 4-bit | MIT | 66 + 30 | 54 + 20 | 7 + 7 | 8 + 8 | 0.36/0.78 s | 0.35/0.80 s | 0.34/0.73 s | 2.2 GB | 3.0 GB |
| Llama-3.1-8B-Instruct 4-bit | Llama 3.1 Community | 67 + 29 | 61 + 28 | 1 + 0 | 0 + 0 | 0.54/0.59 s | 0.53/0.55 s | 0.63/2.05 s | 4.5 GB | 5.3 GB |
| granite-4.1-8b 4-bit | Apache-2.0 | 67 + 30 | 61 + 24 | 3 + 5 | 4 + 5 | 0.63/0.73 s | 0.65/0.70 s | 0.84/2.53 s | 5.2 GB | 5.9 GB |

After the table was made, the main set gained five small-talk cases that expect no acknowledgement. Qwen3-4B passes all 72, with 0 ungrounded. The other models were not rerun.

Latencies are from the 67-case set. The robot waits 0.9 s for an acknowledgement, 1.5 s for progress and 4 s for lines.

Through `serve` with every real engine loaded, over the WebSocket, Qwen3-4B took 0.39/0.44 s to acknowledge, 0.37/0.38 s for progress and 0.53/0.79 s for lines (p50/p95, all 97 cases), while a speak stream ran beside each narration. The server's memory footprint peaked at 12 GB.

Qwen3-4B is the default. Its first answer passed every case in both sets, so it never needed a retry or a fallback. It is also the fastest at the acknowledgement and holds the least memory but one. After the check, no model returned an ungrounded line. Llama said a forbidden fact once on the held-out set, and Phi once on the main set.

To score a model:

```sh
.venv/bin/python evals/narrator/run.py --model mlx-community/Qwen3-4B-Instruct-2507-4bit
.venv/bin/python evals/narrator/run.py --cases holdout.json --model mlx-community/Qwen3-4B-Instruct-2507-4bit
.venv/bin/python evals/narrator/run.py --report
```

Without `--model`, it scores the four candidates. Results go to `evals/narrator/results`, one file per model and case set, with every line and every first answer.
