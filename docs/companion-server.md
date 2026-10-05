# reachy-companion

A Mac on the same network can hear, speak and narrate for a Reachy Mini robot.
This repository is the server that runs on the Mac.

The robot stays in charge of the conversation. It sends microphone audio and text to the companion, and the companion returns transcripts, speech audio and short spoken lines.
The wire protocol belongs to the robot's SDK, [muse-gadget-sdk](../muse-gadget-sdk). This server imports the SDK's codec, so both sides use the same rules. The protocol spec is `linux/docs/reachy-companion-protocol.md` in that repository.

Hearing, speech and narration run real models on the Mac. Each engine is a Protocol in `reachy_companion/engines.py`.

- Hearing uses Voxtral Mini 4B Realtime for the running transcript. Silero VAD and Smart Turn v3 decide when the user has finished. Parakeet TDT v2 then transcribes the whole utterance again for the final text.
- Speech uses Kokoro-82M. It speaks one sentence at a time, at the rate the robot asks for, raised to -20 dBFS RMS with peaks held under -1 dBFS. `serve --voice NAME` picks the voice (default `af_heart`) and `--loudness DBFS` the level.
- Narration uses Qwen3-4B-Instruct-2507, 4-bit, through mlx-lm. It writes the acknowledgement, the progress lines and the condensed answer. See [Narration](#narration).

The `mac-speech-bench-2` study in the SDK repository chose the speech models. `evals/narrator` chose the narrator.

## What the Mac needs

- A Mac with Apple Silicon. The models run on the GPU through MLX. Smart Turn and the VAD run on the CPU.
- About 15 GB of free memory. The speech models hold about 5 to 9 GB in steady use and peaked at 12 GB in the latency benchmark. The narrator adds 2.3 GB of weights and peaked at 3 GB on the eval set. MLX keeps freed GPU buffers for reuse, and capping that cache at 1 GB made the final text ten times slower.
- About 8 GB of disk for the models.

## Install

The SDK checkout must sit next to this one.

```sh
uv venv
uv pip install -e ../muse-gadget-sdk/linux -e '.[test,real]'
```

The `real` extra installs the speech models' packages. Without it, only `serve --engines stub` works.

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

The server loads and warms the speech and narrator models before it listens. This takes about 10 s.
The first run also downloads the models, about 7.9 GB, from Hugging Face. Voxtral is 2.9 GB, Parakeet 2.3 GB, Qwen3-4B 2.3 GB and Kokoro 0.3 GB. They go to the Hugging Face cache, `~/.cache/huggingface/hub`. Set `HF_HOME` to put them somewhere else.

The server listens on port 8765 at the path `/v1`, with TLS and the paired certificate. Stop it with Ctrl-C.

These options change the engines:

- `--engines stub` serves the stub engines instead: a fixed script, a 440 Hz tone and an echo. The tests use it.
- `--delay-ms 240` makes Voxtral's partial text arrive sooner. The default is 480 ms.
- `--narrator-model mlx-community/Llama-3.1-8B-Instruct-4bit` narrates with another MLX instruction model. The eval set below scores any model you name.
- `--no-final-pass` skips Parakeet and uses Voxtral's own final text. This saves about 1 GB of memory and 2.3 GB of download, but the final text arrives about 130 ms later.

`reachy-companion serve --insecure` serves plain `ws://` without TLS, and listens only on 127.0.0.1 unless you pass `--bind`. Use it only for tests.

## Add the companion to the robot

On the robot, paste the values that `pair` printed:

```sh
muse-reachy companion add wss://studio.local:8765 --pin sha256:4f… --token rc1_…
```

The robot then trusts only this certificate and sends the token when it connects. The `companion add` command arrives in the SDK's companion client step (S4).

## Test

```sh
.venv/bin/python -m pytest
```

The conformance tests run every golden vector from the SDK through the codec, then drive the real server with a WebSocket client. They use the stub engines and take about a second.

```sh
.venv/bin/python -m pytest -m real -s
```

These slow tests load the real models. One sends three recorded requests to the server in real time and checks the transcripts for the words that matter, such as "eight" and "not". Then it checks that speech comes back. The other sends three eval cases to the narrator over the WebSocket and scores its lines.

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
- `lines` condenses Muse's reply into one to three sentences, each with an expression from the SDK's vocabulary.

The model answers in JSON. The narrator parses that answer and checks every line before it sends it:

- at most one line for `acknowledge` and `progress`, at most three for `lines`, each at most 240 bytes, with no markup;
- an expression from the SDK's `Expression`;
- no result claimed in an acknowledgement, and no repeat of a progress line already said;
- no fact that the request, status or reply does not state. `reachy_companion/grounding.py` checks numbers, number words, ordinals, decimals, times, amounts, date words and names. It also runs the SDK's `grounded()`, so the robot accepts every line it passes.

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
