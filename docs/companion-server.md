# reachy-companion

A Mac on the same network can hear, speak and narrate for a Reachy Mini robot.
This repository is the server that runs on the Mac.

The robot stays in charge of the conversation. It sends microphone audio and text to the companion, and the companion returns transcripts, speech audio and short spoken lines.
The wire protocol belongs to the robot's SDK, [muse-gadget-sdk](../muse-gadget-sdk). This server imports the SDK's codec, so both sides use the same rules. The protocol spec is `linux/docs/reachy-companion-protocol.md` in that repository.

Hearing and speech run real models on the Mac. Narration is still a stub that repeats the first three sentences it gets. Each engine is a Protocol in `reachy_companion/engines.py`.

- Hearing uses Voxtral Mini 4B Realtime for the running transcript. Silero VAD and Smart Turn v3 decide when the user has finished. Parakeet TDT v2 then transcribes the whole utterance again for the final text.
- Speech uses Kokoro-82M. It speaks one sentence at a time, at the rate the robot asks for.

The `mac-speech-bench-2` study in the SDK repository chose these models.

## What the Mac needs

- A Mac with Apple Silicon. The models run on the GPU through MLX. Smart Turn and the VAD run on the CPU.
- About 12 GB of free memory. The server holds about 5 to 9 GB in steady use and peaked at 12 GB in the latency benchmark. MLX keeps freed GPU buffers for reuse, and capping that cache at 1 GB made the final text ten times slower.
- About 6 GB of disk for the models.

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

The server loads and warms the speech models before it listens. This takes about 10 s.
The first run also downloads the models, about 5.6 GB, from Hugging Face. Voxtral is 2.9 GB, Parakeet 2.3 GB and Kokoro 0.3 GB. They go to the Hugging Face cache, `~/.cache/huggingface/hub`. Set `HF_HOME` to put them somewhere else.

The server listens on port 8765 at the path `/v1`, with TLS and the paired certificate. Stop it with Ctrl-C.

These options change the engines:

- `--engines stub` serves the stub engines instead: a fixed script, a 440 Hz tone and an echo. The tests use it.
- `--delay-ms 240` makes Voxtral's partial text arrive sooner. The default is 480 ms.
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

This slow test loads the real models. It sends three recorded requests to the server in real time and checks the transcripts for the words that matter, such as "eight" and "not". Then it checks that speech comes back.

## Measure latency

```sh
.venv/bin/python bench/latency.py
```

The script starts `serve` and acts as the robot. It reports how long the final text takes after the user stops, how often partial text arrives, and how long the first speech frame takes. Pass `--corpus` to use other WAVs, and any `serve` option such as `--no-final-pass`.
