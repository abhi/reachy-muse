# reachy-companion

A Mac on the same network can hear, speak and narrate for a Reachy Mini robot.
This repository is the server that runs on the Mac.

The robot stays in charge of the conversation. It sends microphone audio and text to the companion, and the companion returns transcripts, speech audio and short spoken lines.
The wire protocol belongs to the robot's SDK, [muse-gadget-sdk](../muse-gadget-sdk). This server imports the SDK's codec, so both sides use the same rules. The protocol spec is `linux/docs/reachy-companion-protocol.md` in that repository.

Today the server runs stub engines. Hearing replays a fixed script, speech is a 440 Hz tone, and narration repeats the first three sentences it gets. Real models replace them in later steps. Each engine is a Protocol in `reachy_companion/engines.py`.

## Install

The SDK checkout must sit next to this one.

```sh
uv venv
uv pip install -e ../muse-gadget-sdk/linux -e '.[test]'
```

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

The server listens on port 8765 at the path `/v1`, with TLS and the paired certificate. Stop it with Ctrl-C.

`reachy-companion serve --insecure` serves plain `ws://` without TLS. Use it only for tests.

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

The conformance tests run every golden vector from the SDK through the codec, then drive the real server with a WebSocket client.
