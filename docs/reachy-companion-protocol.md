# Reachy companion protocol, version 1

This is the normative wire protocol between a Reachy Mini robot and its companion.
The companion is a computer on the same LAN that hears, speaks, narrates, classifies device commands and generates motion for the robot.
The robot owns the conversation. The companion is stateless between requests.

The codec is `src/reachy_muse/protocol.py`.
Golden vectors ship in the package at `src/reachy_muse/data/companion_protocol_v1/`.
A companion implementation must pass every vector.

## Transport

- The robot opens one WebSocket to `wss://HOST:8765/v1`. The robot is the client.
- Pairing stores the base address, `wss://HOST:8765`. The robot adds `/v1`.
- The pin is `sha256:` and 64 lowercase hex digits, the digest of the certificate in DER form.
- TLS is pinned. The robot accepts only the companion's self-signed certificate whose SHA-256 fingerprint it stored at pairing. It does not use a certificate authority.
- The pairing token is 32 random bytes. The robot sends it in `hello`. The companion compares it in constant time.
- Plain `ws://` is for tests only. The robot's client allows it only when a test passes `insecure=True`.
- Both sides send a WebSocket ping every 2 s. A peer that does not answer within 3 s is gone.

## Framing

A frame is either text or binary.

**Text frames.** Each text frame is one UTF-8 JSON object of at most 16384 bytes. The field `t` names the message type. A receiver ignores fields it does not know. A receiver answers an unknown `t` with `error{unknown_type}`. A message type sent in the wrong direction counts as unknown.

**Binary frames.** Each binary frame is an 8-byte header and then mono s16le PCM.

| Bytes | Field | Type |
|---|---|---|
| 0 | `kind` | u8 |
| 1 | reserved | senders write 0; receivers ignore it |
| 2–3 | `stream` | u16, little-endian |
| 4–7 | `seq` | u32, little-endian |
| 8– | samples | s16le, mono, even byte count, at least one sample |

The header is `struct.Struct("<BxHI")`.

| Kind | Direction | Stream | Rate | Size |
|---|---|---|---|---|
| 1, microphone | robot → companion | a hear stream | 16 kHz | at most 20 ms (640 bytes) |
| 2, speech | companion → robot | a speak stream | `hello.out_rate` | at most 100 ms (9600 bytes at 48 kHz) |

The companion resamples speech to `out_rate`. The robot never resamples.

`seq` starts at 0 on each stream and grows by 1 per frame. A skipped or repeated `seq` is a gap. The receiver fails that stream with `error{gap}`.

## Messages

Every text field has a byte limit, measured in UTF-8. Integers are JSON integers. `true` and `false` are not integers.
Fields marked optional may be absent. An optional string or integer may also be `null`. An optional list or object must be absent, not `null`.

### Robot to companion

| `t` | Field | Type and limit |
|---|---|---|
| `hello` | `versions` | list of 1 to 8 integers. Every version the robot supports. |
| | `token` | string, 1 to 128 bytes |
| | `ops` | list of `hear`, `speak`, `narrate`, `classify`, `animate`. The ops the robot wants. A receiver drops names it does not know. |
| | `out_rate` | integer, 8000 to 48000 Hz |
| | `robot` | string, at most 64 bytes. A display name. |
| `hear_open` | `stream` | odd integer, 1 to 65535 |
| | `pre_roll_frames` | integer, 0 to 150. How many of the first microphone frames came before the wake word. |
| `hear_close` | `stream` | odd integer |
| | `how` | `finish` or `abort` |
| `speak_start` | `stream` | even integer, 2 to 65534 |
| | `text` | string, 1 to 240 bytes. Plain text, no markup. |
| | `voice` | `main` or `progress` |
| `speak_cancel` | `stream` | even integer |
| `narrate` | `id` | integer, 0 to 4294967295 |
| | `op` | `acknowledge`, `progress` or `lines` |
| | `request` | string, at most 4096 bytes. What the user asked. |
| | `status` | optional string, 1 to 120 bytes. A public status label. Required when `op` is `progress`. |
| | `already_said` | optional list of at most 8 strings of at most 240 bytes. Progress lines already spoken this turn, or the last audible Reachy lines for `acknowledge`. |
| | `reply` | optional string, 1 to 8192 bytes. The Muse reply to condense. Required when `op` is `lines`. |
| | `previous` | optional non-empty string, at most 240 UTF-8 bytes. Previous user request for `acknowledge`. |
| | `may_chat` | optional boolean, default false. Permission to answer pure social chat locally. The companion also checks the social gate. |
| `classify` | `id` | integer, 0 to 4294967295. Shares the `narrate` id space. |
| | `text` | string, 1 to 512 bytes. What the user said. |
| | `allowed` | list of 1 to 8 command labels the robot can carry out now: `sleep`, `wake_up`, `stop_listening`, `volume_up`, `volume_down`, `face_tracking_on`, `face_tracking_off`. A receiver drops labels it does not know. |
| `animate` | `id` | integer, 0 to 4294967295. Shares the `narrate` id space. |
| | `intent` | string, 1 to 200 bytes. The gesture or feeling to move, such as `thinking hard`. |
| | `duration_ms` | integer, 500 to 15000. About how long the motion should last. |

### Companion to robot

| `t` | Field | Type and limit |
|---|---|---|
| `welcome` | `version` | integer. The version chosen from `hello.versions`. |
| | `ops` | list of `hear`, `speak`, `narrate`, `classify`, `animate`. A subset of `hello.ops`. |
| | `models` | optional object of at most 8 entries, op name to model name (at most 120 bytes). For display. |
| `hear_partial` | `stream` | odd integer |
| | `rev` | integer, 0 to 4294967295. Grows within a stream. |
| | `text` | string, at most 4096 bytes |
| `hear_endpoint` | `stream` | odd integer |
| | `text` | string, at most 4096 bytes. Empty means nothing intelligible. |
| | `forced` | boolean. `true` when `hear_close{finish}` ended the turn. |
| `speak_done` | `stream` | even integer |
| | `frames` | integer. How many speech frames the companion sent. |
| `narrate_result` | `id` | the `narrate.id` it answers |
| | `lines` | list of at most 3 pairs `[text, expression]`. `text` is 1 to 240 bytes. `expression` is a name from `Expression` or `null`. An unknown name means no gesture. An empty list means say nothing. |
| | `intent` | optional `chat`, `question`, `action`, or `follow_up`. Unknown strings decode as absent; non-strings are invalid. A missing intent always keeps the turn Muse-bound. |
| `classify_result` | `id` | the `classify.id` it answers |
| | `command` | optional label from that request's `allowed`. Absent or `null` means the text is not a command. |
| `animate_result` | `id` | the `animate.id` it answers |
| | `fps` | integer, 1 to 50. Frames per second. |
| | `frames` | optional string, standard base64 of at most 375 frames. Absent or empty means no motion. |
| `error` | `code` | `auth`, `version`, `unknown_type`, `bad_request`, `gap`, `overloaded`, `model` or `internal` |
| | `message` | optional string, at most 500 bytes. For logs only. |
| | `stream` | optional integer. The stream that failed. |
| | `id` | optional integer. The narration, classification or animation that failed. |

## Rules

1. **Handshake.** The robot sends `hello` first. The companion answers `welcome`. Or it sends `error{auth}` or `error{version}` and closes. The robot keeps every op missing from `welcome.ops` local. A companion can ship one op at a time.
2. **Stream IDs.** The robot allocates stream IDs. Hear streams are odd. Speak streams are even. Each hear stream gets exactly one `hear_endpoint`, unless the robot aborted it. The companion may send `hear_endpoint` before `hear_close` when it detects the end of the turn itself.
3. **Speech.** After `speak_start`, the companion sends speech frames and then `speak_done`. `speak_cancel` is idempotent. The companion stops within one frame and still sends `speak_done`.
4. **Narration.** `narrate_result` answers the `narrate` with the same `id`. `classify_result` answers the `classify` with the same `id`. At most 4 narrations and classifications together are outstanding at once. A model answer that is not exactly an allowed label is sent as no command, never as `error{model}`.
5. **Motion.** `animate_result` answers the `animate` with the same `id`. At most 2 animations are outstanding at once, apart from narrations. Each frame is 18 bytes: nine little-endian int16 values in tenths, in the order x, y, z (millimetres), roll, pitch, yaw, right antenna, left antenna, body yaw (degrees). A motion the companion cannot make is an `animate_result` with no frames, never `error{model}`. The robot clamps every frame to its own limits before it moves.
6. **Liveness.** A close, a `ProtocolError`, `error{internal}` or `error{model}` fails every op in flight and marks the link down. `error{overloaded}`, `error{bad_request}` and `error{gap}` fail only the op they name.
7. **Reconnect.** The robot backs off from 1 s to 30 s with jitter. Each reconnect starts with a fresh `hello`. No stream or narration state survives a reconnect. The link is up only after `welcome`.

## Versioning

An additive change keeps version 1. Additive means a new optional field, or a new op that `welcome.ops` gates.
Any other change bumps the version.
The robot lists every version it supports in `hello.versions`. The companion picks one.

## Never on the wire

The robot never sends these to the companion:

- Muse credentials or tokens
- Noise keys
- Muse session IDs or message IDs
- raw Muse events

The companion receives only microphone audio, the request text, public status labels, the lines already spoken, the reply text it must condense, the short utterances it classifies, and motion intents from the robot's fixed table.
Status labels pass `validate_public_progress` on the robot before they are sent.

Acknowledgement results carry an intent and either a hold or a local social reply. Old robots omit `may_chat`, so a new companion returns no lines for CHAT. New robots treat an absent intent as an older companion and dispatch to Muse. All new fields are optional and omitted at their defaults; protocol v1 and operation names are unchanged.
