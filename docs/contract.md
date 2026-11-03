# The Reachy Mini contract with Muse

`reachy_muse.contract` defines the registered commands, request text, expressions,
and reply parsers. It uses only the standard library.

`ReachyService.describe()` supplies the SDK's `DeviceDescription`, with
`display_name` set to `Reachy Mini`. Muse retains that name and `commands_v2`.
The SDK sends registration on every connection. Muse reads the command
descriptions as device tool documentation on every turn.

The description of `reachy.say` carries Reachy's identity, the persona from
`contract/persona.py`, the speech-transcript caveat, spoken reply guidance, and
local controls. Ordinary requests contain only the user's words. The optional
300-byte local-chat note adds conversation history that Muse has not seen.
Reconnects and companion changes never add a voice context. The morning
briefing request asks for its own 60 to 90 seconds, roughly 150 to 220 words.

## Registered commands

| Command | Parameters and result |
| --- | --- |
| `reachy.say` | Required `text`, 1 to 240 UTF-8 bytes. Uses the normal local or companion speaker path. Returns `{"ok": true, "payload": {"spoken": true}}` after complete playback, or `{"ok": true, "payload": {"spoken": false, "reason": "..."}}` on refusal. The default timeout is 30 seconds. |
| `reachy.presence` | No parameters. Returns `{"ok": true, "payload": {"present": true, "last_seen_seconds": 0}}`. `present` is true, false, or null; `last_seen_seconds` is a rounded age or null. It returns no images or identity. |
| `reachy.expression` | Required expression `name`, optional `intensity` and `duration`. Built-in or recorded motions remain bounded and serialized. |
| `reachy.move` | Optional head, body, and antenna coordinates, plus `duration` and `hold`. At least one coordinate is required. `POSE_LIMITS` defines the bounds. |

`reachy.say` refuses while asleep, hushed, handling a user turn, or before voice
readiness. It waits for the current line otherwise, then checks again before
synthesis. Refusing an active Muse turn also prevents a tool from waiting for
the turn that invoked it. Stop talking interrupts playback and blocks new
`reachy.say` calls until the next admitted user request or wake. Sleep and hush
survive reconnects. Timeouts and disconnects cancel pending speech. A false
`spoken` result can follow partial playback and does not imply that nothing
was audible. A true result does not establish that someone heard it.

Reasons include `asleep`, `hushed`, `busy`, `not_ready`, `disconnected`,
`speech_unavailable`, `timeout`, `interrupted`, and `speech_failed`.
`muse-voice` cannot synthesize arbitrary command text, so `reachy.say` returns
`speech_unavailable` in that mode. Normal replies still use Muse's own TTS,
with `output_modality: voice` and the reply message ID. No setup prompt is
needed for that path.

Presence reuses the arrival detector's observations while watching. It returns
unknown if the last detection is more than two seconds old, the camera is
moving, or the watcher is unavailable. Motion gating can leave no recent
observation even when frames are arriving. An old arrival never means that a
person is still present. Outside watching, one local person-detector check
reads one frame. Missing frames or detector failures return unknown. Only the
last observation and last-seen time remain in memory. Presence never moves or
wakes the robot. Neither new command logs message text, images, or identity;
command logs contain the name and outcome.

The speaker keeps the existing volume and motion settings. This version has
no quiet-hours schedule, and this change adds none.

`parse_command(name, params, *, antenna_mode, timeout_ms)` checks calls and
returns a `Command` with defaults. Invalid speech or presence parameters return
`{"ok": false, "error": "invalid_parameters"}` at the service boundary.
There is no urgency override, deferred-delivery ledger, or delivery retry.

## Registration size

`tests/pins/registration.json` pins the entire SDK registration with a fixed
36-character request ID and the SDK's 15-character node ID. The compact JSON
is **4,176 bytes**, or **4,180 bytes** including the four-byte length prefix,
under the 8,192-byte cap. `commands_v2` occupies 3,900 bytes.

| Command description | UTF-8 bytes |
| --- | ---: |
| `reachy.say` | 1,439 |
| `reachy.presence` | 499 |
| `reachy.expression` | 262 |
| `reachy.move` | 208 |

`tests/test_registration.py` compares the literal registration fixture and
measures the actual SDK `encode_message` output, including the prefix. It also
pins each description's byte count. In the project's test environment, run
`python -m pytest tests/test_registration.py` to repeat the measurement.

## Chat creation and reply compatibility

`CHAT_BOOTSTRAP` and `with_setup_challenge()` create a missing side chat with
an initialization challenge. A saved UUID alone does not create a server chat.
Reachy drains the exact challenge response without speech before accepting
microphone input. Existing initialized chats need no message, including older
state with instruction digests. Interrupted creation retries the same chat ID
with a fresh challenge. No persona or reply-format setup is sent.

`muse_request(text, note=...)` adds only the optional local-chat note.
`STOP_REQUEST` remains the explicit request to stop Muse's current work.

`ReplyStyle` selects local handling and Muse TTS, without changing request
text. Muse is asked for plain speech through registration. The companion
narrator assigns expressions to its spoken lines. `spoken_reply()` still
accepts legacy `[reachy:NAME]` markers, and `SentenceStream` still accepts
plain sentences and legacy JSON sentence frames. Their validation, progress
handling, and revision protections remain in place. Malformed replies raise
`ReplyProtocolError`; changes to speech already given raise
`ReplyRevisionError`.
