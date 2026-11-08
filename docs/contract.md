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
| `reachy.presence` | No parameters. Returns `{"ok": true, "payload": {"present": true, "last_seen_seconds": 0, "source": "face", "available": true, "place": "office"}}`. `present` is true, false, or null; age and source may be null. `place` is omitted unless the owner supplies `--place`. No images or identity. |
| `reachy.notify` | Required `id` (1 to 128 UTF-8 bytes), `text` (1 to 240 UTF-8 bytes), `mode` (`now` or `when_present`) and `kind` (`notification` or `suggestion`). Optional `expires_in_s`, default 14400, range 0 to 86400. Returns `{"ok": true, "payload": {"status": "queued", "reason": "awaiting_presence"}}`. Status is `delivered`, `queued`, `expired` or `refused`. |
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

Presence fuses the arrival watcher's local camera observations, the face follower's
last detection, admitted voice wakes, recognized voice turns and claimed wave cues.
Camera and face observations stay fresh for two seconds; voice and wave cues stay
fresh for 30 seconds. A fresh positive cue wins over a negative camera check.
With no fresh positive cue, a fresh negative camera observation means absent;
otherwise presence is unknown. `last_seen_seconds` is the rounded age of the
latest positive cue. `source` names the freshest positive cue supporting presence,
or the last observation source when presence is absent or unknown.

A presence command reuses the arrival watcher while watching, or checks one local
camera frame otherwise. Its five-second default deadline includes dispatch and
waiting for another check. Missing frames and detector failures do not establish
absence. Background inbox delivery reads cached observations without starting
additional camera polls. Enable `--face-follow --face-follow-idle` for idle face
presence outside morning watching. The optional `--place office` label has a
64-byte limit and is included in registration and presence results. It describes
where the owner put Reachy; it does not locate or identify a person.

`available` is false while asleep, hushed, processing or hearing a turn, awaiting
the first wake request, speaking, disconnected, not ready, or unable to synthesize
command text. `reachy.notify` never wakes Reachy or overrides hush. Both kinds
follow the same delivery rules. `now` attempts immediate delivery when present
and available; otherwise it queues. `when_present` queues for the next available
delivery check, including if someone is already present. Items play oldest first,
one at a time, after a 1.6-second antenna wave. Reachy rechecks availability after
the gesture and before speech, and uses its normal speech ownership and interruption
path. A cue becoming stale during Reachy's gesture does not revoke the admitted
line; a fresh absence or a new conversation turn defers it. There is no quiet-hours schedule or urgency override in this slice.

The robot stores `reachy-inbox.json` in its protected state directory with mode
0600 and atomic replacement. It contains at most 20 IDs, including completed
receipts until their original expiry; a full ledger refuses new IDs. Expired
entries are removed. Retrying the same ID and content returns its existing state
and does not extend its lifetime. Reusing an ID with different text or kind is
refused. IDs may be reused after their original expiry, so use a unique ID for
each notification. Repeating `reachy.notify` with the same ID is also the status
lookup for this slice; there is no separate inbox command.

Queued text is stored locally. It is removed before speech begins; only the ID,
content digest, expiry and result remain. A durable claim precedes speech so a
restart cannot repeat a possibly audible line. A crash during delivery leaves
`refused` with `delivery_uncertain`, which can mean the line was never heard.
Interrupted or failed speech also returns `refused` and is not automatically
replayed. `delivered` means playback finished, not that anyone heard it. Zero
lifetime or expiry before playback returns `expired`. Invalid parameters,
conflicting IDs, full storage and storage failures return `refused` with reasons
such as `invalid_parameters`, `id_conflict`, `inbox_full` or `storage_unavailable`.
Queued results explain waiting for presence or availability. The inbox binds to
the paired account using a hash of the SDK's username and API endpoint, never a
token. Opening it or checking delivery when credentials are missing or the owner
has changed deletes the old queue. The same check stops speech during playback.
Failed or cancelled pairing, token refresh and reconnecting to the same account
preserve the queue. Older inbox files without an owner binding are discarded.

Presence is not identity verification. Notifications and suggestions go to
whoever is present. Voice verification is not calibrated for delivery recipients.
No command logs notification text, images or identity. The speaker keeps the
existing volume and motion settings. `on-robot` or `companion` synthesis is needed;
`muse-voice` alone cannot deliver arbitrary notification text.

The registration guide and command descriptions ask Muse to prefer Reachy for
notifications, reminders and proactive suggestions while present: check
`reachy.presence`, then `reachy.notify` with `now` if present or `when_present`
otherwise. They preserve the user's channel choice, recommend another channel
for time-critical messages, and grant no additional permissions. SDKs without
device-guide support receive the same routing guidance and place through command
descriptions. Ordinary voice replies already play automatically.

`parse_command(name, params, *, antenna_mode, timeout_ms)` validates calls and
fills defaults. Invalid speech or presence parameters return
`{"ok": false, "error": "invalid_parameters"}`. Notification parameter errors
use the notification result format with `status: refused`.

## Registration size

`tests/pins/registration.json` pins the entire SDK registration with a fixed
36-character request ID and the SDK's 15-character node ID. The compact JSON
is **5,553 bytes**, or **5,557 bytes** including the four-byte length prefix,
under the 8,192-byte cap. `commands_v2` occupies 5,277 bytes.

| Command description | UTF-8 bytes |
| --- | ---: |
| `reachy.say` | 1,471 |
| `reachy.presence` | 537 |
| `reachy.notify` | 475 |
| `reachy.expression` | 446 |
| `reachy.move` | 393 |

The guide is 2,425 UTF-8 bytes, within its 4,096-byte limit. Configuring `office`
makes the current SDK registration 5,593 bytes including its prefix. The worst
escaped 64-byte place label makes it 5,843 bytes, still below 8,192.

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
