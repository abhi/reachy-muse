# The Reachy Mini contract with Muse

`reachy_muse.contract` holds what Muse and Reachy Mini agree on: the robot
commands, the expressions, the prompts and the reply formats. It uses only the
standard library. Its tests are `tests/test_contract_*.py`.

- `contract.commands.COMMAND_SPECS` lists the commands Muse can call on Reachy:
  `reachy.expression` and `reachy.move`.
- `contract.commands.parse_command(name, params, *, antenna_mode, timeout_ms)`
  checks one call and returns a `Command` with the defaults filled in. It
  raises `ValueError` that names the first problem.
- `contract.commands.POSE_LIMITS` maps each pose and antenna parameter to its
  `(low, high)` range in millimetres or degrees.
- `contract.expressions.Expression` lists the expressions that Muse can name in
  a reply. `Expression.parse(name)` returns None for an unknown name.
- `contract.prompts.voice_context(style, ...)` describes Reachy and the reply
  format to Muse. It also records whether motion, each antenna and face
  tracking are on.
- `contract.prompts.SETUP_REQUEST` maps each `ReplyStyle` to the text that asks
  Muse to start answering in that style.
- `contract.prompts.CHAT_SETUP_PREFIX` and `contract.prompts.CHAT_SETUP_SUFFIX`
  go before and after a voice context to make the setup message for Reachy's
  side chat. `contract.prompts.CHAT_SETUP_MESSAGE` is the two with no context
  between.
- `contract.prompts.with_setup_challenge(setup_message, nonce)` appends the
  setup challenge to a setup message. Muse answers it with exactly
  `contract.prompts.setup_challenge_reply(nonce)`.
- `contract.prompts.STOP_REQUEST` asks Muse to stop working on the last request.
- `contract.prompts.spoken_request(context, text)` puts a voice context before a
  spoken request, for a chat that last heard another reply style.
- `contract.replies.ReplyStyle` names the reply formats.
- `contract.replies.spoken_reply(text)` splits a reply into speech and the
  expression from its last `[reachy:NAME]` marker.
- `contract.replies.SentenceStream` reads a reply as it streams, one JSON
  sentence frame per line or plain sentences, and returns `SpokenSentence`
  values. It raises `ReplyProtocolError` for a malformed reply and
  `ReplyRevisionError` when the final reply changes speech already given.
