# Contextual openers

What Reachy says while Muse works, before and after `conversation/contextual-openers`.

Run it with:

```sh
.venv/bin/python bench/contextual_openers.py --label after [--local-chat off] [--src DIR] [--port 8798]
```

The script pairs a fresh companion on its own port (never 8765 or 8766), loads the real models, and acts as the robot.
It sends the 61 utterances in `contextual_openers.json` as one conversation, twice (122 narrations).
Each utterance goes out as `narrate{acknowledge}` the way the robot sends it: with the user's previous request, Reachy's last four lines, and `may_chat` when local chat is on and the robot's gate passes.
The opener is then spoken with `speak_start`, and the time to its first audio frame is recorded.
Frames are raw JSON, so the same script scores the old companion (`--src` pointing at a copy of `075b6fc`'s `src`).

Runs: `contextual_openers_runs/before.json` (075b6fc), `after.json`, `after-local-chat-off.json`.
Model: `mlx-community/Qwen3-4B-Instruct-2507-4bit` on the Mac companion, as live.

## Results

| | before | after | after, local chat off |
|---|---|---|---|
| Intent accuracy (model) | none | 86.1% | 77.9% |
| Intent accuracy (robot grammar, used when the companion is away) | none | 86.9% | 86.9% |
| Utterances with an opener or a local reply | 93.4% | 96.7% | 86.9% |
| Accepted by the robot's `grounded()` | 100% | 100% | 100% |
| Parroting (shares three words in a row with the request, or mostly its words) | 71.9% | 1.7% | 0.0% |
| Same first three words as one of the last three openers | 7.0% | 2.5% | 1.9% |
| Same first word as the previous opener | 94.7% | 8.5% | 12.3% |
| Openers that are a fixed fallback phrase | 0% | 18.6% | 28.3% |
| Narration time p50 / p95 | 0.349 / 0.385 s | 0.246 / 0.543 s | 0.242 / 0.523 s |
| First opener audio after `speak_start`, p50 / p95 | 0.117 / 0.559 s | 0.078 / 0.102 s | 0.077 / 0.096 s |
| Opener audio ready (narration + first audio), p50 / p95 | 0.479 / 0.941 s | 0.342 / 0.615 s | 0.355 / 0.613 s |
| Local chat replies | 0 | 22 of 28 wanted | 0 |
| Local chat false positives (should have gone to Muse) | none | none | none |
| Local reply audio after end of speech, p50 / p95 | none | 0.318 / 0.557 s | none |

Intent confusion in `after` (wanted, got: count): question as question 32, as follow_up 5, as action 3; action as action 28; chat as chat 26, as follow_up 6; follow_up as follow_up 19, as action 2, as question 1.
All six chat-as-follow_up cases are utterances the robot's gate did not pass ("I'm feeling a bit nervous about tomorrow", "okay I'm heading to bed now, good night", "you know what, never mind the weather, how's your day going"), so they went to Muse either way.

Reading the latency rows:

- An opener for a Muse turn still plays only when Muse has said nothing 1.0 s after dispatch (`ACK_GRACE_S`, unchanged), so on the robot it starts at about 1.08 s after dispatch in both versions. What changed is that the opener is ready well before that deadline: p95 0.62 s instead of 0.94 s. The p95 dropped because the companion now synthesizes the opener as soon as its text is final, and `speak_start` plays the buffered audio.
- Narration p50 dropped because the opener's system prompt is cached once at load. Its p95 rose because about one narration in five needs a retry after a check rejects the first answer.
- A local chat reply plays as soon as it is ready, with no grace period: 0.32 s p50 from the end of speech to the first audio, against several seconds for a round trip through Muse.

## Before and after

Round one of each run. "Kind" is what the new narrator decided.

| User said | Before | Kind | After |
|---|---|---|---|
| how are you doing today | (silent) | local reply | Great, thanks! How about you? |
| I'm so tired today | Okay, I hear you're tired today. | local reply | Mmm, bedtime vibes? |
| ugh what a day | Okay, hearing you say ugh what a day | local reply | Ugh, I feel that too. |
| I just got back from a long run | Okay, hearing about your long run. | local reply | Wow, you've got that run energy! |
| I wonder if it'll rain | Okay, checking if it'll rain. | question | Hmm, clouds might be whispering. |
| what's the weather like in Oslo this weekend | Okay, checking the weather in Oslo. | question | Ooh, let me think. |
| why is the sky blue | Okay, explaining why the sky is blue. | question | Ah, light scattering, right? |
| is it safe to eat raw cookie dough | Okay, checking if raw cookie dough is safe to eat. | question | Give me a second on that. |
| remind me to call mom at six | Okay, remind you to call mom at six. | action | On it. |
| book a table for two at an Italian place tonight | Okay, booking a table for two at an Italian place tonight. | action | Got it, booking that table. |
| send a message to Priya saying I'm running late | Okay, sending a message to Priya saying you're running late. | action | Sure, I'll get that going. |
| set a timer for ten minutes | Sure, a timer for ten minutes. | action | Okay, I'll ask Muse to handle that. |
| what did you just say | Okay, repeating what you just said. | follow_up | Right, the recipe, hang on. |
| wait, which one was cheaper | Okay, comparing the prices of both options. | follow_up | Right, the price part, on it. |
| so what does that mean for me | Okay, what does that mean for you? | follow_up | Hang on, I'll clarify. |
| I'm hungry | (silent) | follow_up | Right, hunger check, on it. |

## Known limits

- The checks catch invented figures, names, dates, claims of finished work and lines that state a result ("is", "not", "booked", "started", "how about"). They cannot catch every factual answer. "Ah, light scattering, right?" for "why is the sky blue" hints at the answer and passes. An opener for a Muse turn is at most eight words, which bounds the damage.
- The robot's gate passes only short social utterances with no fact, time, data or action words, so six of the 28 utterances labeled as chat still go to Muse. That is the intended bias.
- The model over-uses `follow_up` for questions that follow another request. The intent only picks the opener's register and the waiting lines, so a wrong label costs tone, not routing.
