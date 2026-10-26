# Device command recognition

`bench/local_commands.py` measures how fast Reachy recognizes a request about itself, and how often it mistakes an ordinary request for one.
Measured on 2026-10-07 on an Apple M3 Ultra with 96 GB, with `mlx-community/Qwen3-4B-Instruct-2507-4bit`.
The paired companion service was running at the same time and holding its own models.

    .venv/bin/python bench/local_commands.py

## Results

| Measure | Result |
|---|---|
| Grammar, `Commands().recognize` with no model, 17 phrases x 200 runs | p50 1 us, p95 4 us, max 67 us |
| Model round trip through the companion WebSocket, 33 calls that passed the prefilter | p50 75 ms, p95 80 ms, max 82 ms |
| Paraphrases recognized as the right command | 29 of 30 |
| Ordinary requests taken as a command (false positives) | 0 of 30 |
| The model alone on all 30 ordinary requests, with no prefilter or evidence check | 0 of 30 labelled |

The 300 ms deadline is about four times the slowest model answer.
24 of the 30 ordinary requests never reach the model: the prefilter turns away questions, negations and anything that names another device or medium, so they cost no time.
The one missed paraphrase, "turn off your microphone", got no label from the model and goes to Muse.

## Every phrase

"decided by" names the stage that answered: the grammar, the prefilter (no model call), or the model.
"model alone" is the label the model gives when asked directly, with no prefilter or evidence check.

| kind | phrase | expected | got | ms | decided by | model alone |
|---|---|---|---|---|---|---|
| paraphrase | time to hit the hay | SLEEP | SLEEP | 70 | model | SLEEP |
| paraphrase | let's call it a night | SLEEP | SLEEP | 69 | model | SLEEP |
| paraphrase | you can rest now | SLEEP | SLEEP | 68 | model | SLEEP |
| paraphrase | power down for the night | SLEEP | SLEEP | 70 | model | SLEEP |
| paraphrase | nap time for you | SLEEP | SLEEP | 70 | model | SLEEP |
| paraphrase | you can go back to sleep now | SLEEP | SLEEP | 0 | grammar | SLEEP |
| paraphrase | rise and shine little robot | WAKE_UP | WAKE_UP | 75 | model | WAKE_UP |
| paraphrase | come on, wake yourself up | WAKE_UP | WAKE_UP | 77 | model | WAKE_UP |
| paraphrase | okay I'm done talking to you | STOP_LISTENING | STOP_LISTENING | 81 | model | STOP_LISTENING |
| paraphrase | we're finished, you can stop listening for now | STOP_LISTENING | STOP_LISTENING | 82 | model | STOP_LISTENING |
| paraphrase | turn off your microphone | STOP_LISTENING | none | 64 | model | none |
| paraphrase | could you please quiet down a bit | VOLUME_DOWN | VOLUME_DOWN | 76 | model | VOLUME_DOWN |
| paraphrase | I need you to pipe down | VOLUME_DOWN | VOLUME_DOWN | 76 | model | VOLUME_DOWN |
| paraphrase | can you be a bit quieter | VOLUME_DOWN | VOLUME_DOWN | 0 | grammar | VOLUME_DOWN |
| paraphrase | you're kind of loud | VOLUME_DOWN | VOLUME_DOWN | 75 | model | VOLUME_DOWN |
| paraphrase | make your voice softer | VOLUME_DOWN | VOLUME_DOWN | 75 | model | VOLUME_DOWN |
| paraphrase | speak a tad more quietly | VOLUME_DOWN | VOLUME_DOWN | 75 | model | VOLUME_DOWN |
| paraphrase | crank up your volume | VOLUME_UP | VOLUME_UP | 75 | model | VOLUME_UP |
| paraphrase | I can barely hear you | VOLUME_UP | VOLUME_UP | 74 | model | VOLUME_DOWN |
| paraphrase | speak a bit more loudly | VOLUME_UP | VOLUME_UP | 75 | model | VOLUME_UP |
| paraphrase | your voice is too soft | VOLUME_UP | VOLUME_UP | 74 | model | VOLUME_UP |
| paraphrase | make yourself louder | VOLUME_UP | VOLUME_UP | 74 | model | VOLUME_UP |
| paraphrase | keep your eyes on me | FACE_TRACKING_ON | FACE_TRACKING_ON | 80 | model | FACE_TRACKING_ON |
| paraphrase | look over at me | FACE_TRACKING_ON | FACE_TRACKING_ON | 80 | model | FACE_TRACKING_ON |
| paraphrase | follow me with your eyes | FACE_TRACKING_ON | FACE_TRACKING_ON | 80 | model | FACE_TRACKING_ON |
| paraphrase | turn toward my face | FACE_TRACKING_ON | FACE_TRACKING_ON | 79 | model | FACE_TRACKING_ON |
| paraphrase | quit staring at me | FACE_TRACKING_OFF | FACE_TRACKING_OFF | 80 | model | FACE_TRACKING_OFF |
| paraphrase | stop following my face around | FACE_TRACKING_OFF | FACE_TRACKING_OFF | 80 | model | FACE_TRACKING_OFF |
| paraphrase | quit tracking me | FACE_TRACKING_OFF | FACE_TRACKING_OFF | 80 | model | FACE_TRACKING_OFF |
| paraphrase | stop watching me | FACE_TRACKING_OFF | FACE_TRACKING_OFF | 0 | grammar | FACE_TRACKING_OFF |
| question | how do I sleep better | none | none | 0 | prefilter | none |
| question | stop the music | none | none | 0 | prefilter | none |
| question | what's the volume of the moon | none | none | 0 | prefilter | none |
| question | tell me about wake-up routines | none | none | 66 | model | none |
| question | what time should I go to bed tonight | none | none | 0 | prefilter | none |
| question | set an alarm to wake me up at seven | none | none | 0 | prefilter | none |
| question | turn up the thermostat | none | none | 0 | prefilter | none |
| question | how loud is a jet engine | none | none | 0 | prefilter | none |
| question | look up the weather in Paris | none | none | 65 | model | none |
| question | who sang stop in the name of love | none | none | 0 | prefilter | none |
| question | can you remind me to call mom | none | none | 0 | prefilter | none |
| question | what's a good bedtime story for kids | none | none | 0 | prefilter | none |
| question | how do I stop snoring | none | none | 0 | prefilter | none |
| question | turn down the lights | none | none | 0 | prefilter | none |
| question | is it bad to sleep with the tv on | none | none | 0 | prefilter | none |
| question | what does REM sleep mean | none | none | 0 | prefilter | none |
| question | follow up on my email to Sam | none | none | 63 | model | none |
| question | find a quiet restaurant nearby | none | none | 64 | model | none |
| question | what are the face tracking features on iphone | none | none | 0 | prefilter | none |
| question | play some relaxing sleep music | none | none | 0 | prefilter | none |
| question | how many hours of sleep do teenagers need | none | none | 0 | prefilter | none |
| question | listen to this joke | none | none | 64 | model | none |
| question | tell me a story about a sleepy dragon | none | none | 0 | prefilter | none |
| question | what's louder a lion or a tiger | none | none | 0 | prefilter | none |
| question | cancel my dentist appointment | none | none | 0 | prefilter | none |
| question | stop my timer | none | none | 0 | prefilter | none |
| question | how do I mute my microphone in zoom | none | none | 0 | prefilter | none |
| question | my neighbor is too loud, what can I do | none | none | 65 | model | none |
| question | why do cats sleep so much | none | none | 0 | prefilter | none |
| question | keep an eye on the weather for me | none | none | 0 | prefilter | none |
