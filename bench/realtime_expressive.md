# Expressive motion for Muse realtime replies

`--muse-path realtime` plays Muse's own voice, so no local text-to-speech step
tells Reachy where a sentence starts. Reachy infers its motion from what the call
delivers: transcript deltas, 24 kHz reply chunks with sample offsets, and the
reply lifecycle. This page measures that inference offline. No robot, Muse
session, network, or model was used.

## How motion follows the reply

- **Sway** comes from the output PCM, as on the chat path: every chunk goes
  through `hardware.play_audio`, which feeds the sway layer.
- **Sentence expressions.** `SentenceCutter` cuts sentences from
  `TranscriptDelta` text, and the local `classify_sentence` rules pick each
  expression. `ReplyAligner` places each sentence's first sample from three
  sources:
  - its character position, at the reply's speaking rate
  - a typical pause for each sentence break (0.35 s) and comma (0.15 s) before it
  - the end of the nearest long pause in the received audio, once that stretch
    has arrived; longer pauses win over nearer ones

  Each pause-confirmed sentence corrects the speaking rate, and
  `TranscriptDone` anchors it to the whole reply. When the speaker is about to
  play a chunk, it asks `ReplyMotion.marks` for the motion inside it and splits
  the chunk there. The expression therefore rides on the first sample of its
  sentence. Recorded clips play, or a motion-library clip when one is ready.
- **Partial sentences.** A sentence whose speech starts before its words are
  complete starts with the expression of its first words. If the full sentence
  shows a different feeling while it still plays, that expression replaces it.
- **Prosody.** `ProsodyTracker` reads loudness every 10 ms and pitch every 20 ms
  from the reply PCM. A loudness peak 5 dB above the running level gives a small
  nod, and 8 dB adds an antenna flick to it. A pitch climb of 15% into a pause
  gives a head tilt with the antennas. Accents are at least 1 s apart and 0.5 s clear of sentence
  starts. They need no words, so motion starts with the first audio.
- **Lifecycle.** `Working` cues the thinking motion when Muse is still silent
  0.3 s later. After `ResponseDone`, the reply's audio drains and the speaker
  stays quiet for 0.3 s. Then a happy or surprised tone in the last two
  sentences gives a celebration and a sad one an apology. The companion motion
  library plays them when it is available, else recorded clips (`success1`,
  then the built-in `happy`; `sad`). `PlaybackFlushed` forgets the flushed
  replies' sentences and accents. The speaker flush clears queued sentence
  events.
- **Muse's own gestures.** While a Muse-invoked `reachy.expression` or
  `reachy.move` runs, local sentence marks and accents are withheld. The
  hardware already skips sentence events during a commanded motion. Local
  inference resumes with the next sentence.

Motion never delays audio. Marks only split a chunk, and the speaker pushes
every chunk as soon as the call accepts it.

## Measurements

The script is `bench/realtime_expressive.py`. It replays 24 synthesized replies
built from 8 texts with 63 sentences in total. The audio is speech-like PCM:
voiced harmonic syllables, word gaps, comma and sentence pauses, about 12% of
words stressed by 7.6 dB, and pitch rises on questions. Every sentence's true
first sample is known. The script replays each reply in virtual time, in a
call's delivery order, and asks for marks chunk by chunk as the speaker would.

There are two speech profiles:

- **clear:** sentence pauses of 0.25 to 0.6 s, comma pauses of 0.18 to 0.3 s,
  and a quiet floor (RMS 40)
- **tight:** sentence pauses of 0.12 to 0.3 s overlap comma pauses of 0.12 to
  0.25 s, over a louder floor (RMS 150)

There are three delivery regimes:

- **ahead:** words and audio arrive in bursts at 2.5x real time, and each word
  arrives just before its audio
- **with_audio:** each word arrives with the chunk that holds its start
- **late:** each word arrives only when playback reaches it

### Sentence start alignment

The error is the absolute difference between the sample where a sentence's
expression was scheduled and the sentence's true first sample.

| Profile | Regime | p50 | p95 | max | ≤150 ms | Pause-confirmed | Committed late |
|---|---|---|---|---|---|---|---|
| clear | ahead | 1.0 ms | 5.2 ms | 5.8 ms | 100% | 100% | 0% |
| clear | with_audio | 1.5 ms | 5.9 ms | 818 ms | 98.4% | 100% | 0% |
| clear | late | 1.7 ms | 117 ms | 159 ms | 96.8% | 100% | 17.5% |
| tight | ahead | 1.1 ms | 309 ms | 641 ms | 92.1% | 100% | 0% |
| tight | with_audio | 0.7 ms | 87 ms | 755 ms | 95.2% | 100% | 1.6% |
| tight | late | 2.2 ms | 689 ms | 909 ms | 82.5% | 98.4% | 15.9% |

Most misses in the tight profile are sentences that open with a short clause,
such as "First, we pick up groceries." When the comma pause is as long as the
sentence pause, the snap lands after the comma. The expression then starts
0.3 to 0.65 s into its own sentence, never in the previous one. "Committed
late" means the sentence had no words yet when playback reached it. Its
expression starts at the next chunk, at most 0.2 s later.

An earlier version used only the character rate and a 3-word minimum before
committing a still-arriving sentence. On the clear profile it measured p95
527 ms and 79% within 150 ms in the ahead regime, and p95 1,200 ms and 44%
within 150 ms in the late regime. Modeling break pauses, preferring long
pauses, and committing on time from the first words gave the table above.

### Expression coverage

- Every sentence got an expression at its start in all six runs (`shown` = 100%;
  the late limit is 1.5 s).
- 30 of 63 sentences (48%) got a non-neutral expression (happy, sad, curious,
  surprised, or nod) when words arrived ahead or with audio. With late words,
  27 to 28 did, because a few started from words that did not show a feeling.
  The rest stay neutral, with sway and accents.
- Prosody gave 0.30 to 0.36 accents per second of speech, about one every 3 s,
  in addition to sentence motion.
- 11 to 12 of 24 replies qualified for a closing celebration or apology.

### First audio and CPU

| Measurement | Value |
|---|---|
| Chunk accepted to first speaker push, through the chunk's motion analysis, the real `RealtimeSpeaker` and its PyAV resampler (30 trials) | p50 0.6 ms, p95 0.7 ms, max 2.1 ms |
| Motion inference CPU (aligner, prosody, classifier, marks) | 1.1 to 1.2 ms per second of reply audio |
| Speaker path CPU (resampling, chunk splitting, pushes, and motion inference) | 9 to 11 ms per second of reply audio, over two runs |

These figures are from an Apple M3 Ultra. The robot's Raspberry Pi CM4 is
slower, so measure there before relying on them. Even at 15 times slower, the
speaker path would use about 17% of one core, and motion inference about 1.8%.
The 25 ms motion loop is unchanged: accents are extra sentence-end
events in its existing bounded queue.

The time from the user stopping to Muse's first audio depends on Muse and the
network, and cannot be measured offline. The live log line
`Muse's realtime reply started N s after the user stopped speaking` records it.

## What is not measured

- Real TTS audio has breaths, reverb tails, and pitch movement that the
  synthetic PCM lacks. The prosody thresholds and the 0.35 s break pause are
  priors, not values fitted to Muse's voice. Replaying recorded Muse replies
  here would test them; Reachy does not record them yet.
- The real arrival pattern of Muse's transcript deltas relative to its audio
  chunks. The three regimes bracket it.
- Visual quality of the motion on the robot.

## Reproduce

```sh
PYTHONPATH=src python bench/realtime_expressive.py --json realtime.json
```

The script needs NumPy and PyAV. Its replies, seeds, and profiles are fixed,
so alignment figures repeat exactly; CPU and timing figures vary with the
machine.
