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
  the full transcript anchors it to the whole reply once playback drains.
  `TranscriptDone` alone does not confirm audio completion. When the speaker is about to
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
  starts. Accents detected up to 0.5 s behind playback play at the current chunk;
  older accents are dropped. They need no words.
- **Lifecycle.** `Working` cues the thinking motion when Muse is still silent
  0.3 s later. `ResponseDone` can precede more audio. The last sentence closes
  when playback drains. Once the response is complete and the speaker stays
  quiet for 0.3 s, a happy or surprised tone in the last two
  sentences gives a celebration and a sad one an apology. The companion motion
  library plays them when it is available, else recorded clips (`success1`,
  then the built-in `happy`; `sad`). `PlaybackFlushed` forgets the flushed
  replies' sentences and accents. The speaker flush clears queued sentence
  events.
- **Muse's own gestures.** While a Muse-invoked `reachy.expression` or
  `reachy.move` runs, local sentence marks and accents are withheld. The
  hardware already skips sentence events during a commanded motion. Local
  inference resumes with the next sentence.

Motion never waits for transcript data. Marks only split a chunk. Realtime
replies share speaker ownership with local speech, so the two play in sequence.

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

`TranscriptDone` follows the last delta in every regime. Audio completion is
delivered separately, after the last sample plays, as in production. The replay
checks that transcript deltas reconstruct the final text and that audio completion
follows playback.

### Sentence start alignment

The error is the absolute difference between the sample where a sentence's
expression was scheduled and the sentence's true first sample.

| Profile | Regime | Scheduled | p50 | p95 | max | ≤150 ms | Pause-confirmed | Committed late |
|---|---|---|---|---|---|---|---|---|
| clear | ahead | 63/63 | 1.0 ms | 5.2 ms | 5.8 ms | 100% | 100% | 0% |
| clear | with_audio | 62/63 | 1.4 ms | 6.3 ms | 907.7 ms | 96.8% | 100% | 0% |
| clear | late | 63/63 | 64.5 ms | 197.2 ms | 1069.8 ms | 77.8% | 96.8% | 57.1% |
| tight | ahead | 61/63 | 1.0 ms | 640.8 ms | 2057.1 ms | 85.2% | 100% | 0% |
| tight | with_audio | 62/63 | 0.8 ms | 819.5 ms | 2107.6 ms | 87.1% | 98.4% | 0% |
| tight | late | 62/63 | 56.8 ms | 537.4 ms | 1889.0 ms | 69.4% | 93.5% | 46.8% |

In the tight profile, sentence and comma pauses overlap, so the aligner can
choose the wrong pause. Speaking-rate estimates can also place a sentence beyond
the available audio. Errors reach 2.1 s without the complete-audio rate anchor
during playback. Late transcripts expose speaking-rate errors before the whole
transcript is available. "Committed late" means the
estimated start is already behind playback when the sentence is scheduled.
These percentages describe scheduled sentences. The table reports coverage
separately because estimates beyond the available audio never receive a mark.

The previous replay delivered the full transcript before the remaining late
deltas. That exposed future words and later appended duplicates. Correcting
the event order reduced clear/late accuracy within 150 ms from 96.8% to 77.8%.
Earlier comparisons using that event order are not valid for the late regime.
The next replay signalled audio completion at final PCM arrival. That exposed
the complete speaking rate before playback drained. Matching production raises
tight/with_audio p95 error from 86.8 ms to 819.5 ms and reduces its coverage from
63 to 62 sentences. The table above uses the corrected playback lifecycle.

### Expression coverage

- All scheduled sentences got an expression (`shown` = 100%; the late limit
  is 1.5 s). Coverage ranges from 61 to 63 of the 63 source sentences.
- 30 sentences got a non-neutral expression (happy, sad, curious, surprised,
  or nod) when words arrived ahead or with audio, except clear/with_audio,
  which produced 29. With late words, 27 did, because a few started from words
  that did not show a feeling.
  The rest stay neutral, with sway and accents.
- Prosody gave 0.30 to 0.36 accents per second of speech, about one every 3 s,
  in addition to sentence motion.
- 12 of 24 replies qualified for a closing celebration or apology.

### First audio and CPU

| Measurement | Value |
|---|---|
| Chunk accepted to first speaker push, through the chunk's motion analysis, the real `RealtimeSpeaker` and its PyAV resampler (30 trials) | p50 0.6 ms, p95 0.7 ms, max 1.7 ms |
| Motion inference CPU (aligner, prosody, classifier, marks) | 1.05 to 1.37 ms per second of reply audio |
| Speaker path CPU (resampling, chunk splitting, pushes, and motion inference) | 12.14 ms per second of reply audio |

These figures are from the development host. Measure on the robot's
Raspberry Pi CM4 before relying on them there.
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

The corrected run's raw results are in [realtime_expressive.json](realtime_expressive.json).
