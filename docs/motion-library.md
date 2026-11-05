# Offline motion library

The companion can answer `animate` from saved trajectories without loading the
motion planner or generator. Live generation competed with hearing, narration and
speech on the Mac GPU. Build motions in a separate process, then serve their frames
from memory.

```sh
reachy-companion motion-library build
reachy-companion serve --motion-library
```

The builder defaults to 12 variants for each intent. The catalogue covers every
phrase in `motion/moments.py`, plus `wake`, `happy`, `curious`, `surprised`, `sad`,
`nod`, `shake`, `listening` and neutral emphasis under `neutral`. `thinking` covers
both moments and sentence expressions. There are 13 intent keys and 156 motions
at the default count. Seeds, phrasings and duration hints vary deterministically.

Use the same state directory for building and serving. It defaults to
`~/.reachy-companion`, or `REACHY_COMPANION_HOME`. The global `--state-dir` option
goes before the subcommand:

```sh
reachy-companion --state-dir "$REACHY_COMPANION_HOME" motion-library build --variants 12 --only greeting wake
reachy-companion --state-dir "$REACHY_COMPANION_HOME" serve --motion-library
```

Weights must already be prepared locally, using the existing motion installation.
The default weight directory is
`~/Library/Application Support/reachy-companion/motion`; pass `--motion-dir DIR` to
the build command if needed. Building requires the `companion-motion` extra.
It checks and hashes the local weights before generation and never downloads them.
The serving path does not require the motion models or their runtime imports.

## Storage and recovery

The builder writes under `STATE_DIR/motion-library/`, outside the checkout. Each
variant has an immutable JSON file under its intent directory. `manifest.json`
records the file hashes, model and generator implementation hashes, package
versions, intent text, and generation settings. Trajectories use the protocol's
nine channel order, 25 FPS and pose limits.

Each file is written to a temporary file in its destination directory, flushed,
and atomically renamed. The manifest is replaced only after the variant file is
complete. An interrupted build preserves published variants. Running the command
again skips valid entries, repairs missing or damaged files, and regenerates
entries whose model, implementation or intent fingerprint changed. Unreferenced
files left by interruption or replacement are harmless and remain on disk.

The server validates and loads a snapshot at startup. Each lookup selects a
variant while avoiding the last three where alternatives exist. It can interpolate
the trajectory to the requested duration within a 0.5 to 2 times retiming ratio.
Requests with no suitable variant get an empty frame list, so the robot uses its
existing expressions. Restart the companion after publishing changed variants to
load the new snapshot. It never builds motions while serving.

`--motion` and `--motion-library` are mutually exclusive. The library advertises
`reachy-motion-library` in the existing welcome engine metadata; the `animate`
request and response format is unchanged. Selection logs contain only the canonical
intent and variant ID.

## Sentence expressions

On the robot, add `--library-expressions` to `muse-reachy run` to opt in. It defaults
off. This mode requests the sentence's existing expression key only when the
companion offers `animate` under the library engine name. It never requests a
sentence motion from a live planner.

The request runs alongside speech preparation with a 50 ms deadline. First audio
does not wait for it. A result already ready at first audio travels with the
sentence's audio-timed start event; otherwise the recorded expression clip starts
as usual. Late results are discarded. Cancellation and sentence epochs keep old
results out of the next sentence. Frame preparation runs outside the motion loop;
the loop samples just two adjacent nine-channel frames per tick.

## Build scheduling

The companion writes `robot-activity.json` at operation boundaries, starting before
model loading. Hear and speak receipt resets the quiet interval immediately. Active
operations across all connections, including narration and voice preparation, also
prevent building. Completion, cancellation and disconnect reset the interval.

The build supervisor holds a single-writer lock and starts a low-priority worker
only after 20 quiet minutes. A manual first build can run without a status file.
`--if-idle 20` requires a valid status file, so the nightly job skips if the upgraded
companion has not reported activity. Missing, invalid, recent or active status
prevents a scheduled build. A skipped job exits successfully; an interrupted build
exits with status 75 and resumes on the next invocation.

During a build the supervisor checks activity every 25 ms and kills its owned worker
process group when activity changes. The worker also exits if its supervisor dies.
This aborts native model work; cancelling an await on MotionEngine's thread would
leave that work running. Robot audio never waits for the builder to finish or exit.

The example [launchd plist](examples/com.reachy.motion-library.plist) schedules a
separate job at 01:30 local time, with `Nice=10` and background I/O. It has no
`RunAtLoad` or `KeepAlive`. Busy jobs skip that invocation. As with other launchd
calendar jobs, a sleeping Mac can run a missed invocation on wake; the quiet check
still applies. The worker also lowers its priority when launched manually.

Before installation, replace `__REACHY_COMPANION__`, `__STATE_DIR__`, and
`__LOG_DIR__` with absolute deployment paths. launchd does not expand shell
variables or `~` in these fields. Create the log directory and add `--motion-dir`
and its absolute value to `ProgramArguments` if the weights use a custom location.
Copy the completed plist to `~/Library/LaunchAgents/` and validate it with
`plutil -lint`. The operator can then install it with:

```sh
launchctl bootstrap "gui/$(id -u)" "$HOME/Library/LaunchAgents/com.reachy.motion-library.plist"
```

Remove the scheduled job with:

```sh
launchctl bootout "gui/$(id -u)" "$HOME/Library/LaunchAgents/com.reachy.motion-library.plist"
```

This repository change prepares the plist only. Installation and service restarts
are deployment steps.

## Design choice and verification

Two independent designs compared a complete JSON snapshot with per-variant files
and an atomic manifest. The per-variant design was selected because it resumes
after each completed motion. Both designs favored validated in-memory serving.
The sentence path consumes only a timely result already ready at first audio, so
network latency cannot delay speech. A separate supervised worker is necessary to
abort native GPU work.

The tests use fake engines and literal frame expectations. They cover resuming,
invalidation, atomic publication, limits, variant selection, model-free serving,
activity abort, and sentence fallback. [The CPU benchmark](../bench/motion_library.md)
measures lookup latency and synthetic-library size. No real motion model was run
to validate this change.
