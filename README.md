# reachy-muse

reachy-muse lets you talk to Muse through a Reachy Mini robot.
You speak to the robot, Muse answers, and the robot speaks the answer.
Its head and antennas move while it listens, thinks and talks.

The repository has two programs:

- `muse-reachy` runs on the robot.
- `reachy-companion` runs on a Mac on the same network. It is optional.

## Pick a mode

`muse-reachy run --mode` decides who hears you and who speaks.

- `muse-voice` is the default. The robot records what you say and sends the audio to Muse. Muse writes the answer and speaks it. The robot needs no speech models.
- `on-robot` keeps speech on the robot. Whisper turns your words into text, and Piper speaks Muse's answer.
- `companion` lets the Mac hear, speak and narrate for the robot. If the Mac stops answering, the robot falls back to its own `on-robot` models.

## Install on the robot

reachy-muse needs Python 3.10 or later. The robot's own environment uses Python 3.12.

Copy this repository to `~/reachy-muse` and the Muse Gadget SDK to `~/muse-gadget-sdk` on the robot.
Then follow [Install on a Wireless robot](docs/reachy-mini.md#install-on-a-wireless-robot) in the Reachy Mini guide.
It creates `~/muse-reachy-venv` next to Reachy's own environment and installs both checkouts into it.
The same guide covers the speech models, pairing with the Muse app, wake words and startup at boot.

## Install the companion on a Mac

The companion needs a Mac with Apple Silicon, Python 3.10 or later and about 15 GB of free memory.
Put the Muse Gadget SDK checkout next to this repository, then run these commands in this repository:

```sh
uv venv --python 3.13
uv pip install -e '.[companion-server,test]' --override <(echo "musegadget @ file://$PWD/../muse-gadget-sdk/linux")
uv pip install -e ../muse-gadget-sdk/linux
reachy-companion pair
reachy-companion serve
```

`pyproject.toml` names the SDK by its Git URL. The `--override` makes uv take it from the local checkout instead, and the last install makes that checkout editable.

`pair` prints the address, certificate pin and token. Add them on the robot with `muse-reachy companion add`.
The [companion server guide](docs/companion-server.md) explains each step and the models it loads.

## How it uses the Muse Gadget SDK

The [Muse Gadget SDK](https://github.com/abhi/muse-gadget-sdk) (`musegadget`) does everything that talks to Muse.
It pairs the robot with the Muse app, keeps the encrypted link to Muse open, sends each spoken request and follows Muse's answer.
Its `musegadget.reachy` package holds what Muse and the robot agree on: the robot commands, the expressions, the prompts and the reply formats.

reachy-muse uses only the parts of the SDK that its `docs/gadget-api.md` lists.
Everything else here is the robot app: audio, wake words, speech models, motors, face tracking and the companion.

## Test

Install the same way with the `robot,test` extras, or `companion-server,test`, then run each suite:

```sh
.venv/bin/python -m pytest tests --ignore=tests/companion_server
.venv/bin/python -m pytest tests/companion_server
```

## License

Apache-2.0. See [LICENSE](LICENSE). The face detection model in `src/reachy_muse/data/face_tracking` has its own MIT license next to it.
