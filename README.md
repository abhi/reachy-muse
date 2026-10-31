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
- `companion` lets the Mac hear, speak and narrate for the robot. If the Mac stops answering, the robot continues with its own `on-robot` models without announcing the outage or voice switch. It logs the fallback once per outage.

In `on-robot` and `companion` modes, the robot handles requests about itself without Muse.
"Go to sleep" and "stop listening" return it to the wake word, "louder" and "quieter" step its speaker volume, "look at me" and "stop looking at me" switch face tracking, and "stop talking" cuts off an answer that is still playing.
After "go to sleep", Reachy bows its head and folds its antennas. "Hey Muse" wakes it: the head lifts and the antennas unfold over one second. Said alone, "Hey Muse" gets "I'm awake."; a question in the same breath goes straight to Muse.
In `companion` mode, the Mac's narrator model also recognizes paraphrases such as "could you quiet down a little".

## Install on the robot

reachy-muse needs Python 3.10 or later. The robot's own environment uses Python 3.12.

Copy this repository to `~/reachy-muse` on the robot.
Copy the Muse Gadget SDK to `~/muse-gadget-sdk`, checked out at the commit that `pyproject.toml` pins.
Then follow [Install on a Wireless robot](docs/reachy-mini.md#install-on-a-wireless-robot) in the Reachy Mini guide.
It creates `~/muse-reachy-venv` next to Reachy's own environment and installs both checkouts into it.
The same guide covers the speech models, pairing with the Muse app, wake words and startup at boot.

## Install the companion on a Mac

The companion needs a Mac with Apple Silicon, Python 3.10 or later and about 15 GB of free memory.
Run these commands in this repository:

```sh
uv venv --python 3.13
uv pip install -e '.[companion-server,test]'
reachy-companion pair
reachy-companion serve
```

`pyproject.toml` pins the SDK to one commit of its Git repository, and uv installs it from there.
To work on the SDK at the same time, put its checkout next to this repository and install it in place of the pin:

```sh
uv pip install -e '.[companion-server,test]' --override <(echo "musegadget @ file://$PWD/../muse-gadget-sdk/linux")
uv pip install -e ../muse-gadget-sdk/linux
```

`pair` prints the address, certificate pin and token. Add them on the robot with `muse-reachy companion add`.
The [companion server guide](docs/companion-server.md) explains each step and the models it loads.

## How it uses the Muse Gadget SDK

The [Muse Gadget SDK](https://github.com/facebookincubator/muse-gadget-sdk) (`musegadget`) does everything that talks to Muse.
It pairs the robot with the Muse app, keeps the encrypted link to Muse open, sends each spoken request and follows Muse's answer.
reachy-muse uses only the parts of the SDK that the "Building a gadget" section of its [`linux/AGENTS.md`](https://github.com/facebookincubator/muse-gadget-sdk/blob/4737636133c9a0253d0b778333605b16fd2626c1/linux/AGENTS.md#building-a-gadget) lists.
Everything else here is the robot app: audio, wake words, speech models, motors, face tracking and the companion.

`pyproject.toml` pins the official upstream repository at `4737636133c9a0253d0b778333605b16fd2626c1`.
All changes from the fork are now upstream, including the public `Service` hooks from [PR #142](https://github.com/facebookincubator/muse-gadget-sdk/pull/142), merged as `729d0c8`.

What Muse and the robot agree on lives here, in `reachy_muse.contract`: the robot commands, the expressions, the prompts and the reply formats.
[The Reachy Mini contract with Muse](docs/contract.md) lists them.

## Test

Install the same way with the `robot,test` extras, or `companion-server,test`, then run each suite:

```sh
.venv/bin/python -m pytest tests --ignore=tests/companion_server
.venv/bin/python -m pytest tests/companion_server
```

## License

Apache-2.0. See [LICENSE](LICENSE). The face detection model in `src/reachy_muse/data/face_tracking` has its own MIT license next to it.
Speech sway and motion crossfades come from [reachy-animation](https://github.com/pham-tuan-binh/reachy-animation), also Apache-2.0. See [NOTICE](NOTICE).
