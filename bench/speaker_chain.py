"""Measure companion speech at each stage of the robot's speaker chain, offline, without playing anything.

    python bench/speaker_chain.py WAV [WAV ...]

Needs GStreamer and PyGObject (on a Mac: brew install gstreamer pygobject3, then a Python that
sees both, such as a venv made with --system-site-packages on Homebrew's Python) plus numpy and soxr.

The stages are the ones the robot runs in companion mode:
  companion  the WAV resampled to 16 kHz and passed through serve's Loudness, as the robot receives it
  after_eq   the robot's own speaker EQ bin (speaker_eq.py, companion profile) built in a real GStreamer pipeline,
             at the 16 kHz the Reachy Mini Wireless sound card accepts
  at_speaker after the ALSA 'PCM' mixer, which the daemon's volume 85 sets to -9 dB (volume 100 is 0 dB)
Levels are dBFS over voiced 20 ms frames; "useful" is the 300 Hz to 8 kHz band a small driver can play.
"""

from __future__ import annotations

import sys
import wave
from pathlib import Path

import gi
import numpy as np
import soxr

gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: E402

from reachy_muse import speaker_eq  # noqa: E402
from reachy_muse.companion_server.voice_kokoro import Loudness  # noqa: E402

RATE = 16_000
MIXER_DB_AT_VOLUME_85 = -9.0
BANDS = {"below_300": (0, 300), "300_1k": (300, 1000), "1k_4k": (1000, 4000), "4k_8k": (4000, 8000), "useful": (300, 8000)}

Gst.init(None)
_add = Gst.Bin.add
# Homebrew's PyGObject ships Gst overrides whose Bin.add returns None; the robot's returns True, which speaker_eq checks.
Gst.Bin.add = lambda self, *elements: _add(self, *elements) or True


def robot_eq(pcm16_mono_16k: np.ndarray) -> np.ndarray:
    pipeline = Gst.Pipeline.new("chain")
    source = Gst.ElementFactory.make("appsrc")
    source.set_property("caps", Gst.Caps.from_string(f"audio/x-raw,format=F32LE,channels=2,rate={RATE},layout=interleaved"))
    source.set_property("format", Gst.Format.TIME)
    out_caps = Gst.ElementFactory.make("capsfilter")
    out_caps.set_property("caps", Gst.Caps.from_string(f"audio/x-raw,format=S16LE,channels=2,rate={RATE}"))
    sink = Gst.ElementFactory.make("appsink")
    sink.set_property("sync", False)
    chain = [source, Gst.ElementFactory.make("audioconvert"), Gst.ElementFactory.make("audioresample"),
             speaker_eq._build_eq_bin(Gst, speaker_eq.COMPANION_SPEAKER_EQ_GAINS), out_caps, sink]
    for element in chain:
        pipeline.add(element)
    for first, second in zip(chain, chain[1:]):
        first.link(second)
    stereo = np.repeat((pcm16_mono_16k.astype(np.float32) / 32768)[:, None], 2, axis=1).astype("<f4")
    pipeline.set_state(Gst.State.PLAYING)
    buffer = Gst.Buffer.new_wrapped(stereo.tobytes())
    buffer.pts = 0
    source.emit("push-buffer", buffer)
    source.emit("end-of-stream")
    chunks = []
    while (sample := sink.emit("pull-sample")) is not None:
        ok, info = sample.get_buffer().map(Gst.MapFlags.READ)
        chunks.append(np.frombuffer(info.data, "<i2").copy())
        sample.get_buffer().unmap(info)
    pipeline.set_state(Gst.State.NULL)
    return np.concatenate(chunks).reshape(-1, 2)[:, 0].astype(np.float32) / 32768


def dbfs(audio: np.ndarray) -> float:
    return float(20 * np.log10(np.sqrt(np.mean(np.square(audio, dtype=np.float64))) + 1e-12))


def voiced(audio: np.ndarray) -> np.ndarray:
    hop = RATE // 50
    frames = audio[: audio.size // hop * hop].reshape(-1, hop)
    energy = np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=1))
    return frames[energy >= energy.max() * 0.01].reshape(-1)


def levels(audio: np.ndarray) -> dict:
    speech = voiced(audio)
    spectrum, hz = np.fft.rfft(speech), np.fft.rfftfreq(speech.size, 1 / RATE)
    bands = {name: dbfs(np.fft.irfft(np.where((hz >= lo) & (hz < hi), spectrum, 0), speech.size)) for name, (lo, hi) in BANDS.items()}
    return {"rms": dbfs(speech), "peak": float(20 * np.log10(np.abs(audio).max() + 1e-12)), **bands}


def stages(path: Path) -> dict:
    with wave.open(str(path)) as wav:
        audio = np.frombuffer(wav.readframes(wav.getnframes()), "<i2").astype(np.float32) / 32767
        rate = wav.getframerate()
    if rate != RATE:
        audio = soxr.resample(audio, rate, RATE).astype(np.float32)
    pcm = (np.clip(Loudness().apply(audio, RATE), -1, 1) * 32767).astype("<i2")
    after_eq = levels(robot_eq(pcm))
    return {"companion": levels(pcm / 32768), "after_eq": after_eq,
            "at_speaker": {name: value + MIXER_DB_AT_VOLUME_85 for name, value in after_eq.items()}}


def main() -> None:
    results = [stages(Path(path)) for path in sys.argv[1:]]
    if not results:
        raise SystemExit(__doc__)
    columns = list(results[0]["companion"])
    print("stage       " + " ".join(f"{c:>10}" for c in columns))
    for stage in results[0]:
        print(f"{stage:<11} " + " ".join(f"{np.mean([r[stage][c] for r in results]):10.1f}" for c in columns))


if __name__ == "__main__":
    main()
