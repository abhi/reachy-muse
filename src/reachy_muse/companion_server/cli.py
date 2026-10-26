from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import sys
from pathlib import Path
from typing import Optional, Sequence

from reachy_muse.companion_server import pair as pairing
from reachy_muse.companion_server.engines import EchoNarrator, NodAnimator, ScriptedHearing, SpeechFirst, ToneVoice
from reachy_muse.companion_server.narrator_mlx import DEFAULT_REPO
from reachy_muse.companion_server.server import CompanionServer, Engines
from reachy_muse.companion_server.voices import DEFAULT as DEFAULT_VOICE, build_voice, parse_voice

# With hearing, the narrator and Chatterbox evaluating on their own threads at once, Metal reported
# GPU hangs and aborted the process in 4 of 6 three-minute runs of bench/gpu_hang.py; with at most
# 16 operations per command buffer, in none of 9. It costs single-model throughput about 18%.
GPU_COMMAND_BUFFER = {"MLX_MAX_OPS_PER_BUFFER": "16"}


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(prog="reachy-companion")
    parser.add_argument("--state-dir", type=Path, default=pairing.default_state_dir(), help="where the certificate, key and token live")
    commands = parser.add_subparsers(dest="command", required=True)

    pair_command = commands.add_parser("pair", help="create the certificate and token, and print what the robot needs")
    pair_command.add_argument("--host", default=pairing.advertised_host(), help="the name the robot uses to reach this Mac")
    pair_command.add_argument("--port", type=int, default=pairing.DEFAULT_PORT)
    pair_command.add_argument("--force", action="store_true", help="replace the existing pairing; every robot must be added again")

    serve_command = commands.add_parser("serve", help="serve the robot")
    serve_command.add_argument("--bind", help="the address to listen on (default: every interface, or 127.0.0.1 with --insecure)")
    serve_command.add_argument("--port", type=int, default=pairing.DEFAULT_PORT)
    serve_command.add_argument("--insecure", action="store_true", help="plain ws:// without TLS, for tests only")
    serve_command.add_argument("--engines", choices=("real", "stub"), default="real", help="real speech models, or the stubs the tests use")
    serve_command.add_argument("--delay-ms", type=int, default=480, help="Voxtral's transcription delay; lower is faster and may be less accurate")
    serve_command.add_argument("--final-pass", action=argparse.BooleanOptionalAction, default=True, help="re-transcribe each whole utterance with Parakeet when the turn ends")
    serve_command.add_argument("--voice", type=_voice, default=DEFAULT_VOICE, metavar="ENGINE[:VOICE]",
                               help="the voice for every spoken line: chatterbox-turbo[:builtin], streamed, with Kokoro af_heart "
                                    "standing in when it falls behind; or kokoro[:VOICE], such as kokoro:af_heart or kokoro:bm_george "
                                    f"(default {DEFAULT_VOICE})")
    serve_command.add_argument("--loudness", type=float, default=-12.0, metavar="DBFS", help="the RMS level of the voiced part of spoken speech, before peaks are limited to -4 dBFS")
    serve_command.add_argument("--narrator-model", default=DEFAULT_REPO, help="the MLX instruction model that narrates; evals/narrator chose the default")
    serve_command.add_argument("--motion", action="store_true", help="also generate motion for the robot (the companion-motion extra); real engines only")
    serve_command.add_argument("--motion-dir", type=Path, help="where the motion planner and generator live "
                               "(default: ~/Library/Application Support/reachy-companion/motion)")

    args = parser.parse_args(argv)
    if args.command == "pair":
        existed = (args.state_dir / "token").exists()
        result = pairing.pair(args.state_dir, hosts=[args.host], force=args.force)
        print(result.line(args.host, args.port))
        print(f"{'kept' if existed and not args.force else 'created'} the pairing in {args.state_dir}", file=sys.stderr)
    else:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
        asyncio.run(_serve(args))


def _voice(spec: str) -> str:
    try:
        parse_voice(spec)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from None
    return spec


async def _serve(args: argparse.Namespace) -> None:
    paired = pairing.load(args.state_dir)
    engines = await _engines(args)
    tls = None if args.insecure else paired.tls_context()
    stop = asyncio.get_running_loop().create_future()
    for sig in (signal.SIGINT, signal.SIGTERM):
        asyncio.get_running_loop().add_signal_handler(sig, lambda: stop.done() or stop.set_result(None))
    async with CompanionServer(engines, paired.token, ssl_context=tls, host=args.bind or ("127.0.0.1" if args.insecure else "0.0.0.0"), port=args.port) as server:
        logging.info("serving %s (%s engines)%s", server.url, args.engines, "  INSECURE: no TLS" if args.insecure else "")
        await stop
    logging.info("stopped")


async def _engines(args: argparse.Namespace) -> Engines:
    if args.engines == "stub":
        narrator = EchoNarrator()
        return Engines(hearing=ScriptedHearing(), voice=ToneVoice(), narrator=narrator, classifier=narrator,
                       motion=NodAnimator() if args.motion else None)
    for name, value in GPU_COMMAND_BUFFER.items():
        os.environ.setdefault(name, value)  # MLX reads it when the GPU is first used
    from reachy_muse.companion_server.hearing_voxtral import HearingConfig, VoxtralHearing
    from reachy_muse.companion_server.narrator_mlx import MlxNarrator
    from reachy_muse.companion_server.voice_kokoro import Loudness

    hearing = VoxtralHearing(HearingConfig(delay_ms=args.delay_ms, final_pass=args.final_pass))
    voice = build_voice(args.voice, Loudness(target_dbfs=args.loudness))
    narrator = MlxNarrator(args.narrator_model)
    speech_first = SpeechFirst()
    motion = None
    if args.motion:
        from reachy_muse.companion_server.motion import DEFAULT_DIR, MotionEngine

        motion = MotionEngine(args.motion_dir or DEFAULT_DIR, yield_to=speech_first)
    logging.info("loading the speech and narrator models; the first run downloads about 9.2 GB")
    # One at a time: warming several models on their own threads at once hung the GPU.
    for engine in (hearing, voice, narrator, *([motion] if motion else [])):
        await engine.load()
    return Engines(hearing=hearing, voice=voice, narrator=narrator, motion=motion, classifier=narrator,
                   speech_first=speech_first)
