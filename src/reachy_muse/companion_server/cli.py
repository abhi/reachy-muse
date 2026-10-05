from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys
from pathlib import Path
from typing import Optional, Sequence

from reachy_muse.companion_server import pair as pairing
from reachy_muse.companion_server.engines import EchoNarrator, ScriptedHearing, ToneVoice
from reachy_muse.companion_server.narrator_mlx import DEFAULT_REPO
from reachy_muse.companion_server.server import CompanionServer, Engines


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
    serve_command.add_argument("--voice", default="af_heart", help="the Kokoro voice for every spoken line, such as af_heart or bm_george")
    serve_command.add_argument("--loudness", type=float, default=-20.0, metavar="DBFS", help="the RMS level each spoken sentence is raised or lowered to")
    serve_command.add_argument("--narrator-model", default=DEFAULT_REPO, help="the MLX instruction model that narrates; evals/narrator chose the default")

    args = parser.parse_args(argv)
    if args.command == "pair":
        existed = (args.state_dir / "token").exists()
        result = pairing.pair(args.state_dir, hosts=[args.host], force=args.force)
        print(result.line(args.host, args.port))
        print(f"{'kept' if existed and not args.force else 'created'} the pairing in {args.state_dir}", file=sys.stderr)
    else:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
        asyncio.run(_serve(args))


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
        return Engines(hearing=ScriptedHearing(), voice=ToneVoice(), narrator=EchoNarrator())
    from reachy_muse.companion_server.hearing_voxtral import HearingConfig, VoxtralHearing
    from reachy_muse.companion_server.narrator_mlx import MlxNarrator
    from reachy_muse.companion_server.voice_kokoro import KokoroVoice, Loudness

    hearing = VoxtralHearing(HearingConfig(delay_ms=args.delay_ms, final_pass=args.final_pass))
    voice = KokoroVoice(args.voice, Loudness(target_dbfs=args.loudness))
    narrator = MlxNarrator(args.narrator_model)
    logging.info("loading the speech and narrator models; the first run downloads about 7.9 GB")
    await asyncio.gather(hearing.load(), voice.load(), narrator.load())
    return Engines(hearing=hearing, voice=voice, narrator=narrator)
