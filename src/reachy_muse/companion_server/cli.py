from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys
from pathlib import Path
from typing import Optional, Sequence

from reachy_companion import pair as pairing
from reachy_companion.engines import EchoNarrator, ScriptedHearing, ToneVoice
from reachy_companion.server import CompanionServer, Engines


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(prog="reachy-companion")
    parser.add_argument("--state-dir", type=Path, default=pairing.default_state_dir(), help="where the certificate, key and token live")
    commands = parser.add_subparsers(dest="command", required=True)

    pair_command = commands.add_parser("pair", help="create the certificate and token, and print what the robot needs")
    pair_command.add_argument("--host", default=pairing.advertised_host(), help="the name the robot uses to reach this Mac")
    pair_command.add_argument("--port", type=int, default=pairing.DEFAULT_PORT)
    pair_command.add_argument("--force", action="store_true", help="replace the existing pairing; every robot must be added again")

    serve_command = commands.add_parser("serve", help="serve the robot")
    serve_command.add_argument("--bind", default="0.0.0.0", help="the address to listen on")
    serve_command.add_argument("--port", type=int, default=pairing.DEFAULT_PORT)
    serve_command.add_argument("--insecure", action="store_true", help="plain ws:// without TLS, for tests only")

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
    engines = Engines(hearing=ScriptedHearing(), voice=ToneVoice(), narrator=EchoNarrator())
    tls = None if args.insecure else paired.tls_context()
    stop = asyncio.get_running_loop().create_future()
    for sig in (signal.SIGINT, signal.SIGTERM):
        asyncio.get_running_loop().add_signal_handler(sig, lambda: stop.done() or stop.set_result(None))
    async with CompanionServer(engines, paired.token, ssl_context=tls, host=args.bind, port=args.port) as server:
        logging.info("serving %s (stub engines)%s", server.url, "  INSECURE: no TLS" if args.insecure else "")
        await stop
    logging.info("stopped")
