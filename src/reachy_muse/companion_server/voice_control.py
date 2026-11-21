"""The owner's own control of the voice print: enroll it or forget it through a local socket.

The socket lives in the companion's state directory, which only its Mac user can enter, so being
that user is the authentication. Nothing on the robot's WebSocket reaches it, and no request the
robot or Muse makes can start an enrollment.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import socket
import stat
from pathlib import Path
from typing import AsyncIterator, Awaitable, Callable, List, Optional, Set

from reachy_muse.companion_server.engines import Speaker
from reachy_muse.companion_server.owner_voice import PRINT_FILE, delete_print

log = logging.getLogger("reachy_muse.companion_server")

SOCKET_FILE = "voice.sock"
REQUEST_TIMEOUT_S = 5.0
ENROLL = {"op": "enroll"}
FORGET = {"op": "forget"}
FAILURES = {
    "busy": "Another enrollment is already running. Nothing changed.",
    "timeout": "Time ran out after {taken} of {of} sentences. Nothing changed.",
    "mismatch": "Sentence {utterance} does not sound like the others, so nothing changed. "
                "Run enroll-voice again with only you speaking.",
    "save": "The voice print could not be saved. Nothing changed.",
    "forgotten": "forget-voice ran during the enrollment, so it kept nothing.",
}
NOT_LISTENING = "No companion is listening for enrollment. Start reachy-companion serve with --speaker-model, then try again."


@contextlib.asynccontextmanager
async def serving(voice: Speaker, state_dir: Path) -> AsyncIterator[None]:
    """Take enroll and forget requests on the state directory's socket while the block runs.

    Leaving the block ends any enrollment still listening, so stopping the companion never waits for one.
    """
    sessions: Set[asyncio.Task] = set()
    closing = False

    async def connected(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        if closing:
            writer.close()
            return
        session = asyncio.current_task()
        sessions.add(session)
        try:
            await _session(voice, reader, writer)
        finally:
            sessions.discard(session)

    server = await _listen(state_dir / SOCKET_FILE, connected)
    try:
        yield
    finally:
        if server is not None:
            closing = True
            server.close()
            for session in list(sessions):
                session.cancel()
            await server.wait_closed()


async def _listen(path: Path, connected: Callable[[asyncio.StreamReader, asyncio.StreamWriter], Awaitable[None]]
                  ) -> Optional[asyncio.AbstractServer]:
    if not _private(path.parent):
        log.warning("voice enrollment is off: the state directory must belong to this user alone (mode 700)")
        return None
    if await _answers(path):
        log.warning("voice enrollment is off: another companion serves this state directory")
        return None
    try:
        server = await asyncio.start_unix_server(connected, path=str(path))
    except OSError as error:
        log.warning("voice enrollment is off: %s", error)
        return None
    os.chmod(path, 0o600)
    log.info("voice enrollment listens on %s", path)
    return server


def _private(directory: Path) -> bool:
    try:
        info = os.stat(directory)
    except FileNotFoundError:
        return False
    return info.st_uid == os.getuid() and not stat.S_IMODE(info.st_mode) & 0o077


async def _answers(path: Path) -> bool:
    """Whether a live server already owns ``path``; binding would silently replace its socket."""
    try:
        _, writer = await asyncio.wait_for(asyncio.open_unix_connection(str(path)), 1)
    except (OSError, asyncio.TimeoutError):
        return False
    writer.close()
    return True


async def _session(voice: Speaker, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    """One CLI connection: a forget, or an enroll and its events until they end or the CLI hangs up."""
    tasks: List[asyncio.Task] = []
    try:
        request = _parsed(await asyncio.wait_for(reader.readline(), REQUEST_TIMEOUT_S))
        if request == FORGET:
            stopped, removed = await voice.forget()
            writer.write(json.dumps({"event": "forgotten", "stopped": stopped, "removed": removed}).encode() + b"\n")
            await writer.drain()
        elif request == ENROLL:
            tasks = [asyncio.create_task(_relay(voice, writer)), asyncio.create_task(reader.read(1))]
            await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    except (asyncio.TimeoutError, ConnectionError, ValueError):
        pass
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        writer.close()


async def _relay(voice: Speaker, writer: asyncio.StreamWriter) -> None:
    async with contextlib.aclosing(voice.enroll()) as events:
        async for event in events:
            writer.write(json.dumps(event).encode() + b"\n")
            await writer.drain()


def _parsed(line: bytes) -> object:
    try:
        return json.loads(line)
    except ValueError:
        return None


def enroll(state_dir: Path, say: Callable[[str], None] = print) -> int:
    """Ask the running companion to enroll the owner's voice, saying each step. Returns the exit status."""
    if not _private(state_dir):
        say(f"{state_dir} must belong to you alone (mode 700) before it can hold a voice print.")
        return 1
    try:
        with _connected(state_dir, ENROLL) as lines:
            for line in lines:
                event = json.loads(line)
                say(_described(event, state_dir))
                if event["event"] in ("enrolled", "failed"):
                    return 0 if event["event"] == "enrolled" else 1
    except (FileNotFoundError, ConnectionRefusedError):
        say(NOT_LISTENING)
        return 1
    except KeyboardInterrupt:
        say("Stopped.")
        return 130
    say("The companion stopped before it confirmed the enrollment.")
    return 1


def forget(state_dir: Path, say: Callable[[str], None] = print) -> int:
    """Delete the voice print. A running companion also stops any enrollment and the print's use at once."""
    reply = None
    if _private(state_dir):
        with contextlib.suppress(FileNotFoundError, ConnectionRefusedError):
            with _connected(state_dir, FORGET) as lines:
                reply = json.loads(next(lines, "null"))
    stopped, removed = (reply["stopped"], reply["removed"]) if reply else (False, delete_print(state_dir / PRINT_FILE))
    said = ["Stopped the enrollment that was running."] if stopped else []
    said.append("Deleted the voice print. The companion labels every voice unknown until you enroll again." if removed
                else "There is no voice print to delete.")
    say(" ".join(said))
    return 0


@contextlib.contextmanager
def _connected(state_dir: Path, request: dict):
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.connect(str(state_dir / SOCKET_FILE))
        connection.sendall(json.dumps(request).encode() + b"\n")
        with connection.makefile("r", encoding="utf-8") as lines:
            yield lines


def _described(event: dict, state_dir: Path) -> str:
    kind = event["event"]
    if kind == "listening":
        return (f"Listening for your voice. Say {event['utterances']} full sentences to Reachy within "
                f"{event['within_s'] / 60:g} minutes, each with at least {event['speech_s']:g} seconds of speech. "
                "Only you should speak. Reachy answers them as usual. Press Ctrl-C to stop.")
    if kind == "taken":
        return f"Heard sentence {event['taken']} of {event['of']}: {event['voiced_s']:.1f} s of speech."
    if kind == "skipped":
        return f"Skipped a sentence with {event['voiced_s']:.1f} s of speech; each needs {event['speech_s']:g} s."
    if kind == "enrolled":
        return (f"Enrolled your voice from {event['taken']} sentences, {event['voiced_s']:.1f} s of speech. "
                f"The voice print stays on this Mac in {state_dir / PRINT_FILE}.")
    return FAILURES.get(event.get("reason"), "Enrollment failed. Nothing changed.").format(**event)
