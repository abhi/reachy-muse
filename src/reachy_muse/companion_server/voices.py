"""The voices `serve --voice` can pick: `engine[:voice]`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Dict, Optional, Protocol, Tuple

from reachy_muse.companion_server.engines import Voice

if TYPE_CHECKING:
    from reachy_muse.companion_server.voice_kokoro import Loudness

CHATTERBOX_VOICE = "builtin"
KOKORO_DEFAULT = "af_heart"


class LoadableVoice(Voice, Protocol):
    async def load(self) -> None: ...


@dataclass(frozen=True)
class Engine:
    default: str
    only: Optional[Tuple[str, ...]]
    build: Callable[[str, "Loudness"], LoadableVoice]


def _chatterbox(voice: str, loudness: Loudness) -> LoadableVoice:
    from reachy_muse.companion_server.voice_chatterbox import ChatterboxVoice
    from reachy_muse.companion_server.voice_fallback import FallbackVoice
    from reachy_muse.companion_server.voice_kokoro import KokoroVoice

    return FallbackVoice(ChatterboxVoice(loudness), KokoroVoice(KOKORO_DEFAULT, loudness))


def _kokoro(voice: str, loudness: Loudness) -> LoadableVoice:
    from reachy_muse.companion_server.voice_kokoro import KokoroVoice

    return KokoroVoice(voice, loudness)


ENGINES: Dict[str, Engine] = {
    "chatterbox-turbo": Engine(CHATTERBOX_VOICE, (CHATTERBOX_VOICE,), _chatterbox),
    "kokoro": Engine(KOKORO_DEFAULT, None, _kokoro),
}
DEFAULT = "chatterbox-turbo"


def parse_voice(spec: str) -> Tuple[str, str]:
    """`engine[:voice]` to (engine, voice), with the engine's default voice when none is named."""
    engine, _, voice = spec.partition(":")
    if engine not in ENGINES:
        raise ValueError(f"unknown voice engine {engine!r}; choose from {', '.join(ENGINES)}")
    choice = ENGINES[engine]
    voice = voice or choice.default
    if choice.only is not None and voice not in choice.only:
        raise ValueError(f"{engine} has only the voice {', '.join(choice.only)}, not {voice!r}")
    return engine, voice


def build_voice(spec: str, loudness: Loudness) -> LoadableVoice:
    engine, voice = parse_voice(spec)
    return ENGINES[engine].build(voice, loudness)
