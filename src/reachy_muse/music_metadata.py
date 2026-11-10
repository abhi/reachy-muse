"""Bounded track identity, returned to the caller and never written to logs."""

from dataclasses import dataclass
import re


def _label(value):
    if not isinstance(value, str):
        return None
    return "".join(c for c in value if c.isprintable()).strip()[:256] or None


@dataclass(frozen=True, repr=False)
class TrackMetadata:
    title: str | None = None
    artist: str | None = None
    album: str | None = None
    genre: str | None = None

    @classmethod
    def from_tags(cls, *tags):
        merged = {str(key).casefold(): value for source in tags for key, value in source.items()}
        return cls(*(_label(merged.get(key)) for key in ("title", "artist", "album", "genre")))

    @classmethod
    def from_icy(cls, packet):
        match = re.search(r"(?:^|;)\s*StreamTitle='(.*?)';", packet, re.IGNORECASE)
        if match is None:
            return None
        label = _label(match.group(1))
        if label and " - " in label:
            artist, title = label.split(" - ", 1)
            return cls(title=_label(title), artist=_label(artist))
        return cls(title=label)

    def payload(self):
        return {key: value for key in ("title", "artist", "album", "genre")
                if (value := getattr(self, key)) is not None}
