# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0 (the "License").

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Tuple

from reachy_muse.capabilities import Heard
from reachy_muse.contract.prompts import clip_words, local_chat_note
from reachy_muse.protocol import MAX_LINE_BYTES


@dataclass
class Recent:
    previous: str = ""
    said: Deque[str] = field(default_factory=lambda: deque(maxlen=4))
    chats: Deque[Tuple[str, str]] = field(default_factory=lambda: deque(maxlen=3))

    def heard(self, text: str, *, may_chat: bool) -> Heard:
        return Heard(text, self.previous, tuple(self.said), may_chat)

    def spoke(self, line: str) -> None:
        self.said.append(clip_words(line, MAX_LINE_BYTES))

    def chatted(self, user: str, reply: str) -> None:
        self.chats.append((user, reply))

    def asked(self, text: str) -> None:
        self.previous = clip_words(text, MAX_LINE_BYTES)

    def chat_note(self) -> Tuple[str, int]:
        return local_chat_note(tuple(self.chats)), len(self.chats)

    def forget_chat(self, count: int) -> None:
        for _ in range(min(count, len(self.chats))):
            self.chats.popleft()
