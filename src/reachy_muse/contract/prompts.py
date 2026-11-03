# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Chat creation, local conversation history, and requests for specific tasks."""

from __future__ import annotations

from typing import Sequence, Tuple

CHAT_BOOTSTRAP = "Initialize Reachy's dedicated voice chat. Do not use tools or take external actions."


def setup_challenge_reply(nonce: str) -> str:
    """The exact reply that answers the side-chat setup challenge for ``nonce``."""
    return f"Ready {nonce}"


def with_setup_challenge(setup_message: str, nonce: str) -> str:
    """The side-chat setup message with the challenge that Muse answers with setup_challenge_reply(nonce)."""
    return f"{setup_message}\n\nInitialization challenge: reply exactly {setup_challenge_reply(nonce)}"


STOP_REQUEST = "Please stop working on my previous request."

LOCAL_CHAT_NOTE_BYTES = 300


def clip_words(text: str, limit: int) -> str:
    text = " ".join(text.split())
    if len(text.encode("utf-8")) <= limit:
        return text
    words = text.split()
    while words and len((" ".join(words) + "…").encode("utf-8")) > limit:
        words.pop()
    return " ".join(words) + "…" if limit >= 3 else ""


def local_chat_note(chats: Sequence[Tuple[str, str]]) -> str:
    if not chats:
        return ""
    prefix = "Earlier, said to Reachy only: "
    recent = list(chats)[-3:]
    whole = prefix + " | ".join(f"user: {user}; Reachy: {reply}" for user, reply in recent)
    if len(whole.encode("utf-8")) <= LOCAL_CHAT_NOTE_BYTES:
        return whole
    budget = LOCAL_CHAT_NOTE_BYTES - len(prefix.encode("utf-8")) - 3 * (len(recent) - 1)
    side = (budget // len(recent) - len("user: ; Reachy: ")) // 2
    return prefix + " | ".join(f"user: {clip_words(user, side)}; Reachy: {clip_words(reply, side)}"
                                for user, reply in recent)


def muse_request(text: str, *, note: str = "") -> str:
    return f"{note}\nThe user's spoken request is: {text}" if note else text


MORNING_BRIEFING_REQUEST = (
    "This is Reachy's scheduled morning briefing, not a user speech transcript. "
    "It will be spoken aloud to the user as they walk into the room. It is {when}. "
    "For this reply, speak for about 60 to 90 seconds, roughly 150 to 220 words. "
    "In this order: wish them a warm good morning; give one "
    "concrete piece of advice to help them keep their focus today; give the latest market news, with the major "
    "stock indices' latest moves and any notable stocks; then the latest technology news, two or three stories. "
    "Use plain, listenable sentences with no lists, headings, links or symbols, and say numbers the way a person "
    "would say them aloud. Check every market and news fact with your tools, and say plainly what you could not "
    "verify.")
