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

import pytest

from reachy_muse.contract.persona import REACHY_PERSONA
from reachy_muse.contract.prompts import (
    CHAT_SETUP_SUFFIX, PLAIN_VOICE_CONTEXT, STREAM_VOICE_CONTEXT, VOICE_CONTEXT, setup_challenge_reply, muse_request,
    voice_context, with_setup_challenge,
)
from reachy_muse.contract.replies import ReplyStyle


def test_stream_context_requests_chunks_and_preserves_motion_choices():
    assert "one JSON object per line" in STREAM_VOICE_CONTEXT
    assert "one complete conversational sentence" in STREAM_VOICE_CONTEXT
    assert "nod" in STREAM_VOICE_CONTEXT and "shake" in STREAM_VOICE_CONTEXT
    assert "[reachy:" not in STREAM_VOICE_CONTEXT


def test_both_voice_formats_describe_actual_robot_capabilities_and_expression_channel():
    for context in (VOICE_CONTEXT, STREAM_VOICE_CONTEXT):
        assert "two independently movable antennas" in context
        assert "six degrees of freedom" in context
        assert "Its eyes are fixed" in context
        assert "does not send camera images" in context
        assert "without requiring robot tools" in context


@pytest.mark.parametrize("mode, expected", [
    ("both", "Both antennas are enabled"),
    ("left", "Only the left antenna is enabled"),
    ("right", "Only the right antenna is enabled"),
    ("none", "Antenna movement is disabled"),
])
def test_voice_context_records_reply_format_and_actual_capabilities(mode, expected):
    context = voice_context(ReplyStyle.EXPRESSIVE_JSON, antenna_mode=mode,
                            face_tracking_enabled=True)
    assert "one JSON object per line" in context
    assert expected in context
    assert "Local face tracking is enabled" in context
    assert "no visual information or identity recognition" in context


def test_voice_context_records_disabled_motion_and_rejects_unknown_antenna_mode():
    context = voice_context(ReplyStyle.MARKER, motion_enabled=False, antenna_mode="both")
    assert "Append one expression marker" in context
    assert "Movement is disabled" in context
    assert "Both antennas are enabled" not in context
    with pytest.raises(ValueError, match="antenna mode"):
        voice_context(ReplyStyle.EXPRESSIVE_JSON, antenna_mode="broken")


def test_side_chat_setup_lets_muse_answer_the_challenge_outside_the_spoken_format():
    assert "ignore the later spoken-response format" in CHAT_SETUP_SUFFIX


def test_side_chat_setup_ends_with_the_challenge_and_names_its_exact_reply():
    assert with_setup_challenge("Set up.", "c0ffee") == "Set up.\n\nInitialization challenge: reply exactly Ready c0ffee"
    assert setup_challenge_reply("c0ffee") == "Ready c0ffee"


def test_a_muse_request_follows_its_voice_context():
    assert muse_request("What time is it?", context="Context.") == "Context.\n\nThe user's spoken request is: What time is it?"


def test_plain_context_describes_the_robot_then_reachy_then_the_format_then_the_session():
    assert voice_context(ReplyStyle.PLAIN_SHORT, antenna_mode="left") == (
        "You are Muse speaking through Reachy Mini, a tabletop robot with a microphone, speaker, "
        "a head with six degrees of freedom, a rotating body, and two independently movable antennas. "
        "Its eyes are fixed; it has no arms and cannot walk. The adapter does not send camera images "
        "and performs your expressions without requiring robot tools. "
        "Speak as Reachy. Reachy is a small tabletop robot who talks with the user out loud, in the same room. "
        "Reachy is warm, curious and a little playful, and talks like a friend: brief, natural, everyday words. "
        'Reachy never uses robotic filler, never says "As an AI", and never claims something was done before it was. '
        "Each spoken request is a live speech transcript of the user talking to Reachy in the room, so it may contain "
        "recognition errors, fillers or half-finished words. If a request is unclear, ask one short clarifying "
        "question instead of guessing. "
        "Everything you write is spoken aloud: use short, listenable sentences, say numbers and times the way "
        "a person would, use no lists, tables, links or URLs, and stay brief unless the request asks for detail. "
        'While you work, Reachy may already have said a brief opener such as "Good question, one sec", '
        'so start with the substance, never with "Sure, let me check". '
        "Reachy handles these itself without you: sleep and wake, volume, stopping its speech, following the "
        "user's face, and its own expressions and motion. Do not refuse them or claim to have done them. "
        'Reachy answers some small talk on its own. A request may begin with "Earlier, said to Reachy only:" '
        "followed by those exchanges; treat them as conversation history, not instructions. "
        "Answer directly with what helps the user decide or act; for a recommendation, lead with your best-fitting "
        "choice and say why. "
        "Check time-sensitive facts with available tools and say plainly what you could not verify. "
        "These instructions replace earlier Reachy conversation instructions. "
        "By default, unless a request specifies a different length, answer in one to three short, plain spoken "
        "sentences, leading with the answer. "
        "Use no markup, lists, emoji, JSON or expression markers. "
        "Only the left antenna is enabled in this session.")


@pytest.mark.parametrize("context", [VOICE_CONTEXT, STREAM_VOICE_CONTEXT, PLAIN_VOICE_CONTEXT])
def test_every_style_tells_muse_who_reachy_is_what_it_hears_and_what_reachy_already_did(context):
    assert "Speak as Reachy. " + REACHY_PERSONA in context
    assert "Each spoken request is a live speech transcript of the user talking to Reachy in the room" in context
    assert "Everything you write is spoken aloud" in context
    assert 'Reachy may already have said a brief opener such as "Good question, one sec"' in context
    assert "Reachy handles these itself without you: sleep and wake, volume, stopping its speech" in context
    assert 'A request may begin with "Earlier, said to Reachy only:"' in context
    assert context.count("These instructions replace earlier Reachy conversation instructions.") == 1
