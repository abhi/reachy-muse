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

from reachy_muse.contract.prompts import (
    CHAT_BOOTSTRAP, muse_request, setup_challenge_reply, with_setup_challenge,
)


def test_side_chat_creation_has_only_an_initialization_challenge():
    assert with_setup_challenge(CHAT_BOOTSTRAP, "c0ffee") == (
        "Initialize Reachy's dedicated voice chat. Do not use tools or take external actions."
        "\n\nInitialization challenge: reply exactly Ready c0ffee")
    assert setup_challenge_reply("c0ffee") == "Ready c0ffee"


def test_a_request_is_exactly_the_users_words():
    assert muse_request("What time is it?") == "What time is it?"


def test_only_local_conversation_history_adds_a_note():
    assert muse_request("Why?", note="Earlier, said to Reachy only: user: Hi; Reachy: Hello.") == (
        "Earlier, said to Reachy only: user: Hi; Reachy: Hello.\nThe user's spoken request is: Why?")
