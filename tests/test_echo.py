"""What Reachy hears while it is talking: its own voice coming back is not a request."""

import asyncio

import pytest

from reachy_muse.follow_up import is_echo, too_short_for_follow_up

np = pytest.importorskip("numpy")
pytest.importorskip("av")

ACK = "tts:Hmm, good question."


@pytest.mark.parametrize("text, heard_over, expected", [
    ("good question", ("Hmm, good question.",), True),
    ("Hmm, good.", ("Hmm, good question.",), True),
    ("What about Paris?", ("Hmm, good question.",), False),
    ("good question", (), False),
    ("", ("Hmm, good question.",), False),
    ("Not eight, nine.", ("Eight, not nine.",), False),
    ("Eight, not nine.", ("Not eight, nine.",), False),
    ("Yes.", ("Yes, I can do that.",), False),
    ("No, thanks.", ("No, thanks for asking.",), False),
    ("Yes, I can do that.", ("Yes, I can do that.",), True),
    ("No, thanks for asking.", ("No, thanks for asking.",), True),
    ("good question", ("Good morning.", "One question."), False),
])
def test_words_heard_over_reachys_lines_are_an_echo_only_when_the_lines_hold_them_all(text, heard_over, expected):
    assert is_echo(text, heard_over) is expected


LONG_REPLY = ("I can't see you from here, but I can still help. Tell me what you're wearing and I'll suggest "
              "something for tonight, or ask me about the weather first.")
NUMBER_REPLY = ("Right now it's 72 degrees and sunny, and by eight tonight it will drop to 64, so bring a jacket "
                "if you go out later.")
STORY = ("Once upon a time there was a little robot who lived in a workshop by the sea. Every morning it would roll "
         "to the window and watch the boats come in, and it would wonder what it would be like to sail. One day the "
         "old fisherman who ran the workshop asked if it would like to come along, and the robot said yes right away. "
         "They packed a lunch, checked the weather, and set out before the sun was up. Can you guess what it asked "
         "the fisherman for the next morning? It asked if it could take a picture of the two of them on the boat, so "
         "it would never forget the day. Do you want me to tell you another one, or would you like to know more "
         "about the sea?")


@pytest.mark.parametrize("text, line, expected", [
    ("Tell me what you're wearing and I'll suggest something", LONG_REPLY, True),
    ("Tell me what you are wearing and I'll suggest", LONG_REPLY, True),
    ("it's seventy two degrees and sunny and by eight", NUMBER_REPLY, True),
    ("Tell me what your wearing and I'll suggest something", LONG_REPLY, True),
    ("wearing suggest something tonight ask about the weather first", LONG_REPLY, True),
    ("Reachy, what should I wear to the dinner tonight?", LONG_REPLY, False),
    ("Can you take a photo of me now, Reachy?", LONG_REPLY, False),
    ("Got it.", LONG_REPLY, False),
    ("Reachy, can you take a photo of me now?", STORY, False),
    ("and the one after that", STORY, False),
    ("", LONG_REPLY, False),
], ids=["exact", "a contraction written out", "a number written out", "a substitution", "fragments",
        "a request", "a request opening with a word of the reply", "a remark", "a request over a long story",
        "a remark over a long story", "no words"])
def test_muse_s_words_written_back_with_some_changed_or_missed_are_its_echo_and_a_request_is_not(
        text, line, expected):
    from reachy_muse.follow_up import echoes_playback

    assert echoes_playback(text, (line,)) is expected


@pytest.mark.parametrize("text, heard_over", [
    ("Not eight, nine.", "Eight, not nine."),
    ("Yes.", "Yes, I can do that."),
    ("No, thanks.", "No, thanks for asking."),
])
def test_corrections_and_short_answers_reach_request_dispatch(text, heard_over):
    from reachy_muse.capabilities import Mode
    from reachy_muse.local_backends import backends_for
    from test_conversation import FakeHardware, FakeSession
    from test_local_chat import DispatchOnly, Speech

    async def scenario():
        session = FakeSession()
        conversation = DispatchOnly(session, FakeHardware(), backends=backends_for(
            Mode.ON_ROBOT, session, speech=Speech()), local_chat=False)
        await conversation.turn(b"voice", recognized_text=text, heard_over=(heard_over,))
        assert session.setup_messages == [(text, None)]

    asyncio.run(scenario())


@pytest.mark.parametrize("text, expected", [
    ("Thanks.", True), ("Okay", True), ("Hmm.", True),
    ("Yes.", False), ("No!", False), ("Stop.", False), ("And Paris?", False),
])
def test_one_word_follow_ups_are_too_short_unless_they_answer_or_stop(text, expected):
    assert too_short_for_follow_up(text) is expected


def slow_weather_turn(monkeypatch, second):
    from test_pins import Rig, Transcripts, run_scenario

    rig = Rig(monkeypatch)
    rig.hardware.echo_cancelled_input = True
    transcripts = Transcripts("What's the weather tomorrow?", second)

    async def muse(session, request):
        if request == 1:
            await rig.wait_until_or_timeout(lambda: not transcripts.texts, 10)
            await rig.wait_until_or_timeout(lambda: False, 1)
        await session.answer(request, f"Answer {request}. [reachy:happy]")

    service = rig.service(muse, speech=rig.voice(), progress_speech=rig.voice("progress_tts"),
                          transcriber=transcripts)

    async def user():
        rig.say()
        await rig.wait_until_or_timeout(lambda: {"robot": "play", "audio": ACK} in rig.recording.events, 5)
        rig.say()
    events = run_scenario(rig, service, user)["events"]
    return ([event["text"] for event in events if event.get("muse") == "send_chat"],
            [event["audio"] for event in events if event.get("robot") == "play"])


def test_reachys_acknowledgement_heard_back_while_it_plays_is_not_sent_to_muse(monkeypatch):
    requests, plays = slow_weather_turn(monkeypatch, "good question")
    assert requests == ["What's the weather tomorrow?"]
    assert plays == [ACK, "tts:Answer 1."]


def test_a_real_question_over_the_acknowledgement_still_waits_its_turn(monkeypatch):
    requests, plays = slow_weather_turn(monkeypatch, "What about Paris?")
    assert requests == ["What's the weather tomorrow?", "What about Paris?"]
    assert plays == [ACK, "tts:Answer 1.", "tts:Answer 2."]


def test_a_one_word_aside_while_muse_works_is_not_sent_as_a_request(monkeypatch):
    requests, plays = slow_weather_turn(monkeypatch, "Thanks.")
    assert requests == ["What's the weather tomorrow?"]
    assert plays == [ACK, "tts:Answer 1."]
