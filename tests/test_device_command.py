import asyncio
import time

import pytest

from reachy_muse.device_command import COMMANDS, Commands, DeviceCommand, executable, parse_command

D = DeviceCommand
EVERYTHING = frozenset(DeviceCommand)


@pytest.mark.parametrize("utterance, expected", [
    ("Go to sleep.", D.SLEEP),
    ("go to sleep now", D.SLEEP),
    ("Okay Reachy, go back to sleep.", D.SLEEP),
    ("Could you please go to sleep?", D.SLEEP),
    ("You can take a nap.", D.SLEEP),
    ("Time to go to bed.", D.SLEEP),
    ("Goodnight, Reachy.", D.SLEEP),
    ("Good night.", D.SLEEP),
    ("Wake up!", D.WAKE_UP),
    ("Hey, wake up please.", D.WAKE_UP),
    ("Stop listening.", D.STOP_LISTENING),
    ("Please stop listening to me.", D.STOP_LISTENING),
    ("That's all, thanks.", D.STOP_LISTENING),
    ("Stop talking.", D.STOP_TALKING),
    ("Be quiet.", D.STOP_TALKING),
    ("Shut up.", D.STOP_TALKING),
    ("Stop.", D.STOP_TALKING),
    ("Never mind.", D.STOP_TALKING),
    ("Okay, stop, stop.", D.STOP_TALKING),
    ("Louder.", D.VOLUME_UP),
    ("Speak up, please.", D.VOLUME_UP),
    ("Could you speak a little louder?", D.VOLUME_UP),
    ("Turn up your volume.", D.VOLUME_UP),
    ("Turn the volume up a bit.", D.VOLUME_UP),
    ("I can't hear you.", D.VOLUME_UP),
    ("A bit quieter.", D.VOLUME_DOWN),
    ("Could you be a bit quieter?", D.VOLUME_DOWN),
    ("You're too loud.", D.VOLUME_DOWN),
    ("Lower your voice.", D.VOLUME_DOWN),
    ("Turn your volume down please.", D.VOLUME_DOWN),
    ("Speak more softly.", D.VOLUME_DOWN),
    ("Look at me.", D.FACE_TRACKING_ON),
    ("Keep looking at me.", D.FACE_TRACKING_ON),
    ("Follow my face.", D.FACE_TRACKING_ON),
    ("Turn on face tracking.", D.FACE_TRACKING_ON),
    ("Stop looking at me.", D.FACE_TRACKING_OFF),
    ("Don't look at me.", D.FACE_TRACKING_OFF),
    ("Stop following my face please.", D.FACE_TRACKING_OFF),
    ("Turn off face tracking.", D.FACE_TRACKING_OFF),
    ("how do I sleep better", None),
    ("stop the music", None),
    ("what's the volume of the moon", None),
    ("tell me about wake-up routines", None),
    ("set an alarm to wake me up at seven", None),
    ("turn up the music", None),
    ("don't go to sleep", None),
    ("cancel my dentist appointment", None),
    ("look up the weather", None),
    ("follow up on my email", None),
    ("go to sleep mode on my laptop", None),
    ("turn it up", None),
    ("wake me up at six", None),
    ("I can't sleep", None),
    ("what's louder, a lion or a tiger", None),
    ("watch me dance", None),
    ("look at me and tell me a joke", None),
    ("stop listening to the radio", None),
    ("sleep is important", None),
    ("good night stories for kids", None),
    ("", None),
])
def test_parse_command(utterance, expected):
    assert parse_command(utterance) is expected


def test_every_command_has_one_row():
    assert set(COMMANDS) == set(DeviceCommand)


def test_executable_follows_what_reachy_has_now():
    assert executable(wake=False, output=False, face=False, volume=False) == frozenset()
    assert executable(wake=True, output=False, face=False, volume=False) == {D.SLEEP, D.WAKE_UP, D.STOP_LISTENING}
    assert executable(wake=False, output=True, face=True, volume=True) == {
        D.STOP_TALKING, D.VOLUME_UP, D.VOLUME_DOWN, D.FACE_TRACKING_ON, D.FACE_TRACKING_OFF}


class FakeModel:
    def __init__(self, answer=None, *, delay_s=0.0, available=True):
        self.answer = answer
        self.delay_s = delay_s
        self.available = available
        self.asked = []

    async def classify(self, text, allowed):
        self.asked.append((text, allowed))
        await asyncio.sleep(self.delay_s)
        return self.answer


def recognize(model, text, allowed=EVERYTHING):
    return asyncio.run(Commands(model).recognize(text, allowed))


def test_a_grammar_hit_never_asks_the_model():
    model = FakeModel(D.VOLUME_UP)
    assert recognize(model, "Go to sleep.") is D.SLEEP
    assert model.asked == []


def test_a_grammar_hit_reachy_cannot_carry_out_goes_to_muse():
    model = FakeModel(D.SLEEP)
    assert recognize(model, "Stop.", allowed=EVERYTHING - {D.STOP_TALKING}) is None
    assert model.asked == []


def test_a_model_label_stands_only_with_its_own_evidence_word():
    assert recognize(FakeModel(D.VOLUME_DOWN), "could you be a tad less noisy and quieter") is D.VOLUME_DOWN
    assert recognize(FakeModel(D.VOLUME_UP), "keep your eyes on me") is None


def test_the_model_is_asked_only_about_commands_it_may_name():
    model = FakeModel(D.FACE_TRACKING_ON)
    assert recognize(model, "keep your eyes on me") is D.FACE_TRACKING_ON
    assert model.asked == [("keep your eyes on me", EVERYTHING - {D.STOP_TALKING})]


@pytest.mark.parametrize("answer", [D.STOP_TALKING, D.SLEEP, "volume_up"])
def test_a_model_label_outside_what_was_asked_is_rejected(answer):
    allowed = frozenset({D.VOLUME_UP, D.VOLUME_DOWN, D.STOP_TALKING})
    assert recognize(FakeModel(answer), "please pipe down a little", allowed) is None


def test_a_model_past_its_deadline_sends_the_words_to_muse():
    started = time.monotonic()
    assert recognize(FakeModel(D.VOLUME_DOWN, delay_s=2), "please pipe down a little") is None
    assert time.monotonic() - started < 0.35


def test_a_failing_or_unavailable_model_sends_the_words_to_muse():
    class Broken(FakeModel):
        async def classify(self, text, allowed):
            raise ConnectionError("link down")

    assert recognize(Broken(D.VOLUME_DOWN), "please pipe down a little") is None
    model = FakeModel(D.VOLUME_DOWN, available=False)
    assert recognize(model, "please pipe down a little") is None
    assert model.asked == []


@pytest.mark.parametrize("utterance", [
    "how do I sleep better",
    "what's the volume of the moon",
    "don't go to sleep",
    "play some relaxing sleep music",
    "turn down the lights",
    "tell me a story about a sleepy dragon who could not find a quiet place to rest",
    "what time is it in Tokyo",
])
def test_a_prefilter_miss_never_calls_the_model(utterance):
    model = FakeModel(D.SLEEP)
    assert recognize(model, utterance) is None
    assert model.asked == []
