import pytest

from reachy_muse.follow_up import FollowUp, classify_follow_up, request_topic, words_besides_back_channels


@pytest.mark.parametrize("utterance, expected", [
    ("Never mind.", FollowUp.CANCEL),
    ("Oh, never mind that.", FollowUp.CANCEL),
    ("Nevermind", FollowUp.CANCEL),
    ("Cancel.", FollowUp.CANCEL),
    ("Cancel that request, please.", FollowUp.CANCEL),
    ("Stop!", FollowUp.CANCEL),
    ("No, stop it.", FollowUp.CANCEL),
    ("Forget it.", FollowUp.CANCEL),
    ("Okay, forget about it.", FollowUp.CANCEL),
    ("Actually, never mind.", FollowUp.CANCEL),
    ("stop stop", FollowUp.CANCEL),
    ("Stop, stop!", FollowUp.CANCEL),
    ("Forget about that.", FollowUp.CANCEL),
    ("Scratch that.", FollowUp.CANCEL),
    ("Don't bother.", FollowUp.CANCEL),
    ("Never mind that.", FollowUp.CANCEL),
    ("Stop talking.", FollowUp.SILENCE),
    ("Shut up!", FollowUp.SILENCE),
    ("Okay, stop talking please.", FollowUp.SILENCE),
    ("Also make it vegetarian.", FollowUp.ADD_DETAIL),
    ("And for tomorrow too?", FollowUp.ADD_DETAIL),
    ("Actually, make it two people.", FollowUp.ADD_DETAIL),
    ("Oh and book the earlier one.", FollowUp.ADD_DETAIL),
    ("also make it vegetarian", FollowUp.ADD_DETAIL),
    ("and for two people", FollowUp.ADD_DETAIL),
    ("actually make it Tuesday", FollowUp.ADD_DETAIL),
    ("And what's the weather in Paris?", FollowUp.NEW),
    ("Actually, what time is it?", FollowUp.NEW),
    ("Also, remind me to call mom.", FollowUp.NEW),
    ("And can you set a timer?", FollowUp.NEW),
    ("Oh and please tell me a joke.", FollowUp.NEW),
    ("Forget about the meeting.", FollowUp.NEW),
    ("Stop the music.", FollowUp.NEW),
    ("Cancel my dentist appointment.", FollowUp.NEW),
    ("What time is it in Tokyo?", FollowUp.NEW),
    ("Oh, what's the weather?", FollowUp.NEW),
    ("Android phones are great.", FollowUp.NEW),
    ("", FollowUp.NEW),
])
def test_classify_follow_up(utterance, expected):
    assert classify_follow_up(utterance) is expected


@pytest.mark.parametrize("request_text, expected", [
    ("Can you search for the latest robot news?", "the latest robot news"),
    ("What's the weather tomorrow?", "the weather tomorrow"),
    ("Tell me about the old castles and dragons near Edinburgh please", "the old castles and dragons near"),
    ("How does a rainbow form?", "a rainbow form"),
    ("Can you?", "Can you"),
    ("", ""),
])
def test_request_topic_uses_only_the_requests_words(request_text, expected):
    assert request_topic(request_text) == expected


@pytest.mark.parametrize("text, fillers", [
    ("Um.", True), ("Uh, hmm.", True), ("Oh.", True), ("Um, what do you see?", False), ("Thanks.", False),
    ("", False),
])
def test_only_fillers_hesitate_and_ask_nothing(text, fillers):
    from reachy_muse.follow_up import only_fillers

    assert only_fillers(text) is fillers


@pytest.mark.parametrize("text, words", [
    ("Got it.", 2), ("Oh, nice.", 1), ("Mm-hmm, okay.", 1), ("Uh huh.", 0), ("Um, what time is it?", 4), ("", 0)])
def test_back_channel_sounds_are_not_counted_among_the_words(text, words):
    assert words_besides_back_channels(text) == words
