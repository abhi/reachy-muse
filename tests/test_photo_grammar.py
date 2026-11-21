"""Which spoken words ask Reachy for a photo, and which never do."""

import asyncio

import pytest

from reachy_muse.device_command import Commands, DeviceCommand, gaze_requested, photo_requested, whole_command
from test_device_command import FakeModel

REQUESTS = [
    "Take a photo.", "Reachy, take a photo", "take a picture please", "Can you take a picture of me please?",
    "Could you snap a quick photo?", "take a photo of this and tell me what it is", "take a photo then read the label",
    "take a photo of my plant and tell me if it needs water", "take a selfie", "please take a picture of the room",
    "What do you see?", "what can you see right now", "what do you see on the table", "What's in front of you?",
    "What am I holding?", "what is this I'm holding", "can you see this?", "do you see that",
    "tell me what you see", "look at this and tell me what it is",
    "take a look at this and tell me what kind of bird it is", "send Muse a photo of this",
    "send a picture of the whiteboard to Muse", "I want you to take a photo", "Go ahead and take a photo",
    "okay muse take a photo now", "take a photo, then tell me what you see", "what can you see in my hand",
    "what do you see here", "what do you see on the whiteboard", "Hey Reachy, what do you see?",
    "Muse, can you take a photo of this?", "Can you take a photo of this for me?",
    "take a photo of what's on my desk", "take a picture of my screen and tell me what the error says",
    "take a photo of this text message and read it to me", "What am I pointing at?",
]

NOT_REQUESTS = [
    # Negated
    "don't take a photo", "Don’t take a photo.", "do not take a picture", "please don't take a photo",
    "never take a photo of me", "no photos please", "stop taking photos", "take a photo and don't send it",
    # Sent, kept or scheduled
    "take a photo and send it to my wife", "take a photo of the receipt and email it to me",
    "take a photo and tell me what it is and text it to mom", "take a photo of this for my wife",
    "take a photo and tell me what it is, then save it", "take a photo of my cat when she comes in",
    "take a photo every morning", "take a photo tomorrow",
    "take a photo of me in five minutes", "take a photo of me at 5 pm", "take a photo of me at five",
    "take a photo of me next week", "take a photo of me this evening", "take a photo of me soon",
    "take a picture of the door in a bit", "take a photo of the stars at night",
    "take a photo and tell me what you think before sending it to my mom",
    "take a photo and tell me if it is worth sharing", "take a photo and tell me is it worth posting",
    "take a photo and could you forward it to dad", "take a photo then could you whatsapp it to her",
    "take a photo and can you put it on instagram", "take a photo and could you e-mail it to me",
    "take a photo and could you add it to the family album", "take a photo and can you remember it",
    "take a photo and tell me what it is, then keep it",
    # Quoted, hypothetical, reported or about the capability
    "if I say take a photo what happens", "how do I make you take a photo", "she said take a photo",
    "remind me to take a photo", "say take a photo", "Say 'take a photo' and I'll look.", '"Take a photo."',
    "“What do you see?”", "‘take a picture’", "can you take photos?", "do you take photos",
    "are you able to take a photo", "you can take a photo later",
    # Open-ended seeing
    "what do you see in the news today", "what do you see as the problem", "what do you see in your future",
    # Device commands, gaze and local chat
    "look at me", "look over here", "keep looking at me", "watch me", "say hi to someone else",
    "look at the other person", "greet my guest", "look at him", "look at this", "what's up", "go to sleep",
    "stop listening", "turn up the volume", "",
]

# Every phrase the device-command and gaze grammars test, whatever they decide about it.
COMMAND_AND_GAZE_PHRASES = [
    "Go to sleep.", "go to sleep now", "Okay Reachy, go back to sleep.", "Could you please go to sleep?",
    "You can take a nap.", "Time to go to bed.", "Goodnight, Reachy.", "Good night.", "Wake up!",
    "Hey, wake up please.", "Stop listening.", "Please stop listening to me.", "That's all, thanks.",
    "Stop talking.", "Be quiet.", "Shut up.", "Stop.", "Never mind.", "Okay, stop, stop.", "Louder.",
    "Speak up, please.", "Could you speak a little louder?", "Turn up your volume.", "Turn the volume up a bit.",
    "I can't hear you.", "A bit quieter.", "Could you be a bit quieter?", "You're too loud.", "Lower your voice.",
    "Turn your volume down please.", "Speak more softly.", "Look at me.", "Keep looking at me.",
    "Follow my face.", "Turn on face tracking.", "Stop looking at me.", "Don't look at me.",
    "Stop following my face please.", "Turn off face tracking.", "how do I sleep better", "stop the music",
    "what's the volume of the moon", "tell me about wake-up routines", "set an alarm to wake me up at seven",
    "turn up the music", "don't go to sleep", "cancel my dentist appointment", "look up the weather",
    "follow up on my email", "go to sleep mode on my laptop", "turn it up", "wake me up at six", "I can't sleep",
    "what's louder, a lion or a tiger", "watch me dance", "look at me and tell me a joke",
    "stop listening to the radio", "sleep is important", "good night stories for kids",
    "Say hi to my friend.", "Say hello to someone else.", "Greet my guest.", "Please look at her.",
    "Could you look at them?", "Okay, say hi to him.", "Reachy, say hi to my friend.",
    "Say hi to my friend, please.", "Can you please say hi to my friend?",
    "Hey Muse, would you greet the other person now?", "Alright, look at her!", "Look over here.", "Watch me.",
    "Say hi to my friend Sarah.", "Say hello to Sarah.", "Say hello to my mom.", "Greet our guests.",
    "Can you say hi to everyone?", "Say hi to the kids.", "Say hi to the camera.", "Look at this.",
    "Look at that dog.", "Look at Sarah.", "Say hi to her for me.", "Say hi to her when you call her.",
]


@pytest.mark.parametrize("text", REQUESTS)
def test_a_spoken_request_takes_a_photo_and_is_neither_a_command_nor_a_gaze(text):
    assert (photo_requested(text), whole_command(text), gaze_requested(text)) == (True, None, False)


@pytest.mark.parametrize("text", NOT_REQUESTS)
def test_other_words_never_take_a_photo(text):
    assert photo_requested(text) is False


@pytest.mark.parametrize("text", COMMAND_AND_GAZE_PHRASES)
def test_no_device_command_or_gaze_phrase_takes_a_photo(text):
    assert photo_requested(text) is False


@pytest.mark.parametrize("text", [
    "Look at this and tell me what it is.", "Look at this and describe it.", "Take a picture of my face.",
    "Can you take a photo of my eyes?",
])
def test_photo_requests_never_reach_the_command_model(text):
    model = FakeModel(DeviceCommand.FACE_TRACKING_ON)
    assert asyncio.run(Commands(model).recognize(text, frozenset(DeviceCommand))) is None
    assert model.asked == []


def test_other_looking_words_still_reach_the_command_model():
    model = FakeModel(DeviceCommand.FACE_TRACKING_ON)
    assert asyncio.run(Commands(model).recognize("Could you look this way?", frozenset(DeviceCommand))) is (
        DeviceCommand.FACE_TRACKING_ON)
    assert [text for text, _ in model.asked] == ["Could you look this way?"]
