"""Which spoken words ask Reachy for a photo, and which never do."""

import asyncio

import pytest

from reachy_muse.device_command import (Commands, DeviceCommand, gaze_requested, names_reachy, photo_requested,
                                       whole_command)
from reachy_muse.turn_timeline import TurnTimeline
from test_device_command import FakeModel

REQUESTS = [
    "Take a photo.", "Reachy, take a photo", "take a picture please", "Can you take a picture of me please?",
    "Could you snap a quick photo?", "take a photo of this and tell me what it is", "take a photo then read the label",
    "take a photo of my plant and tell me if it needs water", "take a selfie", "please take a picture of the room",
    "What do you see?", "what can you see right now", "what do you see on the table", "What's in front of you?",
    "What am I holding?", "what is this I'm holding", "can you see this?", "do you see this",
    "tell me what you see", "look at this and tell me what it is",
    "take a look at this and tell me what kind of bird it is", "send Muse a photo of this",
    "send a picture of the whiteboard to Muse", "I want you to take a photo", "Go ahead and take a photo",
    "okay muse take a photo now", "take a photo, then tell me what you see", "what can you see in my hand",
    "what do you see here", "what do you see on the whiteboard", "Hey Reachy, what do you see?",
    "Muse, can you take a photo of this?", "Can you take a photo of this for me?",
    "take a photo of what's on my desk", "take a picture of my screen and tell me what the error says",
    "take a photo of this text message and read it to me", "What am I pointing at?",
    # Natural forms
    "What is this?", "What's this?", "What's in my hand?", "What am I holding up?", "Can you see me?",
    "Take a photo for Muse", "Take a picture and send it to Muse", "Take a picture, what do you see?",
    "Look at this, what is it?", "Show this to Muse", "Photograph this", "Take a photo of my morning coffee.",
    "Take a photo of the kids' drawing.", "Take a picture of the evening sky.", "Show Muse this.",
    "Send this to Muse.", "Take a photo and ask Muse what it is.", "What kind of plant is this?",
    "What breed is this dog?", "Read this label.", "What does this sign say?", "Identify this.",
    "Can you see my face?", "Do you see the dog?", "What's on the table?", "How do I look?", "Do I look okay?",
    "Describe the room.", "How many fingers am I holding up?", "What do you think of my outfit?",
    "What can you see from where you are?", "Photograph the whiteboard.",
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
    "take a photo and tell me what it is, then keep it", "take a photo and send it to Muse and my wife",
    "take a picture for Muse and my wife", "take a photo and ask her what she thinks",
    "take a photo of me in 2 secs", "take a photo on three", "take a photo of us on three",
    "take a photo of us on the count of three", "take a photo and count down from five",
    "take a photo with a timer", "take a photo of me this morning", "take a photo of us in the evening",
    "take a photo of the garden on weekends", "take a photo of me in a few seconds",
    # Quoted, hypothetical, reported or about the capability
    "if I say take a photo what happens", "how do I make you take a photo", "she said take a photo",
    "remind me to take a photo", "say take a photo", "Say 'take a photo' and I'll look.", '"Take a photo."',
    "“What do you see?”", "‘take a picture’", "can you take photos?", "do you take photos",
    "are you able to take a photo", "you can take a photo later",
    # Open-ended seeing, hearing, understanding, or words that refer back
    "what do you see in the news today", "what do you see as the problem", "what do you see in your future",
    "What kind of music is this?", "What type of song is this?", "What kind of question is that?",
    "What model is this?", "What is this song?", "What's that noise?", "Who is this?", "Who is this calling?",
    "Do you see what I mean?", "Do you see the point?", "Can you see the bigger picture?",
    "Can you see that happening?", "Can you see my perspective?", "Send that to Muse.", "Read that again.",
    "Read that to me.", "What does that say?", "What does it say?", "Identify this song.", "Describe that.",
    "What's on my plate?", "Photograph of the Eiffel Tower.", "Look at that, what a mess.",
    "Look at this, how cute.", "Check this out.", "Take a look.", "Capture this.", "Take a shot of this.",
    "Is this a good idea?", "Say 'take photos' and smile.",
    # "That" and "it" refer back to something said; "this" points at what Reachy can see
    "do you see that", "What's that?", "What is that?", "What does that sign say?", "Can you see it?",
    "Look at that and tell me what it says.", "Tell me what that is.", "What kind of bird is that?",
    "Show that to Muse.",
    # Things in the user's apps rather than in front of Reachy
    "Identify this bird.", "Read this note.", "Read this list.", "Can you see my photo?", "Can you see my screen?",
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
    "Can you take a photo of my eyes?", "Look at this, what is it?", "Can you see my face?",
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


NAMED = [
    "Muse, take a photo.", "Hey Reachy, what do you see?", "OK Muse, what am I holding?", "Okay, Reachy, look at this.",
    "What am I holding, Reachy?", "What's this, Muse, please?", "What do you see, Reachy, right now?",
    "Take a photo of this, Reachy, thank you.",
]
NOT_NAMED = [
    "What do you see?", "Hey, take a picture of this.", "Take a picture and send it to Muse.", "Show this to Muse.",
    "Send a picture of the whiteboard to Muse.", "Take a photo of Reachy.", "Take a photo for Muse.",
    "Ask Reachy what it sees.", "Muse's answer was wrong.", "",
]


@pytest.mark.parametrize("text, named", [(text, True) for text in NAMED] + [(text, False) for text in NOT_NAMED])
def test_words_name_reachy_only_when_they_open_or_close_by_speaking_to_it(text, named):
    assert names_reachy(text) is named


# Lines a television, a phone or video call, or people in the room say to each other, not to Reachy.
BACKGROUND = [
    "What do you see?", "What can you see?", "Do you see that?", "Do you see it?", "Can you see it?",
    "Can you see this?", "Can you see that?", "Can you see it now?", "Can you see this now?",
    "Do you see this?", "Can you see me?", "Can you see me now?", "Do you see what I'm holding?",
    "Can you see what I'm holding?", "Tell me what you see.", "Describe what you see.",
    "Tell me what you can see.", "What are you seeing?", "What do you see there?", "What do you see here?",
    "What do you see now?", "What do you see in front of you?", "What's in front of you?",
    "What is in front of you?", "What do you see in the picture?", "What do you see on the screen?",
    "What do you see on the board?", "What can you see from there?", "What can you see in the photo?",
    "What do you see when you look at me?", "Look at this and tell me what it is.",
    "Look at that and tell me what it says.", "Take a look at this and tell me what you think.",
    "Have a look at this and tell me if it's broken.", "Look at it and tell me the truth.",
    "What am I holding?", "What am I wearing?", "What am I wearing today?", "What am I pointing at?",
    "What am I showing you?", "What is this I'm holding?", "What's this I'm holding?",
    "What's that I'm holding?",
    "Take a picture.", "Take a photo.", "Take a picture!", "Take a photo of me.", "Take a picture of us.",
    "Take a picture of this.", "Take a photo of this.", "Can you take a picture of us?",
    "Could you take a photo of us please?", "Take a selfie!", "Take a quick photo.", "Snap a photo.",
    "Grab a pic.", "Take another picture.", "Take one more picture.", "Okay take a picture now.",
    "Take a picture, it'll last longer.", "Take a picture of the menu.",
    "Can you take a picture of the receipt?", "Take a photo of the license plate.",
    "Take a picture of my outfit.", "Please take a photo of us.", "Will you take a photo of us?",
    "Hey, take a picture of this.", "So, take a picture.", "Now take a picture.", "Well, take a picture.",
    "Um, take a picture.", "Okay, what do you see?", "So what do you see?", "Now what do you see?",
    "Well, what do you see?", "Alright, what am I holding?",
    "I see what you mean.", "See you later.", "What do you think?", "What did you see?",
    "Did you see that?", "Did you see the game?", "What do you see in him?", "What do you want?",
    "Look at this!", "Look at that!", "Look at me.", "Picture this.", "Say cheese!",
    "Smile for the camera.", "Let me take a picture.", "I'll take a picture.", "Let's take a selfie.",
    "Who wants to take a picture?", "Can I take a picture?", "Should I take a picture?",
]


def photo_decider():
    from reachy_muse.capabilities import Mode
    from reachy_muse.conversation import VoiceConversation
    from reachy_muse.local_backends import backends_for
    from test_conversation import FakeHardware, FakeSession

    session = FakeSession()
    return VoiceConversation(session, FakeHardware(), backends=backends_for(Mode.MUSE_VOICE, session),
                             photo_requests=True)


def after_a_wake():
    timeline = TurnTimeline()
    timeline.mark("wake")
    return timeline


def test_room_and_television_speech_never_takes_a_photo():
    decider = photo_decider()
    assert len(BACKGROUND) == 96
    assert [text for text in BACKGROUND if decider._photo_requested(text, TurnTimeline())] == []
    assert sum(decider._photo_requested(text, after_a_wake()) for text in BACKGROUND) == sum(
        map(photo_requested, BACKGROUND)) == 64


# Questions a user asks Muse about their own things, naming it, as an assistant is asked.
ADDRESSED_QUESTIONS = [
    "Muse, can you see my calendar?", "Muse, can you see my email?", "Muse, can you see my messages?",
    "Muse, can you see my last message?", "Muse, do you see the file?", "Muse, can you see my location?",
    "Muse, can you see the weather?", "Muse, can you see my notifications?", "Muse, can you see my schedule?",
    "Muse, can you see my shopping list?", "Muse, can you see my reminders?", "Muse, can you see my account?",
    "Muse, can you see my package?", "Muse, can you see my order?", "Muse, can you see my flight?",
    "Muse, can you see the chat?", "Muse, do you see my texts?", "Muse, can you see my history?",
    "Muse, can you see my steps?", "Muse, can you see my contacts?", "Muse, can you see the document?",
    "Muse, what's that?", "Muse, what is that?", "Muse, tell me what that is.", "Muse, what does that say?",
    "Muse, what kind of bird is that?", "Muse, identify this bird.", "Muse, read this note.",
    "Muse, read this list.", "Can you see my calendar, Muse?", "What's that, Muse?",
    "Hey Muse, can you see my inbox?", "Muse, can you see my photos?", "Muse, can you see my pictures?",
    "Muse, what is this called in French?", "Muse, what's this weather like?",
]


def test_questions_spoken_to_muse_about_the_user_s_own_things_never_take_a_photo():
    decider = photo_decider()
    assert len(ADDRESSED_QUESTIONS) == 36 and all(map(names_reachy, ADDRESSED_QUESTIONS))
    assert [text for text in ADDRESSED_QUESTIONS if decider._photo_requested(text, TurnTimeline())] == []


# Natural requests a user makes of Reachy.
NATURAL = [
    "What is this?", "What's this?", "What's that?", "What is that?", "Who is this?", "What color is this?",
    "What does this say?", "Can you read this?", "Read this for me.", "What does it say?", "Can you read this label?",
    "What kind of plant is this?", "Is this ripe?", "How do I look?", "Do I look okay?", "What do you think of my outfit?",
    "Look at this.", "Look at this!", "Check this out.", "Take a look.", "Have a look at this.",
    "Look at this, what is it?", "Look at this and tell me what you think.", "Take a look at this, what is it?",
    "Can you see me?", "Can you see my face?", "Do you see my cat?", "Can you see the dog?",
    "Show Muse this.", "Show this to Muse.", "Send this to Muse.", "Send Muse a picture.", "Send a picture to Muse.",
    "Take a pic.", "Snap a pic.", "Take a photo for Muse.", "Take a picture and send it to Muse.",
    "Take a picture for me.", "Take a photo please.", "Could you take a photo of this, please?",
    "Take a photo of this and describe it.", "Take a picture, what do you see?", "Take a photo. What is it?",
    "Photograph this.", "Take a snapshot.", "Capture this.", "Take a shot of this.",
    "What am I holding up?", "What's in my hand?", "What do I have in my hand?", "What's on the table?",
    "What's on my desk?", "Describe the room.", "Describe what's in front of you.", "What is in front of you right now?",
    "What do you see right now?", "Hey Reachy, what do you see right now?", "What can you see from where you are?",
    "Muse, what do you see?", "OK Muse, what am I holding?", "Reachy, look at this and tell me what it is.",
    "Tell me what this is.", "Identify this.", "What breed is this dog?", "How many fingers am I holding up?",
    "Is my shirt on inside out?", "Which one is bigger?", "Can you tell me what this is?",
    "Take a photo of my morning coffee.", "Take a picture of the evening sky.", "Take a photo of my weekend bag.",
    "Take a photo of the kids' drawing.", "Take a photo of the kids drawing.", "Take a photo of this receipt for my records.",
]


def test_most_natural_requests_take_a_photo_and_the_rest_are_refused_on_purpose():
    assert len(NATURAL) == 74
    assert [text for text in NATURAL if not photo_requested(text)] == [
        "What's that?", "What is that?",      # "that" usually refers back to something said
        "Who is this?",                       # also a song's singer or a caller
        "What does it say?",                  # "it" refers back, often to a message
        "Is this ripe?",                      # "is this" asks too much that is not seen
        "Look at this.", "Look at this!",     # no question for Muse, and "look at" is also face tracking
        "Check this out.", "Take a look.", "Have a look at this.",
        "Capture this.",                      # often a note to keep
        "Take a shot of this.",               # also a drink
        "Is my shirt on inside out?", "Which one is bigger?",
        "Take a photo of this receipt for my records.",  # keeps the photo
    ]
