from reachy_muse.capabilities import Heard
from reachy_muse.contract.prompts import LOCAL_CHAT_NOTE_BYTES, local_chat_note, muse_request
from reachy_muse.recent import Recent


def test_recent_keeps_audible_lines_previous_request_and_unsent_chat():
    recent = Recent()
    recent.asked("how are you")
    recent.chatted("how are you", "Happy to see you.")
    for text in ("Hi.", "Oh.", "Nice.", "Cool.", "Sure."):
        recent.spoke(text)
    assert recent.heard("why is that", may_chat=False) == Heard(
        "why is that", "how are you", ("Oh.", "Nice.", "Cool.", "Sure."), False)
    assert recent.chat_note() == ("Earlier, said to Reachy only: user: how are you; Reachy: Happy to see you.", 1)
    note, count = recent.chat_note()
    recent.chatted("thanks", "Anytime.")
    recent.forget_chat(count)
    assert recent.chat_note() == ("Earlier, said to Reachy only: user: thanks; Reachy: Anytime.", 1)


def test_notes_are_bounded_in_utf8_and_keep_the_newest_three():
    note = local_chat_note([("old", "old"), ("I'm tired " * 30, "Aw, rest " * 30),
                            ("I love cafés " * 30, "Ooh, cafés " * 30), ("thanks", "Anytime.")])
    assert len(note.encode("utf-8")) <= LOCAL_CHAT_NOTE_BYTES
    assert "old" not in note
    assert note.endswith("user: thanks; Reachy: Anytime.")
    assert "…" in note
    assert local_chat_note([]) == ""


def test_a_note_that_fits_keeps_every_word():
    note = local_chat_note([("I'm so tired today", "Aw, long day? Want to tell me about it?"),
                            ("haha that's funny", "Ha, I try my best!")])
    assert note == ("Earlier, said to Reachy only: user: I'm so tired today; Reachy: Aw, long day? Want to tell me "
                    "about it? | user: haha that's funny; Reachy: Ha, I try my best!")


def test_muse_request_has_no_per_turn_overhead_without_history_or_setup():
    assert muse_request("Why is the sky blue?") == "Why is the sky blue?"
    assert len(muse_request("Hello").encode()) == 5
    assert muse_request("Why?", note="Earlier.") == "Earlier.\nThe user's spoken request is: Why?"
    assert muse_request("Why?", context="Context.", note="Earlier.") == (
        "Context.\n\nEarlier.\nThe user's spoken request is: Why?")
