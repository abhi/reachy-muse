import asyncio
from dataclasses import replace

from reachy_muse.capabilities import Motion
from reachy_muse.contract.expressions import Expression
from reachy_muse.motion.moments import Moment, MomentDirector


class Source:
    available = True

    def __init__(self, delay_s=0.0):
        self.delay_s = delay_s
        self.asked = []
        self.outstanding = self.most_outstanding = 0

    async def generate(self, intent, duration_s):
        self.asked.append(intent)
        self.outstanding += 1
        self.most_outstanding = max(self.most_outstanding, self.outstanding)
        try:
            await asyncio.sleep(self.delay_s)
        finally:
            self.outstanding -= 1
        return Motion(25, ((0.0,) * 9,) * 3)


class Hardware:
    def __init__(self):
        self.calls = []

    def play_motion(self, frames, fps, *, during, loop=False, start_within_s=1.0):
        self.calls.append(("play", sorted(during)))

    def stop_motion(self):
        self.calls.append(("stop",))


def scenario(steps):
    asyncio.run(asyncio.wait_for(steps(), 5))


def test_stop_talking_just_after_the_answer_ends_cancels_its_closing():
    async def steps():
        hardware = Hardware()
        director = MomentDirector(Source(), hardware)
        director.prepare_closing(Expression.HAPPY)
        speaking = [True]
        director.finish_turn(lambda: not speaking[0])
        await asyncio.sleep(.1)
        director.stop()
        speaking[0] = False
        await asyncio.sleep(.3)
        assert hardware.calls == []
    scenario(steps)


def test_a_greeting_cued_as_the_answer_begins_is_still_stopped_by_a_later_barge_in():
    async def steps():
        hardware = Hardware()
        director = MomentDirector(Source(delay_s=.2), hardware)
        director.cue(Moment.THINKING)
        director.stop_cue()
        director.cue(Moment.GREETING)
        await asyncio.sleep(.05)
        director.stop()
        await asyncio.sleep(.4)
        assert hardware.calls == []
    scenario(steps)


def test_three_moments_at_once_share_the_two_requests_the_companion_allows():
    async def steps():
        source = Source(delay_s=.1)
        hardware = Hardware()
        director = MomentDirector(source, hardware)
        director.refill()
        director.cue(Moment.GREETING)
        director.prepare_closing(Expression.SAD)
        await asyncio.sleep(.5)
        assert {"apologetic and sorry", "perking up to say hello", "thinking hard"} <= set(source.asked)
        assert source.most_outstanding == 2
        assert hardware.calls == [("play", ["listening"])]
    scenario(steps)


def test_a_motion_that_arrives_after_its_start_window_is_never_handed_over():
    async def steps():
        hardware = Hardware()
        director = MomentDirector(Source(delay_s=1.6), hardware)
        director.prepare_closing(Expression.HAPPY)
        director.finish_turn(lambda: True)
        await asyncio.sleep(2.0)
        assert hardware.calls == []
    scenario(steps)


def test_conversation_shutdown_stops_inflight_prefetch_before_it_can_request_more():
    from reachy_muse.capabilities import Mode
    from reachy_muse.conversation import VoiceConversation
    from reachy_muse.local_backends import backends_for
    from test_conversation import FakeHardware, FakeSession, cancel_task

    async def steps():
        entered, release, finished = asyncio.Event(), asyncio.Event(), asyncio.Event()

        class WaitingSource(Source):
            async def generate(self, intent, duration_s):
                self.asked.append(intent)
                entered.set()
                try:
                    await release.wait()
                    return Motion(25, ((0.0,) * 9,) * 3)
                finally:
                    await asyncio.sleep(.01)
                    finished.set()

        class Robot(FakeHardware, Hardware):
            pass

        source, session = WaitingSource(), FakeSession()
        conversation = VoiceConversation(session, Robot(), backends=replace(
            backends_for(Mode.MUSE_VOICE, session), motion=source))
        running = asyncio.create_task(conversation.run())
        await entered.wait()
        await cancel_task(running)
        joined = finished.is_set()
        release.set()
        await asyncio.sleep(.1)
        assert source.asked == ["thinking hard"]
        assert joined

    scenario(steps)


def test_close_joins_cues_closings_and_generation_and_prevents_refill():
    async def steps():
        source, hardware = Source(delay_s=10), Hardware()
        director = MomentDirector(source, hardware)
        director.refill()
        director.cue(Moment.GREETING)
        director.prepare_closing(Expression.SAD)
        director.finish_turn(lambda: False)
        while source.outstanding < 2:
            await asyncio.sleep(0)
        await director.close()
        assert source.outstanding == 0
        asked = list(source.asked)
        director.refill()
        director.cue(Moment.THINKING)
        director.prepare_closing(Expression.HAPPY)
        await director.keep_ready()
        await director.close()
        await asyncio.sleep(0)
        assert source.asked == asked
        assert hardware.calls == []

    scenario(steps)
