import asyncio
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from reachy_muse.arrival import WatchPose
from reachy_muse.capabilities import SpokenLine
from reachy_muse.contract.expressions import Expression
from reachy_muse.morning_briefing import (APOLOGY, BriefingSchedule, DayLedger, MorningRoutine, briefing_lines,
                                          briefing_request)

WINDOW = BriefingSchedule.parse("04:50-06:00")
SATURDAY = datetime(2026, 10, 10, 4, 50)
BRIEFING = (SpokenLine("Good morning.", None, "answer"), SpokenLine("Markets rose.", None, "answer"))
REFRESHED = (SpokenLine("Markets fell.", None, "answer"),)


@pytest.mark.parametrize("clock, expected", [
    ((4, 49, 59), None),
    ((4, 50, 0), date(2026, 10, 10)),
    ((5, 59, 59), date(2026, 10, 10)),
    ((6, 0, 0), None),
])
def test_the_window_opens_at_its_start_and_closes_at_its_end(clock, expected):
    assert WINDOW.window_date(datetime(2026, 10, 10, *clock)) == expected


def test_days_limit_which_mornings_have_a_window():
    weekdays = BriefingSchedule.parse("04:50-06:00", "mon-fri")
    some = BriefingSchedule.parse("04:50-06:00", "mon,wed,sat")
    assert [weekdays.window_date(datetime(2026, 10, day, 5)) for day in (9, 10)] == [date(2026, 10, 9), None]
    assert [bool(some.window_date(datetime(2026, 10, day, 5))) for day in range(5, 12)] == [
        True, False, True, False, False, True, False]
    assert (str(weekdays), str(some), str(WINDOW)) == (
        "04:50-06:00 mon,tue,wed,thu,fri", "04:50-06:00 mon,wed,sat", "04:50-06:00 daily")


@pytest.mark.parametrize("window, days", [
    ("06:00-04:50", None), ("04:50-04:50", None), ("23:30-00:30", None), ("4.50-6", None),
    ("04:50", None), ("25:00-26:00", None), ("04:50-06:00", "someday"), ("04:50-06:00", "fri-mon"),
])
def test_a_window_that_crosses_midnight_or_does_not_parse_is_refused(window, days):
    with pytest.raises(ValueError):
        BriefingSchedule.parse(window, days)


@pytest.mark.parametrize("day", [date(2026, 3, 8), date(2026, 11, 1)])
def test_daylight_saving_days_still_open_at_the_wall_clock_time(day):
    new_york = ZoneInfo("America/New_York")
    opened = datetime(day.year, day.month, day.day, 4, 50, tzinfo=new_york)
    assert WINDOW.window_date(opened) == day
    assert WINDOW.window_date(opened - timedelta(seconds=1)) is None
    assert WINDOW.window_date(datetime(day.year, day.month, day.day, 6, 0, tzinfo=new_york)) is None


def test_the_request_names_the_local_morning():
    template = "It is {when}. Mind the {braces} and {}."
    assert briefing_request(template, SATURDAY) == (
        "It is Saturday, October 10, 04:50 local time. Mind the {braces} and {}.")
    assert briefing_request("No placeholder.", SATURDAY) == "No placeholder."


def test_a_marker_reply_becomes_every_sentence_with_the_expression_on_the_last():
    lines = briefing_lines("Good morning! Focus on one thing. Stocks rose. [reachy:happy]")
    assert lines == (SpokenLine("Good morning!", None, "answer"),
                     SpokenLine("Focus on one thing.", None, "answer"),
                     SpokenLine("Stocks rose.", Expression.HAPPY, "answer"))


def test_a_sentence_longer_than_a_voice_line_is_split_at_clauses_without_losing_words():
    clause = "the Nasdaq composite climbed a little more than one percent on chip makers"
    sentence = ", ".join([clause] * 5) + "."
    lines = briefing_lines("Markets were busy. " + sentence)
    assert [len(line.text.encode()) <= 240 for line in lines] == [True, True, True]
    assert " ".join(line.text for line in lines) == "Markets were busy. " + sentence


def test_a_clause_longer_than_a_voice_line_is_split_between_words():
    sentence = " ".join(["word"] * 100) + "."
    lines = briefing_lines(sentence)
    assert [len(line.text.encode()) for line in lines] == [239, 239, 20]
    assert " ".join(line.text for line in lines) == sentence


def test_the_ledger_survives_a_restart_and_ignores_a_corrupt_file(tmp_path):
    path = tmp_path / "morning-briefing.json"
    DayLedger(path).mark(date(2026, 10, 10))
    restarted = DayLedger(path)
    assert (restarted.done(date(2026, 10, 10)), restarted.done(date(2026, 10, 11))) == (True, False)
    path.write_text("{not json")
    assert DayLedger(path).done(date(2026, 10, 10)) is False
    assert DayLedger(tmp_path / "missing.json").done(date(2026, 10, 10)) is False


class FakeClock:
    def __init__(self, start=SATURDAY):
        self.start = start
        self.elapsed = 0.0

    def now_local(self):
        return self.start + timedelta(seconds=self.elapsed)

    def monotonic(self):
        return self.elapsed

    async def sleep(self, seconds):
        if asyncio.current_task().get_name() == "camera":
            until = self.elapsed + seconds
            while self.elapsed < until:
                await asyncio.sleep(0)
            return
        await asyncio.sleep(0)
        self.elapsed += seconds


class FakeCamera:
    def __init__(self, clock, arrive_after_s=None):
        self.clock = clock
        self.arrive_after_s = arrive_after_s
        self.watching_since = None
        self.calls = []

    def start_watching(self, pose):
        self.watching_since = self.clock.elapsed
        self.calls.append(("start", pose))

    def stop_watching(self):
        self.calls.append(("stop",))

    def log_arrival_diagnostics(self):
        pass

    def arrival(self):
        if self.arrive_after_s is None or self.clock.elapsed < self.watching_since + self.arrive_after_s:
            return None
        return self.watching_since + self.arrive_after_s


class FakeVoice:
    def __init__(self, clock, *answers):
        self.clock = clock
        self.answers = list(answers)
        self.said = []

    async def fetch(self, request, *, attended):
        after_s, result = self.answers.pop(0)
        self.said.append(("fetch", self.clock.elapsed, request))
        done = self.clock.elapsed + after_s
        while self.clock.elapsed < done:
            await asyncio.sleep(0)
        self.said.append(("fetched", self.clock.elapsed, attended()))
        return result

    async def greet(self, *, waiting):
        self.said.append(("greet", self.clock.elapsed, waiting))

    async def hold(self, line):
        self.said.append(("hold", self.clock.elapsed, line))

    async def brief(self, lines):
        self.said.append(("brief", self.clock.elapsed, lines))
        return 75.0


def drive(routine, until_s):
    async def main():
        camera = asyncio.create_task(routine.watch_camera(), name="camera")
        task = asyncio.create_task(routine.run(routine.voice))
        while routine.clock.elapsed < until_s and not task.done():
            await asyncio.sleep(0)
        task.cancel()
        camera.cancel()
        result, _ = await asyncio.gather(task, camera, return_exceptions=True)
        if not isinstance(result, asyncio.CancelledError):
            raise AssertionError(result)
    asyncio.run(main())


def routine_with(*answers, arrive_at=None, start=SATURDAY, ledger=None, poll_s=1.0, **options):
    clock = FakeClock(start)
    camera = FakeCamera(clock, arrive_at)
    routine = MorningRoutine(WINDOW, "Brief me for {when}.", camera, ledger or DayLedger(None), clock=clock,
                             poll_s=poll_s, **options)
    routine.voice = FakeVoice(clock, *answers)
    return routine


def kinds(routine):
    return [entry[0] for entry in routine.voice.said]


def test_without_an_arrival_nothing_is_said_and_the_briefing_is_dropped_at_window_end():
    routine = routine_with((30, BRIEFING), (30, REFRESHED))
    drive(routine, 70 * 60)
    assert routine.voice.said == [("fetch", 0, "Brief me for Saturday, October 10, 04:50 local time."),
                                  ("fetched", 30, False),
                                  ("fetch", 2430, "Brief me for Saturday, October 10, 05:30 local time."),
                                  ("fetched", 2460, False)]
    assert routine.camera.calls == [("start", None), ("stop",)]
    assert routine.ledger.done(date(2026, 10, 10)) is False


def test_a_ready_briefing_is_spoken_at_once_after_the_greeting():
    routine = routine_with((30, BRIEFING), arrive_at=300, pose=WatchPose(20.0, 5.0))
    drive(routine, 70 * 60)
    assert routine.voice.said[2:] == [("greet", 300, False), ("brief", 300, BRIEFING)]
    assert kinds(routine).count("fetch") == 1
    assert routine.camera.calls == [("start", WatchPose(20.0, 5.0)), ("stop",)]
    assert routine.ledger.done(date(2026, 10, 10)) is True


def test_an_arrival_before_the_prefetch_finishes_waits_for_it_with_hold_lines():
    routine = routine_with((40, BRIEFING), arrive_at=5)
    drive(routine, 70 * 60)
    assert routine.voice.said[1:] == [
        ("greet", 5, True), ("hold", 20, "Still gathering the market news."),
        ("hold", 35, "Checking the latest tech stories."), ("fetched", 40, True), ("brief", 41, BRIEFING)]


def test_a_failed_prefetch_is_fetched_again_live_on_arrival():
    routine = routine_with((30, None), (10, BRIEFING), arrive_at=600)
    drive(routine, 70 * 60)
    assert [entry[:2] for entry in routine.voice.said] == [
        ("fetch", 0), ("fetched", 30), ("greet", 600), ("fetch", 600), ("fetched", 610), ("brief", 611)]
    assert routine.voice.said[4] == ("fetched", 610, True)
    assert routine.voice.said[-1][2] == BRIEFING


def test_when_the_live_fetch_fails_too_reachy_apologizes():
    routine = routine_with((30, None), (10, None), arrive_at=600)
    drive(routine, 70 * 60)
    assert routine.voice.said[-1] == ("hold", 611, APOLOGY)
    assert "brief" not in kinds(routine)
    assert routine.ledger.done(date(2026, 10, 10)) is True


def test_a_briefing_older_than_forty_minutes_is_refreshed_once():
    routine = routine_with((30, BRIEFING), (30, REFRESHED), arrive_at=65 * 60)
    drive(routine, 70 * 60)
    assert [entry[:2] for entry in routine.voice.said] == [
        ("fetch", 0), ("fetched", 30), ("fetch", 2430), ("fetched", 2460), ("greet", 3900), ("brief", 3900)]
    assert routine.voice.said[-1][2] == REFRESHED


def test_a_failed_refresh_keeps_the_earlier_briefing():
    routine = routine_with((30, BRIEFING), (30, None), arrive_at=65 * 60)
    drive(routine, 70 * 60)
    assert routine.voice.said[-1] == ("brief", 3900, BRIEFING)


def test_a_prefetch_dropped_before_anyone_arrived_is_fetched_live_on_arrival():
    routine = routine_with((30, ()), (10, BRIEFING), arrive_at=600)
    drive(routine, 70 * 60)
    assert [entry[:2] for entry in routine.voice.said] == [
        ("fetch", 0), ("fetched", 30), ("greet", 600), ("fetch", 600), ("fetched", 610), ("brief", 611)]
    assert routine.voice.said[-1][2] == BRIEFING
    assert routine.camera.calls == [("start", None), ("stop",)]


def test_a_prefetch_dropped_with_nobody_arriving_leaves_the_day_open():
    routine = routine_with((30, ()))
    drive(routine, 70 * 60)
    assert kinds(routine) == ["fetch", "fetched"]
    assert routine.ledger.done(date(2026, 10, 10)) is False


def test_a_live_briefing_the_user_dropped_ends_without_a_word_more():
    routine = routine_with((30, None), (10, ()), arrive_at=600)
    drive(routine, 70 * 60)
    assert kinds(routine) == ["fetch", "fetched", "greet", "fetch", "fetched"]


def test_reachy_briefs_once_a_day_even_across_a_restart(tmp_path):
    path = tmp_path / "morning-briefing.json"
    first = routine_with((30, BRIEFING), arrive_at=300, ledger=DayLedger(path))
    drive(first, 20 * 60)
    restarted = routine_with((30, BRIEFING), arrive_at=600, start=SATURDAY + timedelta(minutes=20),
                             ledger=DayLedger(path))
    drive(restarted, 30 * 60)
    assert kinds(first) == ["fetch", "fetched", "greet", "brief"]
    assert restarted.voice.said == []
    assert restarted.camera.calls == []


def test_the_next_morning_opens_a_new_window():
    routine = routine_with((30, BRIEFING), (30, REFRESHED), arrive_at=60, start=datetime(2026, 10, 9, 23, 0))
    drive(routine, 8 * 3600)
    assert [entry[:2] for entry in routine.voice.said] == [
        ("fetch", 21000), ("fetched", 21030), ("greet", 21060), ("brief", 21060)]


def test_a_request_still_running_at_window_end_is_never_spoken():
    routine = routine_with((80 * 60, BRIEFING))
    drive(routine, 90 * 60)
    assert routine.voice.said == [("fetch", 0, "Brief me for Saturday, October 10, 04:50 local time.")]
    assert routine.camera.calls == [("start", None), ("stop",)]


def test_an_arrival_during_a_refresh_speaks_the_earlier_briefing_and_abandons_the_refresh():
    routine = routine_with((30, BRIEFING), (600, REFRESHED), arrive_at=2500)
    drive(routine, 70 * 60)
    assert routine.voice.said[2:] == [
        ("fetch", 2430, "Brief me for Saturday, October 10, 05:30 local time."),
        ("greet", 2500, False), ("brief", 2500, BRIEFING)]


def test_a_disconnect_while_cancelling_a_refresh_preserves_the_arrival():
    routine = routine_with(arrive_at=2500)

    async def check():
        cancelling = asyncio.Event()

        class CancellableVoice(FakeVoice):
            async def fetch(self, request, *, attended):
                try:
                    return await super().fetch(request, attended=attended)
                except asyncio.CancelledError:
                    cancelling.set()
                    await asyncio.Event().wait()
                    raise

        routine.voice = CancellableVoice(routine.clock, (30, BRIEFING), (600, REFRESHED))
        camera = asyncio.create_task(routine.watch_camera(), name="camera")
        task = asyncio.create_task(routine.run(routine.voice))
        try:
            await asyncio.wait_for(cancelling.wait(), 5)
        finally:
            task.cancel()
            camera.cancel()
            await asyncio.gather(task, camera, return_exceptions=True)

    asyncio.run(check())
    routine.camera.arrive_after_s = None
    routine.voice = FakeVoice(routine.clock)
    drive(routine, 2700)
    assert routine.voice.said == [("greet", 2500, False), ("brief", 2500, BRIEFING)]
    assert routine.camera.calls == [("start", None), ("stop",)] * 2
    assert routine.ledger.done(date(2026, 10, 10)) is True


class UnfinishedBriefVoice(FakeVoice):
    async def brief(self, lines):
        self.said.append(("brief", self.clock.elapsed, lines))
        while True:
            await self.clock.sleep(1)


@pytest.mark.parametrize("answers", [((30, BRIEFING),), ((30, None), (20, BRIEFING))])
def test_a_reconnect_after_arrival_still_briefs_that_morning_once(answers):
    routine = routine_with(arrive_at=300)
    routine.voice = UnfinishedBriefVoice(routine.clock, *answers)
    drive(routine, 400)
    assert routine.ledger.done(date(2026, 10, 10)) is False
    routine.camera.arrive_after_s = None
    routine.voice = FakeVoice(routine.clock, (20, REFRESHED))
    drive(routine, 1800)
    assert routine.voice.said == [("greet", 400, False), ("brief", 400, BRIEFING)]
    assert routine.camera.calls == [("start", None), ("stop",)] * 2
    assert routine.ledger.done(date(2026, 10, 10)) is True


@pytest.mark.parametrize("answer, spoken", [
    (REFRESHED, ["Markets fell."]),
    (None, ["Good morning.", "Markets rose."]),
    ((), []),
])
def test_a_reconnect_after_arrival_refreshes_an_expired_briefing(answer, spoken):
    routine = routine_with(arrive_at=300)
    routine.voice = UnfinishedBriefVoice(routine.clock, (30, BRIEFING))
    drive(routine, 301)
    assert routine.ledger.done(date(2026, 10, 10)) is False
    routine.clock.elapsed = 3600
    routine.camera.arrive_after_s = None
    routine.voice = FakeVoice(routine.clock, (30, answer))
    drive(routine, 70 * 60)
    assert [line.text for entry in routine.voice.said if entry[0] == "brief" for line in entry[2]] == spoken
    assert [entry for entry in routine.voice.said if entry[0] == "fetch"] == [
        ("fetch", 3600, "Brief me for Saturday, October 10, 05:50 local time.")]
    assert routine.voice.said[0] == ("greet", 3600, True)
    assert routine.camera.calls == [("start", None), ("stop",)] * 2
    assert routine.ledger.done(date(2026, 10, 10)) is True


def test_a_dropped_refresh_stays_dropped_if_disconnected_before_the_next_poll(tmp_path, monkeypatch):
    path = tmp_path / "morning-briefing.json"
    routine = routine_with(arrive_at=300, ledger=DayLedger(path))
    routine.select_chat("vm-1", "robot-chat")
    routine.voice = UnfinishedBriefVoice(routine.clock, (30, BRIEFING))
    drive(routine, 301)
    routine.clock.elapsed = 3600
    routine.camera.arrive_after_s = None

    async def disconnect_after_drop():
        dropped = asyncio.Event()

        class DroppingVoice(FakeVoice):
            async def fetch(self, request, *, attended):
                result = await super().fetch(request, attended=attended)
                dropped.set()
                return result

        async def wait_for_disconnect(seconds):
            await asyncio.Event().wait()

        routine.voice = DroppingVoice(routine.clock, (0, ()))
        with monkeypatch.context() as patch:
            patch.setattr(routine.clock, "sleep", wait_for_disconnect)
            task = asyncio.create_task(routine.run(routine.voice))
            try:
                await asyncio.wait_for(dropped.wait(), 5)
            finally:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    asyncio.run(disconnect_after_drop())
    assert routine.voice.said[-1] == ("fetched", 3600, True)
    persisted = DayLedger(path).done(date(2026, 10, 10))
    routine.select_chat("vm-1", "robot-chat")
    routine.voice = FakeVoice(routine.clock, (0, REFRESHED))
    drive(routine, 3700)
    assert kinds(routine) == []
    assert persisted is True


def test_a_restart_before_arrival_reuses_the_held_briefing():
    routine = routine_with((30, BRIEFING), (30, REFRESHED), (30, REFRESHED))
    drive(routine, 180)
    drive(routine, 420)
    routine.camera.arrive_after_s = 60
    drive(routine, 600)
    assert routine.voice.said == [
        ("fetch", 0, "Brief me for Saturday, October 10, 04:50 local time."),
        ("fetched", 30, False), ("greet", 480, False), ("brief", 480, BRIEFING)]
    assert routine.camera.calls == [("start", None), ("stop",)] * 3


@pytest.mark.parametrize("answer", [(600, BRIEFING), (30, None), (30, ())])
def test_a_restart_without_a_held_reply_waits_until_arrival_to_retry(answer):
    routine = routine_with(answer, (10, BRIEFING))
    drive(routine, 180)
    drive(routine, 420)
    assert kinds(routine).count("fetch") == 1
    routine.camera.arrive_after_s = 60
    drive(routine, 600)
    assert routine.voice.said[-4:] == [
        ("greet", 480, True),
        ("fetch", 480, "Brief me for Saturday, October 10, 04:58 local time."),
        ("fetched", 490, True), ("brief", 491, BRIEFING)]


def test_cancelling_a_live_retry_finishes_the_fetch_before_returning():
    class CancellableVoice(FakeVoice):
        async def fetch(self, request, *, attended):
            try:
                return await super().fetch(request, attended=attended)
            except asyncio.CancelledError:
                self.said.append(("fetch_cancelled", self.clock.elapsed))
                raise

    async def check():
        routine = routine_with(arrive_at=5)
        routine.voice = CancellableVoice(routine.clock, (0, None), (600, BRIEFING))
        camera = asyncio.create_task(routine.watch_camera(), name="camera")
        task = asyncio.create_task(routine.run(routine.voice))
        try:
            while kinds(routine).count("fetch") < 2:
                await asyncio.sleep(0)
        finally:
            task.cancel()
            camera.cancel()
            await asyncio.gather(task, camera, return_exceptions=True)
        assert kinds(routine) == ["fetch", "fetched", "greet", "fetch", "fetch_cancelled"]

    asyncio.run(check())


def test_a_restart_preserves_the_held_reply_age_and_refresh_limit():
    routine = routine_with((30, BRIEFING), (30, REFRESHED), (30, BRIEFING))
    drive(routine, 180)
    drive(routine, 2500)
    routine.camera.arrive_after_s = 60
    drive(routine, 2700)
    assert routine.voice.said == [
        ("fetch", 0, "Brief me for Saturday, October 10, 04:50 local time."),
        ("fetched", 30, False),
        ("fetch", 2430, "Brief me for Saturday, October 10, 05:30 local time."),
        ("fetched", 2460, False), ("greet", 2560, False), ("brief", 2560, REFRESHED)]


def test_restarting_after_a_failed_refresh_keeps_the_earlier_reply_without_retrying():
    routine = routine_with((30, BRIEFING), (30, None), (30, REFRESHED))
    drive(routine, 2500)
    routine.camera.arrive_after_s = 60
    drive(routine, 2700)
    assert [entry[:2] for entry in routine.voice.said] == [
        ("fetch", 0), ("fetched", 30), ("fetch", 2430), ("fetched", 2460), ("greet", 2560), ("brief", 2560)]
    assert routine.voice.said[-1][2] == BRIEFING


def test_restarting_during_a_refresh_keeps_the_earlier_reply_without_retrying():
    routine = routine_with((30, BRIEFING), (600, REFRESHED))
    drive(routine, 2500)
    routine.camera.arrive_after_s = 60
    drive(routine, 2700)
    assert [entry[:2] for entry in routine.voice.said] == [
        ("fetch", 0), ("fetched", 30), ("fetch", 2430), ("greet", 2560), ("brief", 2560)]
    assert routine.voice.said[-1][2] == BRIEFING


def test_a_held_reply_that_aged_during_a_disconnect_is_refreshed_on_restart():
    routine = routine_with((30, BRIEFING), (30, REFRESHED))
    drive(routine, 180)
    routine.clock.elapsed = 2500
    routine.camera.arrive_after_s = 60
    drive(routine, 2700)
    assert [entry[:2] for entry in routine.voice.said] == [
        ("fetch", 0), ("fetched", 30), ("fetch", 2500), ("fetched", 2530), ("greet", 2560), ("brief", 2560)]
    assert routine.voice.said[-1][2] == REFRESHED


def test_a_new_window_does_not_reuse_an_unspoken_reply_from_yesterday():
    routine = routine_with((30, BRIEFING), (30, REFRESHED))
    drive(routine, 180)
    routine.clock.elapsed = 24 * 3600
    routine.camera.arrive_after_s = 60
    drive(routine, 24 * 3600 + 120)
    assert routine.voice.said[-4:] == [
        ("fetch", 86400, "Brief me for Sunday, October 11, 04:50 local time."),
        ("fetched", 86430, False), ("greet", 86460, False), ("brief", 86460, REFRESHED)]


def test_camera_opens_offline_logs_each_minute_and_closes_at_the_window_boundary(caplog):
    caplog.set_level("INFO", logger="reachy_muse.morning_briefing")

    async def check():
        class SteppedClock(FakeClock):
            changed = asyncio.Event()

            async def sleep(self, seconds):
                await self.changed.wait()

            async def advance(self, elapsed):
                self.elapsed = elapsed
                event, self.changed = self.changed, asyncio.Event()
                event.set()
                await asyncio.sleep(0)

        clock = SteppedClock(SATURDAY - timedelta(minutes=1))
        camera = FakeCamera(clock)
        diagnostics = []
        camera.log_arrival_diagnostics = lambda: diagnostics.append(clock.elapsed)
        routine = MorningRoutine(WINDOW, "Brief me.", camera, DayLedger(None), clock=clock)
        task = asyncio.create_task(routine.watch_camera())
        try:
            await asyncio.sleep(0)
            assert camera.calls == []
            await clock.advance(60)
            assert camera.calls == [("start", None)]
            await clock.advance(119)
            assert diagnostics == []
            await clock.advance(120)
            await clock.advance(180)
            assert diagnostics == [120, 180]
            await clock.advance(4260)
            assert camera.calls == [("start", None), ("stop",)]
            assert routine.ledger.done(date(2026, 10, 10)) is False
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(check())
    assert [record.getMessage() for record in caplog.records] == [
        "Arrival watcher started: window opened",
        "Arrival watcher stopped: window closed"]
