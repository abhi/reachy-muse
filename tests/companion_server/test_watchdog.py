"""Exercise process death, including an engine whose asyncio waiter was cancelled. No models."""

import asyncio
import os
import resource
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from reachy_muse.companion_server.watchdog import EngineExecutor, Watchdog


def test_hearing_hard_limit_dumps_all_threads_and_fault_counts_before_exit(monkeypatch, capfd):
    now = [100.0]
    entered, release, exited = threading.Event(), threading.Event(), threading.Event()
    exits, usage_calls = [], []

    def usage(who):
        usage_calls.append(who)
        return SimpleNamespace(ru_minflt=17, ru_majflt=3)

    def exit_process(status):
        exits.append(status)
        os.write(2, b"INJECTED EXIT\n")
        exited.set()

    def blocked_hearing_for_dump():
        private = "private-hearing-content"
        entered.set()
        assert release.wait(5)
        return private

    monkeypatch.setattr(resource, "getrusage", usage)
    monitor = Watchdog(clock=lambda: now[0], loop_s=1000, check_s=.001, exit_process=exit_process)

    async def scenario():
        monitor.start()
        try:
            with EngineExecutor("hearing", monitor=monitor) as worker:
                pending = worker.submit(blocked_hearing_for_dump)
                try:
                    assert await asyncio.to_thread(entered.wait, 2)
                    now[0] = 219.9
                    assert not await asyncio.to_thread(exited.wait, .02)
                    now[0] = 220.0
                    assert await asyncio.to_thread(exited.wait, 2)
                finally:
                    release.set()
                assert pending.result() == "private-hearing-content"
        finally:
            monitor.stop()

    asyncio.run(scenario())
    output = capfd.readouterr().err
    assert exits == [70]
    assert "hearing made no progress for 120.0s (limit 120s)" in output
    assert "blocked_hearing_for_dump" in output
    assert "page faults: minor=17 major=3" in output
    assert output.index("blocked_hearing_for_dump") < output.index("INJECTED EXIT")
    assert output.index("page faults:") < output.index("INJECTED EXIT")
    assert usage_calls == [resource.RUSAGE_SELF]
    assert "private-hearing-content" not in output


def test_a_quiet_stage_delays_only_the_soft_stall_check(capfd):
    now, exits = [100.0], []
    entered, release = threading.Event(), threading.Event()
    monitor = Watchdog(clock=lambda: now[0], loop_s=1000, check_s=.001, exit_process=exits.append)

    def final_pass():
        monitor.progress(quiet_s=60.0)
        entered.set()
        assert release.wait(5)

    async def scenario():
        monitor.start()
        try:
            with EngineExecutor("hearing", monitor=monitor) as worker:
                worker.submit(final_pass)
                try:
                    assert await asyncio.to_thread(entered.wait, 2)
                    stalled = []
                    for at in (159.9, 160.0, 220.0):
                        now[0] = at
                        stalled.append(monitor.stalled("hearing", 3.0))
                        await asyncio.sleep(.02)
                    return stalled
                finally:
                    release.set()
        finally:
            monitor.stop()

    assert asyncio.run(scenario()) == [False, True, True]
    assert exits == [70]
    assert "hearing made no progress for 120.0s (limit 120s)" in capfd.readouterr().err


@pytest.mark.parametrize("blocked, reason", [
    ("time.sleep(10)", "event loop"),
    ("""
worker = EngineExecutor('stuck-native-call', monitor=monitor)
entered = threading.Event()
def stuck():
    entered.set()
    time.sleep(10)
future = asyncio.get_running_loop().run_in_executor(worker, stuck)
await asyncio.to_thread(entered.wait)
future.cancel()
await asyncio.sleep(10)
""", "stuck-native-call"),
])
def test_stall_exits_nonzero_even_after_asyncio_cancels_the_native_waiter(blocked, reason):
    script = """
import asyncio, threading, time
from reachy_muse.companion_server.watchdog import EngineExecutor, Watchdog
async def main():
    monitor = Watchdog(loop_s=.4, engine_s=.1, check_s=.01)
    monitor.start()
""" + "\n".join("    " + line for line in blocked.splitlines()) + "\nasyncio.run(main())\n"
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=5)
    assert result.returncode == 70, result.stderr
    assert f"companion watchdog: {reason} made no progress" in result.stderr
    assert "exiting 70 for restart" in result.stderr


def test_model_load_idle_time_and_a_long_progressing_briefing_do_not_restart():
    exits = []
    monitor = Watchdog(loop_s=1, engine_s=.12, check_s=.01, exit_process=exits.append)

    async def scenario():
        with EngineExecutor("narrator", monitor=monitor) as engine:
            # Loading is unbounded. The CLI arms the watchdog after all engine.load calls.
            await asyncio.wrap_future(engine.submit(time.sleep, .25))
            monitor.start()
            try:
                await asyncio.sleep(.25)  # idle is healthy

                def briefing():
                    for _ in range(30):
                        time.sleep(.01)
                        monitor.progress()
                    return "The whole briefing."

                long = engine.submit(briefing)
                queued = engine.submit(lambda: "Next line.")
                assert await asyncio.wrap_future(long) == "The whole briefing."
                assert await asyncio.wrap_future(queued) == "Next line."
                await asyncio.sleep(.2)  # completed calls must be forgotten
            finally:
                monitor.stop()
        assert exits == []

    asyncio.run(scenario())


def test_chatterbox_timeout_cannot_hide_its_stuck_worker():
    script = """
import asyncio, threading, time
from reachy_muse.companion_server.voice_chatterbox import ChatterboxVoice
from reachy_muse.companion_server.watchdog import watchdog
from reachy_muse.protocol import VoiceRole
entered = threading.Event()
class Model:
    sample_rate = 24000
    def tokens(self, text):
        entered.set()
        time.sleep(10)
        yield [], True
async def main():
    voice = ChatterboxVoice()
    voice._model = Model()
    watchdog.engine_s, watchdog.loop_s, watchdog.check_s = .15, 2, .01
    watchdog.start()
    speech = voice.speak('A long briefing.', VoiceRole.MAIN, 24000)
    pending = asyncio.create_task(speech.__anext__())
    await asyncio.to_thread(entered.wait)
    pending.cancel()
    await asyncio.gather(pending, return_exceptions=True)
    await speech.aclose()
    await asyncio.sleep(10)
asyncio.run(main())
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=5)
    assert result.returncode == 70, result.stderr
    assert "voice-chatterbox made no progress" in result.stderr


def test_narrator_tokens_keep_a_long_briefing_alive(monkeypatch):
    from reachy_muse.companion_server import narrator_mlx

    exits = []
    monitor = Watchdog(loop_s=1, engine_s=.12, check_s=.01, exit_process=exits.append)
    monkeypatch.setattr(narrator_mlx, "watchdog", monitor)

    def tokens(*args, **kwargs):
        for character in "Here is the whole morning briefing.":
            time.sleep(.01)
            yield SimpleNamespace(text=character)

    monkeypatch.setitem(sys.modules, "mlx_lm", SimpleNamespace(stream_generate=tokens))
    narrator = narrator_mlx.MlxNarrator()
    narrator._thread.shutdown()
    narrator._thread = EngineExecutor("narrator", monitor=monitor)
    narrator._tokenizer = SimpleNamespace(apply_chat_template=lambda *args, **kwargs: "prompt",
                                          encode=lambda *args, **kwargs: [1])

    async def scenario():
        monitor.start()
        try:
            assert await asyncio.get_running_loop().run_in_executor(
                narrator._thread, narrator._generate, [], 100, "") == "Here is the whole morning briefing."
            assert exits == []
        finally:
            monitor.stop()
            narrator._thread.shutdown()

    asyncio.run(scenario())
