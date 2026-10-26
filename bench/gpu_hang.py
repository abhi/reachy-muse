"""Run the narrator, Kokoro and Chatterbox in loops on their own threads, to see whether the GPU hangs.

    MLX_MAX_OPS_PER_BUFFER=16 .venv/bin/python bench/gpu_hang.py [SECONDS] [llm,kokoro,chatterbox]

It prints how many loops each finished and any error. A Metal hang raises in one thread and
discards the others' work; sometimes the process aborts before it prints anything.
"""

from __future__ import annotations

import sys
import threading
import time
import traceback

from mlx_audio.tts.utils import load_model
from mlx_lm import generate, load

SECONDS = float(sys.argv[1]) if len(sys.argv) > 1 else 180
ONLY = set(sys.argv[2].split(",")) if len(sys.argv) > 2 else {"llm", "kokoro", "chatterbox"}

llm, tokenizer = load("mlx-community/Qwen3-4B-Instruct-2507-4bit")
kokoro = load_model("mlx-community/Kokoro-82M-bf16")
list(kokoro.generate("Warm up.", voice="af_heart", lang_code="a"))
chatterbox = load_model("mlx-community/chatterbox-turbo-8bit")

WORK = {
    "llm": lambda: generate(llm, tokenizer, prompt="Tell me a story about Reachy.", max_tokens=128),
    "kokoro": lambda: list(kokoro.generate("Your flight to Lisbon leaves Tuesday at seven forty five.", voice="af_heart", lang_code="a")),
    "chatterbox": lambda: list(chatterbox.generate("Sure, I can help with that. It is 18 degrees in Oslo.")),
}
counts = {name: 0 for name in ONLY}
errors = []
stop = time.time() + SECONDS


def loop(name: str) -> None:
    try:
        while time.time() < stop:
            WORK[name]()
            counts[name] += 1
    except Exception as error:
        errors.append((name, repr(error)[:200]))
        traceback.print_exc()


threads = [threading.Thread(target=loop, args=(name,)) for name in sorted(ONLY)]
for thread in threads:
    thread.start()
for thread in threads:
    thread.join()
print("counts", counts, "errors", errors, flush=True)
