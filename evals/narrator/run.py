"""Score narrator models on the eval set, one model per process so memory is measured alone.

    .venv/bin/python evals/narrator/run.py                       # every candidate, then the table
    .venv/bin/python evals/narrator/run.py --model REPO           # one model, writes results/<name>.json
    .venv/bin/python evals/narrator/run.py --report               # the table from the saved results

A case passes when every line it finally returns is grounded by this repo's check and by the SDK's
grounded(), matches none of the case's forbidden patterns, fits the format (count, 240 bytes, no
markup) and carries a valid expression (a fallback line may carry none). A "silent" case passes only
when nothing is said: small talk gets no acknowledgement. "Raw" scores the model's
first answer alone, before the check, the retry and the fallback.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

from reachy_muse.capabilities import grounded
from reachy_muse.protocol import NarrateOp

from reachy_muse.companion_server.grounding import ungrounded
from reachy_muse.companion_server.narrator_mlx import MARKUP, MAX_LINE_BYTES, OPS, Ask, MlxNarrator, parse

HERE = Path(__file__).resolve().parent

CANDIDATES = (
    "mlx-community/Qwen3-4B-Instruct-2507-4bit",
    "mlx-community/Phi-4-mini-instruct-4bit",
    "mlx-community/Llama-3.1-8B-Instruct-4bit",
    "mlx-community/granite-4.1-8b-4bit",
)
LICENSES = {"Qwen3-4B-Instruct-2507-4bit": "Apache-2.0", "Phi-4-mini-instruct-4bit": "MIT",
            "Llama-3.1-8B-Instruct-4bit": "Llama 3.1 Community", "granite-4.1-8b-4bit": "Apache-2.0"}
BUDGETS_S = {"acknowledge": 0.9, "progress": 1.5, "lines": 4.0}
RESULTS = HERE / "results"


def ask_for(case: Dict[str, Any]) -> Ask:
    return Ask(NarrateOp(case["op"]), case["request"], status=case.get("status", ""),
               already_said=tuple(case.get("already_said", ())), reply=case.get("reply", ""))


def score(case: Dict[str, Any], lines, fell_back: bool) -> Dict[str, Any]:
    ask = ask_for(case)
    texts = [line.text for line in lines]
    forbidden = [p for p in case.get("forbidden", ()) for t in texts if re.search(p, t, re.IGNORECASE)]
    result = {
        "ungrounded": [f"{t}: {p}" for t in texts for p in ungrounded(t, *ask.sources())],
        "robot_rejects": [t for t in texts if not grounded(t, *ask.sources())],
        "forbidden": forbidden,
        "format": [t for t in texts if not t or len(t.encode("utf-8")) > MAX_LINE_BYTES or MARKUP.search(t)]
                  + (["too many lines"] if len(texts) > OPS[ask.op].max_lines else []),
        "expression": [l.text for l in lines if l.expression is None and not fell_back],
        "spoke_to_small_talk": texts if case.get("silent") else [],
    }
    result["pass"] = not any(result.values())
    expect = case.get("expect")
    result["kept_key_fact"] = None if not expect or not texts else any(re.search(p, " ".join(texts), re.IGNORECASE) for p in expect)
    return result


def run_model(repo: str, repeat: int, cases_file: str) -> Dict[str, Any]:
    import mlx.core as mx

    cases = json.loads((HERE / cases_file).read_text())["cases"]
    narrator = MlxNarrator(repo)
    started = time.perf_counter()
    narrator.load_now()
    load_s = time.perf_counter() - started
    weights_gb = mx.get_active_memory() / 1e9
    records = []
    for case in cases:
        for _ in range(repeat):
            began = time.perf_counter()
            narration = narrator.narrate(ask_for(case))
            seconds = time.perf_counter() - began
        first = narration.attempts[0].raw if narration.attempts else '{"lines": []}'  # no attempt: the small-talk rule answered
        raw_lines, raw_problems = parse(ask_for(case), first)
        raw = score(case, raw_lines, fell_back=False)
        records.append({
            "id": case["id"], "op": case["op"], "category": case["category"], "seconds": seconds,
            "lines": [[l.text, l.expression.value if l.expression else None] for l in narration.lines],
            "fell_back": narration.fell_back, "attempts": len(narration.attempts),
            "first_answer": first, "first_problems": list(raw_problems),
            "raw_pass": not raw_problems and raw["pass"], "raw_ungrounded": raw["ungrounded"], "raw_forbidden": raw["forbidden"],
            **score(case, narration.lines, narration.fell_back),
        })
        print(f"{case['id']:<44} {'pass' if records[-1]['pass'] else 'FAIL'} {seconds:5.2f}s"
              f"{' fallback' if narration.fell_back else ''}{' retry' if len(narration.attempts) > 1 else ''}", file=sys.stderr)
    return {"repo": repo, "load_s": load_s, "weights_gb": weights_gb, "peak_gb": mx.get_peak_memory() / 1e9,
            "sdk": _sdk_head(), "cases": records}


def _sdk_head() -> str:
    import musegadget

    root = Path(musegadget.__file__).resolve().parents[3]
    return subprocess.run(["git", "-C", str(root), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()


def summary(result: Dict[str, Any]) -> Dict[str, Any]:
    cases = result["cases"]
    n = len(cases)
    out = {
        "model": result["repo"].split("/")[-1], "cases": n,
        "pass": sum(c["pass"] for c in cases), "raw_pass": sum(c["raw_pass"] for c in cases),
        "raw_ungrounded": sum(bool(c["raw_ungrounded"]) for c in cases),
        "raw_forbidden": sum(bool(c["raw_forbidden"]) for c in cases),
        "ungrounded": sum(bool(c["ungrounded"] or c["robot_rejects"]) for c in cases),
        "forbidden": sum(bool(c["forbidden"]) for c in cases),
        "fell_back": sum(c["fell_back"] for c in cases), "retried": sum(c["attempts"] > 1 for c in cases),
        "kept_key_fact": f"{sum(c['kept_key_fact'] is True for c in cases)}/{sum(c['kept_key_fact'] is not None for c in cases)}",
        "weights_gb": round(result["weights_gb"], 2), "peak_gb": round(result["peak_gb"], 2), "load_s": round(result["load_s"], 1),
    }
    for op, budget in BUDGETS_S.items():
        seconds = sorted(c["seconds"] for c in cases if c["op"] == op)
        out[f"{op}_p50"] = statistics.median(seconds)
        out[f"{op}_p95"] = seconds[min(len(seconds) - 1, round(0.95 * (len(seconds) - 1)))]
        out[f"{op}_over_budget"] = sum(s > budget for s in seconds)
    return out


def table(rows: List[Dict[str, Any]]) -> str:
    head = ["Model", "License", "Pass", "Raw pass", "Raw ungrounded", "Ungrounded after check", "Forbidden", "Fallback",
            "Retried", "Kept key fact", "Ack p50/p95", "Progress p50/p95", "Lines p50/p95", "Weights", "Peak"]
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rows:
        n = r["cases"]
        cells = [r["model"], LICENSES.get(r["model"], "?"), f"{r['pass']}/{n}", f"{r['raw_pass']}/{n}", r["raw_ungrounded"],
                 r["ungrounded"], r["forbidden"], r["fell_back"], r["retried"], r["kept_key_fact"],
                 *(f"{r[op + '_p50']:.2f}/{r[op + '_p95']:.2f} s" for op in BUDGETS_S), f"{r['weights_gb']} GB", f"{r['peak_gb']} GB"]
        out.append("| " + " | ".join(str(c) for c in cells) + " |")
    return "\n".join(out)


def _saved(repo: str, cases_file: str) -> Path:
    suffix = "" if cases_file == "cases.json" else "." + Path(cases_file).stem
    return RESULTS / f"{repo.split('/')[-1]}{suffix}.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="score one model in this process")
    parser.add_argument("--repeat", type=int, default=1, help="run each case this many times and time the last run")
    parser.add_argument("--report", action="store_true", help="print the table from saved results")
    parser.add_argument("--cases", default="cases.json", help="the case file in evals/narrator: cases.json, or holdout.json written blind to the prompts")
    args = parser.parse_args()
    RESULTS.mkdir(exist_ok=True)
    if args.model:
        result = run_model(args.model, args.repeat, args.cases)
        _saved(args.model, args.cases).write_text(json.dumps(result, indent=1) + "\n")
        print(json.dumps(summary(result), indent=1))
        return
    if not args.report:
        for repo in CANDIDATES:
            subprocess.run([sys.executable, __file__, "--model", repo, "--repeat", str(args.repeat), "--cases", args.cases], check=True)
    rows = [summary(json.loads(_saved(repo, args.cases).read_text())) for repo in CANDIDATES if _saved(repo, args.cases).exists()]
    print(table(rows))


if __name__ == "__main__":
    main()
