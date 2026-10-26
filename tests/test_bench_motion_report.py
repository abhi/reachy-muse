import json
import subprocess
import sys
from pathlib import Path

from bench.motion_report import model_name


def test_model_identifiers_keep_repo_ids_and_strip_local_directories(tmp_path):
    local = tmp_path / "4b-8bit"
    local.mkdir()
    assert model_name(str(local)) == "4b-8bit"
    assert model_name(Path("models/4b")) == "4b"
    assert model_name("~/models/4b") == "4b"
    assert model_name("mlx-community/Qwen3-4B-Instruct-2507-4bit") == "mlx-community/Qwen3-4B-Instruct-2507-4bit"
    assert model_name(None) is None


def test_compact_report_cli_preserves_statistics_and_is_safe_to_repeat(tmp_path):
    report = tmp_path / "run.json"
    report.write_text(json.dumps({
        "config": {"bundle": str(tmp_path / "4b"), "mlx_model": None, "contend": "org/model"},
        "latency_ms": {"planner": {"p50": 852, "p95": 982}},
        "quality": {"nodding yes": {"frames": 5, "outside_limits": 0}},
        "runs": [{"prompt": "nodding yes", "recipe": "nod", "planner": 852}],
        "trajectories": {"nodding yes": {
            "channels": ["yaw", "pitch"], "fps": 25,
            "frames": [[0, 0], [8, 5], [-8, -5], [35, 20], [-35, -20]],
        }},
    }))
    command = [sys.executable, str(Path(__file__).resolve().parents[1] / "bench/motion_report.py"), str(report)]
    subprocess.run(command, check=True)
    assert json.loads(report.read_text()) == {
        "config": {"bundle": "4b", "mlx_model": None, "contend": "org/model"},
        "latency_ms": {"planner": {"p50": 852, "p95": 982}},
        "quality": {"nodding yes": {
            "frames": 5, "outside_limits": 0, "frames_at_limit": {"yaw": 2, "pitch": 2},
            "pitch_swings": 4, "yaw_swings": 4,
        }},
        "runs": [{"prompt": "nodding yes", "recipe": "nod", "planner": 852}],
    }
    compacted = report.read_bytes()
    subprocess.run(command, check=True)
    assert report.read_bytes() == compacted
