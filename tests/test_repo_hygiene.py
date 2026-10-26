"""Keep machine-specific paths and account names out of tracked text."""

import subprocess
from pathlib import Path


def test_tracked_text_has_no_machine_local_paths_or_usernames():
    # Split the literals so this check also scans its own source.
    forbidden = ("/private" "/tmp", "/" "Users/", "ry" "thm")
    result = subprocess.run(
        ["git", "grep", "-n", "-I", "-F", *[arg for word in forbidden for arg in ("-e", word)], "--", "."],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )
    assert result.returncode in (0, 1), result.stderr
    assert result.returncode == 1, f"Machine-local identifiers in tracked text:\n{result.stdout}"
