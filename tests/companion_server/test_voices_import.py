import subprocess
import sys


def test_the_serve_cli_checks_a_voice_without_loading_numpy_or_mlx():
    probe = ("import sys; from reachy_muse.companion_server import cli; cli._voice('chatterbox-turbo'); "
             "print(sorted(m for m in ('numpy', 'mlx', 'mlx_audio') if m in sys.modules))")
    assert subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True).stdout == "[]\n"
