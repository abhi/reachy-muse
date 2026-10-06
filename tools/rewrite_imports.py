"""Split the Reachy app out of muse-gadget-sdk and reachy-companion into reachy-muse.

One table drives both steps of the split:

  python tools/rewrite_imports.py paths sdk        # git filter-repo --paths-from-file for muse-gadget-sdk
  python tools/rewrite_imports.py paths companion  # the same for reachy-companion
  python tools/rewrite_imports.py rewrite [ROOT]   # rewrite imports in a reachy-muse checkout
  python tools/rewrite_imports.py fake-vm SDK/linux/tests/test_link_client.py > tests/fake_muse_vm.py
  python tools/rewrite_imports.py contract SDK REF [ROOT]  # copy the SDK's Reachy contract into reachy_muse.contract

The rewrite is idempotent: running it on its own output changes nothing.
"""
import re
import sys
from pathlib import Path

SDK_SRC = "linux/src/musegadget/"
SDK_TESTS = "linux/tests/"

# Old musegadget module -> new reachy_muse module.
MODULES = {
    "reachy_capabilities": "capabilities",
    "reachy_cli": "cli",
    "reachy_companion_client": "companion_client",
    "reachy_companion_protocol": "protocol",
    "reachy_duplex_audio": "duplex_audio",
    "reachy_expression_plan": "expression_plan",
    "reachy_failover": "failover",
    "reachy_follow_up": "follow_up",
    "reachy_hardware": "hardware",
    "reachy_local_backends": "local_backends",
    "reachy_progress": "progress",
    "reachy_speaker_eq": "speaker_eq",
    "reachy_voice": "conversation",
    "local_face_tracking": "local_face_tracking",
    "local_speech": "local_speech",
    "local_transcription": "local_transcription",
    "speech_gate": "speech_gate",
    "speech_playback": "speech_playback",
    "streaming_transcription": "streaming_transcription",
    "voice_audio": "voice_audio",
    "vosk_wake": "vosk_wake",
    "wake_word": "wake_word",
}

# Old test file -> new test file, for tests whose name does not follow from MODULES.
TESTS = {
    "test_reachy_voice.py": "test_conversation.py",
    "test_reachy_companion_protocol.py": "test_protocol.py",
    "test_reachy_capabilities.py": "test_capabilities.py",
    "test_reachy_chat_contract.py": "test_chat_contract.py",
    "test_reachy_cli.py": "test_cli.py",
    "test_reachy_companion.py": "test_companion.py",
    "test_reachy_companion_client.py": "test_companion_client.py",
    "test_reachy_duplex.py": "test_duplex.py",
    "test_reachy_duplex_audio.py": "test_duplex_audio.py",
    "test_reachy_duplex_privacy.py": "test_duplex_privacy.py",
    "test_reachy_duplex_voice.py": "test_duplex_voice.py",
    "test_reachy_expression_plan.py": "test_expression_plan.py",
    "test_reachy_failover.py": "test_failover.py",
    "test_reachy_follow_up.py": "test_follow_up.py",
    "test_reachy_hardware.py": "test_hardware.py",
    "test_reachy_local_backends.py": "test_local_backends.py",
    "test_reachy_pins.py": "test_pins.py",
    "test_reachy_progress.py": "test_progress.py",
    "test_reachy_short_wake.py": "test_short_wake.py",
    "test_reachy_speaker_eq.py": "test_speaker_eq.py",
    "test_reachy_streaming_input.py": "test_streaming_input.py",
    "test_reachy_turn_pacing.py": "test_turn_pacing.py",
    "test_reachy_wake_handoff.py": "test_wake_handoff.py",
    "test_local_face_tracking.py": "test_local_face_tracking.py",
    "test_local_speech.py": "test_local_speech.py",
    "test_local_transcription.py": "test_local_transcription.py",
    "test_speech_gate.py": "test_speech_gate.py",
    "test_speech_playback.py": "test_speech_playback.py",
    "test_streaming_transcription.py": "test_streaming_transcription.py",
    "test_voice_audio.py": "test_voice_audio.py",
    "test_vosk_wake.py": "test_vosk_wake.py",
    "test_wake_word.py": "test_wake_word.py",
}

SDK_PATHS = {
    **{f"{SDK_SRC}{old}.py": f"src/reachy_muse/{new}.py" for old, new in MODULES.items()},
    **{f"{SDK_TESTS}{old}": f"tests/{new}" for old, new in TESTS.items()},
    f"{SDK_SRC}data/face_tracking/": "src/reachy_muse/data/face_tracking/",
    f"{SDK_SRC}data/companion_protocol_v1/": "src/reachy_muse/data/companion_protocol_v1/",
    f"{SDK_TESTS}pins/": "tests/pins/",
    f"{SDK_TESTS}fake_companion.py": "tests/fake_companion.py",
    "linux/docs/reachy-mini.md": "docs/reachy-mini.md",
    "linux/docs/reachy-companion-protocol.md": "docs/reachy-companion-protocol.md",
    "linux/examples/muse-reachy.path": "examples/muse-reachy.path",
    "linux/examples/muse-reachy.service": "examples/muse-reachy.service",
}

# The Reachy contract stayed in the SDK as musegadget.reachy until reachy-muse took it over.
CONTRACT_PATHS = {
    **{f"{SDK_SRC}reachy/{name}.py": f"src/reachy_muse/contract/{name}.py"
       for name in ["__init__", "commands", "expressions", "prompts", "replies"]},
    **{f"{SDK_TESTS}test_reachy_{name}.py": f"tests/test_contract_{name}.py"
       for name in ["commands", "expressions", "prompts", "replies"]},
}

COMPANION_PATHS = {
    "reachy_companion/": "src/reachy_muse/companion_server/",
    "tests/": "tests/companion_server/",
    "evals/": "evals/",
    "bench/": "bench/",
    "README.md": "docs/companion-server.md",
    "pyproject.toml": "pyproject.toml",
    ".gitignore": ".gitignore",
}

MOVED = "|".join(sorted(MODULES, key=len, reverse=True))

# Applied in order to every .py file outside the copied data.
LINE_RULES = [
    # The companion took Expression from reachy_capabilities, which only re-imported it from the contract.
    (re.compile(r"^from musegadget\.reachy_capabilities import Expression$", re.M),
     "from reachy_muse.contract.expressions import Expression"),
    (re.compile(r"\bmusegadget\.reachy\b(?![\w-])"), "reachy_muse.contract"),
    (re.compile(r"\bmusegadget\.(%s)\b" % MOVED), lambda m: "reachy_muse." + MODULES[m.group(1)]),
    (re.compile(r'files\("musegadget"\) / "data"'), 'files("reachy_muse") / "data"'),
    (re.compile(r"""(["'])musegadget\[reachy\]\1"""), r"\1reachy-muse[robot]\1"),
    (re.compile(r"\breachy_companion(?=\.)"), "reachy_muse.companion_server"),
    (re.compile(r'"-m", "reachy_companion"'), '"-m", "reachy_muse.companion_server"'),
    (re.compile(r"^import reachy_companion$", re.M), "import reachy_muse.companion_server"),
    (re.compile(r"^(\s*)from reachy_companion import ", re.M), r"\1from reachy_muse.companion_server import "),
    (re.compile(r'getLogger\("reachy_companion"\)'), 'getLogger("reachy_muse.companion_server")'),
]

# Robot tests import helpers from other test modules by module name.
TEST_MODULES = {old[:-3]: new[:-3] for old, new in TESTS.items()}
TEST_MODULES["test_link_client"] = "fake_muse_vm"
TEST_IMPORT = re.compile(r"^(\s*from )(%s)( import )" % "|".join(sorted(TEST_MODULES, key=len, reverse=True)), re.M)

# The SDK's fake Muse VM, which tests/test_conversation.py drives a real LinkSession against.
FAKE_VM_NAMES = ["DEVICE", "Pipe", "FakeVm", "make_session", "registered_session"]

# A renamed module imported bare, as in `from musegadget import reachy_voice`, becomes
# `import reachy_muse.conversation`, and `reachy_voice.X` becomes `reachy_muse.conversation.X`.
# The full name cannot collide with a local such as `conversation = VoiceConversation(...)`.
BARE = re.compile(r"(?<![\w.'\"/-])(%s)\b(?![\w-])" % "|".join(old for old, new in MODULES.items() if old != new))

FROM_IMPORT = re.compile(r"^(?P<indent>[ \t]*)from musegadget import (?P<names>[^()\n]+)$", re.M)


def split_from_imports(text: str) -> str:
    """Split `from musegadget import a, reachy_voice` into one SDK line and one reachy_muse line."""
    def split(match: re.Match) -> str:
        indent = match.group("indent")
        kept, moved, qualified = [], [], []
        for item in (part.strip() for part in match.group("names").split(",")):
            name, _, alias = item.partition(" as ")
            if name in MODULES and not alias and MODULES[name] != name:
                qualified.append(f"reachy_muse.{MODULES[name]}")
            elif name in MODULES:
                moved.append(MODULES[name] + (f" as {alias}" if alias else ""))
            elif name == "__version__":
                moved.append(item)
            else:
                kept.append(item)
        lines = []
        if kept:
            lines.append(f"{indent}from musegadget import {', '.join(kept)}")
        if moved:
            lines.append(f"{indent}from reachy_muse import {', '.join(moved)}")
        lines += [f"{indent}import {module}" for module in qualified]
        return "\n".join(lines)
    return FROM_IMPORT.sub(split, text)


def rewrite_text(text: str) -> str:
    for pattern, replacement in LINE_RULES:
        text = pattern.sub(replacement, text)
    text = split_from_imports(text)
    text = TEST_IMPORT.sub(lambda m: m.group(1) + TEST_MODULES[m.group(2)] + m.group(3), text)
    return BARE.sub(lambda m: "reachy_muse." + MODULES[m.group(1)], text)


def rewrite_tree(root: Path) -> list:
    fixed_depth = root / "tests/companion_server"
    changed = []
    for path in sorted(root.rglob("*.py")):
        relative = path.relative_to(root)
        if relative.parts[0] in {".git", ".venv"} or relative.parts[:2] == ("tools", "rewrite_imports.py"):
            continue
        before = path.read_text()
        after = rewrite_text(before)
        if fixed_depth in path.parents:
            # The companion tests moved from tests/ to tests/companion_server/, one level deeper.
            after = after.replace("Path(__file__).resolve().parents[1]", "Path(__file__).resolve().parents[2]")
        if after != before:
            path.write_text(after)
            changed.append(str(relative))
    return changed


def fake_vm(sdk_test_link_client: Path) -> str:
    """Copy the fake Muse VM out of the SDK's test_link_client.py: FAKE_VM_NAMES and the imports they use."""
    import ast
    source = sdk_test_link_client.read_text()
    tree = ast.parse(source)
    lines = source.splitlines()
    defs = [node for node in tree.body
            if getattr(node, "name", None) in FAKE_VM_NAMES
            or any(isinstance(t, ast.Name) and t.id in FAKE_VM_NAMES for t in getattr(node, "targets", []))]
    used = {n.id for d in defs for n in ast.walk(d) if isinstance(n, ast.Name)}
    imports = []
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            imports.append(ast.unparse(node))
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            node.names = [a for a in node.names if (a.asname or a.name).split(".")[0] in used]
            if node.names:
                imports.append(ast.unparse(node))
    header = ('"""The fake Muse VM from the Muse Gadget SDK\'s linux/tests/test_link_client.py.\n\n'
              'Generated by tools/rewrite_imports.py fake-vm. Do not edit by hand."""\n\n')
    bodies = ["\n".join(lines[node.lineno - 1:node.end_lineno]) for node in defs]
    return header + "\n".join(imports) + "\n\n\n" + "\n\n\n".join(bodies) + "\n"


def copy_contract(sdk: Path, ref: str, root: Path) -> list:
    """Write the contract files from SDK at REF into ROOT, with their imports rewritten."""
    import subprocess
    for old, new in CONTRACT_PATHS.items():
        source = subprocess.run(["git", "-C", str(sdk), "show", f"{ref}:{old}"],
                                check=True, capture_output=True, text=True).stdout
        target = root / new
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(rewrite_text(source))
    return list(CONTRACT_PATHS.values())


def paths_file(mapping: dict) -> str:
    """Keep each old path, then rename it. filter-repo keeps everything when the file has only renames."""
    return "".join(f"{old}\n{old}==>{new}\n" for old, new in mapping.items())


if __name__ == "__main__":
    if sys.argv[1:2] == ["paths"]:
        sys.stdout.write(paths_file({"sdk": SDK_PATHS, "companion": COMPANION_PATHS}[sys.argv[2]]))
    elif sys.argv[1:2] == ["fake-vm"]:
        sys.stdout.write(fake_vm(Path(sys.argv[2])))
    elif sys.argv[1:2] == ["contract"]:
        root = Path(sys.argv[4] if len(sys.argv) > 4 else ".").resolve()
        for name in copy_contract(Path(sys.argv[2]), sys.argv[3], root):
            print(name)
    elif sys.argv[1:2] == ["rewrite"]:
        root = Path(sys.argv[2] if len(sys.argv) > 2 else ".").resolve()
        for name in rewrite_tree(root):
            print(name)
    else:
        sys.exit(__doc__)
