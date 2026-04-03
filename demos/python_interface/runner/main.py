"""Runner for the Python-interface demo crate."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import tempfile
import venv


def parse_args() -> argparse.Namespace:
    """Parse the generated crate directory used by the runner."""
    parser = argparse.ArgumentParser(
        description="Install the generated Python module into an isolated venv and call it.",
    )
    parser.add_argument(
        "--crate-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "foo",
        help="Path to the generated Python-enabled Rust crate.",
    )
    return parser.parse_args()


def _venv_python(venv_dir: Path) -> Path:
    if sys.platform == "win32":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def main() -> None:
    args = parse_args()
    crate_dir = args.crate_dir.resolve()
    if not crate_dir.is_dir():
        raise FileNotFoundError(f"generated crate directory not found: {crate_dir}")

    with tempfile.TemporaryDirectory(prefix="gradgen-python-iface-runner-") as tmpdir:
        venv_dir = Path(tmpdir) / "venv"
        venv.EnvBuilder(with_pip=True).create(venv_dir)
        python_bin = _venv_python(venv_dir)

        subprocess.run(
            [str(python_bin), "-m", "pip", "install", "-e", str(crate_dir)],
            check=True,
            capture_output=True,
            text=True,
        )

        snippet = """
import foo

workspace = foo.workspace_for_function("energy")
result = foo.call("energy", [1.0, 2.0], [3.0], workspace)

print("workspace_for_function('energy') =", workspace)
print("call('energy', [1.0, 2.0], [3.0], workspace) =", result)
"""
        subprocess.run([str(python_bin), "-c", snippet], check=True, text=True)


if __name__ == "__main__":
    main()
