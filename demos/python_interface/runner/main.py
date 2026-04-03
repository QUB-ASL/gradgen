"""Runner for the Python-interface demo crate."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import tempfile
import venv


def parse_args() -> argparse.Namespace:
    """Parse the generated Python wrapper crate directory used by the runner."""
    parser = argparse.ArgumentParser(
        description="Install the generated Python wrapper into an isolated venv and call it.",
    )
    parser.add_argument(
        "--wrapper-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "foo_python",
        help="Path to the generated Python wrapper crate.",
    )
    return parser.parse_args()


def _venv_python(venv_dir: Path) -> Path:
    if sys.platform == "win32":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def main() -> None:
    args = parse_args()
    wrapper_dir = args.wrapper_dir.resolve()
    if not wrapper_dir.is_dir():
        raise FileNotFoundError(f"generated wrapper directory not found: {wrapper_dir}")

    with tempfile.TemporaryDirectory(prefix="gradgen-python-iface-runner-") as tmpdir:
        venv_dir = Path(tmpdir) / "venv"
        venv.EnvBuilder(with_pip=True).create(venv_dir)
        python_bin = _venv_python(venv_dir)

        subprocess.run(
            [str(python_bin), "-m", "pip", "install", "-e", str(wrapper_dir)],
            check=True,
            capture_output=True,
            text=True,
        )

        snippet = """
import foo

print("all_functions() =", foo.all_functions())
print("function_info('energy') =", foo.function_info("energy"))
workspace = foo.workspace_for_function("energy")
result = foo.energy([1.0, 2.0], [3.0], workspace)

print("workspace =", workspace)
print("foo.energy([1.0, 2.0], [3.0], workspace) =", result)
print("workspace_for_function('energy') =", workspace)
"""
        subprocess.run([str(python_bin), "-c", snippet], check=True, text=True)


if __name__ == "__main__":
    main()
