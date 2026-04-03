import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
import sys


class PythonInterfaceDemoTests(unittest.TestCase):
    @staticmethod
    def _run_python(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, *args],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
        )

    def test_demo_generates_python_enabled_crate(self) -> None:
        with TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "foo"
            completed = self._run_python(
                "demos/python_interface/main.py",
                "--output-dir",
                str(output_dir),
            )

            self.assertIn("Generated Rust crate:", completed.stdout)
            self.assertTrue((output_dir / "Cargo.toml").is_file())
            self.assertTrue((output_dir / "src" / "lib.rs").is_file())
            self.assertFalse((output_dir / "pyproject.toml").exists())

            wrapper_dir = Path(tmpdir) / "foo_python"
            self.assertTrue((wrapper_dir / "Cargo.toml").is_file())
            self.assertTrue((wrapper_dir / "pyproject.toml").is_file())
            self.assertTrue((wrapper_dir / "src" / "lib.rs").is_file())

    def test_demo_runner_builds_and_calls_python_extension(self) -> None:
        with TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "foo"
            self._run_python(
                "demos/python_interface/main.py",
                "--output-dir",
                str(output_dir),
            )

            completed = self._run_python(
                "demos/python_interface/runner/main.py",
                "--wrapper-dir",
                str(Path(tmpdir) / "foo_python"),
            )

            self.assertIn("workspace_for_function(", completed.stdout)
            self.assertIn("call(", completed.stdout)
            self.assertIn("'cost':", completed.stdout)
            self.assertIn("'state':", completed.stdout)
