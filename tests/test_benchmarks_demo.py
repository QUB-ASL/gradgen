"""Smoke tests for the benchmark demos."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import unittest


class BenchmarkDemoTests(unittest.TestCase):
    """Smoke-test the single-shooting benchmark demo."""

    def test_single_shooting_benchmark_prints_runtime_columns(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script = (
            repo_root
            / "demos"
            / "benchmarks"
            / "single_shooting"
            / "main.py"
        )
        env = os.environ.copy()
        env["PYTHONPATH"] = "src"
        completed = subprocess.run(
            [
                sys.executable,
                str(script),
                "--start-horizon",
                "10",
                "--max-horizon",
                "10",
                "--step",
                "10",
                "--num-runs",
                "1",
            ],
            cwd=repo_root,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("Gradgen (us)", completed.stdout)
        self.assertIn("CasADi (us)", completed.stdout)
        self.assertIn("10 |", completed.stdout)

    def test_single_shooting_benchmark_supports_gradgen_f32(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script = (
            repo_root
            / "demos"
            / "benchmarks"
            / "single_shooting"
            / "main.py"
        )
        env = os.environ.copy()
        env["PYTHONPATH"] = "src"
        completed = subprocess.run(
            [
                sys.executable,
                str(script),
                "--start-horizon",
                "10",
                "--max-horizon",
                "10",
                "--step",
                "10",
                "--num-runs",
                "1",
                "--gradgen-scalar-type",
                "f32",
            ],
            cwd=repo_root,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("Gradgen scalar type: f32", completed.stdout)
        self.assertIn("Gradgen (us) [f32]", completed.stdout)
        self.assertIn("10 |", completed.stdout)

    def test_single_shooting_benchmark_supports_flattened_gradgen(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script = (
            repo_root
            / "demos"
            / "benchmarks"
            / "single_shooting"
            / "main.py"
        )
        env = os.environ.copy()
        env["PYTHONPATH"] = "src"
        completed = subprocess.run(
            [
                sys.executable,
                str(script),
                "--start-horizon",
                "10",
                "--max-horizon",
                "10",
                "--step",
                "10",
                "--num-runs",
                "1",
                "--flatten",
                "true",
            ],
            cwd=repo_root,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("Gradgen lowering: flattened to_function()", completed.stdout)
        self.assertIn("Gradgen (us)", completed.stdout)
        self.assertIn("10 |", completed.stdout)
