"""Benchmark Gradgen against CasADi on a bicycle-model single-shooting OCP."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import casadi as ca

# Allow running this file directly from inside the demo directory.
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from gradgen import (  # noqa: E402
    CodeGenerationBuilder,
    Function,
    RustBackendConfig,
    SX,
    SXVector,
    SingleShootingProblem,
)


@contextmanager
def _pushd(path: Path):
    """Temporarily change the process working directory."""
    old_cwd = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old_cwd)


def parse_args() -> argparse.Namespace:
    """Parse the benchmark horizon range."""
    parser = argparse.ArgumentParser(
        description=(
            "Compare Gradgen and CasADi code size for a bicycle-model "
            "single-shooting problem."
        ),
    )
    parser.add_argument(
        "--start-horizon",
        type=int,
        default=25,
        help="First horizon length to benchmark.",
    )
    parser.add_argument(
        "--max-horizon",
        type=int,
        default=200,
        help="Largest horizon length to benchmark.",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=25,
        help="Spacing between benchmark horizons.",
    )
    args = parser.parse_args()
    if args.start_horizon <= 0:
        parser.error("--start-horizon must be positive")
    if args.max_horizon < args.start_horizon:
        parser.error("--max-horizon must be at least --start-horizon")
    if args.step <= 0:
        parser.error("--step must be positive")
    return args


def count_nonempty_lines(source: str) -> int:
    """Return the number of non-empty lines in ``source``."""
    return sum(1 for line in source.splitlines() if line.strip())


def build_bicycle_problem(horizon: int) -> SingleShootingProblem:
    """Build a bicycle-model single-shooting problem for ``horizon``."""
    x = SXVector.sym("x", 4)
    u = SXVector.sym("u", 2)
    p = SXVector.sym("p", 4)

    dt = 0.1
    wheelbase = 2.7

    dynamics = Function(
        "bicycle_dynamics",
        [x, u, p],
        [
            SXVector(
                (
                    x[0] + dt * x[3] * x[2].cos(),
                    x[1] + dt * x[3] * x[2].sin(),
                    x[2] + dt * x[3] * u[1].tan() / wheelbase,
                    x[3] + dt * u[0],
                )
            )
        ],
        input_names=["x", "u", "p"],
        output_names=["x_next"],
    )

    stage_cost = Function(
        "bicycle_stage_cost",
        [x, u, p],
        [_tracking_stage_cost(x, u, p)],
        input_names=["x", "u", "p"],
        output_names=["ell"],
    )

    terminal_cost = Function(
        "bicycle_terminal_cost",
        [x, p],
        [_tracking_terminal_cost(x, p)],
        input_names=["x", "p"],
        output_names=["vf"],
    )

    return (
        SingleShootingProblem("bicycle_benchmark")
        .with_horizon(horizon)
        .with_dynamics(dynamics)
        .with_costs(stage_cost, terminal_cost)
        .with_input_names(
            initial_state_name="x0",
            control_sequence_name="u_seq",
            parameter_name="p",
        )
        .with_simplification("medium")
    )


def _tracking_stage_cost(x: SXVector, u: SXVector, p: SXVector) -> SX:
    """Return the stage tracking cost used in the benchmark."""
    state_weights = (10.0, 10.0, 2.0, 1.0)
    control_weights = (0.2, 0.1)
    cost = SX.const(0.0)
    for index, weight in enumerate(state_weights):
        error = x[index] - p[index]
        cost = cost + weight * error * error
    for index, weight in enumerate(control_weights):
        cost = cost + weight * u[index] * u[index]
    return cost


def _tracking_terminal_cost(x: SXVector, p: SXVector) -> SX:
    """Return the terminal tracking cost used in the benchmark."""
    terminal_weights = (20.0, 20.0, 4.0, 2.0)
    cost = SX.const(0.0)
    for index, weight in enumerate(terminal_weights):
        error = x[index] - p[index]
        cost = cost + weight * error * error
    return cost


def gradgen_loc(horizon: int) -> int:
    """Return the non-empty line count for the generated Gradgen crate."""
    problem = build_bicycle_problem(horizon)
    builder = (
        CodeGenerationBuilder(problem)
        .with_backend_config(
            RustBackendConfig()
            .with_crate_name(f"bicycle_benchmark_{horizon}")
            .with_backend_mode("no_std")
            .with_scalar_type("f64")
        )
        .add_primal()
        .add_gradient()
    )

    with TemporaryDirectory() as tmpdir:
        project = builder.build(Path(tmpdir) / "gradgen_bicycle")
        lib_rs = project.project_dir / "src" / "lib.rs"
        return count_nonempty_lines(lib_rs.read_text(encoding="utf-8"))


def casadi_loc(horizon: int) -> int:
    """Return the non-empty line count for the generated CasADi C code."""
    x0 = ca.SX.sym("x0", 4)
    u_seq = ca.SX.sym("u_seq", 2 * horizon)
    p = ca.SX.sym("p", 4)

    dt = 0.1
    wheelbase = 2.7
    x = x0
    total_cost = ca.SX(0.0)

    for stage in range(horizon):
        u = ca.vertcat(
            u_seq[2 * stage],
            u_seq[2 * stage + 1],
        )
        total_cost += _casadi_stage_cost(x, u, p)
        x = ca.vertcat(
            x[0] + dt * x[3] * ca.cos(x[2]),
            x[1] + dt * x[3] * ca.sin(x[2]),
            x[2] + dt * x[3] * ca.tan(u[1]) / wheelbase,
            x[3] + dt * u[0],
        )

    total_cost += _casadi_terminal_cost(x, p)
    cost_function = ca.Function(
        "bicycle_total_cost",
        [x0, u_seq, p],
        [total_cost],
    )
    gradient_function = ca.Function(
        "bicycle_total_cost_grad_u",
        [x0, u_seq, p],
        [ca.gradient(total_cost, u_seq)],
    )

    with TemporaryDirectory() as tmpdir:
        temp_dir = Path(tmpdir)
        with _pushd(temp_dir):
            generator = ca.CodeGenerator("bicycle_benchmark")
            generator.add(cost_function)
            generator.add(gradient_function)
            generator.generate()
        source_path = temp_dir / "bicycle_benchmark.c"
        return count_nonempty_lines(source_path.read_text(encoding="utf-8"))


def _casadi_stage_cost(x: ca.SX, u: ca.SX, p: ca.SX) -> ca.SX:
    """Return the CasADi stage cost used in the benchmark."""
    state_weights = (10.0, 10.0, 2.0, 1.0)
    control_weights = (0.2, 0.1)
    cost = ca.SX(0.0)
    for index, weight in enumerate(state_weights):
        error = x[index] - p[index]
        cost += weight * error * error
    for index, weight in enumerate(control_weights):
        cost += weight * u[index] * u[index]
    return cost


def _casadi_terminal_cost(x: ca.SX, p: ca.SX) -> ca.SX:
    """Return the CasADi terminal cost used in the benchmark."""
    terminal_weights = (20.0, 20.0, 4.0, 2.0)
    cost = ca.SX(0.0)
    for index, weight in enumerate(terminal_weights):
        error = x[index] - p[index]
        cost += weight * error * error
    return cost


def main() -> None:
    """Run the benchmark and print a compact comparison table."""
    args = parse_args()
    horizons = range(args.start_horizon, args.max_horizon + 1, args.step)

    print(
        "Single-shooting bicycle benchmark "
        "(non-empty lines in generated source)"
    )
    print("N | Gradgen LOC | CasADi LOC | Ratio")
    print("--|-------------|------------|-----------")
    for horizon in horizons:
        gradgen_lines = gradgen_loc(horizon)
        casadi_lines = casadi_loc(horizon)
        x = casadi_lines/gradgen_lines
        print(
            f"{horizon:2d} | {gradgen_lines:11d} | "
            f"{casadi_lines:10d} | {x:+.1f}"
        )


if __name__ == "__main__":
    main()
