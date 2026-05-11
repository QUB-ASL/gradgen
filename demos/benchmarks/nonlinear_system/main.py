"""Benchmark Gradgen against CasADi on a nonlinear single-shooting OCP."""

from __future__ import annotations

import argparse
import os
from contextlib import contextmanager
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

import casadi as ca
from jinja2 import Environment, FileSystemLoader, StrictUndefined

# Allow running this file directly from inside the demo directory.
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))
RUNNERS_DIR = Path(__file__).resolve().parent / "runners"
TEMPLATE_ENV = Environment(
    loader=FileSystemLoader(RUNNERS_DIR),
    autoescape=False,
    trim_blocks=True,
    lstrip_blocks=True,
    undefined=StrictUndefined,
)

from gradgen import (  # noqa: E402
    CodeGenerationBuilder,
    Function,
    RustBackendConfig,
    SX,
    SXVector,
    SingleShootingBundle,
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
            "Compare Gradgen and CasADi code size and runtime for a "
            "larger nonlinear single-shooting problem."
        ),
    )
    parser.add_argument(
        "--start-horizon",
        type=int,
        default=10,
        help="First horizon length to benchmark.",
    )
    parser.add_argument(
        "--max-horizon",
        type=int,
        default=100,
        help="Largest horizon length to benchmark.",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=10,
        help="Spacing between benchmark horizons.",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=1000,
        help="Number of benchmark repetitions per horizon.",
    )
    args = parser.parse_args()
    if args.start_horizon <= 0:
        parser.error("--start-horizon must be positive")
    if args.max_horizon < args.start_horizon:
        parser.error("--max-horizon must be at least --start-horizon")
    if args.step <= 0:
        parser.error("--step must be positive")
    if args.num_runs <= 0:
        parser.error("--num-runs must be positive")
    return args


def count_nonempty_lines(source: str) -> int:
    """Return the number of non-empty lines in ``source``."""
    return sum(1 for line in source.splitlines() if line.strip())


def format_us(mean_ms: float) -> str:
    """Format a timing summary in microseconds."""
    return f"{1000 * mean_ms:.2f}"


def render_template(template_name: str, **context: object) -> str:
    """Render a Jinja2 template from the benchmark runners directory."""
    return TEMPLATE_ENV.get_template(template_name).render(**context)


def build_nonlinear_problem(horizon: int) -> SingleShootingProblem:
    """Build the nonlinear single-shooting problem for ``horizon``."""
    x = SXVector.sym("x", 10)
    u = SXVector.sym("u", 5)
    p = SXVector.sym("p", 10)

    dt = 0.05

    dynamics = Function(
        "nonlinear_dynamics",
        [x, u, p],
        [
            SXVector(
                (
                    x[0]
                    + dt * (0.8 * x[1] + x[2].sin() + u[0] + 0.05 * p[0]),
                    x[1]
                    + dt * (x[2] * x[3] + x[4].cos() + u[1] + 0.05 * p[1]),
                    x[2]
                    + dt * (x[0] * x[1] - 0.2 * x[5] + u[2].tanh()),
                    x[3]
                    + dt * (x[2] * x[4] + u[3] + 0.05 * p[3]),
                    x[4]
                    + dt * (x[0].sin() + x[6] * u[4] + 0.05 * p[4]),
                    x[5]
                    + dt * (x[4] * x[7] - 0.1 * x[8] + u[0] * u[1]),
                    x[6]
                    + dt * (x[1].cos() + x[3] * x[9] + 0.5 * u[2] * u[3]),
                    x[7]
                    + dt * (x[5].sin() + x[6] * x[8] + 0.25 * u[4] * u[4]),
                    x[8]
                    + dt * (x[7] * x[9] + u[0].cos() + 0.3 * u[1] * u[2]),
                    x[9]
                    + dt * (x[0] * x[8] - u[3].sin() + 0.1 * u[4]),
                )
            )
        ],
        input_names=["x", "u", "p"],
        output_names=["x_next"],
    )

    stage_cost = Function(
        "nonlinear_stage_cost",
        [x, u, p],
        [_tracking_stage_cost(x, u, p)],
        input_names=["x", "u", "p"],
        output_names=["ell"],
    )

    terminal_cost = Function(
        "nonlinear_terminal_cost",
        [x, p],
        [_tracking_terminal_cost(x, p)],
        input_names=["x", "p"],
        output_names=["vf"],
    )

    return (
        SingleShootingProblem("nonlinear_benchmark")
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
    state_weights = (12.0, 12.0, 10.0, 10.0, 8.0, 8.0, 6.0, 6.0, 5.0, 5.0)
    control_weights = (0.25, 0.2, 0.15, 0.15, 0.1)
    cost = SX.const(0.0)
    for index, weight in enumerate(state_weights):
        error = x[index] - p[index]
        cost = cost + weight * error * error
    for index, weight in enumerate(control_weights):
        cost = cost + weight * u[index] * u[index]
    return cost


def _tracking_terminal_cost(x: SXVector, p: SXVector) -> SX:
    """Return the terminal tracking cost used in the benchmark."""
    terminal_weights = (
        24.0,
        24.0,
        20.0,
        20.0,
        16.0,
        16.0,
        12.0,
        12.0,
        10.0,
        10.0,
    )
    cost = SX.const(0.0)
    for index, weight in enumerate(terminal_weights):
        error = x[index] - p[index]
        cost = cost + weight * error * error
    return cost


def _build_gradgen_project(horizon: int, project_dir: Path) -> object:
    """Build the Gradgen project for ``horizon``."""
    problem = build_nonlinear_problem(horizon)
    backend_config = (
        RustBackendConfig()
        .with_crate_name(f"nonlinear_benchmark_{horizon}")
        .with_backend_mode("std")
        .with_build_crate(False)
    )
    builder = (
        CodeGenerationBuilder(problem)
        .with_backend_config(backend_config)
        .add_joint(SingleShootingBundle().add_cost().add_gradient())
    )
    return builder.build(project_dir)


def _build_casadi_function(horizon: int) -> ca.Function:
    """Build the CasADi joint cost-and-gradient function for ``horizon``."""
    x0 = ca.SX.sym("x0", 10)
    u_seq = ca.SX.sym("u_seq", 5 * horizon)
    p = ca.SX.sym("p", 10)

    dt = 0.05
    x = x0
    total_cost = ca.SX(0.0)

    for stage in range(horizon):
        u = ca.vertcat(
            u_seq[5 * stage],
            u_seq[5 * stage + 1],
            u_seq[5 * stage + 2],
            u_seq[5 * stage + 3],
            u_seq[5 * stage + 4],
        )
        total_cost += _casadi_stage_cost(x, u, p)
        x = ca.vertcat(
            x[0] + dt * (0.8 * x[1] + ca.sin(x[2]) + u[0] + 0.05 * p[0]),
            x[1] + dt * (x[2] * x[3] + ca.cos(x[4]) + u[1] + 0.05 * p[1]),
            x[2] + dt * (x[0] * x[1] - 0.2 * x[5] + ca.tanh(u[2])),
            x[3] + dt * (x[2] * x[4] + u[3] + 0.05 * p[3]),
            x[4] + dt * (ca.sin(x[0]) + x[6] * u[4] + 0.05 * p[4]),
            x[5] + dt * (x[4] * x[7] - 0.1 * x[8] + u[0] * u[1]),
            x[6] + dt * (ca.cos(x[1]) + x[3] * x[9] + 0.5 * u[2] * u[3]),
            x[7] + dt * (ca.sin(x[5]) + x[6] * x[8] + 0.25 * u[4] * u[4]),
            x[8] + dt * (x[7] * x[9] + ca.cos(u[0]) + 0.3 * u[1] * u[2]),
            x[9] + dt * (x[0] * x[8] - ca.sin(u[3]) + 0.1 * u[4]),
        )

    total_cost += _casadi_terminal_cost(x, p)
    return ca.Function(
        "nonlinear_total_cost_joint",
        [x0, u_seq, p],
        [total_cost, ca.gradient(total_cost, u_seq)],
    )


def _build_gradgen_runner(
    *,
    generated_crate_path: str,
    runner_dir: Path,
    horizon: int,
    timing_runs: int,
    project,
) -> None:
    """Render the native Rust benchmark runner into ``runner_dir``."""
    src_dir = runner_dir / "src"
    src_dir.mkdir(parents=True, exist_ok=True)
    cargo_dir = runner_dir / ".cargo"
    cargo_dir.mkdir(parents=True, exist_ok=True)
    joint_codegen = project.codegens[0]
    (runner_dir / "Cargo.toml").write_text(
        render_template(
            "benchmark_runner/Cargo.toml.j2",
            package_name="benchmark_runner",
            generated_crate_name=project.project_dir.name,
            generated_crate_path=generated_crate_path,
        ),
        encoding="utf-8",
    )
    (cargo_dir / "config.toml").write_text(
        "[build]\n"
        'rustflags = ["-C", "target-cpu=native"]\n',
        encoding="utf-8",
    )
    (src_dir / "main.rs").write_text(
        render_template(
            "benchmark_runner/src/main.rs.j2",
            generated_crate_name=project.project_dir.name,
            function_name=joint_codegen.function_name,
            horizon=horizon,
            timing_runs=timing_runs,
            scalar_type="f64",
            cost_output_size=joint_codegen.output_sizes[0],
            gradient_output_size=joint_codegen.output_sizes[1],
            workspace_size=joint_codegen.workspace_size,
            state_size=10,
            control_size=5,
            parameter_size=10,
            x0_values=(1.0, -0.5, 0.2, 2.0, -0.3, 0.4, 0.1, -0.2, 0.5, 0.7),
            control_stage_values=(0.05, -0.02, 0.03, -0.01, 0.04),
            p_values=(0.2, -0.1, 0.05, 1.8, 0.15, -0.25, 0.4, 0.1, -0.3, 0.2),
        ),
        encoding="utf-8",
    )


def _render_c_runner(
    *,
    runner_path: Path,
    horizon: int,
    timing_runs: int,
) -> None:
    """Render the native CasADi benchmark runner."""
    runner_path.write_text(
        render_template(
            "benchmark_runner.c.j2",
            function_name="nonlinear_total_cost_joint",
            horizon=horizon,
            timing_runs=timing_runs,
            state_size=10,
            control_size=5,
            parameter_size=10,
            control_length=5 * horizon,
            cost_output_size=1,
            gradient_output_size=5 * horizon,
            x0_values=(1.0, -0.5, 0.2, 2.0, -0.3, 0.4, 0.1, -0.2, 0.5, 0.7),
            control_stage_values=(0.05, -0.02, 0.03, -0.01, 0.04),
            p_values=(0.2, -0.1, 0.05, 1.8, 0.15, -0.25, 0.4, 0.1, -0.3, 0.2),
        ),
        encoding="utf-8",
    )


def _compile_c_runner(
    runner_source: Path,
    casadi_source: Path,
    executable: Path,
) -> None:
    """Compile the native C benchmark runner."""
    subprocess.run(
        [
            "cc",
            "-O3",
            str(runner_source),
            str(casadi_source),
            "-lm",
            "-o",
            str(executable),
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def _benchmark_gradgen(
    horizon: int,
    num_runs: int,
    cargo_target_dir_root: Path,
) -> tuple[int, float]:
    """Return Gradgen LOC and runtime for ``horizon``."""
    with TemporaryDirectory() as tmpdir:
        temp_dir = Path(tmpdir)
        cargo_target_dir = cargo_target_dir_root / "staged_f64"
        cargo_target_dir.mkdir(parents=True, exist_ok=True)
        project = _build_gradgen_project(
            horizon,
            temp_dir / "gradgen_nonlinear",
        )
        lib_rs = project.project_dir / "src" / "lib.rs"
        loc = count_nonempty_lines(lib_rs.read_text(encoding="utf-8"))
        runner_dir = temp_dir / "benchmark_runner"
        _build_gradgen_runner(
            generated_crate_path=(
                f"{project.project_dir.parent.name}/{project.project_dir.name}"
            ),
            runner_dir=runner_dir,
            horizon=horizon,
            timing_runs=num_runs,
            project=project,
        )
        completed = subprocess.run(
            ["cargo", "run", "--release", "--quiet"],
            cwd=runner_dir,
            env={
                **os.environ,
                "CARGO_TARGET_DIR": str(cargo_target_dir),
            },
            check=True,
            capture_output=True,
            text=True,
        )
        mean_ms = float(completed.stdout.strip())
        return loc, mean_ms


def _benchmark_casadi(
    horizon: int,
    num_runs: int,
) -> tuple[int, float]:
    """Return CasADi LOC and runtime for ``horizon``."""
    with TemporaryDirectory() as tmpdir:
        temp_dir = Path(tmpdir)
        source_path = _build_casadi_source(horizon, temp_dir)
        loc = count_nonempty_lines(source_path.read_text(encoding="utf-8"))
        runner_path = temp_dir / "benchmark_runner.c"
        _render_c_runner(
            runner_path=runner_path,
            horizon=horizon,
            timing_runs=num_runs,
        )
        executable = temp_dir / "benchmark_runner"
        _compile_c_runner(runner_path, source_path, executable)
        completed = subprocess.run(
            [str(executable)],
            check=True,
            capture_output=True,
            text=True,
        )
        mean_ms = float(completed.stdout.strip())
        return loc, mean_ms


def _casadi_stage_cost(x: ca.SX, u: ca.SX, p: ca.SX) -> ca.SX:
    """Return the CasADi stage cost used in the benchmark."""
    state_weights = (12.0, 12.0, 10.0, 10.0, 8.0, 8.0, 6.0, 6.0, 5.0, 5.0)
    control_weights = (0.25, 0.2, 0.15, 0.15, 0.1)
    cost = ca.SX(0.0)
    for index, weight in enumerate(state_weights):
        error = x[index] - p[index]
        cost += weight * error * error
    for index, weight in enumerate(control_weights):
        cost += weight * u[index] * u[index]
    return cost


def _casadi_terminal_cost(x: ca.SX, p: ca.SX) -> ca.SX:
    """Return the CasADi terminal cost used in the benchmark."""
    terminal_weights = (
        24.0,
        24.0,
        20.0,
        20.0,
        16.0,
        16.0,
        12.0,
        12.0,
        10.0,
        10.0,
    )
    cost = ca.SX(0.0)
    for index, weight in enumerate(terminal_weights):
        error = x[index] - p[index]
        cost += weight * error * error
    return cost


def _build_casadi_source(
    horizon: int,
    directory: Path,
) -> Path:
    """Generate the CasADi C source file for ``horizon``."""
    joint_function = _build_casadi_function(horizon)
    with _pushd(directory):
        generator = ca.CodeGenerator("nonlinear_benchmark")
        generator.add(joint_function)
        generator.generate()
    return directory / "nonlinear_benchmark.c"


def main() -> None:
    """Run the benchmark and print a compact comparison table."""
    args = parse_args()
    horizons = range(args.start_horizon, args.max_horizon + 1, args.step)
    with TemporaryDirectory() as cargo_target_root:
        cargo_target_dir_root = Path(cargo_target_root)

        print(
            "Nonlinear single-shooting benchmark "
            "(non-empty lines in generated source and runtime in us)"
        )
        print(f"Timing runs per horizon: {args.num_runs}")
        print(
            "The system uses 10 states and 5 inputs with trigonometric and "
            "polynomial coupling."
        )
        print(
            "Runtime measures one joint cost-plus-gradient loop divided by "
            "--num-runs."
        )
        print(
            "N   | Gradgen LOC | CasADi LOC  | "
            "Gradgen (us) [f64] | CasADi (us)"
        )
        print(
            "----|-------------|-------------|--------------------|"
            "-----------"
        )
        for horizon in horizons:
            gradgen_lines, gradgen_mean_ms = _benchmark_gradgen(
                horizon,
                args.num_runs,
                cargo_target_dir_root,
            )
            casadi_lines, casadi_mean_ms = _benchmark_casadi(
                horizon, args.num_runs
            )
            print(
                f"{horizon:3d} | {gradgen_lines:11d} | {casadi_lines:11d} | "
                f"{format_us(gradgen_mean_ms):>18} | "
                f"{format_us(casadi_mean_ms):>10}"
            )


if __name__ == "__main__":
    main()
