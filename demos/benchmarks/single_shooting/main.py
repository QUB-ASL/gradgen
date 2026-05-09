"""Benchmark Gradgen against CasADi on a bicycle-model single-shooting OCP."""

from __future__ import annotations

import argparse
import json
from contextlib import contextmanager
import os
from pathlib import Path
import sys
import subprocess
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
    FunctionBundle,
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


def _parse_bool(value: str) -> bool:
    """Parse a human-friendly boolean command-line value."""
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(
        "expected one of: true, false, yes, no, on, off, 1, 0"
    )


def parse_args() -> argparse.Namespace:
    """Parse the benchmark horizon range."""
    parser = argparse.ArgumentParser(
        description=(
            "Compare Gradgen and CasADi code size and runtime for a "
            "bicycle-model single-shooting problem."
        ),
    )
    parser.add_argument(
        "--start-horizon",
        type=int,
        default=40,
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
        default=20,
        help="Spacing between benchmark horizons.",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=1000,
        help="Number of benchmark repetitions per horizon.",
    )
    parser.add_argument(
        "--flatten",
        nargs="?",
        const=True,
        default=False,
        type=_parse_bool,
        help=(
            "Flatten the Gradgen path through to_function() before "
            "code generation."
        ),
    )
    parser.add_argument(
        "--gradgen-scalar-type",
        choices=("f64", "f32"),
        default="f64",
        help="Scalar type to use for the generated Gradgen Rust code.",
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


def build_casadi_function(
    horizon: int,
) -> ca.Function:
    """Build the CasADi joint cost-and-gradient function for ``horizon``."""
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
    return ca.Function(
        "bicycle_total_cost_joint",
        [x0, u_seq, p],
        [total_cost, ca.gradient(total_cost, u_seq)],
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


def _build_gradgen_project(
    horizon: int,
    project_dir: Path,
    *,
    scalar_type: str = "f64",
    flatten: bool = False,
):
    """Build the Gradgen project for ``horizon``."""
    problem = build_bicycle_problem(horizon)
    crate_suffix = "flat" if flatten else "staged"
    backend_config = (
        RustBackendConfig()
        .with_crate_name(f"bicycle_benchmark_{horizon}_{crate_suffix}")
        .with_backend_mode("std")
        .with_scalar_type(scalar_type)
        .with_build_crate(False)
    )
    if flatten:
        builder = (
            CodeGenerationBuilder(problem.to_function())
            .with_backend_config(backend_config)
            .add_joint(FunctionBundle().add_f().add_gradient(wrt="u_seq"))
        )
    else:
        builder = (
            CodeGenerationBuilder(problem)
            .with_backend_config(backend_config)
            .add_joint(SingleShootingBundle().add_cost().add_gradient())
        )
    return builder.build(project_dir)


def _build_casadi_source(
    horizon: int,
    directory: Path,
) -> Path:
    """Generate the CasADi C source file for ``horizon``."""
    joint_function = build_casadi_function(horizon)
    with _pushd(directory):
        generator = ca.CodeGenerator("bicycle_benchmark")
        generator.add(joint_function)
        generator.generate()
    return directory / "bicycle_benchmark.c"


def _render_gradgen_runner(
    *,
    generated_crate_path: str,
    runner_dir: Path,
    horizon: int,
    timing_runs: int,
    scalar_type: str,
    metadata: dict[str, object],
) -> None:
    """Render the native Rust benchmark runner into ``runner_dir``."""
    functions = metadata["functions"]
    assert isinstance(functions, list)
    assert len(functions) == 1
    joint_metadata = functions[0]
    assert isinstance(joint_metadata, dict)

    src_dir = runner_dir / "src"
    src_dir.mkdir(parents=True, exist_ok=True)
    cargo_dir = runner_dir / ".cargo"
    cargo_dir.mkdir(parents=True, exist_ok=True)
    (runner_dir / "Cargo.toml").write_text(
        render_template(
            "benchmark_runner/Cargo.toml.j2",
            package_name="benchmark_runner",
            generated_crate_name=metadata["crate_name"],
            generated_crate_path=generated_crate_path,
        ),
        encoding="utf-8",
    )
    (cargo_dir / "config.toml").write_text(
        render_template("benchmark_runner/.cargo/config.toml.j2"),
        encoding="utf-8",
    )
    (src_dir / "main.rs").write_text(
        render_template(
            "benchmark_runner/src/main.rs.j2",
            generated_crate_name=metadata["crate_name"],
            function_name=joint_metadata["function_name"],
            horizon=horizon,
            timing_runs=timing_runs,
            scalar_type=scalar_type,
            cost_output_size=joint_metadata["output_sizes"][0],
            gradient_output_size=joint_metadata["output_sizes"][1],
            workspace_size=max(joint_metadata["workspace_size"], 1),
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
            function_name="bicycle_total_cost_joint",
            horizon=horizon,
            timing_runs=timing_runs,
            control_length=2 * horizon,
            cost_output_size=1,
            gradient_output_size=2 * horizon,
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
    scalar_type: str,
    flatten: bool,
    cargo_target_dir_root: Path,
) -> tuple[int, float]:
    """Return Gradgen LOC and runtime for ``horizon``."""
    with TemporaryDirectory() as tmpdir:
        temp_dir = Path(tmpdir)
        cargo_target_dir = cargo_target_dir_root / (
            f"{'flattened' if flatten else 'staged'}_{scalar_type}"
        )
        cargo_target_dir.mkdir(parents=True, exist_ok=True)
        project = _build_gradgen_project(
            horizon,
            temp_dir / "gradgen_bicycle",
            scalar_type=scalar_type,
            flatten=flatten,
        )
        lib_rs = project.project_dir / "src" / "lib.rs"
        loc = count_nonempty_lines(lib_rs.read_text(encoding="utf-8"))
        metadata = json.loads(project.metadata_json.read_text(encoding="utf-8"))
        runner_dir = temp_dir / "benchmark_runner"
        _render_gradgen_runner(
            generated_crate_path=(
                f"{project.project_dir.parent.name}/{project.project_dir.name}"
            ),
            runner_dir=runner_dir,
            horizon=horizon,
            timing_runs=num_runs,
            scalar_type=scalar_type,
            metadata=metadata,
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
    with TemporaryDirectory() as cargo_target_root:
        cargo_target_dir_root = Path(cargo_target_root)

        print(
            "Single-shooting bicycle benchmark "
            "(non-empty lines in generated source and runtime in us)"
        )
        print(f"Timing runs per horizon: {args.num_runs}")
        print(f"Gradgen scalar type: {args.gradgen_scalar_type}")
        gradgen_lowering = (
            "flattened to_function()"
            if args.flatten
            else "staged SingleShootingProblem"
        )
        print(f"Gradgen lowering: {gradgen_lowering}")
        print(
            "Runtime measures one joint cost-plus-gradient loop divided by "
            "--num-runs."
        )
        print(
            "N   | Gradgen LOC | CasADi LOC  | "
            f"Gradgen (us) [{args.gradgen_scalar_type}] | "
            "CasADi (us)"
        )
        print(
            "----|-------------|-------------|--------------------|"
            "-----------"
        )
        for horizon in horizons:
            gradgen_lines, gradgen_mean_ms = _benchmark_gradgen(
                horizon,
                args.num_runs,
                args.gradgen_scalar_type,
                args.flatten,
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
