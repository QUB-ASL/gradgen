"""Profile the generated single-shooting Rust kernel."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

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
    """Parse profiling options."""
    parser = argparse.ArgumentParser(
        description=(
            "Profile the generated Rust kernel for the single-shooting "
            "bicycle problem."
        ),
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=100,
        help="Prediction horizon length used in the generated kernel.",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=1000000,
        help="Number of benchmark repetitions per kernel.",
    )
    parser.add_argument(
        "--gradgen-scalar-type",
        choices=("f64", "f32"),
        default="f64",
        help="Scalar type to use for the generated Gradgen Rust code.",
    )
    args = parser.parse_args()
    if args.horizon <= 0:
        parser.error("--horizon must be a positive integer")
    if args.num_runs <= 0:
        parser.error("--num-runs must be positive")
    return args


def count_nonempty_lines(source: str) -> int:
    """Return the number of non-empty lines in ``source``."""
    return sum(1 for line in source.splitlines() if line.strip())


def render_template(template_name: str, **context: object) -> str:
    """Render a Jinja2 template from the profiling runners directory."""
    return TEMPLATE_ENV.get_template(template_name).render(**context)


def build_bicycle_problem(horizon: int) -> SingleShootingProblem:
    """Build the bicycle-model single-shooting problem for ``horizon``."""
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
        SingleShootingProblem("bicycle_profile")
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
    """Return the stage tracking cost used in the profile."""
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
    """Return the terminal tracking cost used in the profile."""
    terminal_weights = (20.0, 20.0, 4.0, 2.0)
    cost = SX.const(0.0)
    for index, weight in enumerate(terminal_weights):
        error = x[index] - p[index]
        cost = cost + weight * error * error
    return cost


def build_gradgen_project(horizon: int, project_dir: Path, scalar_type: str):
    """Build the Gradgen project for ``horizon``."""
    problem = build_bicycle_problem(horizon)
    backend_config = (
        RustBackendConfig()
        .with_crate_name(f"bicycle_profile_{horizon}")
        .with_backend_mode("std")
        .with_scalar_type(scalar_type)
        .with_build_crate(False)
    )
    builder = (
        CodeGenerationBuilder(problem)
        .with_backend_config(backend_config)
        .add_primal(include_states=True)
        .add_gradient(include_states=True)
        .add_joint(
            SingleShootingBundle()
            .add_cost()
            .add_gradient()
            .add_rollout_states()
        )
    )
    return builder.build(project_dir)


def _select_codegen(codegens, suffix: str):
    """Select a generated function by suffix."""
    for codegen in codegens:
        if codegen.function_name.endswith(suffix):
            return codegen
    raise ValueError(f"unable to find codegen ending in {suffix!r}")


def _render_profile_runner(
    *,
    generated_crate_path: str,
    generated_crate_name: str,
    runner_dir: Path,
    horizon: int,
    timing_runs: int,
    scalar_type: str,
    primal_codegen,
    gradient_codegen,
    joint_codegen,
) -> None:
    """Render the native Rust profiling runner into ``runner_dir``."""
    src_dir = runner_dir / "src"
    src_dir.mkdir(parents=True, exist_ok=True)
    cargo_dir = runner_dir / ".cargo"
    cargo_dir.mkdir(parents=True, exist_ok=True)
    (runner_dir / "Cargo.toml").write_text(
        render_template(
            "profile_runner/Cargo.toml.j2",
            package_name="profile_runner",
            generated_crate_name=generated_crate_name,
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
            "profile_runner/src/main.rs.j2",
            generated_crate_name=generated_crate_name,
            primal_function_name=primal_codegen.function_name,
            gradient_function_name=gradient_codegen.function_name,
            joint_function_name=joint_codegen.function_name,
            horizon=horizon,
            timing_runs=timing_runs,
            scalar_type=scalar_type,
            primal_cost_output_size=primal_codegen.output_sizes[0],
            primal_state_output_size=primal_codegen.output_sizes[1],
            gradient_output_size=gradient_codegen.output_sizes[0],
            gradient_state_output_size=gradient_codegen.output_sizes[1],
            joint_cost_output_size=joint_codegen.output_sizes[0],
            joint_gradient_output_size=joint_codegen.output_sizes[1],
            joint_state_output_size=joint_codegen.output_sizes[2],
            primal_workspace_size=primal_codegen.workspace_size,
            gradient_workspace_size=gradient_codegen.workspace_size,
            joint_workspace_size=joint_codegen.workspace_size,
        ),
        encoding="utf-8",
    )


def _benchmark_profile(
    horizon: int,
    num_runs: int,
    scalar_type: str,
) -> tuple[dict[str, float], dict[str, int]]:
    """Build and run the profiling benchmark."""
    with TemporaryDirectory() as tmpdir:
        temp_dir = Path(tmpdir)
        project = build_gradgen_project(
            horizon,
            temp_dir / "gradgen_bicycle",
            scalar_type,
        )
        lib_rs = project.project_dir / "src" / "lib.rs"
        loc = count_nonempty_lines(lib_rs.read_text(encoding="utf-8"))
        primal_codegen = _select_codegen(project.codegens, "f_states")
        gradient_codegen = _select_codegen(
            project.codegens,
            "grad_states_u_seq",
        )
        joint_codegen = _select_codegen(
            project.codegens,
            "f_grad_states_u_seq",
        )

        runner_dir = temp_dir / "profile_runner"
        _render_profile_runner(
            generated_crate_path=(
                f"{project.project_dir.parent.name}/{project.project_dir.name}"
            ),
            generated_crate_name=project.project_dir.name,
            runner_dir=runner_dir,
            horizon=horizon,
            timing_runs=num_runs,
            scalar_type=scalar_type,
            primal_codegen=primal_codegen,
            gradient_codegen=gradient_codegen,
            joint_codegen=joint_codegen,
        )
        completed = subprocess.run(
            ["cargo", "run", "--release", "--quiet"],
            cwd=runner_dir,
            env={**os.environ, "CARGO_TARGET_DIR": str(temp_dir / "target")},
            check=True,
            capture_output=True,
            text=True,
        )
        timings: dict[str, float] = {}
        for line in completed.stdout.splitlines():
            parts = line.split()
            if len(parts) == 2:
                timings[parts[0]] = float(parts[1])
        return timings, {"loc": loc}


def main() -> None:
    """Run the profiling benchmark and print a compact report."""
    args = parse_args()
    timings, stats = _benchmark_profile(
        args.horizon,
        args.num_runs,
        args.gradgen_scalar_type,
    )

    state_copy_us = timings["joint_state_copy_us"]
    gradient_copy_us = timings["joint_gradient_copy_us"]
    copy_total_us = state_copy_us + gradient_copy_us
    joint_core_us = timings["joint_total_us"] - copy_total_us
    primal_core_us = timings["primal_total_us"] - state_copy_us
    gradient_core_us = timings["gradient_total_us"] - gradient_copy_us
    separate_total_us = timings["primal_total_us"] + timings[
        "gradient_total_us"
    ]
    separate_core_us = separate_total_us - copy_total_us
    helper_gap_us = separate_core_us - joint_core_us

    print(
        "Single-shooting Rust profiling "
        "(average runtime in microseconds)"
    )
    print(f"Horizon: {args.horizon}")
    print(f"Timing runs: {args.num_runs}")
    print(f"Gradgen scalar type: {args.gradgen_scalar_type}")
    print(f"Gradgen LOC: {stats['loc']}")
    print("Phase | Total us | Core us")
    print("------|----------|--------")
    print(
        f"joint | {timings['joint_total_us']:8.3f} | {joint_core_us:6.3f}"
    )
    print(
        f"primal | {timings['primal_total_us']:7.3f} | "
        f"{primal_core_us:6.3f}"
    )
    print(
        f"gradient | {timings['gradient_total_us']:8.3f} | "
        f"{gradient_core_us:6.3f}"
    )
    print(f"copy | {copy_total_us:9.3f} |    n/a")
    print(f"separate total | {separate_total_us:7.3f} |    n/a")
    print(f"separate core | {separate_core_us:8.3f} |    n/a")
    print(f"helper gap | {helper_gap_us:10.3f} |    n/a")


if __name__ == "__main__":
    main()
