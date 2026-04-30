"""Demo for single-shooting residual-penalty code generation."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Allow running this file directly from inside the demo directory.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from gradgen import (  # noqa: E402
    CodeGenerationBuilder,
    Function,
    RustBackendConfig,
    SX,
    SXVector,
    SingleShootingBundle,
    SingleShootingProblem,
    maximum,
)


def parse_args() -> argparse.Namespace:
    """Parse the user-configurable horizon length."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate a single-shooting OCP demo with vector residual "
            "penalties in the stage and terminal costs."
        ),
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=5,
        help="Prediction horizon length used in the generated kernel.",
    )
    args = parser.parse_args()
    if args.horizon <= 0:
        parser.error("--horizon must be a positive integer")
    return args


args = parse_args()


# The generated total cost is
#
#   sum_k ell(x_k, u_k, p) + c/2 * ||q(x_k, u_k, p)||_2^2
#   + V_f(x_N, p) + c/2 * ||q_N(x_N, p)||_2^2.
#
# The residual functions q and q_N are vector-valued, while c is a scalar
# runtime input supplied to the generated Rust kernels.
x = SXVector.sym("x", 2)
u = SXVector.sym("u", 1)
p = SXVector.sym("p", 2)
c = SX.sym("c")


dynamics = Function(
    "dynamics",
    [x, u, p],
    [
        SXVector(
            (
                x[0] + p[0] * x[1] + u[0],
                x[1] + p[1] * u[0] - 0.5 * x[0],
            )
        )
    ],
    input_names=["x", "u", "p"],
    output_names=["x_next"],
)


stage_cost = Function(
    "stage_cost",
    [x, u, p],
    [x[0] * x[0] + 0.5 * x[1] * x[1] + 0.1 * u[0] * u[0]],
    input_names=["x", "u", "p"],
    output_names=["ell"],
)


# q(x, u, p) can encode soft constraints, tracking residuals, or both. This
# example softly tracks x_0 + u toward p_0 and x_1 toward p_1.
stage_penalty = Function(
    "stage_penalty",
    [x, u, p],
    [SXVector((x[0] + u[0] - p[0], x[1] - p[1]))],
    input_names=["x", "u", "p"],
    output_names=["q"],
)


terminal_cost = Function(
    "terminal_cost",
    [x, p],
    [0.0 * x[0]],
    input_names=["x", "p"],
    output_names=["vf"],
)


# q_N(x_N, p) is a terminal constraint residual. The residual uses a hinge
# expression, but the total cost squares q_N, so max(0, z)^2 is differentiable.
terminal_penalty = Function(
    "terminal_penalty",
    [x, p],
    [maximum(0.0, x.norm2sq() - 1.0 - p[0])],
    input_names=["x", "p"],
    output_names=["q_n"],
)


problem = (
    SingleShootingProblem("penalized_mpc_cost")
    .with_horizon(args.horizon)
    .with_dynamics(dynamics)
    .with_costs(stage_cost, terminal_cost)
    .with_penalties(stage_penalty, terminal_penalty, c)
    .with_input_names(
        initial_state_name="x0",
        control_sequence_name="u_seq",
        parameter_name="p",
    )
)

pf = problem.to_function()
sample_x0 = [1.0, 2.0]
sample_u_seq = [0.1, 0.2, 0.3, 0.4, 0.5]
sample_p = [-0.1, 0.5]
sample_c = 1000.0

val = pf(x0=sample_x0, u_seq=sample_u_seq, p=sample_p, c=sample_c)
print(f"val = {val}")

backend_config = (
    RustBackendConfig()
    .with_backend_mode("no_std")
    .with_scalar_type("f64")
    .with_crate_name("single_shooting_penalty_kernel")
    .with_enable_python_interface()
)

print("Building residual-penalty single-shooting project...")
project = (
    CodeGenerationBuilder()
    .with_backend_config(backend_config)
    .for_function(problem)
    .add_primal(include_states=True)
    .add_gradient(include_states=True)
    .add_hvp(include_states=True)
    .add_joint(
        SingleShootingBundle()
        .add_cost()
        .add_gradient()
        .add_rollout_states()
    )
    .with_simplification("medium")
    .done()
    .build(Path(__file__).resolve().parent)
)
print("Building project done!")

print("Generated Rust crate:", project.project_dir)
print("Generated Rust functions:")
for codegen in project.codegens:
    print(" -", codegen.function_name)

if project.python_interface is not None:
    print("Generated Python wrapper crate:", project.python_interface.project_dir)

import single_shooting_penalty_kernel  # noqa: E402

workspace = single_shooting_penalty_kernel.workspace_for_function(
    "penalized_mpc_cost_f_grad_states_u_seq",
)
python_result = (
    single_shooting_penalty_kernel.penalized_mpc_cost_f_grad_states_u_seq(
        sample_x0,
        sample_u_seq,
        sample_p,
        sample_c,
        workspace,
    )
)
print("Python wrapper result:", python_result)
