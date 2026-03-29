"""Demo for deterministic single-shooting OCP code generation."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


# Allow running this file directly from inside the demo directory.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from gradgen import (
    CodeGenerationBuilder,
    Function,
    RustBackendConfig,
    SXVector,
    SingleShootingBundle,
    SingleShootingProblem,
)


def parse_args() -> argparse.Namespace:
    """Parse the user-configurable horizon length."""
    parser = argparse.ArgumentParser(
        description="Generate a loop-based deterministic single-shooting demo.",
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


# The dynamics, stage cost, and terminal cost all share the same one-stage
# symbolic signatures. The generated Rust code will expose x0, U, and p as the
# runtime inputs and will use a for loop over the fixed horizon.
x = SXVector.sym("x", 2)
u = SXVector.sym("u", 1)
p = SXVector.sym("p", 2)


# f(x, u, p) =
#   [x_1 + p_1 * x_2 + u,
#    x_2 + p_2 * u - 0.5 * x_1]
dynamics = Function(
    "dynamics",
    [x, u, p],
    [SXVector((x[0] + p[0] * x[1] + u[0], x[1] + p[1] * u[0] - 0.5 * x[0]))],
    input_names=["x", "u", "p"],
    output_names=["x_next"],
)


# ell(x, u, p) = x_1^2 + 2 x_2^2 + 0.3 u^2 + p_1 u
stage_cost = Function(
    "stage_cost",
    [x, u, p],
    [x[0] * x[0] + 2.0 * x[1] * x[1] + 0.3 * u[0] * u[0] + p[0] * u[0]],
    input_names=["x", "u", "p"],
    output_names=["ell"],
)


# Vf(x, p) = 3 x_1^2 + 0.5 x_2^2 + p_2 x_1
terminal_cost = Function(
    "terminal_cost",
    [x, p],
    [3.0 * x[0] * x[0] + 0.5 * x[1] * x[1] + p[1] * x[0]],
    input_names=["x", "p"],
    output_names=["vf"],
)


problem = SingleShootingProblem(
    name="mpc_cost",
    horizon=args.horizon,
    dynamics=dynamics,
    stage_cost=stage_cost,
    terminal_cost=terminal_cost,
    initial_state_name="x0",
    control_sequence_name="u_seq",
    parameter_name="p",
)


x0_value = [1.0, -0.5]
U_value = [0.2 if stage_index % 2 == 0 else -0.1 for stage_index in range(args.horizon)]
p_value = [0.4, -1.2]
v_U_value = [0.5 if stage_index % 2 == 0 else -1.0 for stage_index in range(args.horizon)]

cost_function = problem.to_function()
gradient_function = problem.gradient(include_states=True).to_function()
hvp_function = problem.hvp(include_states=True).to_function()

print("horizon =", args.horizon)
print("cost(x0, u_seq, p) =", cost_function(x0_value, U_value, p_value))
grad_value, x_traj_value = gradient_function(x0_value, U_value, p_value)
print("grad cost(x0, u_seq, p) =", grad_value)
print("hvp cost(x0, u_seq, p; v_u_seq) =", hvp_function(x0_value, U_value, p_value, v_U_value)[0])
print("x_traj(x0, u_seq, p) =", x_traj_value)


backend_config = (
    RustBackendConfig()
    .with_backend_mode("no_std")
    .with_scalar_type("f64")
    .with_crate_name("single_shooting_kernel")
)

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
    .build(Path(__file__).resolve().parent / "single_shooting_kernel")
)

print("Generated Rust crate:", project.project_dir)
print("Generated Rust functions:")
for codegen in project.codegens:
    print(" -", codegen.function_name)
