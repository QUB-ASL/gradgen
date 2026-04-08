"""Demo for staged composition with packed parameters."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


# Allow running this file directly from inside the demo directory.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from gradgen import CodeGenerationBuilder, ComposedFunction, Function, RustBackendConfig, SXVector


def parse_args() -> argparse.Namespace:
    """Parse the user-configurable repeat count."""
    parser = argparse.ArgumentParser(
        description="Generate a staged composition demo with a user-selected repeat count.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Number of repeated applications of G to include in the composition.",
    )
    args = parser.parse_args()
    if args.repeats <= 0:
        parser.error("--repeats must be a positive integer")
    return args


def build_parameters_value(repeats: int) -> list[float]:
    """Build one concrete packed parameter vector for the chosen repeat count."""
    values: list[float] = []
    for repeat_index in range(repeats):
        values.extend(
            [float(2 * repeat_index + 3),
             float(2 * repeat_index + 4)])
    values.append(float(2 * repeats + 3))
    return values


args = parse_args()


# Define the composed state input and reusable stage/terminal signatures.
x = SXVector.sym("x", 2)
state = SXVector.sym("state", 2)
p = SXVector.sym("p", 2)
pf = SXVector.sym("pf", 1)


# G(a, p) = [0.9 * a_1 + p_1,
#            0.1 * a_2 * p_2]
stage = Function(
    "g",
    [state, p],
    [SXVector((0.9 * state[0] + p[0], 0.1 * state[1] * p[1]))],
    input_names=["state", "p"],
    output_names=["next_state"],
)


# h(a, pf) = 2 * a_1 - a_2 + pf
terminal = Function(
    "h",
    [state, pf],
    [2 * state[0] - state[1] + pf[0]],
    input_names=["state", "pf"],
    output_names=["y"],
)


# Build a staged composition that starts directly with N repeated
# applications of the same stage. Symbolic stage parameters are packed into one
# runtime vector named "parameters", with one 2D block per stage followed by
# the terminal scalar parameter.
composed = (
    ComposedFunction("composed_demo", x)
    .repeat(stage, params=[p] * args.repeats)
    .finish(terminal, p=pf)
)
gradient = composed.gradient()


# Evaluate both kernels in Python for one concrete packed-parameter vector.
x_value = [1.0, 2.0]
parameters_value = build_parameters_value(args.repeats)

print("repeat count =", args.repeats)
print("f(x, parameters) =", composed(x_value, parameters_value))
print("grad f(x, parameters) =", gradient.to_function()(x_value, parameters_value))


# Generate one Rust crate directly from the staged composed-function object.
# The multi-function builder now accepts staged sources directly, so the
# generated crate can contain both the primal and staged gradient kernels while
# still preserving the loop structure of the repeated stage.
backend_config = (
    RustBackendConfig()
    .with_backend_mode("no_std")
    .with_scalar_type("f64")
    .with_crate_name("composed_kernel")
)

project = (
    CodeGenerationBuilder()
    .with_backend_config(backend_config)
    .for_function(composed)
    .add_primal()
    .add_gradient()
    .with_simplification("medium")
    .done()
    .build(Path(__file__).resolve().parent)
)

print("Generated Rust crate:", project.project_dir)
