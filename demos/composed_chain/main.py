"""Demo for staged composition with heterogeneous chained stages."""

from __future__ import annotations

from pathlib import Path
import sys


# Allow running this file directly from inside the demo directory.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from gradgen import (  # noqa: E402
    CodeGenerationBuilder,
    ComposedFunction,
    Function,
    RustBackendConfig,
    SXVector,
)


def build_parameters_value() -> list[float]:
    """Return one concrete packed parameter vector for the demo chain.

    The same symbolic parameter ``p`` is reused across all chained stages,
    so the generated composed function only needs one packed parameter block.
    """
    return [0.6, -1.5]


# Define the composed state input and reusable stage signature.
x = SXVector.sym("x", 2)
state = SXVector.sym("state", 2)
p = SXVector.sym("p", 2)


# stage_1(a) = [a_1 + 1, 0.5 a_2 + 2]
stage_1 = Function(
    "warmup",
    [state, p],
    [SXVector((state[0] + p[0], 0.5 * state[1] + p[1]))],
    input_names=["state", "p"],
    output_names=["next_state"],
)


# stage_2(a, p) = [a_1 + p_1 a_2, a_2 + p_2]
stage_2 = Function(
    "mix",
    [state, p],
    [SXVector((state[0] + p[0] * state[1], state[1] + p[1]))],
    input_names=["state", "p"],
    output_names=["next_state"],
)


# stage_3(a, p) = [0.25 a_1 + p_1, 2 a_2 - p_2]
stage_3 = Function(
    "settle",
    [state, p],
    [SXVector((0.25 * state[0] + p[0], 2.0 * state[1] - p[1]))],
    input_names=["state", "p"],
    output_names=["next_state"],
)


# Build a staged composition that chains together heterogeneous stages.
# The first stage is fixed, while the second and third stages bind a packed
# runtime parameter vector named "parameters".
composed = (
    ComposedFunction("chain_demo", x)
    .chain(
        [
            (stage_1, [1.0, 2.0]),
            (stage_2, p),
            (stage_2, p),
            (stage_2, p),
            (stage_2, p),
            (stage_2, p),
            (stage_2, p),
            (stage_2, p),
            (stage_2, p),
            (stage_2, p),
            (stage_2, p),
            (stage_2, p),
            (stage_3, p),
        ]
    )
    .finish()
)
# gradient = composed.gradient()


# # Evaluate both kernels in Python for one concrete packed-parameter vector.
# x_value = [0.5, -0.4]
# parameters_value = build_parameters_value()

# print("packed parameter size =", composed.parameter_size)
# print("f(x, parameters) =", composed(x_value, parameters_value))
# print(
#     "jacobian f(x, parameters) =",
#     gradient.to_function()(x_value, parameters_value),
# )


# Generate one Rust crate directly from the staged composed-function object.
# The multi-function builder accepts staged sources directly, so the generated
# crate can contain both the primal and staged Jacobian kernels.
backend_config = (
    RustBackendConfig()
    .with_backend_mode("no_std")
    .with_scalar_type("f64")
    .with_crate_name("composed_chain_kernel")
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
