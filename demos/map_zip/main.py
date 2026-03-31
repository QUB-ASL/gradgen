"""Demo showing staged loop-preserving map/zip function generation."""

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
    map_function,
    zip_function,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate and evaluate staged map/zip demo kernels.")
    parser.add_argument(
        "--count",
        type=int,
        default=3,
        help="Number of map/zip stages (must be positive).",
    )
    return parser.parse_args()


def _build_sequence_values(count: int) -> tuple[list[float], list[float], list[float]]:
    """Create packed stage-major demo inputs sized to ``count`` stages."""
    x_seq: list[float] = []
    a_seq: list[float] = []
    b_seq: list[float] = []
    for stage_index in range(count):
        x_seq.extend([1.0 - (0.25 * stage_index), 0.5 * ((-1.0) ** stage_index) + (0.2 * stage_index)])
        a_seq.extend([1.0 - (0.2 * stage_index), 0.5 * ((-1.0) ** stage_index) + (0.3 * stage_index)])
        b_seq.extend([0.2 + (0.15 * stage_index), 1.1 * ((-1.0) ** stage_index) - (0.1 * stage_index)])
    return x_seq, a_seq, b_seq


# Build one unary function and one binary function to demonstrate
# map_function(...) and zip_function(...).
x = SXVector.sym("x", 2)
a = SXVector.sym("a", 2)
b = SXVector.sym("b", 2)


# unary(x) = [x1^2 + sin(x2), x1 - 0.5 x2]
unary = Function(
    "unary",
    [x],
    [SXVector((x[0] * x[0] + x[1].sin(), x[0] - 0.5 * x[1]))],
    input_names=["x"],
    output_names=["y"],
)


# binary(a, b) = [a1 + 2 b1, a2 b2 + cos(a1)]
binary = Function(
    "binary",
    [a, b],
    [SXVector((a[0] + 2.0 * b[0], a[1] * b[1] + a[0].cos()))],
    input_names=["a", "b"],
    output_names=["z"],
)


args = _parse_args()
count = args.count
if count <= 0:
    raise ValueError("count must be a positive integer")

# Stage-preserving mapped and zipped kernels over packed stage-major inputs.
unary_map = map_function(unary, count, input_name="x_seq", name="unary_map")
binary_zip = zip_function(binary, count, input_names=("a_seq", "b_seq"), name="binary_zip")

# Jacobian of the zipped function with respect to the packed a-sequence input.
binary_zip_jac_a = binary_zip.jacobian(wrt_index=0)
unary_map_fn = unary_map.to_function()
binary_zip_fn = binary_zip.to_function()
binary_zip_jac_a_fn = binary_zip_jac_a.to_function()

# Compose mapped unary outputs into both inputs of the zipped binary kernel:
# composed(x_seq, b_seq) = binary_zip(unary_map(x_seq), unary_map(b_seq))
x_seq_symbol = SXVector.sym("x_seq", count * len(x))
b_seq_symbol = SXVector.sym("b_seq", count * len(b))
mapped_x_seq_symbol = unary_map_fn(x_seq_symbol)
mapped_b_seq_symbol = unary_map_fn(b_seq_symbol)
composed_output = binary_zip_fn(mapped_x_seq_symbol, mapped_b_seq_symbol)
composed = Function(
    "composed_map_zip",
    [x_seq_symbol, b_seq_symbol],
    [composed_output],
    input_names=["x_seq", "b_seq"],
    output_names=["z_seq"],
)
composed_jac_x = composed.jacobian(wrt_index=0, name="composed_map_zip_jf_x_seq")
composed_fn = composed
composed_jac_x_fn = composed_jac_x


# Evaluate in Python first.
x_seq_value, a_seq_value, b_seq_value = _build_sequence_values(count)

print("count =", count)
print("unary_map(x_seq) =", unary_map_fn(x_seq_value))
print("binary_zip(a_seq, b_seq) =", binary_zip_fn(a_seq_value, b_seq_value))
print(
    "composed_map_zip(x_seq, b_seq) =",
    composed_fn(x_seq_value, b_seq_value),
)
print(
    "J_binary_zip wrt a_seq (flat row-major) =",
    binary_zip_jac_a_fn(a_seq_value, b_seq_value),
)
print(
    "J_composed_map_zip wrt x_seq (flat row-major) =",
    composed_jac_x_fn(x_seq_value, b_seq_value),
)


# Generate one Rust crate containing map, zip, and zipped Jacobian kernels.
project = (
    CodeGenerationBuilder()
    .with_backend_config(
        RustBackendConfig()
        .with_crate_name("map_zip_kernel")
        .with_backend_mode("no_std")
        .with_scalar_type("f64")
    )
    .for_function(unary_map)
        .add_primal()
        .add_jacobian()
        .with_simplification("medium")
        .done()
    .for_function(binary_zip)
        .add_primal()
        .add_jacobian()
        .with_simplification("medium")
        .done()
    .for_function(composed)
        .add_primal()
        .add_jacobian()
        .with_simplification("medium")
        .done()
    .build(Path(__file__).resolve().parent / "map_zip_kernel")
)

print("Generated Rust crate:", project.project_dir)
print("Generated Rust functions:")
for codegen in project.codegens:
    print(" -", codegen.function_name)
