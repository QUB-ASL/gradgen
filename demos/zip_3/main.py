"""Demo showing zip_function with a three-input stage function."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


# Allow running this file directly from inside the demo directory.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from gradgen import CodeGenerationBuilder, Function, RustBackendConfig, SXVector, zip_function, sin, SX


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate and evaluate a 3-input zip demo kernel.")
    parser.add_argument(
        "--count",
        type=int,
        default=5,
        help="Number of zip stages (must be positive).",
    )
    return parser.parse_args()


def _build_sequence_values(count: int) -> tuple[list[float], list[float], list[float]]:
    """Create packed stage-major values for one vector and two scalar sequences."""
    a_seq: list[float] = []
    for stage in range(count):
        a_seq.extend([1.0 + (0.5 * stage), -0.2 + (0.25 * stage)])
    b_seq = [2.0 - (0.25 * stage) for stage in range(count)]
    c_seq = [0.1 * ((-1.0) ** stage) + (0.2 * stage) for stage in range(count)]
    return a_seq, b_seq, c_seq


# Stage function h(a, b, c) with three inputs and one scalar output.
# Here a is a 2-vector, while b and c are scalars.
a = SXVector.sym("a", 2)
b = SX.sym("b")
c = SX.sym("c")
h = Function(
    "h",
    [a, b, c],
    [a[0] * b + a[1] + sin(c)],
    input_names=["a", "b", "c"],
    output_names=["y"],
)

args = _parse_args()
count = args.count
if count <= 0:
    raise ValueError("count must be a positive integer")

# Zip h stage-wise over packed sequences:
# ((a1_0,a1_1), ..., (aN_0,aN_1)), (b1, ..., bN), (c1, ..., cN)
# -> (h(a1,b1,c1), ..., h(aN,bN,cN)).
zipped = zip_function(h, count, input_names=("a_seq", "b_seq", "c_seq"), name="zip3")
zipped_jac_a = zipped.jacobian(wrt_index=0)
zipped_fn = zipped.to_function()
zipped_jac_a_fn = zipped_jac_a.to_function()

a_seq_value, b_seq_value, c_seq_value = _build_sequence_values(count)

print("count =", count)
print("a_seq =", a_seq_value)
print("b_seq =", b_seq_value)
print("c_seq =", c_seq_value)
print("zip3(a_seq, b_seq, c_seq) =", zipped_fn(a_seq_value, b_seq_value, c_seq_value))
print(
    "J_zip3 wrt a_seq (flat row-major) =",
    zipped_jac_a_fn(a_seq_value, b_seq_value, c_seq_value),
)

# Generate a Rust crate containing zip3 primal + jacobian kernels.
project = (
    CodeGenerationBuilder()
    .with_backend_config(
        RustBackendConfig()
        .with_crate_name("zip_3_kernel")
        .with_backend_mode("no_std")
        .with_scalar_type("f64")
    )
    .for_function(zipped)
        .add_primal()
        .add_jacobian()
        .with_simplification("medium")
        .done()
    .build(Path(__file__).resolve().parent / "zip_3_kernel")
)

print("Generated Rust crate:", project.project_dir)
print("Generated Rust functions:")
for codegen in project.codegens:
    print(" -", codegen.function_name)
