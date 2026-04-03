"""Demo showing loop-preserving map + reduce function generation."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from gradgen import (
    CodeGenerationBuilder,
    Function,
    RustBackendConfig,
    SX,
    map_function,
    reduce_function,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate and evaluate map+reduce demo kernels.")
    parser.add_argument(
        "--count",
        type=int,
        default=4,
        help="Number of stages N (must be positive).",
    )
    return parser.parse_args()


def _build_inputs(count: int) -> tuple[list[float], float]:
    x_seq: list[float] = []
    for stage_index in range(count):
        x_seq.append(0.25 + (0.2 * stage_index) - (0.05 * (stage_index * stage_index)))
    acc0 = 0.3
    return x_seq, acc0


x = SX.sym("x")
acc = SX.sym("acc")

# Stage map kernel:
#   m(x_k) = sin(x_k) + x_k^2
map_kernel = Function(
    "map_kernel",
    [x],
    [x.sin() + x * x],
    input_names=["x"],
    output_names=["m"],
)

# Stage reduce kernel (left fold):
#   a_{k+1} = r(a_k, m_k) = sin(a_k) + a_k * m_k + m_k^2
reduce_kernel = Function(
    "reduce_kernel",
    [acc, x],
    [acc.sin() + acc * x + x * x],
    input_names=["acc", "m"],
    output_names=["acc_next"],
)

args = _parse_args()
count = args.count
if count <= 0:
    raise ValueError("count must be a positive integer")

mapped = map_function(map_kernel, count, input_name="x_seq", name="mapped_seq")
reduced = reduce_function(
    reduce_kernel,
    count,
    accumulator_input_name="acc0",
    input_name="m_seq",
    output_name="acc_final",
    name="reduced_scalar",
)


# Rust code generation
project = (
    CodeGenerationBuilder()
    .with_backend_config(
        RustBackendConfig()
        .with_crate_name("reduce_map_kernel")
        .with_backend_mode("no_std")
        .with_scalar_type("f64")
    )
    .for_function(mapped)
        .add_primal()
        .add_jacobian()
        .with_simplification("medium")
        .done()
    .for_function(reduced)
        .add_primal()
        .with_simplification("medium")
        .done()
    .build(Path(__file__).resolve().parent)
)

print("Generated Rust crate:", project.project_dir)
print("Generated Rust functions:")
for codegen in project.codegens:
    print(" -", codegen.function_name)


# Computations in Python
print("Computations in Python (these are generally slower!)")
mapped_fn = mapped.to_function()
reduced_fn = reduced.to_function()

x_seq_value, acc0_value = _build_inputs(count)
mapped_value = mapped_fn(x_seq_value)
reduced_value = reduced_fn(acc0_value, mapped_value)

print("count =", count)
print("x_seq =", x_seq_value)
print("acc0 =", acc0_value)
print("mapped_seq(x_seq) =", mapped_value)
print("reduced_scalar(acc0, mapped_seq(x_seq)) =", reduced_value)
