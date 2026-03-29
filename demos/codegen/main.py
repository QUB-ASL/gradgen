"""Minimal demo showing how gradgen generates a Rust crate."""

from __future__ import annotations

from pathlib import Path
import sys


# Allow running this file directly from inside the demo directory.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from gradgen import CodeGenerationBuilder, Function, RustBackendConfig, SXVector


# Define the symbolic inputs.
x = SXVector.sym("x", 3)
u = SXVector.sym("u", 1)


# Build a simple scalar-valued function of x and u.
# f(x, u) = ||x||_2^2 + u_1 * sin(x_1) + x_2 * x_3
f_expr = x.norm2sq() + u[0] * x[0].sin() + x[1] * x[2]

f = Function(
    "codegen_demo",
    [x, u],
    [f_expr],
    input_names=["x", "u"],
    output_names=["y"],
)


# Evaluate the function once in Python so the demo has a concrete reference.
x_value = [1.0, 2.0, -0.5]
u_value = [3.0]
print("f(x, u) =", f(x_value, u_value))


# Generate one Rust crate containing the primal kernel and its Jacobian blocks.
project = (
    CodeGenerationBuilder(f)
    .with_backend_config(
        RustBackendConfig()
        .with_crate_name("codegen_kernel")
        .with_backend_mode("std")
        .with_scalar_type("f64")
    )
    .with_simplification("medium")
    .add_primal()
    .add_jacobian()
    .build(Path(__file__).resolve().parent / "codegen_kernel")
)

print("Generated Rust crate:", project.project_dir)
