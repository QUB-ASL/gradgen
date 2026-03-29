"""Demo showing Jacobian and runtime-seeded VJP generation for a vector function."""

from __future__ import annotations

from pathlib import Path
import sys


# Allow running this file directly from inside the demo directory.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from gradgen import CodeGenerationBuilder, Function, RustBackendConfig, SXVector


# Define the vector input x in R^2.
x = SXVector.sym("x", 2)


# Define a vector-valued function g : R^2 -> R^3.
# G(x) = [x1 + x2, x1 * x2, sin(x2)]
G = Function(
    "g",
    [x],
    [SXVector((x[0] + x[1], x[0] * x[1], x[1].sin()))],
    input_names=["x"],
    output_names=["y"],
)


# Build the Jacobian block J_G(x) as a flat row-major vector and a
# runtime-seeded VJP helper that computes J_G(x)^T v for a cotangent vector v
# supplied at runtime.
JG = G.jacobian(0)
reverse_x = G.vjp(wrt_index=0)


# Evaluate both symbolic functions at one concrete point.
x_value = [3.0, 4.0]
cotangent_y = [2.0, -1.0, 5.0]
print("G(x) =", G(x_value))
print("J_G(x) flat row-major =", JG(x_value))
print("J_G(x)^T v =", reverse_x(x_value, cotangent_y))


# Generate one Rust crate containing the primal function, the Jacobian,
# and the runtime-seeded VJP kernel.
project = (
    CodeGenerationBuilder()
    .with_backend_config(
        RustBackendConfig()
        .with_crate_name("vjp_kernel")
        .with_backend_mode("no_std")
        .with_scalar_type("f64")
    )
    .for_function(
        G,
        lambda b: (
            b.add_primal()
            .add_jacobian()
            .add_vjp()
            .with_simplification("medium")
        ),
    )
    .build(Path(__file__).resolve().parent / "vjp_kernel")
)

print("Generated Rust crate:", project.project_dir)
print("Generated Rust functions:")
for codegen in project.codegens:
    print(" -", codegen.function_name)
