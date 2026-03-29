"""Demo showing how to generate one Rust crate from multiple source functions."""

from __future__ import annotations

from pathlib import Path
import sys


# Allow running this file directly from inside the demo directory.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from gradgen import CodeGenerationBuilder, Function, FunctionBundle, RustBackendConfig, SXVector


# Define the shared symbolic inputs.
x = SXVector.sym("x", 2)
u = SXVector.sym("u", 1)


# First source function:
# f(x, u) = x_1^2 + x_2^2 + u_1 x_1
f = Function(
    "energy",
    [x, u],
    [x.norm2sq() + u[0] * x[0]],
    input_names=["x", "u"],
    output_names=["energy"],
)


# Second source function:
# g(x, u) = x_1 x_2 + cos(u_1)
g = Function(
    "coupling",
    [x, u],
    [x[0] * x[1] + u[0].cos()],
    input_names=["x", "u"],
    output_names=["coupling"],
)


# Evaluate both symbolic functions in Python first.
x_value = [1.5, -0.25]
u_value = [0.75]
v_x_value = [0.1, -0.3]
print("f(x, u) =", f(x_value, u_value))
print("g(x, u) =", g(x_value, u_value))
print("grad g wrt x =", g.gradient(0)(x_value, u_value))
print("hvp g wrt x =", g.hvp(0)(x_value, u_value, v_x_value))


# Build one Rust crate containing kernels for both source functions.
# Each for_function(...) block controls what gets generated for one source
# function.
project = (
    CodeGenerationBuilder()
    .with_backend_config(
        RustBackendConfig()
        .with_crate_name("multi_function_kernel")
        .with_backend_mode("no_std")
        .with_scalar_type("f64")
    )
    .for_function(
        f,
        lambda b: (
            b.add_primal()
            .add_jacobian()
            .with_simplification("medium")
        ),
    )
    .for_function(
        g,
        lambda b: (
            b.add_primal()
            .add_gradient()
            .add_hvp()
            .add_joint(
                FunctionBundle()
                .add_f()
                .add_jf(wrt=0)
            )
            .with_simplification("medium")
        ),
    )
    .build(Path(__file__).resolve().parent / "multi_function_kernel")
)

print("Generated Rust crate:", project.project_dir)
print("Generated Rust functions:")
for codegen in project.codegens:
    print(" -", codegen.function_name)
