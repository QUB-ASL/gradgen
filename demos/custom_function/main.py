"""Demo for registering an opaque custom function and generating Rust code."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


# Make the repository's ``src`` directory importable when this file is run
# directly from inside ``demos/custom_function``.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from gradgen import CodeGenerationBuilder, Function, RustBackendConfig, SXVector, register_elementary_function


def custom_energy_eval(x: tuple[float, float], w: tuple[float, float]) -> float:
    """Return the custom scalar energy value f(x, w)."""
    return np.exp2(w[0]) * x[0] * x[0] + w[1] * x[1] * x[1] + np.sin(x[0] * x[1])


def custom_energy_jacobian(x: tuple[float, float], w: tuple[float, float]) -> list[float]:
    """Return the Jacobian of f with respect to x as a length-2 vector."""
    return [
        2.0 * np.exp2(w[0]) * x[0] + x[1] * np.cos(x[0] * x[1]),
        2.0 * w[1] * x[1] + x[0] * np.cos(x[0] * x[1]),
    ]


def custom_energy_hessian(x: tuple[float, float], w: tuple[float, float]) -> list[list[float]]:
    """Return the Hessian of f with respect to x as a 2x2 matrix."""
    xy = x[0] * x[1]
    sin_xy = np.sin(xy)
    cross = np.cos(xy) - x[0] * x[1] * sin_xy
    return [
        [2.0 * np.exp2(w[0]) - (x[1] ** 2) * sin_xy, cross],
        [cross, 2.0 * w[1] - (x[0] ** 2) * sin_xy],
    ]


def custom_energy_hvp(x: tuple[float, float], v: tuple[float, float], w: tuple[float, float]) -> list[float]:
    """Return the Hessian-vector product H(x, w) v."""
    return list(np.matmul(np.asarray(custom_energy_hessian(x, w), dtype=float), np.asarray(v, dtype=float)))


# Rust implementation of the primal function f(x, w).
RUST_PRIMAL = """
fn custom_energy_demo(
    x: &[{{ scalar_type }}],
    w: &[{{ scalar_type }}],
) -> {{ scalar_type }} {
    w[0].exp2() * x[0] * x[0] + w[1] * x[1] * x[1] + (x[0] * x[1]).sin()
}
"""

# Rust implementation of the Jacobian \nabla_x f(x, w).
RUST_JACOBIAN = """
fn custom_energy_demo_jacobian(
    x: &[{{ scalar_type }}],
    w: &[{{ scalar_type }}],
    out: &mut [{{ scalar_type }}],
) {
    let xy = x[0] * x[1];
    out[0] = 2.0_{{ scalar_type }} * w[0].exp2() * x[0] + x[1] * xy.cos();
    out[1] = 2.0_{{ scalar_type }} * w[1] * x[1] + x[0] * xy.cos();
}
"""

# Rust implementation of the flat row-major Hessian H_x f(x, w).
RUST_HESSIAN = """
fn custom_energy_demo_hessian(
    x: &[{{ scalar_type }}],
    w: &[{{ scalar_type }}],
    out: &mut [{{ scalar_type }}],
) {
    let xy = x[0] * x[1];
    let sin_xy = xy.sin();
    let cross = xy.cos() - x[0] * x[1] * sin_xy;
    out[0] = 2.0_{{ scalar_type }} * w[0].exp2() - x[1] * x[1] * sin_xy;
    out[1] = cross;
    out[2] = cross;
    out[3] = 2.0_{{ scalar_type }} * w[1] - x[0] * x[0] * sin_xy;
}
"""

# Rust implementation of the Hessian-vector product H_x f(x, w) v.
RUST_HVP = """
fn custom_energy_demo_hvp(
    x: &[{{ scalar_type }}],
    v_x: &[{{ scalar_type }}],
    w: &[{{ scalar_type }}],
    out: &mut [{{ scalar_type }}],
) {
    let xy = x[0] * x[1];
    let sin_xy = xy.sin();
    let cross = xy.cos() - x[0] * x[1] * sin_xy;
    let h00 = 2.0_{{ scalar_type }} * w[0].exp2() - x[1] * x[1] * sin_xy;
    let h11 = 2.0_{{ scalar_type }} * w[1] - x[0] * x[0] * sin_xy;
    out[0] = h00 * v_x[0] + cross * v_x[1];
    out[1] = cross * v_x[0] + h11 * v_x[1];
}
"""


# Register a custom scalar-valued function of a 2D vector x and a 2D parameter
# vector w. The derivative callbacks are intentionally opaque Python/NumPy code.
custom_energy = register_elementary_function(
    name="custom_energy_demo",
    input_dimension=2,
    parameter_dimension=2,
    eval_python=custom_energy_eval,
    jacobian=custom_energy_jacobian,
    hessian=custom_energy_hessian,
    hvp=custom_energy_hvp,
    rust_primal=RUST_PRIMAL,
    rust_jacobian=RUST_JACOBIAN,
    rust_hessian=RUST_HESSIAN,
    rust_hvp=RUST_HVP,
)


# Build a gradgen Function using the custom primitive.
x = SXVector.sym("x", 2)
f = Function(
    "custom_energy",
    [x],
    [custom_energy(x, w=[1.5, 3.0])],
    input_names=["x"],
    output_names=["y"],
)


# Evaluate the primal and derivative kernels on one concrete point so the demo
# shows what the opaque Python callbacks do.
x_value = [1.2, -0.7]
v_value = [0.5, -1.0]
print("f(x) =", f(x_value))
print("grad f(x) =", f.gradient(0)(x_value))
print("hessian f(x) =", f.hessian(0)(x_value))
print("hvp f(x, v) =", f.hvp(0)(x_value, v_value))


# Generate one Rust crate in this demo folder containing the primal, gradient,
# Hessian, and HVP kernels.
project = (
    CodeGenerationBuilder(f)
    .with_backend_config(
        RustBackendConfig()
        .with_crate_name("custom_function_kernel")
        .with_backend_mode("std")
        .with_scalar_type("f64")
    )
    .with_simplification("medium")
    .add_primal()
    .add_gradient()
    .add_hessian()
    .add_hvp()
    .build(Path(__file__).resolve().parent / "custom_function_kernel")
)

print("Generated Rust crate:", project.project_dir)
