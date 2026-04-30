"""Demo for projection-backed half-squared distance code generation."""

from __future__ import annotations

from pathlib import Path
import sys


# Make the repository's ``src`` directory importable when this file is run
# directly from inside ``demos/squared_distance_to_set``.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from gradgen import (  # noqa: E402
    CodeGenerationBuilder,
    Function,
    RustBackendConfig,
    SX,
    SXVector,
    SquaredDistanceToSet,
)

x = SXVector.sym("x", 2)
projection = Function(
    "projection",
    [x],
    [SXVector((x[0], SX.const(0.0)))],
    input_names=["x"],
    output_names=["p"],
)
sq_distance = Function(
    "sq_distance",
    [x],
    [0.5 * x[1] * x[1]],
    input_names=["x"],
    output_names=["d"],
)

# The represented set is the x-axis in R^2, so the half-squared distance is
# 0.5 * x_2^2 and the projection is (x_1, 0). We compose the primitive with
# z = 2x to demonstrate that symbolic chain rules still work.
distance = (
    SquaredDistanceToSet(name="dist_to_axis_demo")
    .with_projection_function(projection)
    .with_sq_distance_function(sq_distance)
)

z = 2.0 * x
f = Function(
    "distance_energy",
    [x],
    [distance(z)],
    input_names=["x"],
    output_names=["y"],
)
grad_x = f.gradient(0, name="distance_energy_grad_x")

sample_x = [1.5, -3.0]
print("distance([1.0, 100.0]) =", distance([1.0, 100.0]))
print("distance.jacobian()([1.0, 100.0]) =", distance.jacobian()([1.0, 100.0]))
print("symbolic expr =", distance(x))
print("f(x) =", f(sample_x))
print("grad_x f(x) =", grad_x(sample_x))

project = (
    CodeGenerationBuilder()
    .with_backend_config(
    RustBackendConfig()
        .with_crate_name("squared_distance_to_set_kernel")
        .with_backend_mode("no_std")
        .with_build_profile("debug") # release | debug
        .with_scalar_type("f64")
    )
    .for_function(f)
        .add_primal()
        .with_simplification("medium")
        .done()
    .for_function(grad_x)
        .add_primal()
        .with_simplification("medium")
        .done()
    .build(Path(__file__).resolve().parent)
)

print("Generated Rust crate:", project.project_dir)
print("Generated Rust functions:")
for codegen in project.codegens:
    print(" -", codegen.function_name)
