"""Demo for Rust-only squared-distance-to-set registration."""

from __future__ import annotations

from pathlib import Path
import sys


# Make the repository's ``src`` directory importable when this file is run
# directly from inside ``demos/squared_distance_to_set_rust_only``.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from gradgen import (  # noqa: E402
    CodeGenerationBuilder,
    Function,
    RustBackendConfig,
    SXVector,
    SquaredDistanceToSet,
)


RUST_PRIMAL = """
fn rust_only_sqdist(
    x: &[{{ scalar_type }}],
    w: &[{{ scalar_type }}],
) -> {{ scalar_type }} {
    let _ = w;
    0.5_{{ scalar_type }} * x[1] * x[1]
}
"""

RUST_PROJECTION = """
fn rust_only_sqdist_projection(
    x: &[{{ scalar_type }}],
    out: &mut [{{ scalar_type }}],
) {
    out[0] = x[0];
    out[1] = 0.0_{{ scalar_type }};
}
"""


x = SXVector.sym("x", 2)
distance = (
    SquaredDistanceToSet(name="rust_only_sqdist")
    .with_rust_sq_distance(RUST_PRIMAL)
    .with_rust_projection(RUST_PROJECTION)
)

f = Function(
    "distance_energy",
    [x],
    [distance(2.0 * x)],
    input_names=["x"],
    output_names=["y"],
)
grad_x = f.gradient(0, name="distance_energy_grad_x")

print("symbolic distance =", distance(x))
try:
    print("distance([1.0, 100.0]) =", distance([1.0, 100.0]))
except ValueError as exc:
    print("distance([1.0, 100.0]) ->", exc)

project = (
    CodeGenerationBuilder()
    .with_backend_config(
        RustBackendConfig()
        .with_crate_name("squared_distance_to_set_rust_only_kernel")
        .with_backend_mode("no_std")
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
