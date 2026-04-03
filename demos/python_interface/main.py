"""Demo showing how to generate a Rust crate plus a separate Python wrapper."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


# Allow running this file directly from inside the demo directory.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from gradgen import Function, RustBackendConfig, SXVector, create_rust_project


def parse_args() -> argparse.Namespace:
    """Parse the output directory used for the generated low-level crate."""
    parser = argparse.ArgumentParser(
        description="Generate a Rust crate and a sibling PyO3 Python wrapper.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "foo",
        help="Directory where the generated crate should be written.",
    )
    return parser.parse_args()


args = parse_args()


# f(x, w) = ||x||^2 + w_1
# g(x, w) = x_1 + x_2
x = SXVector.sym("x", 2)
w = SXVector.sym("w", 1)

energy = Function(
    "energy",
    [x, w],
    [x.norm2sq() + w[0], x[0] + x[1]],
    input_names=["x", "w"],
    output_names=["cost", "state"],
)

sample_x = [1.0, 2.0]
sample_w = [3.0]
print("energy(x, w) =", energy(sample_x, sample_w))

project = create_rust_project(
    energy,
    args.output_dir,
    config=RustBackendConfig()
    .with_crate_name("foo")
    .with_enable_python_interface(True),
)

print("Generated Rust crate:", project.project_dir)
print("Generated Rust function:", project.codegen.function_name)
if project.python_interface is not None:
    print("Generated Python wrapper crate:", project.python_interface.project_dir)
    print("Generated Python module: foo")
    print("Install the wrapper crate with `pip install -e` to import `foo`.")
