"""Second-order-cone distance demo with a Python wrapper."""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
import sys


# Allow running this file directly from inside the demo directory.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from gradgen import (  # noqa: E402
    Function,
    RustBackendConfig,
    SXVector,
    SquaredDistanceToSet,
    create_multi_function_rust_project,
)


def parse_args() -> argparse.Namespace:
    """Parse the output directory used for the generated Rust project."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate a Rust crate and a PyO3 wrapper for a second-order "
            "cone distance demo."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "soc_kernel",
        help="Directory where the generated Rust crate should be written.",
    )
    return parser.parse_args()


def _project_soc_onto_cone(
    x_values: list[float],
    alpha: float,
) -> list[float]:
    """Return the Euclidean projection onto the scaled second-order cone."""
    y_values = x_values[:-1]
    t_value = x_values[-1]
    norm_y = sum(value * value for value in y_values) ** 0.5

    if alpha * norm_y <= -t_value:
        return [0.0 for _ in x_values]

    if norm_y <= alpha * t_value:
        return list(x_values)

    beta = (alpha * norm_y + t_value) / (alpha * alpha + 1.0)
    scale = alpha * beta / norm_y
    return [scale * value for value in y_values] + [beta]


def main() -> None:
    """Generate the demo crate and print the generated wrapper results."""
    args = parse_args()

    alpha = 2.0
    x = SXVector.sym("x", 3)
    distance = SquaredDistanceToSet.second_order_cone(
        name="soc_penalty",
        alpha=alpha,
        dimension=3,
    )

    energy = Function(
        "soc_kernel_energy",
        [x],
        [distance(x)],
        input_names=["x"],
        output_names=["energy"],
    )
    energy_grad_x = energy.gradient(0, name="soc_kernel_energy_grad_x")

    sample_x = [3.0, 4.0, 1.0]
    projected_x = _project_soc_onto_cone(sample_x, alpha)
    expected_energy = 0.5 * sum(
        (value - projection)
        * (value - projection)
        for value, projection in zip(sample_x, projected_x, strict=True)
    )
    expected_gradient = [
        value - projection
        for value, projection in zip(sample_x, projected_x, strict=True)
    ]

    print("symbolic energy =", energy(x))
    print("sample x =", sample_x)
    print("projected x =", projected_x)
    print("expected energy =", expected_energy)
    print("expected gradient =", expected_gradient)

    project = create_multi_function_rust_project(
        (energy, energy_grad_x),
        args.output_dir,
        config=(
            RustBackendConfig()
            .with_crate_name("soc_kernel")
            .with_enable_python_interface(True)
        ),
    )

    print("Generated Rust crate:", project.project_dir)
    print("Generated Rust functions:")
    for codegen in project.codegens:
        print(" -", codegen.function_name)

    assert project.python_interface is not None
    wrapper = project.python_interface
    print("Generated Python wrapper:", wrapper.project_dir)
    print("Generated Python module:", wrapper.module_name)

    soc = importlib.import_module(wrapper.module_name)
    print("all_functions() =", soc.all_functions())
    print("function_info('energy') =", soc.function_info("energy"))
    print(
        "function_info('energy_grad_x') =",
        soc.function_info("energy_grad_x"),
    )

    energy_workspace = soc.workspace_for_function("energy")
    gradient_workspace = soc.workspace_for_function("energy_grad_x")
    print("workspace_for_function('energy') =", energy_workspace)
    print(
        "workspace_for_function('energy_grad_x') =",
        gradient_workspace,
    )

    energy_result = soc.energy(sample_x, energy_workspace)
    gradient_result = soc.energy_grad_x(sample_x, gradient_workspace)
    print("soc.energy(sample_x, workspace) =", energy_result)
    print("soc.energy_grad_x(sample_x, workspace) =", gradient_result)


if __name__ == "__main__":
    main()
