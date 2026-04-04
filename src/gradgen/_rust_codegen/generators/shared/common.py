"""Common helper functions for Rust code generation families."""

from __future__ import annotations

from ....ad import jvp
from ....function import Function, _add_like, _make_symbolic_input_like, _zero_like
from ....sx import SX, SXVector


def _build_directional_derivative_function(
    function: Function,
    *,
    active_indices: tuple[int, ...],
    tangent_names: tuple[str, ...],
    name: str,
) -> Function:
    """Build a function that returns the JVP of ``function`` for selected inputs."""
    if len(active_indices) != len(tangent_names):
        raise ValueError("active_indices and tangent_names must have the same length")

    tangent_inputs = tuple(
        _make_symbolic_input_like(function.inputs[index], tangent_name)
        for index, tangent_name in zip(active_indices, tangent_names)
    )
    tangent_mapping = dict(zip(active_indices, tangent_inputs))

    differentiated_outputs: list[SX | SXVector] = []
    for output in function.outputs:
        total = _zero_like(output)
        for index, tangent_input in tangent_mapping.items():
            total = _add_like(total, jvp(output, function.inputs[index], tangent_input))
        differentiated_outputs.append(total)

    return Function(
        name,
        (*function.inputs, *tangent_inputs),
        differentiated_outputs,
        input_names=(*function.input_names, *tangent_names),
        output_names=function.output_names,
    )
