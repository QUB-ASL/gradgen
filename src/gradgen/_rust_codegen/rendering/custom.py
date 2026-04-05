"""Custom-elementary helper emission for Rust rendering."""

from __future__ import annotations

from ..._custom_elementary import (
    get_registered_elementary_function,
    parse_custom_vector_hessian_entry_args,
    parse_custom_vector_hvp_component_args,
    parse_custom_vector_jacobian_component_args,
    render_custom_rust_snippet,
)
from ...sx import SX, SXNode, SXVector, parse_matvec_component_args
from ..config import RustBackendMode, RustScalarType
from .expression import _emit_matrix_literal, _emit_matrix_vector_argument


def _build_custom_vector_jacobian_wrapper_lines(
    spec: object,
    scalar_type: RustScalarType,
) -> tuple[str, ...]:
    spec = spec  # satisfy type checkers without extra protocols
    vector_dim = getattr(spec, "vector_dim")
    name = getattr(spec, "name")
    return (
        f"fn {name}_jacobian_component(index: usize, x: &[{scalar_type}], w: &[{scalar_type}]) -> {scalar_type} {{",
        f"    let mut out = [0.0_{scalar_type}; {vector_dim}];",
        f"    {name}_jacobian(x, w, &mut out);",
        "    out[index]",
        "}",
    )


def _build_custom_vector_hvp_wrapper_lines(
    spec: object,
    scalar_type: RustScalarType,
) -> tuple[str, ...]:
    spec = spec
    vector_dim = getattr(spec, "vector_dim")
    name = getattr(spec, "name")
    return (
        f"fn {name}_hvp_component(index: usize, x: &[{scalar_type}], v_x: &[{scalar_type}], w: &[{scalar_type}]) -> {scalar_type} {{",
        f"    let mut out = [0.0_{scalar_type}; {vector_dim}];",
        f"    {name}_hvp(x, v_x, w, &mut out);",
        "    out[index]",
        "}",
    )


def _build_custom_vector_hessian_wrapper_lines(
    spec: object,
    scalar_type: RustScalarType,
) -> tuple[str, ...]:
    spec = spec
    vector_dim = getattr(spec, "vector_dim")
    name = getattr(spec, "name")
    flat_size = vector_dim * vector_dim
    return (
        f"fn {name}_hessian_entry(row: usize, col: usize, x: &[{scalar_type}], w: &[{scalar_type}]) -> {scalar_type} {{",
        f"    let mut out = [0.0_{scalar_type}; {flat_size}];",
        f"    {name}_hessian(x, w, &mut out);",
        f"    out[(row * {vector_dim}) + col]",
        "}",
    )


def _build_custom_helper_lines(
    nodes: tuple[SXNode, ...],
    scalar_type: RustScalarType,
    math_library: str | None,
    *,
    suppressed_wrappers: set[tuple[str, str]],
) -> tuple[str, ...]:
    """Return shared helper lines for registered custom elementary functions."""
    emitted: list[str] = []
    seen: set[tuple[str, str]] = set()

    for node in nodes:
        if not node.op.startswith("custom_"):
            continue
        if node.name is None:
            raise ValueError("custom elementary nodes must carry a registered name")
        spec = get_registered_elementary_function(node.name)
        helper_kind: str
        snippet: str | None
        if node.op in {"custom_scalar", "custom_vector"}:
            helper_kind = "primal"
            snippet = spec.rust_primal
        elif node.op in {"custom_scalar_jacobian", "custom_vector_jacobian_component"}:
            helper_kind = "jacobian"
            snippet = spec.rust_jacobian
        elif node.op in {"custom_scalar_hvp", "custom_vector_hvp_component"}:
            helper_kind = "hvp"
            snippet = spec.rust_hvp
        elif node.op in {"custom_scalar_hessian", "custom_vector_hessian_entry"}:
            helper_kind = "hessian"
            snippet = spec.rust_hessian
        else:
            continue

        marker = (spec.name, helper_kind)
        if marker in seen:
            continue
        if snippet is None:
            raise ValueError(
                f"custom function {spec.name!r} requires rust_{helper_kind} for Rust code generation"
            )
        seen.add(marker)
        emitted.extend(render_custom_rust_snippet(snippet, scalar_type=scalar_type, math_library=math_library))
        if not spec.is_scalar and helper_kind == "jacobian" and (spec.name, "jacobian") not in suppressed_wrappers:
            emitted.extend(_build_custom_vector_jacobian_wrapper_lines(spec, scalar_type))
        if not spec.is_scalar and helper_kind == "hvp" and (spec.name, "hvp") not in suppressed_wrappers:
            emitted.extend(_build_custom_vector_hvp_wrapper_lines(spec, scalar_type))
        if not spec.is_scalar and helper_kind == "hessian" and (spec.name, "hessian") not in suppressed_wrappers:
            emitted.extend(_build_custom_vector_hessian_wrapper_lines(spec, scalar_type))

    return tuple(emitted)


def _emit_matvec_output_helper_call(
    output_arg: SX | SXVector,
    output_name: str,
    scalar_bindings: dict[SXNode, str],
    workspace_map: dict[SXNode, int],
    backend_mode: RustBackendMode,
    scalar_type: RustScalarType,
    math_library: str | None,
) -> str | None:
    """Return a single helper call when ``output_arg`` is exactly a matvec."""
    if not isinstance(output_arg, SXVector) or not output_arg.elements:
        return None
    parsed_exprs: list[SX] = []
    for element in output_arg:
        matched = _match_passthrough_matvec_component(element)
        if matched is None:
            return None
        parsed_exprs.append(matched)

    parsed = [parse_matvec_component_args(element.args) for element in parsed_exprs]
    rows, cols, _row, matrix_values, x_values = parsed[0]
    if rows != len(output_arg):
        return None
    for index, candidate in enumerate(parsed):
        cand_rows, cand_cols, cand_row, cand_matrix_values, cand_x_values = candidate
        if (
            cand_rows != rows
            or cand_cols != cols
            or cand_row != index
            or cand_matrix_values != matrix_values
            or cand_x_values != x_values
        ):
            return None

    matrix_ref = _emit_matrix_literal(matrix_values, scalar_type)
    x_ref = _emit_matrix_vector_argument(
        x_values, scalar_bindings, workspace_map, backend_mode, scalar_type, math_library
    )
    return f"matvec({matrix_ref}, {rows}, {cols}, {x_ref}, {output_name});"


def _emit_custom_vector_hessian_output_helper_call(
    output_arg: SXVector,
    output_name: str,
    scalar_bindings: dict[SXNode, str],
    workspace_map: dict[SXNode, int],
    backend_mode: RustBackendMode,
    scalar_type: RustScalarType,
    math_library: str | None,
) -> str | None:
    """Return a direct flat-output helper call for custom vector Hessians."""
    if not output_arg.elements:
        return None

    matched = _match_custom_vector_hessian_output(
        output_arg,
        scalar_bindings=scalar_bindings,
        workspace_map=workspace_map,
        backend_mode=backend_mode,
        scalar_type=scalar_type,
        math_library=math_library,
    )
    if matched is None:
        return None

    name, x_ref, w_ref = matched
    args = [x_ref, w_ref, output_name]
    return f"{name}_hessian(" + ", ".join(args) + ");"


def _emit_custom_vector_output_helper_call(
    output_arg: SX | SXVector,
    output_name: str,
    scalar_bindings: dict[SXNode, str],
    workspace_map: dict[SXNode, int],
    backend_mode: RustBackendMode,
    scalar_type: RustScalarType,
    math_library: str | None,
) -> str | None:
    """Return a direct output helper call for custom vector Jacobian/HVP results."""
    if not isinstance(output_arg, SXVector) or not output_arg.elements:
        return None

    jacobian_call = _match_custom_vector_derivative_output(
        output_arg,
        derivative_kind="jacobian",
        scalar_bindings=scalar_bindings,
        workspace_map=workspace_map,
        backend_mode=backend_mode,
        scalar_type=scalar_type,
        math_library=math_library,
    )
    if jacobian_call is not None:
        name, x_ref, tangent_ref, w_ref = jacobian_call
        _ = tangent_ref
        args = [x_ref, w_ref, output_name]
        return f"{name}_jacobian(" + ", ".join(args) + ");"

    hvp_call = _match_custom_vector_derivative_output(
        output_arg,
        derivative_kind="hvp",
        scalar_bindings=scalar_bindings,
        workspace_map=workspace_map,
        backend_mode=backend_mode,
        scalar_type=scalar_type,
        math_library=math_library,
    )
    if hvp_call is not None:
        name, x_ref, tangent_ref, w_ref = hvp_call
        if tangent_ref is None:
            return None
        args = [x_ref, tangent_ref, w_ref, output_name]
        return f"{name}_hvp(" + ", ".join(args) + ");"

    hessian_call = _emit_custom_vector_hessian_output_helper_call(
        output_arg,
        output_name,
        scalar_bindings,
        workspace_map,
        backend_mode,
        scalar_type,
        math_library,
    )
    if hessian_call is not None:
        return hessian_call

    return None


def _identify_direct_custom_output_marker(
    output_arg: SX | SXVector,
    scalar_bindings: dict[SXNode, str],
    workspace_map: dict[SXNode, int],
    backend_mode: RustBackendMode,
    scalar_type: RustScalarType,
    math_library: str | None,
) -> tuple[str, str] | None:
    """Return the custom helper marker used by a direct output helper call."""
    if not isinstance(output_arg, SXVector) or not output_arg.elements:
        return None

    jacobian_call = _match_custom_vector_derivative_output(
        output_arg,
        derivative_kind="jacobian",
        scalar_bindings=scalar_bindings,
        workspace_map=workspace_map,
        backend_mode=backend_mode,
        scalar_type=scalar_type,
        math_library=math_library,
    )
    if jacobian_call is not None:
        return jacobian_call[0], "jacobian"

    hvp_call = _match_custom_vector_derivative_output(
        output_arg,
        derivative_kind="hvp",
        scalar_bindings=scalar_bindings,
        workspace_map=workspace_map,
        backend_mode=backend_mode,
        scalar_type=scalar_type,
        math_library=math_library,
    )
    if hvp_call is not None:
        return hvp_call[0], "hvp"

    hessian_call = _match_custom_vector_hessian_output(
        output_arg,
        scalar_bindings=scalar_bindings,
        workspace_map=workspace_map,
        backend_mode=backend_mode,
        scalar_type=scalar_type,
        math_library=math_library,
    )
    if hessian_call is not None:
        return hessian_call[0], "hessian"

    return None


def _match_custom_vector_derivative_output(
    output_arg: SXVector,
    *,
    derivative_kind: str,
    scalar_bindings: dict[SXNode, str],
    workspace_map: dict[SXNode, int],
    backend_mode: RustBackendMode,
    scalar_type: RustScalarType,
    math_library: str | None,
) -> tuple[str, str, str | None, str] | None:
    """Match a full custom vector derivative output emitted as components."""
    expected_op = (
        "custom_vector_jacobian_component"
        if derivative_kind == "jacobian"
        else "custom_vector_hvp_component"
    )
    if any(element.op != expected_op for element in output_arg):
        return None

    if derivative_kind == "jacobian":
        parsed = [parse_custom_vector_jacobian_component_args(element.name, element.args) for element in output_arg]
        spec, first_index, value, params = parsed[0]
        if first_index != 0:
            return None
        for index, candidate in enumerate(parsed):
            cand_spec, cand_index, cand_value, cand_params = candidate
            if cand_spec.name != spec.name or cand_index != index or cand_value != value or cand_params != params:
                return None
        x_ref = _emit_matrix_vector_argument(
            value.elements, scalar_bindings, workspace_map, backend_mode, scalar_type, math_library
        )
        w_ref = _emit_matrix_vector_argument(
            params, scalar_bindings, workspace_map, backend_mode, scalar_type, math_library
        )
        return spec.name, x_ref, None, w_ref

    parsed_hvp = [parse_custom_vector_hvp_component_args(element.name, element.args) for element in output_arg]
    spec, first_index, value, tangent, params = parsed_hvp[0]
    if first_index != 0:
        return None
    for index, candidate in enumerate(parsed_hvp):
        cand_spec, cand_index, cand_value, cand_tangent, cand_params = candidate
        if (
            cand_spec.name != spec.name
            or cand_index != index
            or cand_value != value
            or cand_tangent != tangent
            or cand_params != params
        ):
            return None
    x_ref = _emit_matrix_vector_argument(
        value.elements, scalar_bindings, workspace_map, backend_mode, scalar_type, math_library
    )
    tangent_ref = _emit_matrix_vector_argument(
        tangent.elements, scalar_bindings, workspace_map, backend_mode, scalar_type, math_library
    )
    w_ref = _emit_matrix_vector_argument(
        params, scalar_bindings, workspace_map, backend_mode, scalar_type, math_library
    )
    return spec.name, x_ref, tangent_ref, w_ref


def _match_custom_vector_hessian_output(
    output_arg: SXVector,
    *,
    scalar_bindings: dict[SXNode, str],
    workspace_map: dict[SXNode, int],
    backend_mode: RustBackendMode,
    scalar_type: RustScalarType,
    math_library: str | None,
) -> tuple[str, str, str] | None:
    """Match a full flat custom vector Hessian output emitted row-major."""
    parsed: list[tuple[object, int, int, SXVector, tuple[SX, ...]]] = []
    for element in output_arg:
        matched = _match_passthrough_custom_vector_hessian_entry(element)
        if matched is None:
            return None
        parsed.append(parse_custom_vector_hessian_entry_args(matched.name, matched.args))

    spec = parsed[0][0]
    vector_dim = getattr(spec, "vector_dim")
    if vector_dim is None or len(output_arg) != vector_dim * vector_dim:
        return None

    _, _, _, x_value, params = parsed[0]
    for flat_index, candidate in enumerate(parsed):
        cand_spec, row, col, cand_x_value, cand_params = candidate
        if cand_spec != spec:
            return None
        expected_row, expected_col = divmod(flat_index, vector_dim)
        if row != expected_row or col != expected_col:
            return None
        if cand_x_value != x_value or cand_params != params:
            return None

    x_ref = _emit_matrix_vector_argument(
        x_value.elements, scalar_bindings, workspace_map, backend_mode, scalar_type, math_library
    )
    w_ref = _emit_matrix_vector_argument(
        params, scalar_bindings, workspace_map, backend_mode, scalar_type, math_library
    )
    return getattr(spec, "name"), x_ref, w_ref


def _match_passthrough_custom_vector_hessian_entry(expr: SX) -> SX | None:
    """Return the underlying custom Hessian entry through trivial wrappers."""
    if expr.op == "custom_vector_hessian_entry":
        return expr
    if expr.op == "mul":
        left, right = expr.args
        if _is_passthrough_one(left):
            return _match_passthrough_custom_vector_hessian_entry(right)
        if _is_passthrough_one(right):
            return _match_passthrough_custom_vector_hessian_entry(left)
        return None
    if expr.op == "add":
        left, right = expr.args
        if _is_passthrough_zero(left):
            return _match_passthrough_custom_vector_hessian_entry(right)
        if _is_passthrough_zero(right):
            return _match_passthrough_custom_vector_hessian_entry(left)
        return None
    return None


def _match_passthrough_matvec_component(expr: SX) -> SX | None:
    """Return the underlying matvec component through trivial wrappers."""
    if expr.op == "matvec_component":
        return expr
    if expr.op == "mul":
        left, right = expr.args
        if _is_passthrough_one(left):
            return _match_passthrough_matvec_component(right)
        if _is_passthrough_one(right):
            return _match_passthrough_matvec_component(left)
        return None
    if expr.op == "add":
        left, right = expr.args
        if _is_passthrough_zero(left):
            return _match_passthrough_matvec_component(right)
        if _is_passthrough_zero(right):
            return _match_passthrough_matvec_component(left)
        return None
    return None


def _is_passthrough_zero(expr: SX) -> bool:
    """Return whether ``expr`` is structurally equivalent to zero."""
    if expr.op == "const" and expr.value == 0.0:
        return True
    if expr.op == "add":
        left, right = expr.args
        return _is_passthrough_zero(left) and _is_passthrough_zero(right)
    if expr.op == "mul":
        left, right = expr.args
        return _is_passthrough_zero(left) or _is_passthrough_zero(right)
    return False


def _is_passthrough_one(expr: SX) -> bool:
    """Return whether ``expr`` is structurally equivalent to one."""
    if expr.op == "const" and expr.value == 1.0:
        return True
    if expr.op == "add":
        left, right = expr.args
        return (_is_passthrough_zero(left) and _is_passthrough_one(right)) or (
            _is_passthrough_one(left) and _is_passthrough_zero(right)
        )
    if expr.op == "mul":
        left, right = expr.args
        return _is_passthrough_one(left) and _is_passthrough_one(right)
    return False


def _collect_suppressed_custom_wrappers(
    function,
    backend_mode: RustBackendMode,
    scalar_type: RustScalarType,
    math_library: str | None,
) -> set[tuple[str, str]]:
    """Return custom wrapper kinds that are superseded by direct output helpers."""
    scalar_bindings: dict[SXNode, str] = {}
    from .util import _flatten_arg

    for input_index, input_arg in enumerate(function.inputs):
        for scalar_index, scalar in enumerate(_flatten_arg(input_arg)):
            scalar_bindings[scalar.node] = f"arg_{input_index}[{scalar_index}]"

    suppressed: set[tuple[str, str]] = set()
    for output_arg in function.outputs:
        marker = _identify_direct_custom_output_marker(
            output_arg,
            scalar_bindings,
            {},
            backend_mode,
            scalar_type,
            math_library,
        )
        if marker is not None:
            suppressed.add(marker)
    return suppressed


def _reemit_direct_output_helper_call(
    direct_helper_call: str,
    output_arg: SX | SXVector,
    output_name: str,
    scalar_bindings: dict[SXNode, str],
    workspace_map: dict[SXNode, int],
    backend_mode: RustBackendMode,
    scalar_type: RustScalarType,
    math_library: str | None,
) -> str:
    """Rebuild a direct output helper call using the final workspace map."""
    custom_helper_call = _emit_custom_vector_output_helper_call(
        output_arg,
        output_name,
        scalar_bindings,
        workspace_map,
        backend_mode,
        scalar_type,
        math_library,
    )
    if custom_helper_call is not None:
        return custom_helper_call
    matvec_helper_call = _emit_matvec_output_helper_call(
        output_arg,
        output_name,
        scalar_bindings,
        workspace_map,
        backend_mode,
        scalar_type,
        math_library,
    )
    if matvec_helper_call is not None:
        return matvec_helper_call
    return direct_helper_call
