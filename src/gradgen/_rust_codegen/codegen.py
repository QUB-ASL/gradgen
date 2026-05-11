"""Rust code generation orchestration for symbolic functions."""

from __future__ import annotations

from dataclasses import replace
import logging
from pathlib import Path

from . import sanitize_ident
from .config import RustBackendConfig, RustBackendMode, RustScalarType
from .models import (
    RustCodegenResult,
    RustMultiFunctionProjectResult,
    _ArgSpec
)
from .project_support import (
    _derive_python_function_name,
    _render_cargo_dependency_lines,
    _render_metadata_json,
    _run_cargo_build,
    _try_run_cargo_fmt,
    _render_multi_function_lib,
)
from .project import _create_python_interface_project
from .rendering import (
    _allocate_workspace_slots,
    _arg_size,
    _build_shared_helper_lines,
    _collect_reachable_nodes,
    _collect_suppressed_custom_wrappers,
    _describe_input_arg,
    _describe_output_arg,
    _emit_custom_vector_output_helper_call,
    _emit_exact_length_assert,
    _emit_expr_ref,
    _emit_matvec_output_helper_call,
    _emit_min_length_assert,
    _emit_matrix_literal_reference,
    _emit_matrix_vector_argument,
    _emit_node_expr,
    _emit_sincos_call,
    _emit_workspace_assignment,
    _flatten_arg,
    _format_float,
    _format_rust_string_literal,
    _identify_direct_custom_output_marker,
    _reemit_direct_output_helper_call,
    _emit_matrix_literal,
    _use_sincos_bindings,
)
from .templates import _get_template, _render_custom_rust_header
from .validation import (
    resolve_backend_config as _resolve_backend_config,
    validate_backend_mode as _validate_backend_mode,
    validate_generated_argument_names as _validate_generated_argument_names,
    validate_scalar_type as _validate_scalar_type,
)
from ..function import Function
from ..map_zip import (
    ReducedFunction,
    BatchedFunction,
    BatchedJacobianFunction
)
from ..single_shooting import (
    SingleShootingGradientFunction,
    SingleShootingHvpFunction,
    SingleShootingJointFunction,
    SingleShootingPrimalFunction,
    SingleShootingProblem,
)
from ..sx import (
    SX,
    SXNode,
    parse_bilinear_form_args,
    parse_matvec_component_args,
    parse_quadform_args,
    parse_transpose_matvec_component_args,
)

SET_TUPLE_STR = set[tuple[str, str]]

_LOGGER = logging.getLogger(__name__)
_SCALARIZED_PRIVATE_INPUT_LIMIT = 32


def _flatten_output_terms(expr: SX) -> list[tuple[float, SX]]:
    """Return signed additive terms for a scalar output expression."""
    terms: list[tuple[float, SX]] = []

    def visit(node: SX, sign: float) -> None:
        if node.op == "add":
            visit(node.args[0], sign)
            visit(node.args[1], sign)
            return
        if node.op == "sub":
            visit(node.args[0], sign)
            visit(node.args[1], -sign)
            return
        if node.op == "neg":
            visit(node.args[0], -sign)
            return
        terms.append((sign, node))

    visit(expr, 1.0)
    return [
        (sign, term)
        for sign, term in terms
        if not (term.op == "const" and term.value == 0.0)
    ]


def _emit_output_accumulation_lines(
    expr: SX,
    target: str,
    scalar_bindings: dict[SXNode, str],
    workspace_map: dict[SXNode, int],
    backend_mode: RustBackendMode,
    scalar_type: RustScalarType,
    math_library: str | None,
    matrix_literal_bindings: dict[tuple[float, ...], str] | None = None,
) -> list[str]:
    """Emit readable accumulation statements for a scalar output expression."""
    loop_lines = _emit_matrix_component_accumulation_loop_lines(
        expr,
        target,
        scalar_bindings,
        workspace_map,
        backend_mode,
        scalar_type,
        math_library,
        matrix_literal_bindings,
    )
    if loop_lines is not None:
        return loop_lines

    terms = _flatten_output_terms(expr)
    zero_ref = _format_float(0.0, scalar_type)
    if not terms:
        return [f"{target} = {zero_ref};"]

    def emit_term(term: SX) -> str:
        return _emit_expr_ref(
            term,
            scalar_bindings,
            workspace_map,
            backend_mode,
            scalar_type,
            math_library,
            matrix_literal_bindings,
        )

    first_sign, first_term = terms[0]
    first_ref = emit_term(first_term)
    if len(terms) == 1:
        if first_sign >= 0.0:
            return [f"{target} = {first_ref};"]
        return [f"{target} = -{first_ref};"]

    lines = [
        f"{target} = {first_ref};"
        if first_sign >= 0.0
        else f"{target} = -{first_ref};"
    ]
    for sign, term in terms[1:]:
        ref = emit_term(term)
        operator = "+=" if sign >= 0.0 else "-="
        lines.append(f"{target} {operator} {ref};")
    return lines


def _should_scalarize_private_inputs(
    function_keyword: str,
    input_specs: tuple[_ArgSpec, ...],
) -> bool:
    """Return whether a private helper should scalarize slice inputs."""
    if function_keyword.startswith("pub"):
        return False
    return sum(spec.size for spec in input_specs) <= _SCALARIZED_PRIVATE_INPUT_LIMIT


def _build_scalarized_input_binding_lines(
    input_specs: tuple[_ArgSpec, ...],
    input_args: tuple[object, ...],
    reachable_nodes: set[SXNode],
    scalar_bindings: dict[SXNode, str],
) -> list[str]:
    """Return local scalar bindings for reachable helper inputs."""
    lines: list[str] = []
    for input_spec, input_arg in zip(input_specs, input_args):
        for scalar_index, scalar in enumerate(_flatten_arg(input_arg)):
            if scalar.node not in reachable_nodes:
                continue
            binding_name = sanitize_ident(
                f"{input_spec.rust_name}_{scalar_index}"
            )
            lines.append(
                f"let {binding_name} = {input_spec.rust_name}[{scalar_index}];"
            )
            scalar_bindings[scalar.node] = binding_name
    return lines


def _collect_mul_factors(expr: SX) -> list[SX]:
    """Return the flattened multiplicative factors for ``expr``."""
    if expr.op != "mul":
        return [expr]
    return [
        *_collect_mul_factors(expr.args[0]),
        *_collect_mul_factors(expr.args[1]),
    ]


def _match_matvec_component_product(
    expr: SX,
    scalar_bindings: dict[SXNode, str],
    workspace_map: dict[SXNode, int],
    backend_mode: RustBackendMode,
    scalar_type: RustScalarType,
    math_library: str | None,
    matrix_literal_bindings: dict[tuple[float, ...], str] | None,
) -> tuple[int, int, int, tuple[float, ...], tuple[SX, ...], str] | None:
    """Return metadata for a ``matvec_component``-times-vector term."""
    factors = _collect_mul_factors(expr)
    if len(factors) != 2:
        return None

    matvec_factor: SX | None = None
    vector_factor: SX | None = None
    for factor in factors:
        if factor.op == "matvec_component":
            matvec_factor = factor
        else:
            vector_factor = factor
    if matvec_factor is None or vector_factor is None:
        return None

    rows, cols, row, matrix_values, x_values = parse_matvec_component_args(
        matvec_factor.args
    )
    x_ref = _emit_matrix_vector_argument(
        x_values,
        scalar_bindings,
        workspace_map,
        backend_mode,
        scalar_type,
        math_library,
    )
    vector_ref = _emit_expr_ref(
        vector_factor,
        scalar_bindings,
        workspace_map,
        backend_mode,
        scalar_type,
        math_library,
        matrix_literal_bindings,
    )
    if vector_ref != f"{x_ref}[{row}]":
        return None
    return rows, cols, row, matrix_values, x_values, x_ref


def _match_transpose_matvec_component_product(
    expr: SX,
    scalar_bindings: dict[SXNode, str],
    workspace_map: dict[SXNode, int],
    backend_mode: RustBackendMode,
    scalar_type: RustScalarType,
    math_library: str | None,
    matrix_literal_bindings: dict[tuple[float, ...], str] | None,
) -> tuple[int, int, int, tuple[float, ...], tuple[SX, ...], str] | None:
    """Return metadata for a ``transpose_matvec_component`` term."""
    factors = _collect_mul_factors(expr)
    if len(factors) != 2:
        return None

    transpose_factor: SX | None = None
    vector_factor: SX | None = None
    for factor in factors:
        if factor.op == "transpose_matvec_component":
            transpose_factor = factor
        else:
            vector_factor = factor
    if transpose_factor is None or vector_factor is None:
        return None

    rows, cols, col, matrix_values, x_values = (
        parse_transpose_matvec_component_args(transpose_factor.args)
    )
    x_ref = _emit_matrix_vector_argument(
        x_values,
        scalar_bindings,
        workspace_map,
        backend_mode,
        scalar_type,
        math_library,
    )
    vector_ref = _emit_expr_ref(
        vector_factor,
        scalar_bindings,
        workspace_map,
        backend_mode,
        scalar_type,
        math_library,
        matrix_literal_bindings,
    )
    if vector_ref != f"{x_ref}[{col}]":
        return None
    return rows, cols, col, matrix_values, x_values, x_ref


def _emit_matrix_component_accumulation_loop_lines(
    expr: SX,
    target: str,
    scalar_bindings: dict[SXNode, str],
    workspace_map: dict[SXNode, int],
    backend_mode: RustBackendMode,
    scalar_type: RustScalarType,
    math_library: str | None,
    matrix_literal_bindings: dict[tuple[float, ...], str] | None,
) -> list[str] | None:
    """Return loop-based accumulation for repeated matrix-component terms."""
    terms = _flatten_output_terms(expr)
    if len(terms) <= 4:
        return None

    parsed_terms: list[
        tuple[str, tuple[int, int, int, tuple[float, ...], tuple[SX, ...], str]]
    ] = []
    for expected_index, (sign, term) in enumerate(terms):
        if sign < 0.0:
            return None
        parsed = _match_matvec_component_product(
            term,
            scalar_bindings,
            workspace_map,
            backend_mode,
            scalar_type,
            math_library,
            matrix_literal_bindings,
        )
        helper_name = "matvec_component"
        if parsed is None:
            parsed = _match_transpose_matvec_component_product(
                term,
                scalar_bindings,
                workspace_map,
                backend_mode,
                scalar_type,
                math_library,
                matrix_literal_bindings,
            )
            helper_name = "transpose_matvec_component"
        if parsed is None:
            return None
        rows, cols, index, matrix_values, x_values, x_ref = parsed
        if index != expected_index:
            return None
        parsed_terms.append((helper_name, parsed))

    first_helper_name, first_parsed = parsed_terms[0]
    rows, cols, _index, matrix_values, x_values, x_ref = first_parsed
    if rows <= 4 and cols <= 4:
        return None
    for helper_name, parsed in parsed_terms[1:]:
        cand_rows, cand_cols, _cand_index, cand_matrix_values, cand_x_values, cand_x_ref = parsed
        if (
            helper_name != first_helper_name
            or cand_rows != rows
            or cand_cols != cols
            or cand_matrix_values != matrix_values
            or cand_x_values != x_values
            or cand_x_ref != x_ref
        ):
            return None

    matrix_ref = _emit_matrix_literal_reference(
        matrix_values, scalar_type, matrix_literal_bindings
    )
    zero_ref = _format_float(0.0, scalar_type)
    helper_name = first_helper_name.replace("_component", "")
    return [
        f"{target} = {zero_ref};",
        f"for index in 0..{len(parsed_terms)} {{",
        (
            f"    {target} += {helper_name}_component("
            f"{matrix_ref}, {rows}, {cols}, index, {x_ref}) * {x_ref}[index];"
        ),
        "}",
    ]


def _direct_output_helper_requires_shared_helper_source(
    direct_output_helper: str | None,
) -> bool:
    """Return whether a direct output helper still needs module helper code."""
    if direct_output_helper is None:
        return False
    return "\n" not in direct_output_helper


def _collect_emitted_helper_nodes(
    function: Function,
    *,
    materialized_output_refs: tuple[SX, ...],
    direct_output_helpers: tuple[str | None, ...],
) -> tuple[SXNode, ...]:
    """Return helper nodes still needed after direct-output lowering."""
    reachable_nodes = _collect_reachable_nodes(materialized_output_refs)

    for output_arg, direct_output_helper in zip(
        function.outputs,
        direct_output_helpers,
    ):
        if not _direct_output_helper_requires_shared_helper_source(
            direct_output_helper
        ):
            continue
        helper_name = direct_output_helper.split("(", 1)[0]
        if helper_name in {"matvec", "transpose_matvec"}:
            continue
        reachable_nodes.update(_collect_reachable_nodes(_flatten_arg(output_arg)))

    return tuple(node for node in function.nodes if node in reachable_nodes)


def _collect_matrix_literal_bindings(
    nodes: tuple[SXNode, ...],
) -> dict[tuple[float, ...], str]:
    """Return hoisted Rust bindings for repeated constant matrix literals."""
    bindings: dict[tuple[float, ...], str] = {}
    for node in nodes:
        if node.op == "matvec_component":
            _, _, _, matrix_values, _ = parse_matvec_component_args(node.args)
        elif node.op == "transpose_matvec_component":
            _, _, _, matrix_values, _ = parse_transpose_matvec_component_args(
                node.args
            )
        elif node.op == "quadform":
            _, matrix_values, _ = parse_quadform_args(node.args)
        elif node.op == "bilinear_form":
            _, _, matrix_values, _, _ = parse_bilinear_form_args(node.args)
        else:
            continue
        if matrix_values not in bindings:
            bindings[matrix_values] = f"matrix_{len(bindings)}"
    return bindings


def _collect_required_matrix_helpers(
    direct_output_helpers: tuple[str | None, ...],
) -> set[str]:
    """Return whole-kernel matrix helpers required by direct output calls."""
    required_helpers: set[str] = set()
    for helper_call in direct_output_helpers:
        if helper_call is None or "\n" in helper_call:
            continue
        helper_name = helper_call.split("(", 1)[0]
        if helper_name in {"matvec", "transpose_matvec"}:
            required_helpers.add(helper_name)
    return required_helpers


def _collect_sincos_bindings(
    nodes: tuple[SXNode, ...],
) -> dict[SXNode, tuple[str, str]]:
    """Return fused sine/cosine bindings for repeated shared arguments."""
    ops_by_arg: dict[SXNode, set[str]] = {}
    for node in nodes:
        if node.op not in {"sin", "cos"}:
            continue
        arg = node.args[0]
        ops = ops_by_arg.setdefault(arg, set())
        ops.add(node.op)

    bindings: dict[SXNode, tuple[str, str]] = {}
    for arg, ops in ops_by_arg.items():
        if {"sin", "cos"} <= ops:
            index = len(bindings)
            bindings[arg] = (
                f"sincos_{index}_sin",
                f"sincos_{index}_cos",
            )
    return bindings


def _render_sincos_binding_lines(
    bindings: dict[SXNode, tuple[str, str]],
    scalar_bindings: dict[SXNode, str],
    workspace_map: dict[SXNode, int],
    backend_mode: RustBackendMode,
    scalar_type: RustScalarType,
    math_library: str | None,
) -> list[str]:
    """Render fused sine/cosine bindings for shared arguments."""
    lines: list[str] = []
    for arg, (sin_name, cos_name) in bindings.items():
        arg_ref = _emit_expr_ref(
            SX(arg),
            scalar_bindings,
            workspace_map,
            backend_mode,
            scalar_type,
            math_library,
        )
        lines.append(
            f"let ({sin_name}, {cos_name}) = "
            f"{_emit_sincos_call(arg_ref, backend_mode, scalar_type, math_library)};"
        )
    return lines


def _render_matrix_literal_binding_lines(
    matrix_literal_bindings: dict[tuple[float, ...], str],
    scalar_type: RustScalarType,
) -> list[str]:
    """Render hoisted Rust bindings for reused matrix literals."""
    return [
        f"let {name} = {_emit_matrix_literal(values, scalar_type)};"
        for values, name in matrix_literal_bindings.items()
    ]


def _emit_simple_output_ref(
    expr: SX,
    scalar_bindings: dict[SXNode, str],
    workspace_map: dict[SXNode, int],
    backend_mode: RustBackendMode,
    scalar_type: RustScalarType,
    math_library: str | None,
    matrix_literal_bindings: dict[tuple[float, ...], str] | None = None,
) -> str | None:
    """Return a direct reference when an output is a single positive term."""
    terms = _flatten_output_terms(expr)
    if len(terms) != 1:
        return None
    sign, term = terms[0]
    if sign < 0.0:
        return None
    return _emit_expr_ref(
        term,
        scalar_bindings,
        workspace_map,
        backend_mode,
        scalar_type,
        math_library,
        matrix_literal_bindings,
    )


def _parse_workspace_ref(ref: str, workspace_name: str) -> int | None:
    """Return the workspace index encoded in a generated slice reference."""
    prefix = f"{workspace_name}["
    if not ref.startswith(prefix) or not ref.endswith("]"):
        return None
    index_text = ref[len(prefix):-1]
    if not index_text.isdigit():
        return None
    return int(index_text)


def generate_rust(
    function: object,
    *,
    config: RustBackendConfig | None = None,
    function_name: str | None = None,
    backend_mode: RustBackendMode = "std",
    scalar_type: RustScalarType = "f64",
    math_library: str | None = None,
    function_index: int = 0,
    shared_helper_nodes: tuple[SXNode, ...] | None = None,
    shared_helper_suppressed_custom_wrappers: SET_TUPLE_STR | None = None,
    prioritize_expensive_workspace_nodes: bool = False,
    emit_crate_header: bool = True,
    function_keyword: str = "pub fn",
) -> RustCodegenResult:
    """Generate Rust code for a symbolic function or function family.

    This dispatcher inspects the supplied symbolic object and routes it to
    the appropriate Rust generator. It supports plain symbolic functions,
    composed functions, batched functions, single-shooting helpers, and the
    derivative variants built from those abstractions.

    Args:
        function: The symbolic object to lower to Rust. This may be a
            :class:`~gradgen.function.Function`, a composed function, a
            batched function, or one of the single-shooting helper types.
        config: Optional Rust backend configuration. When omitted, the
            generator uses the default backend settings.
        function_name: Optional override for the generated Rust function
            name.
        backend_mode: Backend mode used for code generation. The default is
            ``"std"``.
        scalar_type: Scalar type used in the generated Rust source. The
            default is ``"f64"``.
        math_library: Optional math library name used in ``no_std`` mode.
        function_index: Index of the function within a multi-function render
            family.
        shared_helper_nodes: Optional shared symbolic nodes used to keep
            nested helper kernels consistent.
        shared_helper_suppressed_custom_wrappers: Optional set of custom
            wrapper names to suppress when rendering shared helper kernels.
        emit_crate_header: Whether to emit the crate-level header comments and
            attributes in the generated source.
        function_keyword: Rust keyword used for the generated function
            declaration. The default is ``"pub fn"``.

    Returns:
        A :class:`~gradgen._rust_codegen.models.RustCodegenResult` containing
        the rendered Rust source and its metadata.

    Raises:
        ValueError: If the supplied object is not supported by the Rust
            codegen pipeline or if the backend configuration is invalid.

    Example:
        >>> from gradgen import Function, SX, generate_rust
        >>> x = SX.sym("x")
        >>> f = Function("square", [x], [x * x], input_names=["x"], output_names=["y"])
        >>> result = generate_rust(f)
        >>> result.python_name
        'square'
    """
    from ..composer import FunctionComposition
    from ..composed_function import (
        ComposedFunction,
        ComposedGradientFunction,
        ComposedJacobianFunction,
        ComposedJointFunction,
    )
    from .generators import (
        _generate_composed_gradient_rust,
        _generate_composed_jacobian_rust,
        _generate_composed_joint_rust,
        _generate_composed_primal_rust,
        _generate_reduced_primal_rust,
        _generate_single_shooting_gradient_rust,
        _generate_single_shooting_hvp_rust,
        _generate_single_shooting_joint_rust,
        _generate_single_shooting_primal_rust,
        _generate_batched_jacobian_rust,
        _generate_batched_primal_rust,
    )

    if isinstance(function, FunctionComposition):
        return function.generate_rust(
            config=config,
            function_name=function_name,
            backend_mode=backend_mode,
            scalar_type=scalar_type,
        )
    if isinstance(function, ComposedFunction):
        return _generate_composed_primal_rust(
            function,
            config=config,
            function_name=function_name,
            backend_mode=backend_mode,
            scalar_type=scalar_type,
            math_library=math_library,
            function_index=function_index,
        )
    if isinstance(function, ComposedGradientFunction):
        return _generate_composed_gradient_rust(
            function,
            config=config,
            function_name=function_name,
            backend_mode=backend_mode,
            scalar_type=scalar_type,
            math_library=math_library,
            function_index=function_index,
        )
    if isinstance(function, ComposedJacobianFunction):
        return _generate_composed_jacobian_rust(
            function,
            config=config,
            function_name=function_name,
            backend_mode=backend_mode,
            scalar_type=scalar_type,
            math_library=math_library,
            function_index=function_index,
        )
    if isinstance(function, ComposedJointFunction):
        return _generate_composed_joint_rust(
            function,
            config=config,
            function_name=function_name,
            backend_mode=backend_mode,
            scalar_type=scalar_type,
            math_library=math_library,
            function_index=function_index,
        )
    if isinstance(function, BatchedFunction):
        return _generate_batched_primal_rust(
            function,
            config=config,
            function_name=function_name,
            backend_mode=backend_mode,
            scalar_type=scalar_type,
            math_library=math_library,
            function_index=function_index,
        )
    if isinstance(function, BatchedJacobianFunction):
        return _generate_batched_jacobian_rust(
            function,
            config=config,
            function_name=function_name,
            backend_mode=backend_mode,
            scalar_type=scalar_type,
            math_library=math_library,
            function_index=function_index,
        )
    if isinstance(function, ReducedFunction):
        return _generate_reduced_primal_rust(
            function,
            config=config,
            function_name=function_name,
            backend_mode=backend_mode,
            scalar_type=scalar_type,
            math_library=math_library,
            function_index=function_index,
        )
    if isinstance(function, SingleShootingProblem):
        return _generate_single_shooting_primal_rust(
            function,
            include_states=False,
            config=config,
            function_name=function_name,
            backend_mode=backend_mode,
            scalar_type=scalar_type,
            math_library=math_library,
            function_index=function_index,
        )
    if isinstance(function, SingleShootingPrimalFunction):
        return _generate_single_shooting_primal_rust(
            function.problem,
            include_states=function.include_states,
            config=config,
            function_name=function_name or function.name,
            backend_mode=backend_mode,
            scalar_type=scalar_type,
            math_library=math_library,
            function_index=function_index,
        )
    if isinstance(function, SingleShootingGradientFunction):
        return _generate_single_shooting_gradient_rust(
            function,
            config=config,
            function_name=function_name,
            backend_mode=backend_mode,
            scalar_type=scalar_type,
            math_library=math_library,
            function_index=function_index,
        )
    if isinstance(function, SingleShootingHvpFunction):
        return _generate_single_shooting_hvp_rust(
            function,
            config=config,
            function_name=function_name,
            backend_mode=backend_mode,
            scalar_type=scalar_type,
            math_library=math_library,
            function_index=function_index,
        )
    if isinstance(function, SingleShootingJointFunction):
        return _generate_single_shooting_joint_rust(
            function,
            config=config,
            function_name=function_name,
            backend_mode=backend_mode,
            scalar_type=scalar_type,
            math_library=math_library,
            function_index=function_index,
        )

    resolved_config = _resolve_backend_config(
        config,
        function_name=function_name,
        backend_mode=backend_mode,
        scalar_type=scalar_type,
        math_library=math_library,
    )
    _validate_backend_mode(resolved_config.backend_mode)
    _validate_scalar_type(resolved_config.scalar_type)
    if resolved_config.backend_mode == "std":
        if math_library is not None:
            raise ValueError(
                "math_library is only supported for no_std backend mode")
        resolved_math_library = None
    else:
        resolved_math_library = math_library or "libm"

    name = sanitize_ident(resolved_config.function_name or function.name)
    input_sizes = tuple(_arg_size(arg) for arg in function.inputs)
    output_sizes = tuple(_arg_size(arg) for arg in function.outputs)
    input_specs: list[_ArgSpec] = []
    reachable_nodes = _collect_reachable_nodes(
        tuple(scalar for arg in function.outputs
              for scalar in _flatten_arg(arg)))
    public_function = function_keyword.startswith("pub")
    for raw_name, size, input_arg in zip(
            function.input_names, input_sizes, function.inputs):
        rust_name = sanitize_ident(raw_name)
        if not public_function and all(
                scalar.node not in reachable_nodes
                for scalar in _flatten_arg(input_arg)):
            rust_name = sanitize_ident(f"_{raw_name}")
        input_specs.append(
            _ArgSpec(
                raw_name=raw_name,
                rust_name=rust_name,
                rust_label=_format_rust_string_literal(raw_name),
                doc_description=_describe_input_arg(raw_name),
                size=size,
            )
        )
    input_specs = tuple(input_specs)
    output_specs = tuple(
        _ArgSpec(
            raw_name=raw_name,
            rust_name=sanitize_ident(raw_name),
            rust_label=_format_rust_string_literal(raw_name),
            doc_description=_describe_output_arg(raw_name),
            size=size,
        )
        for raw_name, size in zip(function.output_names, output_sizes)
    )
    _validate_generated_argument_names(input_specs, output_specs)

    scalar_bindings: dict[SXNode, str] = {}
    input_assert_lines: list[str] = []
    input_return_lines: list[str] = []
    output_assert_lines: list[str] = []
    output_return_lines: list[str] = []
    output_write_lines: list[str] = []
    computation_lines: list[str] = []
    suppressed_custom_wrappers: set[tuple[str, str]] = set()
    scalarized_input_binding_lines: list[str] = []

    for input_spec, input_arg in zip(input_specs, function.inputs):
        _assert, _in_return, _out_return = _emit_exact_length_assert(
            input_spec.rust_name,
            input_spec.raw_name,
            input_spec.size,
        )
        input_assert_lines.append(_assert)
        input_return_lines.append(_in_return)
        for scalar_index, scalar in enumerate(_flatten_arg(input_arg)):
            scalar_bindings[scalar.node] = \
                f"{input_spec.rust_name}[{scalar_index}]"

    if _should_scalarize_private_inputs(function_keyword, input_specs):
        scalarized_input_binding_lines = _build_scalarized_input_binding_lines(
            input_specs,
            function.inputs,
            reachable_nodes,
            scalar_bindings,
        )

    direct_output_helpers: list[str | None] = []
    materialized_output_refs: list[SX] = []
    for output_spec, output_arg in zip(output_specs, function.outputs):
        custom_helper_call = _emit_custom_vector_output_helper_call(
            output_arg,
            output_spec.rust_name,
            scalar_bindings,
            {},
            resolved_config.backend_mode,
            resolved_config.scalar_type,
            resolved_math_library,
        )
        if custom_helper_call is not None:
            direct_output_helpers.append(custom_helper_call)
            marker = _identify_direct_custom_output_marker(
                output_arg,
                scalar_bindings,
                {},
                resolved_config.backend_mode,
                resolved_config.scalar_type,
                resolved_math_library,
            )
            if marker is not None:
                suppressed_custom_wrappers.add(marker)
            continue

        matvec_helper_call = _emit_matvec_output_helper_call(
            output_arg,
            output_spec.rust_name,
            scalar_bindings,
            {},
            resolved_config.backend_mode,
            resolved_config.scalar_type,
            resolved_math_library,
            None,
        )
        if matvec_helper_call is not None:
            direct_output_helpers.append(matvec_helper_call)
            continue

        direct_output_helpers.append(None)
        materialized_output_refs.extend(_flatten_arg(output_arg))

    workspace_map, workspace_size = _allocate_workspace_slots(
        function,
        output_refs=tuple(materialized_output_refs),
        prioritize_expensive_nodes=prioritize_expensive_workspace_nodes,
    )
    emitted_helper_nodes = _collect_emitted_helper_nodes(
        function,
        materialized_output_refs=tuple(materialized_output_refs),
        direct_output_helpers=tuple(direct_output_helpers),
    )
    matrix_literal_bindings = _collect_matrix_literal_bindings(
        emitted_helper_nodes
    )
    required_matrix_helpers = _collect_required_matrix_helpers(
        tuple(direct_output_helpers)
    )
    emitted_nodes_for_sincos = _collect_reachable_nodes(
        tuple(materialized_output_refs)
    )
    emitted_nodes_for_sincos.update(emitted_helper_nodes)
    sincos_bindings = _collect_sincos_bindings(
        tuple(
            node
            for node in function.nodes
            if node in emitted_nodes_for_sincos
        )
    )
    workspace_name = "work" if workspace_map else "_work"

    if matrix_literal_bindings:
        computation_lines.extend(
            _render_matrix_literal_binding_lines(
                matrix_literal_bindings,
                resolved_config.scalar_type,
            )
        )

    with _use_sincos_bindings(sincos_bindings):
        if scalarized_input_binding_lines:
            computation_lines.extend(scalarized_input_binding_lines)
        if sincos_bindings:
            computation_lines.extend(
                _render_sincos_binding_lines(
                    sincos_bindings,
                    scalar_bindings,
                    workspace_map,
                    resolved_config.backend_mode,
                    resolved_config.scalar_type,
                    resolved_math_library,
                )
            )

        for node, work_index in workspace_map.items():
            rhs = _emit_node_expr(
                SX(node),
                scalar_bindings,
                workspace_map,
                resolved_config.backend_mode,
                resolved_config.scalar_type,
                resolved_math_library,
                matrix_literal_bindings,
            )
            assignment = _emit_workspace_assignment(
                node,
                work_index,
                rhs,
                scalar_bindings,
                workspace_map,
                resolved_config.backend_mode,
                resolved_config.scalar_type,
                resolved_math_library,
            )
            if assignment:
                computation_lines.append(assignment)

        for output_spec, output_arg, direct_helper_call in zip(
            output_specs, function.outputs, direct_output_helpers
        ):
            _assert, _in_return, _out_return = _emit_exact_length_assert(
                output_spec.rust_name,
                output_spec.raw_name,
                output_spec.size,
            )
            output_assert_lines.append(_assert)
            output_return_lines.append(_out_return)
            if direct_helper_call is not None:
                regenerated_call = _reemit_direct_output_helper_call(
                    direct_helper_call,
                    output_arg,
                    output_spec.rust_name,
                    scalar_bindings,
                    workspace_map,
                    resolved_config.backend_mode,
                    resolved_config.scalar_type,
                    resolved_math_library,
                    matrix_literal_bindings,
                )
                output_write_lines.extend(regenerated_call.splitlines())
                continue
            scalar_output_lines: list[list[str]] = []
            scalar_output_workspace_indices: list[int | None] = []
            can_copy_from_workspace = workspace_name == "work" and (
                output_spec.size > 1
            )
            for scalar_index, scalar in enumerate(_flatten_arg(output_arg)):
                output_ref = _emit_simple_output_ref(
                    scalar,
                    scalar_bindings,
                    workspace_map,
                    resolved_config.backend_mode,
                    resolved_config.scalar_type,
                    resolved_math_library,
                    matrix_literal_bindings,
                )
                if output_ref is not None:
                    workspace_index = _parse_workspace_ref(
                        output_ref, workspace_name
                    )
                    if workspace_index is None:
                        can_copy_from_workspace = False
                    scalar_output_workspace_indices.append(workspace_index)
                    scalar_output_lines.append(
                        [
                            f"{output_spec.rust_name}[{scalar_index}] = "
                            f"{output_ref};"
                        ]
                    )
                    continue
                can_copy_from_workspace = False
                scalar_output_workspace_indices.append(None)
                scalar_output_lines.append(
                    _emit_output_accumulation_lines(
                        scalar,
                        f"{output_spec.rust_name}[{scalar_index}]",
                        scalar_bindings,
                        workspace_map,
                        resolved_config.backend_mode,
                        resolved_config.scalar_type,
                        resolved_math_library,
                        matrix_literal_bindings,
                    )
                )
            if can_copy_from_workspace and all(
                index is not None for index in scalar_output_workspace_indices
            ):
                start = scalar_output_workspace_indices[0]
                if all(
                    index == start + offset
                    for offset, index in enumerate(
                        scalar_output_workspace_indices
                    )
                ):
                    output_write_lines.append(
                        f"{output_spec.rust_name}.copy_from_slice("
                        f"&{workspace_name}[{start}..{start + output_spec.size}]);"
                    )
                    continue
            for lines in scalar_output_lines:
                output_write_lines.extend(lines)

    parameters = ", ".join(
        [
            *[f"{spec.rust_name}: &[{resolved_config.scalar_type}]"
              for spec in input_specs],
            *[f"{spec.rust_name}: &mut [{resolved_config.scalar_type}]"
              for spec in output_specs],
            f"{workspace_name}: &mut [{resolved_config.scalar_type}]",
        ]
    )

    # workspace assertion emits both legacy assert and Result-returning form
    if workspace_size > 0:
        _ws_assert, _ws_return = _emit_min_length_assert(
            "work", "work", workspace_size)
    else:
        _ws_assert = None
        _ws_return = None

    source = _get_template("lib.rs.j2").render(
        function_name=name,
        function_label=_format_rust_string_literal(name),
        function_index=function_index,
        upper_name=name.upper(),
        emit_crate_header=emit_crate_header,
        function_keyword=function_keyword,
        backend_mode=resolved_config.backend_mode,
        scalar_type=resolved_config.scalar_type,
        math_library=resolved_math_library,
        header=_render_custom_rust_header(
            resolved_config.header,
            backend_mode=resolved_config.backend_mode,
            scalar_type=resolved_config.scalar_type,
            math_library=resolved_math_library,
            emit_metadata_helpers=resolved_config.emit_metadata_helpers,
        ),
        workspace_size=workspace_size,
        workspace_assert_line=_ws_assert,
        workspace_return_line=_ws_return,
        emit_metadata_helpers=resolved_config.emit_metadata_helpers,
        input_specs=input_specs,
        output_specs=output_specs,
        function_parameter_count=len(input_specs) + len(output_specs) + 1,
        parameters=parameters,
        input_assert_lines=input_assert_lines,
        input_return_lines=input_return_lines,
        output_assert_lines=output_assert_lines,
        output_return_lines=output_return_lines,
        computation_lines=computation_lines,
        output_write_lines=output_write_lines,
        shared_helper_lines=_build_shared_helper_lines(
            emitted_helper_nodes
            if shared_helper_nodes is None else shared_helper_nodes,
            resolved_config.backend_mode,
            resolved_config.scalar_type,
            resolved_math_library,
            required_matrix_helpers=required_matrix_helpers,
            suppressed_custom_wrappers=(
                suppressed_custom_wrappers
                if shared_helper_nodes is None
                else (shared_helper_suppressed_custom_wrappers or set())
            ),
        ),
    ).rstrip()

    codegen = RustCodegenResult(
        source=source if source.endswith("\n") else f"{source}\n",
        python_name=_derive_python_function_name(
            name,
            resolved_config.crate_name,
        ),
        function_name=name,
        workspace_size=workspace_size,
        input_names=tuple(spec.raw_name for spec in input_specs),
        input_sizes=input_sizes,
        output_names=tuple(spec.raw_name for spec in output_specs),
        output_sizes=output_sizes,
        backend_mode=resolved_config.backend_mode,
        scalar_type=resolved_config.scalar_type,
        math_library=resolved_math_library,
    )
    return codegen


def create_multi_function_rust_project(
    functions: tuple[object, ...],
    path: str | Path,
    *,
    config: RustBackendConfig | None = None,
) -> RustMultiFunctionProjectResult:
    """Create a single Rust crate containing multiple generated kernels."""
    if not functions:
        raise ValueError("at least one function must be generated")

    resolved_config = config or RustBackendConfig()
    _validate_backend_mode(resolved_config.backend_mode)
    _validate_scalar_type(resolved_config.scalar_type)
    resolved_math_library = "libm" \
        if resolved_config.backend_mode == "no_std" else None
    resolved_math_library_version = "0.2" \
        if resolved_math_library == "libm" else None
    dependency_lines = _render_cargo_dependency_lines(
        resolved_math_library,
        resolved_math_library_version,
        getattr(resolved_config, "additional_dependencies", ()),
    )

    project_dir = Path(path).expanduser().resolve()
    crate = sanitize_ident(resolved_config.crate_name or functions[0].name)

    src_dir = project_dir / "src"
    cargo_toml = project_dir / "Cargo.toml"
    readme = project_dir / "README.md"
    metadata_json = project_dir / "metadata.json"
    lib_rs = src_dir / "lib.rs"

    src_dir.mkdir(parents=True, exist_ok=True)
    generation_config = (
        replace(resolved_config, enable_python_interface=False)
        if resolved_config.enable_python_interface
        else resolved_config
    )
    shared_helper_nodes = tuple(
        node
        for generated_function in functions
        if isinstance(generated_function, Function)
        for node in generated_function.nodes
    )

    codegens = tuple(
        generate_rust(
            function,
            config=generation_config,
            function_name=function.name,
            function_index=index,
            shared_helper_nodes=shared_helper_nodes if index == 0 else (),
            shared_helper_suppressed_custom_wrappers=(
                {
                    marker
                    for generated_function in functions
                    if isinstance(generated_function, Function)
                    for marker in _collect_suppressed_custom_wrappers(
                        generated_function,
                        resolved_config.backend_mode,
                        resolved_config.scalar_type,
                        "libm"
                        if resolved_config.backend_mode == "no_std" else None,
                    )
                }
                if index == 0
                else set()
            ),
        )
        for index, function in enumerate(functions)
    )

    cargo_toml.write_text(
        _get_template("Cargo.toml.j2").render(
            crate_name=crate,
            backend_mode=resolved_config.backend_mode,
            scalar_type=resolved_config.scalar_type,
            dependency_lines=dependency_lines,
        ),
        encoding="utf-8",
    )
    readme.write_text(
        _get_template("rust_multi_project_README.md.j2").render(
            crate_name=crate,
            backend_mode=resolved_config.backend_mode,
            scalar_type=resolved_config.scalar_type,
            math_library=resolved_math_library,
            codegens=codegens,
            enable_python_interface=resolved_config.enable_python_interface,
            python_interface_project_name=(
                f"{crate}_python"
                if resolved_config.enable_python_interface else None
            ),
        ),
        encoding="utf-8",
    )
    metadata_json.write_text(
        _render_metadata_json(crate, codegens),
        encoding="utf-8",
    )
    lib_source = _render_multi_function_lib(codegens, resolved_config)
    lib_rs.write_text(lib_source, encoding="utf-8")
    stale_pyproject = project_dir / "pyproject.toml"
    if stale_pyproject.exists():
        stale_pyproject.unlink()
    if resolved_config.build_crate:
        _run_cargo_build(project_dir, resolved_config.build_profile)
    python_interface = None
    if resolved_config.enable_python_interface:
        python_interface = _create_python_interface_project(
            low_level_project_dir=project_dir,
            low_level_crate_name=crate,
            codegens=codegens,
            build_python_interface=resolved_config.build_python_interface,
        )
    _try_run_cargo_fmt(project_dir)

    return RustMultiFunctionProjectResult(
        project_dir=project_dir,
        cargo_toml=cargo_toml,
        readme=readme,
        metadata_json=metadata_json,
        lib_rs=lib_rs,
        codegens=codegens,
        python_interface=python_interface,
    )


__all__ = [name for name in globals() if not name.startswith("__")]
