"""Family-specific Rust generation helpers."""

from __future__ import annotations

from . import shared as _shared
from ..config import RustBackendConfig, RustBackendMode, RustScalarType
from ..models import RustCodegenResult, _ArgSpec, _ComposedRepeatPlan, _ComposedSinglePlan, _SingleShootingHelperBundle
from ..naming import sanitize_ident
from ...function import Function
from ...map_zip import ReducedFunction, ZippedFunction, ZippedJacobianFunction
from ...single_shooting import (
    SingleShootingGradientFunction,
    SingleShootingHvpFunction,
    SingleShootingJointFunction,
    SingleShootingPrimalFunction,
    SingleShootingProblem,
)
from ...sx import SX, SXNode, SXVector

generate_rust = _shared.generate_rust
_resolve_backend_config = _shared._resolve_backend_config
_validate_backend_mode = _shared._validate_backend_mode
_validate_scalar_type = _shared._validate_scalar_type
_validate_generated_argument_names = _shared._validate_generated_argument_names
_maybe_simplify_derivative_function = _shared._maybe_simplify_derivative_function
_derive_python_function_name = _shared._derive_python_function_name
_flatten_arg = _shared._flatten_arg
_arg_size = _shared._arg_size
_scaled_index_expr = _shared._scaled_index_expr
_format_float = _shared._format_float
_format_rust_string_literal = _shared._format_rust_string_literal
_describe_input_arg = _shared._describe_input_arg
_describe_output_arg = _shared._describe_output_arg
_emit_exact_length_assert = _shared._emit_exact_length_assert
_emit_min_length_assert = _shared._emit_min_length_assert
_allocate_workspace_slots = _shared._allocate_workspace_slots
_collect_required_workspace_nodes = _shared._collect_required_workspace_nodes
_collect_reachable_nodes = _shared._collect_reachable_nodes
_reemit_direct_output_helper_call = _shared._reemit_direct_output_helper_call
_collect_suppressed_custom_wrappers = _shared._collect_suppressed_custom_wrappers
_build_shared_helper_lines = _shared._build_shared_helper_lines
_get_template = _shared._get_template
_build_composed_input_specs = _shared._build_composed_input_specs
_emit_composed_fixed_repeat_constants = _shared._emit_composed_fixed_repeat_constants
_emit_composed_parameter_ref = _shared._emit_composed_parameter_ref
_compose_offset_expr = _shared._compose_offset_expr
_compose_composed_helper_base_name = _shared._compose_composed_helper_base_name
_emit_composed_primal_single_block = _shared._emit_composed_primal_single_block
_emit_composed_primal_repeat_block = _shared._emit_composed_primal_repeat_block
_emit_composed_gradient_forward_single_block = _shared._emit_composed_gradient_forward_single_block
_emit_composed_gradient_forward_repeat_block = _shared._emit_composed_gradient_forward_repeat_block
_emit_composed_gradient_reverse_single_block = _shared._emit_composed_gradient_reverse_single_block
_emit_composed_gradient_reverse_repeat_block = _shared._emit_composed_gradient_reverse_repeat_block
_build_directional_derivative_function = _shared._build_directional_derivative_function
_build_single_shooting_helpers = _shared._build_single_shooting_helpers
_build_single_shooting_input_specs = _shared._build_single_shooting_input_specs
_build_single_shooting_output_specs = _shared._build_single_shooting_output_specs
_compose_single_shooting_helper_base_name = _shared._compose_single_shooting_helper_base_name
_emit_single_shooting_control_slice = _shared._emit_single_shooting_control_slice
_emit_single_shooting_stage_range = _shared._emit_single_shooting_stage_range
_emit_small_accumulate = _shared._emit_small_accumulate

def _generate_zipped_primal_rust(
    zipped: ZippedFunction,
    *,
    config: RustBackendConfig | None = None,
    function_name: str | None = None,
    backend_mode: RustBackendMode = "std",
    scalar_type: RustScalarType = "f64",
    math_library: str | None = None,
    function_index: int = 0,
) -> RustCodegenResult:
    """Generate loop-based Rust for a staged map/zip primal kernel."""
    resolved_config = _resolve_backend_config(
        config,
        function_name=function_name,
        backend_mode=backend_mode,
        scalar_type=scalar_type,
        math_library=math_library,
    )
    resolved_math_library = "libm" if resolved_config.backend_mode == "no_std" else None
    name = sanitize_ident(resolved_config.function_name or zipped.name)
    helper_name = sanitize_ident(f"{name}_helper")

    helper_function = zipped.function
    if zipped.simplification is not None:
        helper_function = helper_function.simplify(
            max_effort=zipped.simplification,
            name=helper_function.name,
        )

    helper_config = resolved_config.with_emit_metadata_helpers(False)
    helper_codegen = generate_rust(
        helper_function,
        config=helper_config,
        function_name=helper_name,
        function_index=1,
        shared_helper_nodes=(),
        emit_crate_header=False,
        emit_docs=False,
        function_keyword="fn",
    )

    packed_input_sizes = tuple(zipped.count * _arg_size(arg) for arg in zipped.function.inputs)
    packed_output_sizes = tuple(zipped.count * _arg_size(arg) for arg in zipped.function.outputs)
    input_specs = tuple(
        _ArgSpec(
            raw_name=raw_name,
            rust_name=sanitize_ident(raw_name),
            rust_label=_format_rust_string_literal(raw_name),
            doc_description=_describe_input_arg(raw_name),
            size=size,
        )
        for raw_name, size in zip(zipped.input_names, packed_input_sizes)
    )
    output_specs = tuple(
        _ArgSpec(
            raw_name=raw_name,
            rust_name=sanitize_ident(raw_name),
            rust_label=_format_rust_string_literal(raw_name),
            doc_description=_describe_output_arg(raw_name),
            size=size,
        )
        for raw_name, size in zip(zipped.output_names, packed_output_sizes)
    )
    _validate_generated_argument_names(input_specs, output_specs)

    input_assert_lines: list[str] = []
    input_return_lines: list[str] = []
    output_assert_lines: list[str] = []
    output_return_lines: list[str] = []

    for spec in input_specs:
        _assert, _in_return, _out_return = _emit_exact_length_assert(
            spec.rust_name,
            spec.raw_name,
            spec.size,
        )
        input_assert_lines.append(_assert)
        input_return_lines.append(_in_return)
    for spec in output_specs:
        _assert, _in_return, _out_return = _emit_exact_length_assert(
            spec.rust_name,
            spec.raw_name,
            spec.size,
        )
        output_assert_lines.append(_assert)
        output_return_lines.append(_out_return)

    helper_workspace_size = helper_codegen.workspace_size
    computation_lines: list[str] = []
    if helper_workspace_size > 0:
        computation_lines.append(f"let helper_work = &mut work[..{helper_workspace_size}];")
    else:
        computation_lines.append("let helper_work = &mut work[..0];")
    computation_lines.append(f"for stage_index in 0..{zipped.count} {{")
    for input_spec, formal in zip(input_specs, zipped.function.inputs):
        block_size = _arg_size(formal)
        start_expr = _scaled_index_expr("stage_index", block_size)
        end_expr = _scaled_index_expr("stage_index + 1", block_size)
        computation_lines.append(
            f"    let {input_spec.rust_name}_stage = "
            f"&{input_spec.rust_name}[{start_expr}..{end_expr}];"
        )
    for output_spec, formal in zip(output_specs, zipped.function.outputs):
        block_size = _arg_size(formal)
        start_expr = _scaled_index_expr("stage_index", block_size)
        end_expr = _scaled_index_expr("stage_index + 1", block_size)
        computation_lines.append(
            f"    let {output_spec.rust_name}_stage = "
            f"&mut {output_spec.rust_name}[{start_expr}..{end_expr}];"
        )
    helper_args = ", ".join(
        [
            *[f"{spec.rust_name}_stage" for spec in input_specs],
            *[f"{spec.rust_name}_stage" for spec in output_specs],
            "helper_work",
        ]
    )
    computation_lines.append(f"    {helper_name}({helper_args});")
    computation_lines.append("}")

    parameters = ", ".join(
        [
            *[f"{spec.rust_name}: &[{resolved_config.scalar_type}]" for spec in input_specs],
            *[f"{spec.rust_name}: &mut [{resolved_config.scalar_type}]" for spec in output_specs],
            f"work: &mut [{resolved_config.scalar_type}]",
        ]
    )
    if helper_workspace_size > 0:
        _ws_assert, _ws_return = _emit_min_length_assert("work", "work", helper_workspace_size)
    else:
        _ws_assert = None
        _ws_return = None

    driver_source = _get_template("lib.rs.j2").render(
        function_name=name,
        function_label=_format_rust_string_literal(name),
        function_index=function_index,
        emit_crate_header=True,
        emit_docs=True,
        function_keyword="pub fn",
        backend_mode=resolved_config.backend_mode,
        scalar_type=resolved_config.scalar_type,
        math_library=resolved_math_library,
        workspace_size=helper_workspace_size,
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
        output_write_lines=[],
        shared_helper_lines=[],
    ).rstrip()
    source = "\n\n".join([driver_source, helper_codegen.source.rstrip()])

    return RustCodegenResult(
        source=source if source.endswith("\n") else f"{source}\n",
        python_name=_derive_python_function_name(name, resolved_config.crate_name),
        function_name=name,
        workspace_size=helper_workspace_size,
        input_names=tuple(spec.raw_name for spec in input_specs),
        input_sizes=tuple(spec.size for spec in input_specs),
        output_names=tuple(spec.raw_name for spec in output_specs),
        output_sizes=tuple(spec.size for spec in output_specs),
        backend_mode=resolved_config.backend_mode,
        scalar_type=resolved_config.scalar_type,
        math_library=resolved_math_library,
    )


def _generate_zipped_jacobian_rust(
    zipped_jacobian: ZippedJacobianFunction,
    *,
    config: RustBackendConfig | None = None,
    function_name: str | None = None,
    backend_mode: RustBackendMode = "std",
    scalar_type: RustScalarType = "f64",
    math_library: str | None = None,
    function_index: int = 0,
) -> RustCodegenResult:
    """Generate loop-based Rust for a staged map/zip Jacobian kernel."""
    resolved_config = _resolve_backend_config(
        config,
        function_name=function_name,
        backend_mode=backend_mode,
        scalar_type=scalar_type,
        math_library=math_library,
    )
    resolved_math_library = "libm" if resolved_config.backend_mode == "no_std" else None
    name = sanitize_ident(resolved_config.function_name or zipped_jacobian.name)
    helper_name = sanitize_ident(f"{name}_helper")

    zipped = zipped_jacobian.zipped
    local_jacobian = zipped.function.jacobian(
        zipped_jacobian.wrt_index,
        name=helper_name,
    )
    if zipped_jacobian.simplification is not None:
        local_jacobian = local_jacobian.simplify(
            max_effort=zipped_jacobian.simplification,
            name=local_jacobian.name,
        )

    helper_config = resolved_config.with_emit_metadata_helpers(False)
    helper_codegen = generate_rust(
        local_jacobian,
        config=helper_config,
        function_name=helper_name,
        function_index=1,
        shared_helper_nodes=(),
        emit_crate_header=False,
        emit_docs=False,
        function_keyword="fn",
    )

    packed_input_sizes = tuple(zipped.count * _arg_size(arg) for arg in zipped.function.inputs)
    input_specs = tuple(
        _ArgSpec(
            raw_name=raw_name,
            rust_name=sanitize_ident(raw_name),
            rust_label=_format_rust_string_literal(raw_name),
            doc_description=_describe_input_arg(raw_name),
            size=size,
        )
        for raw_name, size in zip(zipped.input_names, packed_input_sizes)
    )
    wrt_size = _arg_size(zipped.function.inputs[zipped_jacobian.wrt_index])
    packed_wrt_size = zipped.count * wrt_size
    output_specs: list[_ArgSpec] = []
    local_output_sizes: list[int] = []
    row_sizes: list[int] = []
    for raw_name, output in zip(zipped.function.output_names, zipped.function.outputs):
        row_size = _arg_size(output)
        row_sizes.append(row_size)
        local_block_size = row_size * wrt_size
        local_output_sizes.append(local_block_size)
        output_specs.append(
            _ArgSpec(
                raw_name=f"jacobian_{raw_name}",
                rust_name=sanitize_ident(f"jacobian_{raw_name}"),
                rust_label=_format_rust_string_literal(f"jacobian_{raw_name}"),
                doc_description=_describe_output_arg(f"jacobian_{raw_name}"),
                size=(zipped.count * row_size) * packed_wrt_size,
            )
        )
    output_specs_tuple = tuple(output_specs)
    _validate_generated_argument_names(input_specs, output_specs_tuple)

    temp_workspace_size = sum(local_output_sizes)
    total_workspace_size = temp_workspace_size + helper_codegen.workspace_size

    input_assert_lines: list[str] = []
    input_return_lines: list[str] = []
    output_assert_lines: list[str] = []
    output_return_lines: list[str] = []
    for spec in input_specs:
        _assert, _in_return, _out_return = _emit_exact_length_assert(
            spec.rust_name,
            spec.raw_name,
            spec.size,
        )
        input_assert_lines.append(_assert)
        input_return_lines.append(_in_return)
    for spec in output_specs_tuple:
        _assert, _in_return, _out_return = _emit_exact_length_assert(
            spec.rust_name,
            spec.raw_name,
            spec.size,
        )
        output_assert_lines.append(_assert)
        output_return_lines.append(_out_return)

    zero_literal = _format_float(0.0, resolved_config.scalar_type)
    computation_lines: list[str] = [f"{spec.rust_name}.fill({zero_literal});" for spec in output_specs_tuple]
    remaining_work_name = "work"
    for index, (spec, local_size) in enumerate(zip(output_specs_tuple, local_output_sizes)):
        next_remaining = "helper_work" if index == len(output_specs_tuple) - 1 else f"rest_work_{index}"
        computation_lines.append(
            f"let (temp_{spec.rust_name}, {next_remaining}) = {remaining_work_name}.split_at_mut({local_size});"
        )
        remaining_work_name = next_remaining
    computation_lines.append(f"for stage_index in 0..{zipped.count} {{")
    for input_spec, formal in zip(input_specs, zipped.function.inputs):
        block_size = _arg_size(formal)
        start_expr = _scaled_index_expr("stage_index", block_size)
        end_expr = _scaled_index_expr("stage_index + 1", block_size)
        computation_lines.append(
            f"    let {input_spec.rust_name}_stage = "
            f"&{input_spec.rust_name}[{start_expr}..{end_expr}];"
        )
    helper_args = ", ".join(
        [
            *[f"{spec.rust_name}_stage" for spec in input_specs],
            *[f"temp_{spec.rust_name}" for spec in output_specs_tuple],
            "helper_work",
        ]
    )
    computation_lines.append(f"    {helper_name}({helper_args});")
    for spec, row_size, local_size in zip(output_specs_tuple, row_sizes, local_output_sizes):
        computation_lines.append(f"    for local_row in 0..{row_size} {{")
        dest_row_base = _scaled_index_expr("stage_index", row_size)
        dest_stage_offset = _scaled_index_expr("stage_index", wrt_size)
        src_row_base = _scaled_index_expr("local_row", wrt_size)
        if wrt_size > 1:
            src_row_base = src_row_base.replace("(", "", 1).rsplit(")", 1)[0]
        computation_lines.append(
            f"        let dest_row = {dest_row_base} + local_row;"
        )
        computation_lines.append(
            f"        let dest_start = (dest_row * {packed_wrt_size}) + {dest_stage_offset};"
        )
        computation_lines.append(f"        let src_start = {src_row_base};")
        computation_lines.append(
            f"        {spec.rust_name}[dest_start..(dest_start + {wrt_size})]"
            f".copy_from_slice(&temp_{spec.rust_name}[src_start..(src_start + {wrt_size})]);"
        )
        computation_lines.append("    }")
    computation_lines.append("}")

    parameters = ", ".join(
        [
            *[f"{spec.rust_name}: &[{resolved_config.scalar_type}]" for spec in input_specs],
            *[f"{spec.rust_name}: &mut [{resolved_config.scalar_type}]" for spec in output_specs_tuple],
            f"work: &mut [{resolved_config.scalar_type}]",
        ]
    )
    if total_workspace_size > 0:
        _ws_assert, _ws_return = _emit_min_length_assert("work", "work", total_workspace_size)
    else:
        _ws_assert = None
        _ws_return = None

    driver_source = _get_template("lib.rs.j2").render(
        function_name=name,
        function_label=_format_rust_string_literal(name),
        function_index=function_index,
        emit_crate_header=True,
        emit_docs=True,
        function_keyword="pub fn",
        backend_mode=resolved_config.backend_mode,
        scalar_type=resolved_config.scalar_type,
        math_library=resolved_math_library,
        workspace_size=total_workspace_size,
        workspace_assert_line=_ws_assert,
        workspace_return_line=_ws_return,
        emit_metadata_helpers=resolved_config.emit_metadata_helpers,
        input_specs=input_specs,
        output_specs=output_specs_tuple,
        function_parameter_count=len(input_specs) + len(output_specs_tuple) + 1,
        parameters=parameters,
        input_assert_lines=input_assert_lines,
        input_return_lines=input_return_lines,
        output_assert_lines=output_assert_lines,
        output_return_lines=output_return_lines,
        computation_lines=computation_lines,
        output_write_lines=[],
        shared_helper_lines=[],
    ).rstrip()
    source = "\n\n".join([driver_source, helper_codegen.source.rstrip()])

    return RustCodegenResult(
        source=source if source.endswith("\n") else f"{source}\n",
        python_name=_derive_python_function_name(name, resolved_config.crate_name),
        function_name=name,
        workspace_size=total_workspace_size,
        input_names=tuple(spec.raw_name for spec in input_specs),
        input_sizes=tuple(spec.size for spec in input_specs),
        output_names=tuple(spec.raw_name for spec in output_specs_tuple),
        output_sizes=tuple(spec.size for spec in output_specs_tuple),
        backend_mode=resolved_config.backend_mode,
        scalar_type=resolved_config.scalar_type,
        math_library=resolved_math_library,
    )

def _generate_reduced_primal_rust(
    reduced: ReducedFunction,
    *,
    config: RustBackendConfig | None = None,
    function_name: str | None = None,
    backend_mode: RustBackendMode = "std",
    scalar_type: RustScalarType = "f64",
    math_library: str | None = None,
    function_index: int = 0,
) -> RustCodegenResult:
    """Generate loop-based Rust for a staged reduce primal kernel."""
    resolved_config = _resolve_backend_config(
        config,
        function_name=function_name,
        backend_mode=backend_mode,
        scalar_type=scalar_type,
        math_library=math_library,
    )
    resolved_math_library = "libm" if resolved_config.backend_mode == "no_std" else None
    name = sanitize_ident(resolved_config.function_name or reduced.name)
    helper_name = sanitize_ident(f"{name}_helper")

    helper_function = reduced.function
    if reduced.simplification is not None:
        helper_function = helper_function.simplify(
            max_effort=reduced.simplification,
            name=helper_function.name,
        )

    helper_config = resolved_config.with_emit_metadata_helpers(False)
    helper_codegen = generate_rust(
        helper_function,
        config=helper_config,
        function_name=helper_name,
        function_index=1,
        shared_helper_nodes=(),
        emit_crate_header=False,
        emit_docs=False,
        function_keyword="fn",
    )

    accumulator_formal = reduced.function.inputs[0]
    sequence_formal = reduced.function.inputs[1]

    accumulator_size = _arg_size(accumulator_formal)
    sequence_size = reduced.count * _arg_size(sequence_formal)
    output_size = accumulator_size

    input_specs = (
        _ArgSpec(
            raw_name=reduced.accumulator_input_name,
            rust_name=sanitize_ident(reduced.accumulator_input_name),
            rust_label=_format_rust_string_literal(reduced.accumulator_input_name),
            doc_description=_describe_input_arg(reduced.accumulator_input_name),
            size=accumulator_size,
        ),
        _ArgSpec(
            raw_name=reduced.input_name,
            rust_name=sanitize_ident(reduced.input_name),
            rust_label=_format_rust_string_literal(reduced.input_name),
            doc_description=_describe_input_arg(reduced.input_name),
            size=sequence_size,
        ),
    )
    output_specs = (
        _ArgSpec(
            raw_name=reduced.output_name,
            rust_name=sanitize_ident(reduced.output_name),
            rust_label=_format_rust_string_literal(reduced.output_name),
            doc_description=_describe_output_arg(reduced.output_name),
            size=output_size,
        ),
    )
    _validate_generated_argument_names(input_specs, output_specs)

    input_assert_lines: list[str] = []
    input_return_lines: list[str] = []
    output_assert_lines: list[str] = []
    output_return_lines: list[str] = []

    for spec in input_specs:
        _assert, _in_return, _out_return = _emit_exact_length_assert(
            spec.rust_name,
            spec.raw_name,
            spec.size,
        )
        input_assert_lines.append(_assert)
        input_return_lines.append(_in_return)
    for spec in output_specs:
        _assert, _in_return, _out_return = _emit_exact_length_assert(
            spec.rust_name,
            spec.raw_name,
            spec.size,
        )
        output_assert_lines.append(_assert)
        output_return_lines.append(_out_return)

    helper_workspace_size = helper_codegen.workspace_size
    temp_acc_size = 2 * accumulator_size
    total_workspace_size = helper_workspace_size + temp_acc_size

    zero_literal = _format_float(0.0, resolved_config.scalar_type)
    acc0_name = input_specs[0].rust_name
    seq_name = input_specs[1].rust_name
    out_name = output_specs[0].rust_name

    computation_lines: list[str] = []
    if helper_workspace_size > 0:
        computation_lines.append(
            f"let (acc_work, helper_work) = work.split_at_mut({temp_acc_size});"
        )
    else:
        computation_lines.append("let acc_work = work;")
        computation_lines.append("let helper_work = &mut work[..0];")
    computation_lines.append(f"let (acc_curr_buf, acc_next_buf) = acc_work.split_at_mut({accumulator_size});")
    computation_lines.append(f"acc_curr_buf.copy_from_slice({acc0_name});")
    computation_lines.append(f"for stage_index in 0..{reduced.count} {{")
    block_size = _arg_size(sequence_formal)
    start_expr = _scaled_index_expr("stage_index", block_size)
    end_expr = _scaled_index_expr("stage_index + 1", block_size)
    computation_lines.append(f"    let x_stage = &{seq_name}[{start_expr}..{end_expr}];")
    computation_lines.append("    acc_next_buf.fill(" + zero_literal + ");")
    computation_lines.append(f"    {helper_name}(acc_curr_buf, x_stage, acc_next_buf, helper_work);")
    computation_lines.append("    acc_curr_buf.copy_from_slice(acc_next_buf);")
    computation_lines.append("}")
    computation_lines.append(f"{out_name}.copy_from_slice(acc_curr_buf);")

    parameters = ", ".join(
        [
            *[f"{spec.rust_name}: &[{resolved_config.scalar_type}]" for spec in input_specs],
            *[f"{spec.rust_name}: &mut [{resolved_config.scalar_type}]" for spec in output_specs],
            f"work: &mut [{resolved_config.scalar_type}]",
        ]
    )

    if total_workspace_size > 0:
        _ws_assert, _ws_return = _emit_min_length_assert("work", "work", total_workspace_size)
    else:
        _ws_assert = None
        _ws_return = None

    driver_source = _get_template("lib.rs.j2").render(
        function_name=name,
        function_label=_format_rust_string_literal(name),
        function_index=function_index,
        emit_crate_header=True,
        emit_docs=True,
        function_keyword="pub fn",
        backend_mode=resolved_config.backend_mode,
        scalar_type=resolved_config.scalar_type,
        math_library=resolved_math_library,
        workspace_size=total_workspace_size,
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
        output_write_lines=[],
        shared_helper_lines=[],
    ).rstrip()
    source = "\n\n".join([driver_source, helper_codegen.source.rstrip()])

    return RustCodegenResult(
        source=source if source.endswith("\n") else f"{source}\n",
        python_name=_derive_python_function_name(name, resolved_config.crate_name),
        function_name=name,
        workspace_size=total_workspace_size,
        input_names=tuple(spec.raw_name for spec in input_specs),
        input_sizes=tuple(spec.size for spec in input_specs),
        output_names=tuple(spec.raw_name for spec in output_specs),
        output_sizes=tuple(spec.size for spec in output_specs),
        backend_mode=resolved_config.backend_mode,
        scalar_type=resolved_config.scalar_type,
        math_library=resolved_math_library,
    )

