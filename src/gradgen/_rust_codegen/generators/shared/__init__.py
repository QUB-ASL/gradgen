"""Shared helpers for Rust code generation families."""

from ....rust_codegen import generate_rust
from ....rust_codegen import (
    _allocate_workspace_slots,
    _arg_size,
    _build_shared_helper_lines,
    _collect_reachable_nodes,
    _collect_required_workspace_nodes,
    _collect_suppressed_custom_wrappers,
    _derive_python_function_name,
    _describe_input_arg,
    _describe_output_arg,
    _emit_exact_length_assert,
    _emit_min_length_assert,
    _flatten_arg,
    _format_float,
    _format_rust_string_literal,
    _maybe_simplify_derivative_function,
    _reemit_direct_output_helper_call,
    _resolve_backend_config,
    _scaled_index_expr,
    _validate_backend_mode,
    _validate_generated_argument_names,
    _validate_scalar_type,
)
from .common import _build_directional_derivative_function
from .composed import (
    _build_composed_input_specs,
    _compose_composed_helper_base_name,
    _compose_offset_expr,
    _emit_composed_fixed_repeat_constants,
    _emit_composed_gradient_forward_repeat_block,
    _emit_composed_gradient_forward_single_block,
    _emit_composed_gradient_reverse_repeat_block,
    _emit_composed_gradient_reverse_single_block,
    _emit_composed_parameter_ref,
    _emit_composed_primal_repeat_block,
    _emit_composed_primal_single_block,
)
from .single_shooting import (
    _build_single_shooting_helpers,
    _build_single_shooting_input_specs,
    _build_single_shooting_output_specs,
    _compose_single_shooting_helper_base_name,
    _emit_single_shooting_control_slice,
    _emit_single_shooting_stage_range,
    _emit_small_accumulate,
)
from ..rendering import KernelRenderContext, render_kernel_source
