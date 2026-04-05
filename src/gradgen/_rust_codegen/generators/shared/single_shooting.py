"""Single-shooting helper utilities for Rust code generation."""

from __future__ import annotations

import re

from ...models import _ArgSpec, _SingleShootingHelperBundle
from ...naming import sanitize_ident
from .common import _build_directional_derivative_function
from ....function import Function
from ....sx import SX, SXNode, SXVector
from .. import shared as _shared


def _build_single_shooting_helpers(
    problem,
    *,
    helper_base_name: str,
    config,
    simplification,
    include_cost: bool,
    include_gradient: bool,
    include_hvp: bool,
) -> _SingleShootingHelperBundle:
    """Generate stage helper kernels for a single-shooting problem."""
    helper_config = config.with_emit_metadata_helpers(False)
    helper_sources: list[str] = []
    helper_nodes: list[SXNode] = []
    max_workspace = 0

    dynamics_name = sanitize_ident(f"{helper_base_name}_dynamics")
    dynamics_jvp_name: str | None = None
    dynamics_function = _shared._maybe_simplify_derivative_function(problem.dynamics, simplification)
    dynamics_codegen = _shared.generate_rust(
        dynamics_function,
        config=helper_config,
        function_name=dynamics_name,
        function_index=1,
        shared_helper_nodes=(),
        emit_crate_header=False,
        emit_docs=False,
        function_keyword="fn",
    )
    helper_sources.append(dynamics_codegen.source.rstrip())
    helper_nodes.extend(dynamics_function.nodes)
    max_workspace = max(max_workspace, dynamics_codegen.workspace_size)

    if include_hvp:
        dynamics_jvp_name = sanitize_ident(f"{helper_base_name}_dynamics_jvp")
        dynamics_jvp_function = _shared._maybe_simplify_derivative_function(
            _build_directional_derivative_function(
                problem.dynamics,
                active_indices=(0, 1),
                tangent_names=("tangent_x", "tangent_u"),
                name=dynamics_jvp_name,
            ),
            simplification,
        )
        dynamics_jvp_codegen = _shared.generate_rust(
            dynamics_jvp_function,
            config=helper_config,
            function_name=dynamics_jvp_name,
            function_index=1,
            shared_helper_nodes=(),
            emit_crate_header=False,
            emit_docs=False,
            function_keyword="fn",
        )
        helper_sources.append(dynamics_jvp_codegen.source.rstrip())
        helper_nodes.extend(dynamics_jvp_function.nodes)
        max_workspace = max(max_workspace, dynamics_jvp_codegen.workspace_size)

    stage_cost_name = sanitize_ident(f"{helper_base_name}_stage_cost")
    terminal_cost_name = sanitize_ident(f"{helper_base_name}_terminal_cost")
    if include_cost:
        stage_cost_function = _shared._maybe_simplify_derivative_function(problem.stage_cost, simplification)
        stage_cost_codegen = _shared.generate_rust(
            stage_cost_function,
            config=helper_config,
            function_name=stage_cost_name,
            function_index=1,
            shared_helper_nodes=(),
            emit_crate_header=False,
            emit_docs=False,
            function_keyword="fn",
        )
        helper_sources.append(stage_cost_codegen.source.rstrip())
        helper_nodes.extend(stage_cost_function.nodes)
        max_workspace = max(max_workspace, stage_cost_codegen.workspace_size)

        terminal_cost_function = _shared._maybe_simplify_derivative_function(problem.terminal_cost, simplification)
        terminal_cost_codegen = _shared.generate_rust(
            terminal_cost_function,
            config=helper_config,
            function_name=terminal_cost_name,
            function_index=1,
            shared_helper_nodes=(),
            emit_crate_header=False,
            emit_docs=False,
            function_keyword="fn",
        )
        helper_sources.append(terminal_cost_codegen.source.rstrip())
        helper_nodes.extend(terminal_cost_function.nodes)
        max_workspace = max(max_workspace, terminal_cost_codegen.workspace_size)

    dynamics_vjp_x_name: str | None = None
    dynamics_vjp_u_name: str | None = None
    dynamics_vjp_x_jvp_name: str | None = None
    dynamics_vjp_u_jvp_name: str | None = None
    stage_cost_grad_x_name: str | None = None
    stage_cost_grad_u_name: str | None = None
    stage_cost_grad_x_jvp_name: str | None = None
    stage_cost_grad_u_jvp_name: str | None = None
    terminal_cost_grad_x_name: str | None = None
    terminal_cost_grad_x_jvp_name: str | None = None
    if include_gradient or include_hvp:
        dynamics_vjp_x_name = sanitize_ident(f"{helper_base_name}_dynamics_vjp_x")
        dynamics_vjp_u_name = sanitize_ident(f"{helper_base_name}_dynamics_vjp_u")
        stage_cost_grad_x_name = sanitize_ident(f"{helper_base_name}_stage_cost_grad_x")
        stage_cost_grad_u_name = sanitize_ident(f"{helper_base_name}_stage_cost_grad_u")
        terminal_cost_grad_x_name = sanitize_ident(f"{helper_base_name}_terminal_cost_grad_x")

        helper_functions = (
            _shared._maybe_simplify_derivative_function(
                problem.dynamics.vjp(wrt_index=0, name=dynamics_vjp_x_name),
                simplification,
            ),
            _shared._maybe_simplify_derivative_function(
                problem.dynamics.vjp(wrt_index=1, name=dynamics_vjp_u_name),
                simplification,
            ),
            _shared._maybe_simplify_derivative_function(
                problem.stage_cost.gradient(0, name=stage_cost_grad_x_name),
                simplification,
            ),
            _shared._maybe_simplify_derivative_function(
                problem.stage_cost.gradient(1, name=stage_cost_grad_u_name),
                simplification,
            ),
            _shared._maybe_simplify_derivative_function(
                problem.terminal_cost.gradient(0, name=terminal_cost_grad_x_name),
                simplification,
            ),
        )
        helper_names = (
            dynamics_vjp_x_name,
            dynamics_vjp_u_name,
            stage_cost_grad_x_name,
            stage_cost_grad_u_name,
            terminal_cost_grad_x_name,
        )
        for helper_function, helper_name in zip(helper_functions, helper_names):
            helper_codegen = _shared.generate_rust(
                helper_function,
                config=helper_config,
                function_name=helper_name,
                function_index=1,
                shared_helper_nodes=(),
                emit_crate_header=False,
                emit_docs=False,
                function_keyword="fn",
            )
            helper_sources.append(helper_codegen.source.rstrip())
            helper_nodes.extend(helper_function.nodes)
            max_workspace = max(max_workspace, helper_codegen.workspace_size)

    if include_hvp:
        dynamics_vjp_x_jvp_name = sanitize_ident(f"{helper_base_name}_dynamics_vjp_x_jvp")
        dynamics_vjp_u_jvp_name = sanitize_ident(f"{helper_base_name}_dynamics_vjp_u_jvp")
        stage_cost_grad_x_jvp_name = sanitize_ident(f"{helper_base_name}_stage_cost_grad_x_jvp")
        stage_cost_grad_u_jvp_name = sanitize_ident(f"{helper_base_name}_stage_cost_grad_u_jvp")
        terminal_cost_grad_x_jvp_name = sanitize_ident(f"{helper_base_name}_terminal_cost_grad_x_jvp")

        dynamics_vjp_x_function = _shared._maybe_simplify_derivative_function(
            problem.dynamics.vjp(wrt_index=0, name=dynamics_vjp_x_name),
            simplification,
        )
        dynamics_vjp_u_function = _shared._maybe_simplify_derivative_function(
            problem.dynamics.vjp(wrt_index=1, name=dynamics_vjp_u_name),
            simplification,
        )
        stage_cost_grad_x_function = _shared._maybe_simplify_derivative_function(
            problem.stage_cost.gradient(0, name=stage_cost_grad_x_name),
            simplification,
        )
        stage_cost_grad_u_function = _shared._maybe_simplify_derivative_function(
            problem.stage_cost.gradient(1, name=stage_cost_grad_u_name),
            simplification,
        )
        terminal_cost_grad_x_function = _shared._maybe_simplify_derivative_function(
            problem.terminal_cost.gradient(0, name=terminal_cost_grad_x_name),
            simplification,
        )

        hvp_helper_functions = (
            _shared._maybe_simplify_derivative_function(
                _build_directional_derivative_function(
                    dynamics_vjp_x_function,
                    active_indices=(0, 1, 3),
                    tangent_names=("tangent_x", "tangent_u", "tangent_cotangent_x_next"),
                    name=dynamics_vjp_x_jvp_name,
                ),
                simplification,
            ),
            _shared._maybe_simplify_derivative_function(
                _build_directional_derivative_function(
                    dynamics_vjp_u_function,
                    active_indices=(0, 1, 3),
                    tangent_names=("tangent_x", "tangent_u", "tangent_cotangent_x_next"),
                    name=dynamics_vjp_u_jvp_name,
                ),
                simplification,
            ),
            _shared._maybe_simplify_derivative_function(
                _build_directional_derivative_function(
                    stage_cost_grad_x_function,
                    active_indices=(0, 1),
                    tangent_names=("tangent_x", "tangent_u"),
                    name=stage_cost_grad_x_jvp_name,
                ),
                simplification,
            ),
            _shared._maybe_simplify_derivative_function(
                _build_directional_derivative_function(
                    stage_cost_grad_u_function,
                    active_indices=(0, 1),
                    tangent_names=("tangent_x", "tangent_u"),
                    name=stage_cost_grad_u_jvp_name,
                ),
                simplification,
            ),
            _shared._maybe_simplify_derivative_function(
                _build_directional_derivative_function(
                    terminal_cost_grad_x_function,
                    active_indices=(0,),
                    tangent_names=("tangent_x",),
                    name=terminal_cost_grad_x_jvp_name,
                ),
                simplification,
            ),
        )
        hvp_helper_names = (
            dynamics_vjp_x_jvp_name,
            dynamics_vjp_u_jvp_name,
            stage_cost_grad_x_jvp_name,
            stage_cost_grad_u_jvp_name,
            terminal_cost_grad_x_jvp_name,
        )
        for helper_function, helper_name in zip(hvp_helper_functions, hvp_helper_names):
            helper_codegen = _shared.generate_rust(
                helper_function,
                config=helper_config,
                function_name=helper_name,
                function_index=1,
                shared_helper_nodes=(),
                emit_crate_header=False,
                emit_docs=False,
                function_keyword="fn",
            )
            helper_sources.append(helper_codegen.source.rstrip())
            helper_nodes.extend(helper_function.nodes)
            max_workspace = max(max_workspace, helper_codegen.workspace_size)

    return _SingleShootingHelperBundle(
        dynamics_name=dynamics_name,
        dynamics_jvp_name=dynamics_jvp_name,
        stage_cost_name=stage_cost_name,
        terminal_cost_name=terminal_cost_name,
        dynamics_vjp_x_name=dynamics_vjp_x_name,
        dynamics_vjp_u_name=dynamics_vjp_u_name,
        dynamics_vjp_x_jvp_name=dynamics_vjp_x_jvp_name,
        dynamics_vjp_u_jvp_name=dynamics_vjp_u_jvp_name,
        stage_cost_grad_x_name=stage_cost_grad_x_name,
        stage_cost_grad_u_name=stage_cost_grad_u_name,
        stage_cost_grad_x_jvp_name=stage_cost_grad_x_jvp_name,
        stage_cost_grad_u_jvp_name=stage_cost_grad_u_jvp_name,
        terminal_cost_grad_x_name=terminal_cost_grad_x_name,
        terminal_cost_grad_x_jvp_name=terminal_cost_grad_x_jvp_name,
        sources=tuple(helper_sources),
        helper_nodes=tuple(helper_nodes),
        max_workspace_size=max_workspace,
    )


def _build_single_shooting_input_specs(
    problem,
    *,
    include_hvp: bool,
) -> tuple[_ArgSpec, ...]:
    specs = [
        _ArgSpec(
            raw_name=problem.initial_state_name,
            rust_name=sanitize_ident(problem.initial_state_name),
            rust_label=f'"{problem.initial_state_name}"',
            doc_description="initial state vector for the single-shooting rollout",
            size=problem.state_size,
        ),
        _ArgSpec(
            raw_name=problem.control_sequence_name,
            rust_name=sanitize_ident(problem.control_sequence_name),
            rust_label=f'"{problem.control_sequence_name}"',
            doc_description="packed control-sequence slice laid out stage-major over the horizon",
            size=problem.horizon * problem.control_size,
        ),
        _ArgSpec(
            raw_name=problem.parameter_name,
            rust_name=sanitize_ident(problem.parameter_name),
            rust_label=f'"{problem.parameter_name}"',
            doc_description="shared parameter slice used at every stage and terminal evaluation",
            size=problem.parameter_size,
        ),
    ]
    if include_hvp:
        specs.append(
            _ArgSpec(
                raw_name=f"v_{problem.control_sequence_name}",
                rust_name=sanitize_ident(f"v_{problem.control_sequence_name}"),
                rust_label=f'"v_{problem.control_sequence_name}"',
                doc_description="packed control-direction vector for the single-shooting HVP",
                size=problem.horizon * problem.control_size,
            )
        )
    return tuple(specs)


def _build_single_shooting_output_specs(
    problem,
    *,
    include_cost: bool,
    include_gradient: bool,
    include_hvp: bool,
    include_states: bool,
) -> tuple[_ArgSpec, ...]:
    specs: list[_ArgSpec] = []
    if include_cost:
        specs.append(
            _ArgSpec(
                raw_name="cost",
                rust_name="cost",
                rust_label='"cost"',
                doc_description="scalar rollout cost",
                size=1,
            )
        )
    if include_gradient:
        specs.append(
            _ArgSpec(
                raw_name=f"gradient_{problem.control_sequence_name}",
                rust_name=sanitize_ident(f"gradient_{problem.control_sequence_name}"),
                rust_label=f'"gradient_{problem.control_sequence_name}"',
                doc_description="packed gradient with respect to the control sequence",
                size=problem.horizon * problem.control_size,
            )
        )
    if include_hvp:
        specs.append(
            _ArgSpec(
                raw_name=f"hvp_{problem.control_sequence_name}",
                rust_name=sanitize_ident(f"hvp_{problem.control_sequence_name}"),
                rust_label=f'"hvp_{problem.control_sequence_name}"',
                doc_description="packed Hessian-vector product with respect to the control sequence",
                size=problem.horizon * problem.control_size,
            )
        )
    if include_states:
        specs.append(
            _ArgSpec(
                raw_name="x_traj",
                rust_name="x_traj",
                rust_label='"x_traj"',
                doc_description="packed rollout state trajectory",
                size=(problem.horizon + 1) * problem.state_size,
            )
        )
    return tuple(specs)


def _compose_single_shooting_helper_base_name(crate_name: str | None, problem_name: str) -> str:
    base_label = sanitize_ident(problem_name)
    if crate_name is None:
        return base_label
    crate_label = sanitize_ident(crate_name)
    if base_label == crate_label or base_label.startswith(f"{crate_label}_"):
        return base_label
    return sanitize_ident(f"{crate_label}_{base_label}")


def _emit_half_open_range(index_expr: str, block_size: int) -> str:
    if block_size == 1:
        if index_expr == "0":
            return "0..1"
        return f"{index_expr}..({index_expr} + 1)"
    if index_expr == "0":
        return f"0..{block_size}"
    start = f"({index_expr} * {block_size})"
    end = f"(({index_expr} + 1) * {block_size})"
    return f"{start}..{end}"


def _emit_single_shooting_control_slice(sequence_name: str, index_expr: str, control_size: int) -> str:
    return f"&{sequence_name}[{_emit_half_open_range(index_expr, control_size)}]"


def _emit_single_shooting_stage_range(index_expr: str, block_size: int) -> str:
    return _emit_half_open_range(index_expr, block_size)


def _emit_small_accumulate(target_name: str, source_name: str, size: int, *, indent: str = "") -> list[str]:
    lines: list[str] = []
    for index in range(size):
        lines.append(f"{indent}{target_name}[{index}] += {source_name}[{index}];")
    return lines
