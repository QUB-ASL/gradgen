"""Utilities for composing function-like stages into one pipeline."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ._rust_codegen.codegen import generate_rust as _generate_rust
from ._rust_codegen.models import RustCodegenResult, RustProjectResult
from ._rust_codegen.project import _create_python_interface_project
from ._rust_codegen.project_support import _derive_python_function_name
from ._rust_codegen.project_support import _render_metadata_json
from ._rust_codegen.project_support import _render_multi_function_lib
from ._rust_codegen.project_support import _run_cargo_build
from ._rust_codegen.project_support import _try_run_cargo_fmt
from ._rust_codegen.templates import _get_template
from ._rust_codegen.validation import resolve_backend_config
from ._rust_codegen.validation import validate_backend_mode
from ._rust_codegen.validation import validate_scalar_type
from ._rust_codegen.naming import sanitize_ident
from .function import Function
from .map_zip import ReducedFunction
from .map_zip import ZippedFunction
from .sx import SX
from .sx import SXVector

FunctionArg = SX | SXVector
FunctionLike = Any


@dataclass(frozen=True, slots=True)
class _ComposerStage:
    """One stage in a composed pipeline."""

    function: FunctionLike
    feed_arg: str | None = None


class FunctionComposer:
    """Build a linear composition of function-like stages.

    The composer preserves staged wrappers such as
    :class:`~gradgen.map_zip.ReducedFunction` and
    :class:`~gradgen.map_zip.ZippedFunction` when Rust code is generated.
    """

    __slots__ = ("_stages",)

    def __init__(self, function: FunctionLike):
        """Start a new composer from the first stage."""
        self._stages = (_ComposerStage(function=function),)

    @property
    def stages(self) -> tuple[_ComposerStage, ...]:
        """Return the configured composition stages."""
        return self._stages

    def feed_into(
        self,
        function: FunctionLike,
        *,
        arg: str,
    ) -> FunctionComposer:
        """Append a stage that receives the previous stage output."""
        _validate_stage_connection(self._stages[-1].function, function, arg)
        return _append_stage(self, function, arg)

    def compose(self, name: str | None = None) -> FunctionComposition:
        """Finalize the composer into a function-like composed pipeline."""
        return FunctionComposition(
            name=name or _default_composition_name(self._stages),
            stages=self._stages,
        )


@dataclass(frozen=True, slots=True)
class FunctionComposition:
    """A composed function-like pipeline built from staged stages."""

    name: str
    stages: tuple[_ComposerStage, ...]

    @property
    def input_names(self) -> tuple[str, ...]:
        """Return the free input names of the composed pipeline."""
        inputs = self._build_input_bindings()
        return tuple(inputs.keys())

    @property
    def output_names(self) -> tuple[str, ...]:
        """Return the output names of the composed pipeline."""
        return _function_like_to_function(
            self.stages[-1].function
        ).output_names

    @property
    def nodes(self):
        """Return dependency nodes for shared-helper discovery."""
        return self.to_function().nodes

    def to_function(self, name: str | None = None) -> Function:
        """Lower the composed pipeline into a symbolic :class:`Function`."""
        inputs = self._build_input_bindings()
        outputs = self._build_symbolic_outputs(inputs)
        function = Function(
            name or self.name,
            tuple(inputs.values()),
            outputs,
            input_names=tuple(inputs.keys()),
            output_names=self.output_names,
        )
        return function

    def __call__(self, *args, **kwargs):
        """Evaluate the composed pipeline through its symbolic expansion."""
        return self.to_function()(*args, **kwargs)

    def gradient(self, name: str | None = None) -> Function:
        """Return a gradient function for the composed pipeline."""
        return self.to_function().gradient(name=name)

    def hessian(self, wrt_index: int = 0, name: str | None = None) -> Function:
        """Return a Hessian function for the composed pipeline."""
        return self.to_function().hessian(wrt_index=wrt_index, name=name)

    def hvp(
        self,
        wrt_index: int = 0,
        name: str | None = None,
        tangent_name: str | None = None,
    ) -> Function:
        """Return a Hessian-vector product function for the pipeline."""
        return self.to_function().hvp(
            wrt_index=wrt_index,
            name=name,
            tangent_name=tangent_name,
        )

    def generate_rust(
        self,
        *,
        config=None,
        function_name: str | None = None,
        backend_mode: str = "std",
        scalar_type: str = "f64",
    ) -> RustCodegenResult:
        """Generate Rust that calls each composed stage in sequence."""
        resolved_config = resolve_backend_config(
            config,
            function_name=function_name or self.name,
            backend_mode=backend_mode,
            scalar_type=scalar_type,
        )
        validate_backend_mode(resolved_config.backend_mode)
        validate_scalar_type(resolved_config.scalar_type)
        if resolved_config.backend_mode == "std":
            resolved_math_library = None
        else:
            resolved_math_library = "libm"

        input_bindings = self._build_input_bindings()
        output_values = self._build_symbolic_outputs(input_bindings)
        input_names = tuple(input_bindings.keys())
        input_sizes = tuple(
            _arg_size(value) for value in input_bindings.values()
        )
        output_names = self.output_names
        output_sizes = tuple(_arg_size(value) for value in output_values)
        pipeline_stage_indices = _codegen_stage_indices(self.stages)
        stage_names = ["" for _ in self.stages]
        stage_codegens = [None for _ in self.stages]
        plain_function_nodes = tuple(
            node
            for stage in self.stages
            if isinstance(stage.function, Function)
            for node in stage.function.nodes
        )
        for source_index, stage_index in enumerate(pipeline_stage_indices):
            stage = self.stages[stage_index]
            stage_name = _stage_function_name(stage, source_index)
            stage_names[stage_index] = stage_name
            stage_codegens[stage_index] = _generate_rust(
                stage.function,
                config=resolved_config,
                function_name=stage_name,
                backend_mode=resolved_config.backend_mode,
                scalar_type=resolved_config.scalar_type,
                function_index=source_index,
                shared_helper_nodes=(
                    plain_function_nodes if source_index == 0 else ()
                ),
            )
        ordered_codegens = tuple(
            stage_codegens[index] for index in pipeline_stage_indices
        )
        stage_workspace_sizes = tuple(
            codegen.workspace_size for codegen in stage_codegens
        )
        total_workspace = _composition_workspace_size(
            stage_workspace_sizes,
            output_sizes,
        )
        wrapper_source = _render_composed_wrapper_source(
            function_name=resolved_config.function_name or self.name,
            stage_names=tuple(stage_names),
            stages=self.stages,
            stage_codegens=tuple(stage_codegens),
            input_names=input_names,
            input_sizes=input_sizes,
            output_names=output_names,
            output_sizes=output_sizes,
            workspace_size=total_workspace,
            scalar_type=resolved_config.scalar_type,
        )
        wrapper_codegen = RustCodegenResult(
            source=wrapper_source,
            python_name=_derive_python_function_name(
                resolved_config.function_name or self.name,
                resolved_config.crate_name,
            ),
            function_name=resolved_config.function_name or self.name,
            workspace_size=total_workspace,
            input_names=input_names,
            input_sizes=input_sizes,
            output_names=output_names,
            output_sizes=output_sizes,
            backend_mode=resolved_config.backend_mode,
            scalar_type=resolved_config.scalar_type,
            math_library=resolved_math_library,
        )
        source = _render_multi_function_lib(
            (*ordered_codegens, wrapper_codegen),
            resolved_config,
        )
        return RustCodegenResult(
            source=source,
            python_name=wrapper_codegen.python_name,
            function_name=wrapper_codegen.function_name,
            workspace_size=wrapper_codegen.workspace_size,
            input_names=wrapper_codegen.input_names,
            input_sizes=wrapper_codegen.input_sizes,
            output_names=wrapper_codegen.output_names,
            output_sizes=wrapper_codegen.output_sizes,
            backend_mode=wrapper_codegen.backend_mode,
            scalar_type=wrapper_codegen.scalar_type,
            math_library=wrapper_codegen.math_library,
        )

    def create_rust_project(
        self,
        path: str,
        *,
        config=None,
        crate_name: str | None = None,
        function_name: str | None = None,
        backend_mode: str = "std",
        scalar_type: str = "f64",
    ) -> RustProjectResult:
        """Create a Rust crate containing the composed pipeline."""
        resolved_config = resolve_backend_config(
            config,
            crate_name=crate_name,
            function_name=function_name or self.name,
            backend_mode=backend_mode,
            scalar_type=scalar_type,
        )
        validate_backend_mode(resolved_config.backend_mode)
        validate_scalar_type(resolved_config.scalar_type)

        project_dir = Path(path).expanduser().resolve()
        crate = sanitize_ident(resolved_config.crate_name or self.name)
        resolved_math_library = (
            "libm" if resolved_config.backend_mode == "no_std" else None
        )
        resolved_math_library_version = (
            "0.2" if resolved_math_library == "libm" else None
        )
        codegen = self.generate_rust(
            config=resolved_config,
            function_name=function_name,
            backend_mode=backend_mode,
            scalar_type=scalar_type,
        )

        src_dir = project_dir / "src"
        cargo_toml = project_dir / "Cargo.toml"
        readme = project_dir / "README.md"
        metadata_json = project_dir / "metadata.json"
        lib_rs = src_dir / "lib.rs"

        src_dir.mkdir(parents=True, exist_ok=True)
        cargo_toml.write_text(
            _get_template("Cargo.toml.j2").render(
                crate_name=crate,
                backend_mode=resolved_config.backend_mode,
                scalar_type=resolved_config.scalar_type,
                math_library=resolved_math_library,
                math_library_version=resolved_math_library_version,
            ),
            encoding="utf-8",
        )
        readme.write_text(
            _get_template("rust_project_README.md.j2").render(
                crate_name=crate,
                codegen=codegen,
                backend_mode=resolved_config.backend_mode,
                scalar_type=resolved_config.scalar_type,
                math_library=resolved_math_library,
                enable_python_interface=(
                    resolved_config.enable_python_interface
                ),
                python_interface_project_name=(
                    f"{crate}_python"
                    if resolved_config.enable_python_interface
                    else None
                ),
            ),
            encoding="utf-8",
        )
        metadata_json.write_text(
            _render_metadata_json(crate, (codegen,)),
            encoding="utf-8",
        )
        lib_rs.write_text(codegen.source, encoding="utf-8")
        stale_pyproject = project_dir / "pyproject.toml"
        if stale_pyproject.exists():
            stale_pyproject.unlink()
        if resolved_config.build_crate:
            _run_cargo_build(project_dir)
        python_interface = None
        if resolved_config.enable_python_interface:
            python_interface = _create_python_interface_project(
                low_level_project_dir=project_dir,
                low_level_crate_name=crate,
                codegens=(codegen,),
                build_python_interface=resolved_config.build_python_interface,
            )
        _try_run_cargo_fmt(project_dir)

        return RustProjectResult(
            project_dir=project_dir,
            cargo_toml=cargo_toml,
            readme=readme,
            metadata_json=metadata_json,
            lib_rs=lib_rs,
            codegen=codegen,
            python_interface=python_interface,
        )

    def _build_input_bindings(self) -> OrderedDict[str, FunctionArg]:
        bindings: OrderedDict[str, FunctionArg] = OrderedDict()
        for stage_index, stage in enumerate(self.stages):
            function = _function_like_to_function(stage.function)
            for formal, input_name in zip(
                function.inputs, function.input_names
            ):
                if stage_index > 0 and input_name == stage.feed_arg:
                    continue
                existing = bindings.get(input_name)
                if existing is None:
                    bindings[input_name] = _make_symbolic_input_like(
                        formal, input_name
                    )
                    continue
                if not _same_arg_shape(existing, formal):
                    raise ValueError(
                        "shared input names must have the same shape"
                    )
        return bindings

    def _build_symbolic_outputs(
        self,
        bindings: OrderedDict[str, FunctionArg],
    ) -> tuple[FunctionArg, ...]:
        current_value: FunctionArg | None = None
        final_outputs: tuple[FunctionArg, ...] | None = None

        for stage_index, stage in enumerate(self.stages):
            function = _function_like_to_function(stage.function)
            stage_args: list[FunctionArg] = []
            for formal, input_name in zip(
                function.inputs, function.input_names
            ):
                if stage_index > 0 and input_name == stage.feed_arg:
                    if current_value is None:
                        raise ValueError(
                            "composed stages are missing a connected output"
                        )
                    stage_args.append(current_value)
                    continue
                stage_args.append(bindings[input_name])

            stage_result = function(*stage_args)
            if stage_index < len(self.stages) - 1:
                if len(function.outputs) != 1:
                    raise ValueError(
                        "non-terminal composed stages must produce one output"
                    )
                current_value = _coerce_single_output(stage_result)
                continue

            if len(function.outputs) == 1:
                final_outputs = (_coerce_single_output(stage_result),)
            else:
                if not isinstance(stage_result, tuple):
                    raise ValueError(
                        "expected multiple outputs from final stage"
                    )
                final_outputs = tuple(stage_result)

        if final_outputs is None:
            raise ValueError("composition requires at least one stage")
        return final_outputs


def _append_stage(
    composer: FunctionComposer,
    function: FunctionLike,
    arg: str,
) -> FunctionComposer:
    stages = (
        *composer.stages,
        _ComposerStage(function=function, feed_arg=arg),
    )
    new_composer = FunctionComposer(composer.stages[0].function)
    new_composer._stages = stages
    return new_composer


def _validate_stage_connection(
    previous: FunctionLike,
    function: FunctionLike,
    arg: str,
) -> None:
    previous_function = _function_like_to_function(previous)
    next_function = _function_like_to_function(function)
    if len(previous_function.outputs) != 1:
        raise ValueError(
            "composed stages must produce one output before feeding forward"
        )
    if arg not in next_function.input_names:
        raise ValueError(f"unknown composed input argument {arg!r}")
    previous_output = previous_function.outputs[0]
    next_input = next_function.inputs[next_function.input_names.index(arg)]
    if not _same_arg_shape(previous_output, next_input):
        raise ValueError("connected composed stages must share the same shape")


def _function_like_to_function(function: FunctionLike) -> Function:
    if isinstance(function, Function):
        return function
    if hasattr(function, "to_function"):
        return function.to_function()
    raise TypeError("function-like stage must provide to_function()")


def _default_composition_name(stages: tuple[_ComposerStage, ...]) -> str:
    names = [
        sanitize_ident(getattr(stage.function, "name", f"stage_{index}"))
        for index, stage in enumerate(stages)
    ]
    return "_".join(names) if names else "composed"


def _codegen_stage_indices(
    stages: tuple[_ComposerStage, ...],
) -> tuple[int, ...]:
    host_index = next(
        (
            index
            for index, stage in enumerate(stages)
            if isinstance(stage.function, Function)
        ),
        0,
    )
    ordered = [host_index]
    ordered.extend(
        index for index in range(len(stages)) if index != host_index
    )
    return tuple(ordered)


def _stage_function_name(stage: _ComposerStage, index: int) -> str:
    base_name = sanitize_ident(
        getattr(stage.function, "name", f"stage_{index}")
    )
    return f"{base_name}_{index}"


def _same_arg_shape(left: FunctionArg, right: FunctionArg) -> bool:
    if isinstance(left, SX) and isinstance(right, SX):
        return True
    if isinstance(left, SXVector) and isinstance(right, SXVector):
        return len(left) == len(right)
    return False


def _make_symbolic_input_like(
    value: FunctionArg, base_name: str
) -> FunctionArg:
    if isinstance(value, SX):
        return SX.sym(base_name)
    return SXVector.sym(base_name, len(value))


def _arg_size(value: FunctionArg) -> int:
    if isinstance(value, SX):
        return 1
    return len(value)


def _coerce_single_output(value: object) -> FunctionArg:
    if isinstance(value, (SX, SXVector)):
        return value
    raise TypeError("composed stages require symbolic outputs")


def _composition_workspace_size(
    stage_workspace_sizes: tuple[int, ...],
    output_sizes: tuple[int, ...],
) -> int:
    if not stage_workspace_sizes:
        return 0
    intermediate_outputs = (
        sum(output_sizes[:-1]) if len(output_sizes) > 1 else 0
    )
    return sum(stage_workspace_sizes) + intermediate_outputs


def _render_composed_wrapper_source(
    function_name: str,
    stage_names: tuple[str, ...],
    stages: tuple[_ComposerStage, ...],
    stage_codegens: tuple[RustCodegenResult, ...],
    input_names: tuple[str, ...],
    input_sizes: tuple[int, ...],
    output_names: tuple[str, ...],
    output_sizes: tuple[int, ...],
    workspace_size: int,
    scalar_type: str,
) -> str:
    stage_output_sizes = tuple(
        sum(
            _arg_size(output)
            for output in _function_like_to_function(stage.function).outputs
        )
        for stage in stages
    )
    lines: list[str] = [
        f"/// Evaluate the composed function `{function_name}`.",
        "///",
        f"pub fn {function_name}(",
    ]
    for name, size in zip(input_names, input_sizes):
        lines.append(
            f"    {name}: &[{scalar_type}],  // Expected length: {size}."
        )
    for name, size in zip(output_names, output_sizes):
        lines.append(
            f"    {name}: &mut [{scalar_type}],  // Expected length: {size}."
        )
    lines.append(
        f"    work: &mut [{scalar_type}],  // Expected length: at least "
        f"{workspace_size}."
    )
    lines.append(") -> Result<(), GradgenError> {")
    if workspace_size > 0:
        lines.append(
            f"    if work.len() < {workspace_size} {{ return Err("
            f'GradgenError::WorkspaceTooSmall("work expected at least '
            f'{workspace_size}")); }}'
        )
    for name, size in zip(input_names, input_sizes):
        lines.append(
            f"    if {name}.len() != {size} {{ return Err("
            f'GradgenError::InputTooSmall("{name} expected length '
            f'{size}")); }}'
        )
    for name, size in zip(output_names, output_sizes):
        lines.append(
            f"    if {name}.len() != {size} {{ return Err("
            f'GradgenError::OutputTooSmall("{name} expected length '
            f'{size}")); }}'
        )

    lines.append("    let mut remaining_work = work;")
    for index, stage_name in enumerate(stage_names):
        stage_work_size = stage_codegens[index].workspace_size
        if index < len(stage_names) - 1:
            lines.append(
                f"    let (stage_{index}_work, rest_{index}) = "
                f"remaining_work.split_at_mut({stage_work_size});"
            )
        else:
            lines.append(
                f"    let (stage_{index}_work, _) = "
                f"remaining_work.split_at_mut({stage_work_size});"
            )

        stage = stages[index]
        stage_function = _function_like_to_function(stage.function)
        stage_input_refs: list[str] = []
        for formal, input_name in zip(
            stage_function.inputs, stage_function.input_names
        ):
            if index > 0 and input_name == stage.feed_arg:
                stage_input_refs.append(f"stage_{index - 1}_out")
                continue
            stage_input_refs.append(input_name)

        if index < len(stage_names) - 1:
            buffer_size = stage_output_sizes[index]
            lines.append(
                f"    let (stage_{index}_out, rest_out_{index}) = "
                f"rest_{index}.split_at_mut({buffer_size});"
            )
            stage_output_refs = [f"stage_{index}_out"]
            remaining_ref = f"rest_out_{index}"
        else:
            stage_output_refs = list(output_names)
            remaining_ref = f"rest_{index}"

        call_args = [
            *stage_input_refs,
            *stage_output_refs,
            f"stage_{index}_work",
        ]
        lines.append(f"    {stage_name}(" + ", ".join(call_args) + ")?;")
        if index < len(stage_names) - 1:
            lines.append(f"    remaining_work = {remaining_ref};")

    lines.append("    Ok(())")
    lines.append("}")
    return "\n".join(lines) + "\n"
