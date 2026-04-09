"""Staged function composition with packed-parameter support."""

from __future__ import annotations

from dataclasses import dataclass, replace
import re
from typing import Iterable

from .function import Function
from ._staged import _create_rust_project
from ._staged import _generate_rust
from ._staged import _simplify_function
from .sx import SX, SXVector

FunctionArg = SX | SXVector
StageValue = (
    SX | SXVector | float | int | list[object] | tuple[object, ...] | None
)
ChainItem = Function | tuple[Function, StageValue]


@dataclass(frozen=True, slots=True)
class _PackedParameter:
    """One bound stage-parameter declaration."""

    kind: str
    size: int
    formal: FunctionArg
    values: tuple[float, ...] = ()

    @property
    def symbolic_size(self) -> int:
        """Return the packed runtime scalar count for this parameter."""
        if self.kind == "symbolic":
            return self.size
        return 0


@dataclass(frozen=True, slots=True)
class _SingleStage:
    """One explicit stage application."""

    function: Function
    parameter: _PackedParameter

    @property
    def stage_count(self) -> int:
        """Return the number of concrete stage applications."""
        return 1

    @property
    def symbolic_parameter_size(self) -> int:
        """Return the number of packed symbolic parameter scalars."""
        return self.parameter.symbolic_size


@dataclass(frozen=True, slots=True)
class _RepeatStage:
    """One repeated stage application block."""

    function: Function
    parameters: tuple[_PackedParameter, ...]

    @property
    def stage_count(self) -> int:
        """Return the number of concrete stage applications."""
        return len(self.parameters)

    @property
    def symbolic_parameter_size(self) -> int:
        """Return the number of packed symbolic parameter scalars."""
        return sum(parameter.symbolic_size for parameter in self.parameters)


StageStep = _SingleStage | _RepeatStage


@dataclass(frozen=True, slots=True)
class ComposedGradientFunction:
    """Derivative kernel for a finished ``ComposedFunction``.

    The expanded derivative is the Jacobian of the composed rollout with
    respect to the state input.
    """

    composed: ComposedFunction
    name: str
    simplification: int | str | None = None

    def to_function(self, name: str | None = None) -> Function:
        """Expand this staged derivative into a regular symbolic ``Function``.

        Returns:
            A symbolic function whose output is the flattened Jacobian of the
            finished rollout with respect to the state input.
        """
        gradient = self.composed.to_function().jacobian(
            0,
            name=name or self.name,
        )
        return _simplify_function(gradient, self.simplification)

    @property
    def nodes(self):
        """Return dependency nodes for shared-helper discovery."""
        return self.to_function().nodes

    def generate_rust(
        self,
        *,
        config=None,
        function_name: str | None = None,
        backend_mode: str = "std",
        scalar_type: str = "f64",
    ):
        """Generate compact Rust for the staged derivative kernel."""
        return _generate_rust(
            self,
            config=config,
            function_name=function_name,
            backend_mode=backend_mode,
            scalar_type=scalar_type,
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
    ):
        """Create a Rust crate containing the staged derivative kernel."""
        return _create_rust_project(
            self,
            path,
            config=config,
            crate_name=crate_name,
            function_name=function_name,
            backend_mode=backend_mode,
            scalar_type=scalar_type,
        )


@dataclass(frozen=True, slots=True)
class ComposedJacobianFunction:
    """Jacobian kernel for a finished ``ComposedFunction``."""

    composed: ComposedFunction
    name: str
    simplification: int | str | None = None

    def to_function(self, name: str | None = None) -> Function:
        """Expand this staged Jacobian into a regular symbolic ``Function``."""
        packed_parameters = (
            SXVector.sym(
                self.composed.parameter_name,
                self.composed.parameter_size,
            )
            if self.composed.parameter_size > 0
            else None
        )
        jacobian = self._build_symbolic_jacobian(packed_parameters)
        output_name = f"jacobian_{self.composed.output_names[0]}"
        function = Function(
            name or self.name,
            self.composed._compiled_inputs(packed_parameters),
            [jacobian],
            input_names=self.composed.input_names,
            output_names=(output_name,),
        )
        return _simplify_function(function, self.simplification)

    @property
    def nodes(self):
        """Return dependency nodes for shared-helper discovery."""
        return self.to_function().nodes

    def generate_rust(
        self,
        *,
        config=None,
        function_name: str | None = None,
        backend_mode: str = "std",
        scalar_type: str = "f64",
    ):
        """Generate compact Rust for the staged Jacobian kernel."""
        return _generate_rust(
            self,
            config=config,
            function_name=function_name,
            backend_mode=backend_mode,
            scalar_type=scalar_type,
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
    ):
        """Create a Rust crate containing the staged Jacobian kernel."""
        return _create_rust_project(
            self,
            path,
            config=config,
            crate_name=crate_name,
            function_name=function_name,
            backend_mode=backend_mode,
            scalar_type=scalar_type,
        )

    def _build_symbolic_jacobian(
        self, packed_parameters: SXVector | None
    ) -> SXVector:
        """Build the flattened Jacobian using staged symbolic backprop."""
        state_size = _arg_size(self.composed.state_input)

        state: FunctionArg = self.composed.state_input
        parameter_offset = 0
        applications: list[tuple[Function, FunctionArg, FunctionArg]] = []

        for step in self.composed.steps:
            if isinstance(step, _SingleStage):
                parameter_arg, parameter_offset = _resolve_compiled_parameter(
                    step.parameter,
                    packed_parameters,
                    parameter_offset,
                )
                applications.append((step.function, state, parameter_arg))
                state = _coerce_single_output(
                    step.function(state, parameter_arg)
                )
                continue

            for parameter in step.parameters:
                parameter_arg, parameter_offset = _resolve_compiled_parameter(
                    parameter,
                    packed_parameters,
                    parameter_offset,
                )
                applications.append((step.function, state, parameter_arg))
                state = _coerce_single_output(
                    step.function(state, parameter_arg)
                )

        if parameter_offset != self.composed.parameter_size:
            raise AssertionError(
                "packed parameter offsets did not consume the expected size"
            )

        rows: list[SX] = []
        for row_index in range(state_size):
            if state_size == 1:
                current_lambda: FunctionArg = SX.const(1.0)
            else:
                current_lambda = SXVector(
                    tuple(
                        SX.const(1.0) if index == row_index else SX.const(0.0)
                        for index in range(state_size)
                    )
                )

            for function, state_arg, parameter_arg in reversed(applications):
                current_lambda = _coerce_single_output(
                    function.vjp(wrt_index=0)(
                        state_arg,
                        parameter_arg,
                        current_lambda,
                    )
                )

            rows.extend(_flatten_symbolic_arg(current_lambda))

        return SXVector(tuple(rows))


@dataclass(frozen=True, slots=True)
class ComposedJointFunction:
    """Joint kernel for a finished ``ComposedFunction``."""

    composed: ComposedFunction
    name: str
    components: tuple[str, ...]
    wrt_index: int = 0
    simplification: int | str | None = None

    def to_function(self, name: str | None = None) -> Function:
        """Expand this staged joint kernel into a regular symbolic function."""
        self.composed._require_finished()
        if self.wrt_index != 0:
            raise ValueError(
                "ComposedFunction joint kernels only support wrt_index=0"
            )
        resolved_components = _resolve_composed_joint_components(
            self.components
        )
        packed_parameters = (
            SXVector.sym(
                self.composed.parameter_name,
                self.composed.parameter_size,
            )
            if self.composed.parameter_size > 0
            else None
        )
        primal_function = self.composed.to_function()
        jacobian_function = self.composed.jacobian().to_function()
        outputs: list[FunctionArg] = []
        output_names: list[str] = []

        for component in resolved_components:
            if component == "f":
                outputs.extend(primal_function.outputs)
                output_names.extend(primal_function.output_names)
                continue
            if component == "jf":
                outputs.extend(jacobian_function.outputs)
                output_names.extend(jacobian_function.output_names)
                continue
            raise AssertionError(
                f"unexpected composed joint component {component!r}"
            )

        function = Function(
            name or self.name,
            self.composed._compiled_inputs(packed_parameters),
            outputs,
            input_names=self.composed.input_names,
            output_names=tuple(output_names),
        )
        return _simplify_function(function, self.simplification)

    @property
    def nodes(self):
        """Return dependency nodes for shared-helper discovery."""
        return self.to_function().nodes

    def generate_rust(
        self,
        *,
        config=None,
        function_name: str | None = None,
        backend_mode: str = "std",
        scalar_type: str = "f64",
    ):
        """Generate compact Rust for the staged joint kernel."""
        return _generate_rust(
            self,
            config=config,
            function_name=function_name,
            backend_mode=backend_mode,
            scalar_type=scalar_type,
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
    ):
        """Create a Rust crate containing the staged joint kernel."""
        return _create_rust_project(
            self,
            path,
            config=config,
            crate_name=crate_name,
            function_name=function_name,
            backend_mode=backend_mode,
            scalar_type=scalar_type,
        )


@dataclass(frozen=True, slots=True)
class ComposedFunction:
    """Staged composition of vector state transforms.

    A ``ComposedFunction`` represents a chain such as

    ``x -> G1(x, p1) -> G2(..., p2) -> ... -> GN(...)``

    while keeping the stage structure available for Rust code generation.
    Symbolic stage parameters are packed into one additional runtime input
    slice named ``parameter_name``.
    """

    name: str
    state_input: FunctionArg
    input_name: str | None = None
    parameter_name: str = "parameters"
    simplification: int | str | None = None
    steps: tuple[StageStep, ...] = ()
    finished: bool = False

    def __post_init__(self) -> None:
        """Validate and normalize constructor inputs."""
        _validate_state_input(self.state_input)
        object.__setattr__(
            self,
            "input_name",
            self.input_name or _infer_input_name(self.state_input, "x"),
        )

    @property
    def parameter_size(self) -> int:
        """Return the packed runtime parameter length."""
        return sum(step.symbolic_parameter_size for step in self.steps)

    @property
    def stage_count(self) -> int:
        """Return the total number of concrete stage applications."""
        return sum(step.stage_count for step in self.steps)

    @property
    def inputs(self) -> tuple[FunctionArg, ...]:
        """Return the compiled symbolic inputs for this composition."""
        compiled = self._compiled_inputs()
        return compiled

    @property
    def outputs(self) -> tuple[FunctionArg, ...]:
        """Return the compiled symbolic outputs for this composition."""
        return self.to_function().outputs

    @property
    def nodes(self):
        """Return dependency nodes for shared-helper discovery."""
        return self.to_function().nodes

    @property
    def input_names(self) -> tuple[str, ...]:
        """Return the compiled symbolic input names."""
        names = (self.input_name,)
        if self.parameter_size > 0:
            names = (*names, self.parameter_name)
        return names

    @property
    def output_names(self) -> tuple[str, ...]:
        """Return the compiled symbolic output names."""
        self._require_finished()
        return ("y",)

    def then(
        self, function: Function, *, p: StageValue = None
    ) -> ComposedFunction:
        """Append one explicit state-transform stage."""
        self._ensure_not_finished()
        _validate_stage_function(function, self.state_input)
        parameter = _normalize_stage_parameter(
            function.inputs[1], p, role="stage"
        )
        return replace(
            self, steps=(*self.steps, _SingleStage(function, parameter))
        )

    def chain(self, stages: Iterable[ChainItem]) -> ComposedFunction:
        """Append a heterogeneous list of stages."""
        self._ensure_not_finished()
        steps = list(self.steps)
        for item in stages:
            function, parameter_value = _parse_chain_item(item)
            _validate_stage_function(function, self.state_input)
            parameter = _normalize_stage_parameter(
                function.inputs[1],
                parameter_value,
                role="stage",
            )
            _append_chained_stage(steps, function, parameter)
        return replace(self, steps=tuple(steps))

    def repeat(
        self, function: Function, *, params: Iterable[StageValue]
    ) -> ComposedFunction:
        """Append repeated applications of the same state-transform stage."""
        self._ensure_not_finished()
        _validate_stage_function(function, self.state_input)
        normalized = tuple(
            _normalize_stage_parameter(function.inputs[1], value, role="stage")
            for value in params
        )
        if not normalized:
            raise ValueError("repeat requires at least one parameter set")
        kinds = {parameter.kind for parameter in normalized}
        if len(kinds) != 1:
            raise ValueError(
                "repeat currently requires all parameter sets to "
                "be either fixed or symbolic"
            )
        return replace(
            self, steps=(*self.steps, _RepeatStage(function, normalized))
        )

    def finish(self) -> ComposedFunction:
        """Finalize the staged composition without adding another stage.

        Returns:
            A finished copy of the staged composition. The expanded function
            evaluates the repeated stage sequence and returns the final
            state.
        """
        self._ensure_not_finished()
        if not self.steps:
            raise ValueError(
                "finish requires at least one stage before finalizing"
            )
        return replace(self, finished=True)

    def to_function(self, name: str | None = None) -> Function:
        """Expand the staged composition into a symbolic ``Function``."""
        self._require_finished()
        packed_parameters = (
            SXVector.sym(self.parameter_name, self.parameter_size)
            if self.parameter_size > 0
            else None
        )
        compiled_output = self._build_symbolic_output(packed_parameters)
        inputs = list(self._compiled_inputs(packed_parameters))
        function = Function(
            name or self.name,
            inputs,
            [compiled_output],
            input_names=self.input_names,
            output_names=self.output_names,
        )
        return _simplify_function(function, self.simplification)

    def gradient(self, name: str | None = None) -> ComposedGradientFunction:
        """Return a staged derivative kernel with respect to the state input."""
        self._require_finished()
        gradient_name = name or f"{self.name}_gradient_{self.input_name}"
        return ComposedGradientFunction(
            self,
            gradient_name,
            self.simplification,
        )

    def jacobian(self, name: str | None = None) -> ComposedJacobianFunction:
        """Return a staged Jacobian kernel with respect to the state input."""
        self._require_finished()
        jacobian_name = name or f"{self.name}_jacobian_{self.input_name}"
        return ComposedJacobianFunction(
            self,
            jacobian_name,
            self.simplification,
        )

    def joint(
        self,
        components: Iterable[str],
        wrt_index: int = 0,
        name: str | None = None,
        simplify_joint: int | str | None = None,
    ) -> ComposedJointFunction:
        """Return a staged joint kernel for supported composed components."""
        self._require_finished()
        resolved_components = _resolve_composed_joint_components(components)
        joint_name = name or _composed_joint_function_name(
            self.name,
            self.input_name,
            resolved_components,
        )
        return ComposedJointFunction(
            self,
            joint_name,
            resolved_components,
            wrt_index=wrt_index,
            simplification=simplify_joint,
        )

    def generate_rust(
        self,
        *,
        config=None,
        function_name: str | None = None,
        backend_mode: str = "std",
        scalar_type: str = "f64",
    ):
        """Generate compact Rust for the staged primal kernel."""
        return _generate_rust(
            self,
            config=config,
            function_name=function_name,
            backend_mode=backend_mode,
            scalar_type=scalar_type,
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
    ):
        """Create a Rust crate containing the staged primal kernel."""
        return _create_rust_project(
            self,
            path,
            config=config,
            crate_name=crate_name,
            function_name=function_name,
            backend_mode=backend_mode,
            scalar_type=scalar_type,
        )

    def __call__(self, *args, **kwargs):
        """Evaluate or symbolically call the expanded composition."""
        return self.to_function()(*args, **kwargs)

    def _compiled_inputs(
        self, packed_parameters: SXVector | None = None
    ) -> tuple[FunctionArg, ...]:
        """Return the symbolic inputs used by the compiled expansion."""
        if packed_parameters is None and self.parameter_size > 0:
            packed_parameters = SXVector.sym(
                self.parameter_name, self.parameter_size
            )
        if packed_parameters is None:
            return (self.state_input,)
        return (self.state_input, packed_parameters)

    def _build_symbolic_output(
        self, packed_parameters: SXVector | None
    ) -> FunctionArg:
        """Build the final symbolic state using a packed parameter vector."""
        parameter_offset = 0
        state: FunctionArg = self.state_input

        for step in self.steps:
            if isinstance(step, _SingleStage):
                parameter_arg, parameter_offset = _resolve_compiled_parameter(
                    step.parameter,
                    packed_parameters,
                    parameter_offset,
                )
                state = _coerce_single_output(
                    step.function(state, parameter_arg)
                )
                continue

            for parameter in step.parameters:
                parameter_arg, parameter_offset = _resolve_compiled_parameter(
                    parameter,
                    packed_parameters,
                    parameter_offset,
                )
                state = _coerce_single_output(
                    step.function(state, parameter_arg)
                )
        if parameter_offset != self.parameter_size:
            raise AssertionError(
                "packed parameter offsets did not consume the expected size"
            )
        return state

    def _require_finished(self) -> None:
        """Raise when the composition has not been finalized yet."""
        if not self.finished:
            raise ValueError(
                "ComposedFunction is not finished; call finish() first"
            )

    def _ensure_not_finished(self) -> None:
        """Reject stage edits after the composition is finalized."""
        if self.finished:
            raise ValueError("ComposedFunction is already finished")


def _validate_state_input(state_input: FunctionArg) -> None:
    """Validate the symbolic state input used by a composition."""
    if isinstance(state_input, SX):
        if state_input.op != "symbol":
            raise ValueError(
                "ComposedFunction state_input must be a symbolic variable"
            )
        return
    for element in state_input:
        if not isinstance(element, SX) or element.op != "symbol":
            raise ValueError(
                "ComposedFunction state_input must contain "
                "symbolic variables only"
            )


def _infer_input_name(value: FunctionArg, fallback: str) -> str:
    """Infer a user-facing input name from symbolic leaves."""
    if isinstance(value, SX):
        return value.name or fallback
    if not value.elements:
        return fallback
    match = re.fullmatch(r"(.+)_\d+", value[0].name or "")
    if match is None:
        return fallback
    prefix = match.group(1)
    if all((element.name or "").startswith(f"{prefix}_") for element in value):
        return prefix
    return fallback


def _validate_stage_function(
    function: Function, state_input: FunctionArg
) -> None:
    """Validate one state-transform stage."""
    if len(function.inputs) != 2:
        raise ValueError(
            "stage functions must take exactly " "two inputs: (state, p)"
        )
    if len(function.outputs) != 1:
        raise ValueError("stage functions must produce exactly one output")
    _validate_matching_shape(
        function.inputs[0], state_input, "stage state input"
    )
    _validate_matching_shape(function.outputs[0], state_input, "stage output")


def _validate_matching_shape(
    actual: FunctionArg, expected: FunctionArg, label: str
) -> None:
    """Ensure two symbolic values share the same scalar/vector shape."""
    if isinstance(actual, SX) != isinstance(expected, SX):
        raise ValueError(
            f"{label} must match the composed state shape exactly"
        )
    if _arg_size(actual) != _arg_size(expected):
        raise ValueError(
            f"{label} must match the composed state dimension exactly"
        )


def _normalize_stage_parameter(
    formal: FunctionArg, value: StageValue, *, role: str
) -> _PackedParameter:
    """Normalize one stage parameter binding."""
    size = _arg_size(formal)

    if value is None:
        if size != 0:
            raise ValueError(f"{role} parameter value is required")
        return _PackedParameter("fixed", 0, formal)

    if isinstance(value, (SX, SXVector)):
        _validate_matching_shape(value, formal, f"{role} parameter")
        return _PackedParameter("symbolic", size, formal)

    if isinstance(formal, SX):
        numeric = _coerce_numeric_scalar(value, role)
        return _PackedParameter("fixed", 1, formal, (numeric,))

    numeric_values = _coerce_numeric_vector(value, size, role)
    return _PackedParameter("fixed", size, formal, numeric_values)


def _coerce_numeric_scalar(value: StageValue, role: str) -> float:
    """Coerce a numeric scalar parameter."""
    if isinstance(value, (int, float)):
        return float(value)
    if (
        isinstance(value, (list, tuple))
        and len(value) == 1
        and isinstance(value[0], (int, float))
    ):
        return float(value[0])
    raise TypeError(f"{role} scalar parameters must be numeric")


def _coerce_numeric_vector(
    value: StageValue, expected_size: int, role: str
) -> tuple[float, ...]:
    """Coerce a numeric vector parameter."""
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{role} vector parameters must be numeric sequences")
    if len(value) != expected_size:
        raise ValueError(
            f"{role} parameter length must match the "
            f"declared stage parameter size"
        )
    if not all(isinstance(item, (int, float)) for item in value):
        raise TypeError(
            f"{role} vector parameters must contain only numeric values"
        )
    return tuple(float(item) for item in value)


def _parse_chain_item(item: ChainItem) -> tuple[Function, StageValue]:
    """Normalize one ``chain([...])`` entry."""
    if isinstance(item, Function):
        return item, None
    if (
        isinstance(item, tuple)
        and len(item) == 2
        and isinstance(item[0], Function)
    ):
        return item[0], item[1]
    raise TypeError(
        "chain entries must be Function instances or "
        "(Function, parameter) tuples"
    )


def _append_chained_stage(
    steps: list[StageStep], function: Function, parameter: _PackedParameter
) -> None:
    """Append one normalized chain stage, merging adjacent repeats."""
    if not steps:
        steps.append(_SingleStage(function, parameter))
        return

    last = steps[-1]
    if isinstance(last, _SingleStage):
        if last.function == function and last.parameter.kind == parameter.kind:
            steps[-1] = _RepeatStage(
                function,
                (last.parameter, parameter),
            )
            return
        steps.append(_SingleStage(function, parameter))
        return

    if last.function == function and last.parameters[0].kind == parameter.kind:
        steps[-1] = _RepeatStage(function, (*last.parameters, parameter))
        return

    steps.append(_SingleStage(function, parameter))


def _resolve_compiled_parameter(
    parameter: _PackedParameter,
    packed_parameters: SXVector | None,
    offset: int,
) -> tuple[FunctionArg, int]:
    """Return the symbolic argument bound to one stage parameter."""
    if parameter.kind == "fixed":
        return (
            _parameter_arg_from_numeric_values(
                parameter.formal, parameter.values
            ),
            offset,
        )

    if packed_parameters is None:
        raise ValueError(
            "symbolic stage parameters require a packed parameter input"
        )

    if isinstance(parameter.formal, SX):
        return packed_parameters[offset], offset + 1
    return (
        packed_parameters[offset: offset + parameter.size],
        offset + parameter.size,
    )


def _parameter_arg_from_numeric_values(
    formal: FunctionArg, values: tuple[float, ...]
) -> FunctionArg:
    """Build a symbolic constant argument matching one stage parameter."""
    if isinstance(formal, SX):
        return SX.const(values[0]) if values else SX.const(0.0)
    return SXVector(tuple(SX.const(value) for value in values))


def _coerce_single_output(value: object) -> FunctionArg:
    """Normalize a one-output ``Function`` call result."""
    if isinstance(value, (SX, SXVector)):
        return value
    raise TypeError(
        "stage functions must evaluate to a single SX or SXVector output"
    )


def _flatten_symbolic_arg(arg: FunctionArg) -> tuple[SX, ...]:
    """Return scalar leaves from a symbolic scalar or vector argument."""
    if isinstance(arg, SX):
        return (arg,)
    return tuple(arg)


def _resolve_composed_joint_components(
    components: Iterable[str],
) -> tuple[str, ...]:
    """Validate joint components supported by ``ComposedFunction``."""
    resolved = tuple(components)
    if not resolved:
        raise ValueError("joint functions require at least one component")
    allowed = {"f", "jf"}
    if any(component not in allowed for component in resolved):
        raise ValueError(
            "ComposedFunction joint kernels currently support only 'f' and "
            "'jf' components"
        )
    if len(set(resolved)) != len(resolved):
        raise ValueError("joint components must be unique")
    return resolved


def _composed_joint_function_name(
    base_name: str,
    input_name: str,
    components: tuple[str, ...],
) -> str:
    """Build the default name for a composed joint kernel."""
    return f"{base_name}_joint_{'_'.join(components)}_{input_name}"


def _arg_size(arg: FunctionArg) -> int:
    """Return the scalar length of a scalar or vector argument."""
    if isinstance(arg, SX):
        return 1
    return len(arg)
