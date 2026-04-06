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
class _TerminalStage:
    """Terminal scalar stage."""

    function: Function
    parameter: _PackedParameter

    @property
    def symbolic_parameter_size(self) -> int:
        """Return the number of packed symbolic parameter scalars."""
        return self.parameter.symbolic_size


@dataclass(frozen=True, slots=True)
class ComposedGradientFunction:
    """Gradient kernel for a finished ``ComposedFunction``."""

    composed: ComposedFunction
    name: str
    simplification: int | str | None = None

    def to_function(self, name: str | None = None) -> Function:
        """Expand this staged gradient into a regular symbolic ``Function``."""
        gradient = self.composed.to_function().gradient(
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
        """Generate compact Rust for the staged gradient kernel."""
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
        """Create a Rust crate containing the staged gradient kernel."""
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
    """Staged composition of vector state transforms ending in a scalar output.

    A ``ComposedFunction`` represents a chain such as

    ``x -> G1(x, p1) -> G2(..., p2) -> ... -> h(..., pf)``

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
    terminal: _TerminalStage | None = None

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
        return sum(step.symbolic_parameter_size for step in self.steps) + (
            0
            if self.terminal is None
            else self.terminal.symbolic_parameter_size
        )

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
        terminal = self._require_terminal()
        return terminal.function.output_names

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
        composed = self
        for item in stages:
            function, parameter = _parse_chain_item(item)
            composed = composed.then(function, p=parameter)
        return composed

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

    def finish(
        self, function: Function, *, p: StageValue = None
    ) -> ComposedFunction:
        """Attach the terminal scalar stage."""
        self._ensure_not_finished()
        if not self.steps:
            raise ValueError(
                "finish requires at least one stage before "
                "the terminal scalar function"
            )
        _validate_terminal_function(function, self.state_input)
        parameter = _normalize_stage_parameter(
            function.inputs[1], p, role="terminal"
        )
        return replace(self, terminal=_TerminalStage(function, parameter))

    def to_function(self, name: str | None = None) -> Function:
        """Expand the staged composition into a symbolic ``Function``."""
        terminal = self._require_terminal()
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
            output_names=terminal.function.output_names,
        )
        return _simplify_function(function, self.simplification)

    def gradient(self, name: str | None = None) -> ComposedGradientFunction:
        """Return a staged gradient kernel with respect to the state input."""
        self._require_terminal()
        gradient_name = name or f"{self.name}_gradient_{self.input_name}"
        return ComposedGradientFunction(
            self,
            gradient_name,
            self.simplification,
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

    def __call__(self, *args):
        """Evaluate or symbolically call the expanded composition."""
        return self.to_function()(*args)

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

    def _build_symbolic_output(self, packed_parameters: SXVector | None) -> SX:
        """Build the scalar symbolic output using a packed parameter vector."""
        terminal = self._require_terminal()
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

        terminal_parameter, parameter_offset = _resolve_compiled_parameter(
            terminal.parameter,
            packed_parameters,
            parameter_offset,
        )
        output = _coerce_single_output(
            terminal.function(state, terminal_parameter)
        )
        if not isinstance(output, SX):
            raise TypeError(
                "terminal function must produce a scalar SX output"
            )
        if parameter_offset != self.parameter_size:
            raise AssertionError(
                "packed parameter offsets did " "not consume the expected size"
            )
        return output

    def _require_terminal(self) -> _TerminalStage:
        """Return the terminal stage or raise when incomplete."""
        if self.terminal is None:
            raise ValueError(
                "ComposedFunction is not finished; call finish(...) first"
            )
        return self.terminal

    def _ensure_not_finished(self) -> None:
        """Reject stage edits after the terminal stage is attached."""
        if self.terminal is not None:
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


def _validate_terminal_function(
    function: Function, state_input: FunctionArg
) -> None:
    """Validate the terminal scalar stage."""
    if len(function.inputs) != 2:
        raise ValueError(
            "terminal function must take exactly two inputs: (state, p)"
        )
    if len(function.outputs) != 1 or not isinstance(function.outputs[0], SX):
        raise ValueError(
            "terminal function must produce exactly one scalar output"
        )
    _validate_matching_shape(
        function.inputs[0], state_input, "terminal state input"
    )


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
    """Normalize one stage or terminal parameter binding."""
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
        packed_parameters[offset : offset + parameter.size],
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


def _arg_size(arg: FunctionArg) -> int:
    """Return the scalar length of a scalar or vector argument."""
    if isinstance(arg, SX):
        return 1
    return len(arg)
