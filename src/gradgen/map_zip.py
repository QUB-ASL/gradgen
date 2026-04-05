"""Loop-structured batched map/zip/reduce function abstractions."""

from __future__ import annotations

from dataclasses import dataclass

from .function import Function
from .sx import SX, SXVector


FunctionArg = SX | SXVector


@dataclass(frozen=True,
           slots=True)
class ZippedJacobianFunction:
    """Staged Jacobian wrapper for a zipped function.

    This object represents the Jacobian of a staged batched function without
    immediately expanding it back into an ordinary :class:`Function`.

    Attributes:
        zipped: The underlying staged batched function.
        wrt_index: The input index with respect to which the Jacobian is
            taken.
        name: The symbolic name of the staged Jacobian wrapper.
        simplification: Optional simplification effort forwarded to the
            expanded symbolic function.
    """

    zipped: ZippedFunction
    wrt_index: int
    name: str
    simplification: int | str | None = None

    def to_function(self,
                    name: str | None = None) -> Function:
        """Expand this staged Jacobian into a regular symbolic function.

        Args:
            name: Optional name for the expanded symbolic function. When not
                provided, the wrapper name is reused.

        Returns:
            A symbolic :class:`~gradgen.function.Function` representing the
            expanded Jacobian.
        """
        function = self.zipped\
            .to_function()\
            .jacobian(
                self.wrt_index,
                name=name or self.name)
        if self.simplification is None:
            return function
        return function.simplify(
            max_effort=self.simplification,
            name=function.name)

    @property
    def input_names(self) -> tuple[str, ...]:
        """Return the packed input names used by the staged Jacobian."""
        return self.zipped.input_names

    @property
    def output_names(self) -> tuple[str, ...]:
        """Return the generated Jacobian output names."""
        return tuple(f"jacobian_{name}"
                     for name in self.zipped.function.output_names)

    def generate_rust(
        self,
        *,
        config=None,
        function_name: str | None = None,
        backend_mode: str = "std",
        scalar_type: str = "f64",
    ):
        """Generate compact Rust for the staged Jacobian kernel.

        Args:
            config: Optional backend configuration object.
            function_name: Optional exported Rust function name.
            backend_mode: Rust backend mode to target.
            scalar_type: Rust scalar type to emit.

        Returns:
            The generated Rust code bundle for this staged Jacobian.
        """
        from ._rust_codegen.codegen import generate_rust

        return generate_rust(
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
        """Create a Rust crate containing the staged Jacobian kernel.

        Args:
            path: Directory where the Rust crate should be created.
            config: Optional backend configuration object.
            crate_name: Optional crate name override.
            function_name: Optional exported Rust function name.
            backend_mode: Rust backend mode to target.
            scalar_type: Rust scalar type to emit.

        Returns:
            A Rust project bundle rooted at ``path``.
        """
        from ._rust_codegen.project import create_rust_project

        return create_rust_project(
            self,
            path,
            config=config,
            crate_name=crate_name,
            function_name=function_name,
            backend_mode=backend_mode,
            scalar_type=scalar_type,
        )


@dataclass(frozen=True, slots=True)
class ZippedFunction:
    """Staged wrapper that batches a function over packed inputs.

    The wrapper represents a symbolic function evaluated repeatedly over
    packed input sequences, producing packed outputs suitable for Rust code
    generation or symbolic expansion.

    Attributes:
        function: The symbolic function to stage.
        count: Number of repeated stages in the packed sequence.
        name: Symbolic name of the staged wrapper.
        input_sequence_names: Packed input sequence names.
        simplification: Optional simplification effort forwarded to the
            expanded symbolic function.
    """

    function: Function
    count: int
    name: str
    input_sequence_names: tuple[str, ...]
    simplification: int | str | None = None

    def __post_init__(self) -> None:
        """Validate batching metadata and staged input arity."""
        if self.count <= 0:
            raise ValueError("count must be a positive integer")
        if len(self.input_sequence_names) != len(self.function.inputs):
            raise ValueError(
                "input_sequence_names must match the number "
                "of function inputs")

    @property
    def input_names(self) -> tuple[str, ...]:
        """Return the packed input sequence names."""
        return self.input_sequence_names

    @property
    def output_names(self) -> tuple[str, ...]:
        """Return the packed output names."""
        return self.function.output_names

    @property
    def nodes(self):
        """Return dependency nodes for symbolic fallback usage."""
        return self.to_function().nodes

    def jacobian(self,
                 wrt_index: int = 0,
                 *,
                 name: str | None = None) -> ZippedJacobianFunction:
        """Return a staged Jacobian wrapper for one packed input.

        Args:
            wrt_index: Input index to differentiate with respect to.
            name: Optional symbolic name for the staged Jacobian wrapper.

        Returns:
            A :class:`ZippedJacobianFunction` describing the staged Jacobian.
        """
        if wrt_index < 0 or wrt_index >= len(self.function.inputs):
            raise IndexError("wrt_index out of range")
        return ZippedJacobianFunction(
            zipped=self,
            wrt_index=wrt_index,
            name=name or f"{self.name}_jacobian_{self.input_names[wrt_index]}",
            simplification=self.simplification,
        )

    def to_function(self, name: str | None = None) -> Function:
        """Expand this staged batching into a regular symbolic function.

        Args:
            name: Optional name for the expanded symbolic function. When not
                provided, the wrapper name is reused.

        Returns:
            A symbolic :class:`~gradgen.function.Function` representing the
            expanded staged batch.
        """
        packed_inputs = tuple(
            SXVector.sym(input_name, self.count * _arg_size(formal))
            for formal, input_name in zip(
                self.function.inputs, self.input_sequence_names)
        )
        packed_outputs: list[SXVector] = []
        for output_index, _ in enumerate(self.function.outputs):
            scalars: list[SX] = []
            for stage_index in range(self.count):
                stage_inputs = tuple(
                    _slice_packed_input(sequence, stage_index, formal)
                    for sequence, formal in zip(
                        packed_inputs, self.function.inputs)
                )
                stage_result = self.function(*stage_inputs)
                stage_output = _normalize_function_result(
                    stage_result, self.function.outputs)[output_index]
                scalars.extend(_flatten_arg(stage_output))
            packed_outputs.append(SXVector(tuple(scalars)))

        function = Function(
            name or self.name,
            packed_inputs,
            packed_outputs,
            input_names=self.input_sequence_names,
            output_names=self.function.output_names,
        )
        if self.simplification is None:
            return function
        return function.simplify(max_effort=self.simplification,
                                 name=function.name)

    def generate_rust(
        self,
        *,
        config=None,
        function_name: str | None = None,
        backend_mode: str = "std",
        scalar_type: str = "f64",
    ):
        """Generate compact Rust for the staged batched kernel.

        Args:
            config: Optional backend configuration object.
            function_name: Optional exported Rust function name.
            backend_mode: Rust backend mode to target.
            scalar_type: Rust scalar type to emit.

        Returns:
            The generated Rust code bundle for this staged batch.
        """
        from ._rust_codegen.codegen import generate_rust

        return generate_rust(
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
        """Create a Rust crate containing the staged batched kernel.

        Args:
            path: Directory where the Rust crate should be created.
            config: Optional backend configuration object.
            crate_name: Optional crate name override.
            function_name: Optional exported Rust function name.
            backend_mode: Rust backend mode to target.
            scalar_type: Rust scalar type to emit.

        Returns:
            A Rust project bundle rooted at ``path``.
        """
        from ._rust_codegen.project import create_rust_project

        return create_rust_project(
            self,
            path,
            config=config,
            crate_name=crate_name,
            function_name=function_name,
            backend_mode=backend_mode,
            scalar_type=scalar_type,
        )


@dataclass(frozen=True, slots=True)
class ReducedFunction:
    """Staged wrapper that reduces a packed input sequence.

    The wrapper represents a repeated left-fold reduction over a packed input
    sequence and is designed for symbolic expansion or Rust generation.

    Attributes:
        function: The binary stage function being reduced.
        count: Number of reduction stages.
        name: Symbolic name of the staged reducer.
        accumulator_input_name: Name of the accumulator input sequence.
        input_name: Name of the packed sequence input.
        output_name: Name of the final reduced output.
        simplification: Optional simplification effort forwarded to the
            expanded symbolic function.
    """

    function: Function
    count: int
    name: str
    accumulator_input_name: str
    input_name: str
    output_name: str
    simplification: int | str | None = None

    def __post_init__(self) -> None:
        """Validate reduction metadata and stage-function contract."""
        if self.count <= 0:
            raise ValueError("count must be a positive integer")
        if len(self.function.inputs) != 2:
            raise ValueError(
                "reduce_function requires a function with exactly two inputs")
        if len(self.function.outputs) != 1:
            raise ValueError(
                "reduce_function requires a function with exactly one output")
        accumulator_formal = self.function.inputs[0]
        accumulator_output = self.function.outputs[0]
        if not _same_arg_shape(accumulator_formal, accumulator_output):
            raise ValueError(
                "reduce_function output shape must match "
                "accumulator input shape")

    @property
    def input_names(self) -> tuple[str, ...]:
        """Return packed reduce input names: ``(acc0, x_seq)``."""
        return (self.accumulator_input_name, self.input_name)

    @property
    def output_names(self) -> tuple[str, ...]:
        """Return the reduction output names."""
        return (self.output_name,)

    @property
    def nodes(self):
        """Return dependency nodes for symbolic fallback usage."""
        return self.to_function().nodes

    def to_function(self, name: str | None = None) -> Function:
        """Expand this staged reduce into a regular symbolic function.

        Args:
            name: Optional name for the expanded symbolic function. 
                When not provided, the wrapper name is reused.

        Returns:
            A symbolic :class:`~gradgen.function.Function` representing the
            expanded reduction.
        """
        accumulator_formal = self.function.inputs[0]
        sequence_formal = self.function.inputs[1]

        accumulator_initial: FunctionArg
        if isinstance(accumulator_formal, SX):
            accumulator_initial = SX.sym(self.accumulator_input_name)
        else:
            accumulator_initial = SXVector.sym(
                self.accumulator_input_name,
                len(accumulator_formal))

        sequence_input = SXVector.sym(
            self.input_name,
            self.count * _arg_size(sequence_formal))

        accumulator_state = accumulator_initial
        for stage_index in range(self.count):
            sequence_stage = _slice_packed_input(
                sequence_input, stage_index, sequence_formal)
            stage_result = self.function(accumulator_state, sequence_stage)
            accumulator_state = _normalize_function_result(
                stage_result, self.function.outputs)[0]

        function = Function(
            name or self.name,
            (accumulator_initial, sequence_input),
            (accumulator_state,),
            input_names=(self.accumulator_input_name, self.input_name),
            output_names=(self.output_name,),
        )
        if self.simplification is None:
            return function
        return function.simplify(
            max_effort=self.simplification,
            name=function.name)

    def generate_rust(
        self,
        *,
        config=None,
        function_name: str | None = None,
        backend_mode: str = "std",
        scalar_type: str = "f64",
    ):
        """Generate compact Rust for the staged reduce kernel.

        Args:
            config: Optional backend configuration object.
            function_name: Optional exported Rust function name.
            backend_mode: Rust backend mode to target.
            scalar_type: Rust scalar type to emit.

        Returns:
            The generated Rust code bundle for this staged reduction.
        """
        from ._rust_codegen.codegen import generate_rust

        return generate_rust(
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
        """Create a Rust crate containing the staged reduce kernel.

        Args:
            path: Directory where the Rust crate should be created.
            config: Optional backend configuration object.
            crate_name: Optional crate name override.
            function_name: Optional exported Rust function name.
            backend_mode: Rust backend mode to target.
            scalar_type: Rust scalar type to emit.

        Returns:
            A Rust project bundle rooted at ``path``.
        """
        from ._rust_codegen.project import create_rust_project

        return create_rust_project(
            self,
            path,
            config=config,
            crate_name=crate_name,
            function_name=function_name,
            backend_mode=backend_mode,
            scalar_type=scalar_type,
        )


def map_function(
    function: Function,
    count: int,
    *,
    input_name: str | None = None,
    name: str | None = None,
    simplification: int | str | None = None,
) -> ZippedFunction:
    """Return a staged mapped function for a unary function.

    Args:
        function: The symbolic function to stage. It must accept exactly one
            input argument.
        count: Number of repeated applications in the packed sequence.
        input_name: Optional name for the packed input sequence. When not
            provided, the name defaults to ``"<input_name>_seq"``.
        name: Optional name for the staged mapped function. When not
            provided, the generated name defaults to ``"<function.name>_map"``.
        simplification: Optional simplification effort passed through to the
            expanded symbolic function.

    Returns:
        A :class:`ZippedFunction` representing the staged batched function.
    """
    if len(function.inputs) != 1:
        raise ValueError("map_function requires a function "
                         "with exactly one input")
    sequence_name = input_name or f"{function.input_names[0]}_seq"
    return ZippedFunction(
        function=function,
        count=count,
        name=name or f"{function.name}_map",
        input_sequence_names=(sequence_name,),
        simplification=simplification,
    )


def zip_function(
    function: Function,
    count: int,
    *,
    input_names: tuple[str, ...] | list[str] | None = None,
    name: str | None = None,
    simplification: int | str | None = None,
) -> ZippedFunction:
    """Return a staged zipped function for a multi-input ``function``.

    This helper creates a loop-structured wrapper around ``function`` so it can
    be evaluated repeatedly over packed input sequences. The returned object is
    a :class:`ZippedFunction`, which can later be expanded back into a regular
    symbolic :class:`~gradgen.function.Function` or used for Rust code
    generation.

    Args:
        function: The symbolic function to stage. It may accept one or more
            inputs, and each input can be either a scalar ``SX`` or an
            ``SXVector``.
        count: The number of times the function should be applied over the
            packed input sequences.
        input_names: Optional names for the packed input sequences. When not
            provided, each input name defaults to ``"<input_name>_seq"``.
        name: Optional name for the staged zipped function. When not provided,
            the generated name defaults to ``"<function.name>_zip"``.
        simplification: Optional simplification effort passed through to the
            generated symbolic function when the staged wrapper is expanded.

    Returns:
        A :class:`ZippedFunction` describing the staged batched version of
        ``function``.

    Example:
        >>> from gradgen import Function, SX
        >>> x = SX.sym("x")
        >>> y = SX.sym("y")
        >>> f = Function("add", (x, y), (x + y,))
        >>> zipped = zip_function(f, 4)
        >>> zipped.name
        'add_zip'
        >>> zipped.input_names
        ('x_seq', 'y_seq')
    """
    resolved_names = tuple(input_names) if input_names is not None else tuple(
        f"{input_name}_seq" for input_name in function.input_names
    )
    return ZippedFunction(
        function=function,
        count=count,
        name=name or f"{function.name}_zip",
        input_sequence_names=resolved_names,
        simplification=simplification,
    )


def reduce_function(
    function: Function,
    count: int,
    *,
    accumulator_input_name: str | None = None,
    input_name: str | None = None,
    output_name: str | None = None,
    name: str | None = None,
    simplification: int | str | None = None,
) -> ReducedFunction:
    """Return a staged left-fold reduction for a binary stage function.

    Args:
        function: The symbolic function to stage. It must accept exactly two
            inputs and produce exactly one output.
        count: Number of repeated reduction stages.
        accumulator_input_name: Optional name for the accumulator input.
            When not provided, the name defaults to ``"<input0>0"``.
        input_name: Optional name for the packed sequence input. When not
            provided, the name defaults to ``"<input1>_seq"``.
        output_name: Optional name for the final reduced output. When not
            provided, the name defaults to the stage function's output name.
        name: Optional name for the staged reduction. When not provided, the
            generated name defaults to ``"<function.name>_reduce"``.
        simplification: Optional simplification effort passed through to the
            expanded symbolic function.

    Returns:
        A :class:`ReducedFunction` describing the staged reduction.
    """
    if len(function.inputs) != 2:
        raise ValueError(
            "reduce_function requires a function with exactly two inputs")
    if len(function.outputs) != 1:
        raise ValueError(
            "reduce_function requires a function with exactly one output")
    accum_inp_name = accumulator_input_name or f"{function.input_names[0]}0"
    return ReducedFunction(
        function=function,
        count=count,
        name=name or f"{function.name}_reduce",
        accumulator_input_name=accum_inp_name,
        input_name=input_name or f"{function.input_names[1]}_seq",
        output_name=output_name or function.output_names[0],
        simplification=simplification,
    )


def _same_arg_shape(left: FunctionArg, right: FunctionArg) -> bool:
    """
    Return whether two function arguments share the same
    scalar/vector shape.
    """
    if isinstance(left, SX) and isinstance(right, SX):
        return True
    if isinstance(left, SXVector) and isinstance(right, SXVector):
        return len(left) == len(right)
    return False


def _arg_size(value: FunctionArg) -> int:
    """Return the packed scalar count for one argument."""
    if isinstance(value, SX):
        return 1
    return len(value)


def _slice_packed_input(sequence: SXVector,
                        stage_index: int,
                        formal: FunctionArg) -> FunctionArg:
    """Return one stage input block from a packed batched input."""
    block_size = _arg_size(formal)
    start = stage_index * block_size
    if isinstance(formal, SX):
        return sequence[start]
    return SXVector(sequence.elements[start: start + block_size])


def _flatten_arg(value: FunctionArg) -> tuple[SX, ...]:
    """Flatten a scalar or vector symbolic value to scalar expressions."""
    if isinstance(value, SX):
        return (value,)
    return value.elements


def _normalize_function_result(
    value: object,
    output_formals: tuple[FunctionArg, ...],
) -> tuple[FunctionArg, ...]:
    """Normalize a function call result to a tuple of symbolic outputs."""
    output_count = len(output_formals)

    if output_count == 1:
        return (_coerce_function_arg_like(value, output_formals[0]),)

    if isinstance(value, tuple):
        if len(value) != output_count:
            raise ValueError("unexpected number of function outputs")
        return tuple(
            _coerce_function_arg_like(item, formal)
            for item, formal in zip(value, output_formals)
        )
    raise ValueError("expected multiple outputs from the staged function")


def _coerce_function_arg_like(value: object,
                              formal: FunctionArg) -> FunctionArg:
    """Coerce a staged output value to the symbolic shape of ``formal``."""
    if isinstance(formal, SX):
        return _coerce_scalar_output(value)

    if isinstance(value, SXVector):
        if len(value) != len(formal):
            raise ValueError("unexpected vector output length")
        return value

    if isinstance(value, (list, tuple)):
        if len(value) != len(formal):
            raise ValueError("unexpected vector output length")
        return SXVector(tuple(_coerce_scalar_output(item) for item in value))

    raise TypeError("zipped functions require symbolic outputs")


def _coerce_scalar_output(value: object) -> SX:
    """Coerce a scalar staged output to ``SX``."""
    if isinstance(value, SX):
        return value
    if isinstance(value, (int, float)):
        return SX.const(value)
    raise TypeError("zipped functions require symbolic outputs")
