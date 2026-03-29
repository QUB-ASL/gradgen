"""Loop-structured single-shooting optimal-control abstractions."""

from __future__ import annotations

from dataclasses import dataclass, replace

from .function import Function
from .sx import SX, SXVector, vector


FunctionArg = SX | SXVector


@dataclass(frozen=True, slots=True)
class SingleShootingBundle:
    """Requested outputs for a joint single-shooting kernel."""

    include_cost: bool = False
    include_gradient: bool = False
    include_states: bool = False

    def add_cost(self) -> SingleShootingBundle:
        """Include the total cost in the joint kernel outputs."""
        return replace(self, include_cost=True)

    def add_gradient(self) -> SingleShootingBundle:
        """Include the gradient with respect to the packed control sequence."""
        return replace(self, include_gradient=True)

    def add_rollout_states(self) -> SingleShootingBundle:
        """Include the packed rollout state trajectory."""
        return replace(self, include_states=True)


@dataclass(frozen=True, slots=True)
class SingleShootingPrimalFunction:
    """Primal single-shooting cost kernel."""

    problem: SingleShootingProblem
    name: str
    include_states: bool = False
    simplification: int | str | None = None

    def to_function(self, name: str | None = None) -> Function:
        """Expand this staged kernel into a regular symbolic ``Function``."""
        function = self.problem.to_function(
            include_states=self.include_states,
            name=name or self.name,
        )
        if self.simplification is None:
            return function
        return function.simplify(max_effort=self.simplification, name=function.name)

    @property
    def nodes(self):
        """Return dependency nodes for shared helper discovery."""
        return self.to_function().nodes

    @property
    def input_names(self) -> tuple[str, ...]:
        """Return the exposed input names."""
        return self.problem.input_names

    @property
    def output_names(self) -> tuple[str, ...]:
        """Return the exposed output names."""
        names = ["cost"]
        if self.include_states:
            names.append("x_traj")
        return tuple(names)

    def generate_rust(
        self,
        *,
        config=None,
        function_name: str | None = None,
        backend_mode: str = "std",
        scalar_type: str = "f64",
        math_library: str | None = None,
    ):
        """Generate compact Rust for the staged primal kernel."""
        from .rust_codegen import generate_rust

        return generate_rust(
            self,
            config=config,
            function_name=function_name,
            backend_mode=backend_mode,
            scalar_type=scalar_type,
            math_library=math_library,
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
        math_library: str | None = None,
    ):
        """Create a Rust crate containing the staged primal kernel."""
        from .rust_codegen import create_rust_project

        return create_rust_project(
            self,
            path,
            config=config,
            crate_name=crate_name,
            function_name=function_name,
            backend_mode=backend_mode,
            scalar_type=scalar_type,
            math_library=math_library,
        )


@dataclass(frozen=True, slots=True)
class SingleShootingGradientFunction:
    """Gradient kernel for a single-shooting optimal-control problem."""

    problem: SingleShootingProblem
    name: str
    include_states: bool = False
    simplification: int | str | None = None

    def to_function(self, name: str | None = None) -> Function:
        """Expand this staged gradient into a regular symbolic ``Function``."""
        gradient_function = self.problem._expanded_gradient_function(
            include_states=self.include_states,
            name=name or self.name,
        )
        if self.simplification is None:
            return gradient_function
        return gradient_function.simplify(
            max_effort=self.simplification,
            name=gradient_function.name,
        )

    @property
    def nodes(self):
        """Return dependency nodes for shared helper discovery."""
        return self.to_function().nodes

    @property
    def input_names(self) -> tuple[str, ...]:
        """Return the exposed input names."""
        return self.problem.input_names

    @property
    def output_names(self) -> tuple[str, ...]:
        """Return the exposed output names."""
        names = [f"gradient_{self.problem.control_sequence_name}"]
        if self.include_states:
            names.append("x_traj")
        return tuple(names)

    def generate_rust(
        self,
        *,
        config=None,
        function_name: str | None = None,
        backend_mode: str = "std",
        scalar_type: str = "f64",
        math_library: str | None = None,
    ):
        """Generate compact Rust for the staged gradient kernel."""
        from .rust_codegen import generate_rust

        return generate_rust(
            self,
            config=config,
            function_name=function_name,
            backend_mode=backend_mode,
            scalar_type=scalar_type,
            math_library=math_library,
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
        math_library: str | None = None,
    ):
        """Create a Rust crate containing the staged gradient kernel."""
        from .rust_codegen import create_rust_project

        return create_rust_project(
            self,
            path,
            config=config,
            crate_name=crate_name,
            function_name=function_name,
            backend_mode=backend_mode,
            scalar_type=scalar_type,
            math_library=math_library,
        )


@dataclass(frozen=True, slots=True)
class SingleShootingHvpFunction:
    """HVP kernel for a single-shooting optimal-control problem."""

    problem: SingleShootingProblem
    name: str
    include_states: bool = False
    simplification: int | str | None = None

    def to_function(self, name: str | None = None) -> Function:
        """Expand this staged HVP kernel into a regular symbolic ``Function``."""
        hvp_function = self.problem._expanded_hvp_function(
            include_states=self.include_states,
            name=name or self.name,
        )
        if self.simplification is None:
            return hvp_function
        return hvp_function.simplify(
            max_effort=self.simplification,
            name=hvp_function.name,
        )

    @property
    def nodes(self):
        """Return dependency nodes for shared helper discovery."""
        return self.to_function().nodes

    @property
    def input_names(self) -> tuple[str, ...]:
        """Return the exposed input names."""
        return (*self.problem.input_names, f"v_{self.problem.control_sequence_name}")

    @property
    def output_names(self) -> tuple[str, ...]:
        """Return the exposed output names."""
        names = [f"hvp_{self.problem.control_sequence_name}"]
        if self.include_states:
            names.append("x_traj")
        return tuple(names)

    def generate_rust(
        self,
        *,
        config=None,
        function_name: str | None = None,
        backend_mode: str = "std",
        scalar_type: str = "f64",
        math_library: str | None = None,
    ):
        """Generate compact Rust for the staged HVP kernel."""
        from .rust_codegen import generate_rust

        return generate_rust(
            self,
            config=config,
            function_name=function_name,
            backend_mode=backend_mode,
            scalar_type=scalar_type,
            math_library=math_library,
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
        math_library: str | None = None,
    ):
        """Create a Rust crate containing the staged HVP kernel."""
        from .rust_codegen import create_rust_project

        return create_rust_project(
            self,
            path,
            config=config,
            crate_name=crate_name,
            function_name=function_name,
            backend_mode=backend_mode,
            scalar_type=scalar_type,
            math_library=math_library,
        )


@dataclass(frozen=True, slots=True)
class SingleShootingJointFunction:
    """Joint cost/gradient/state kernel for a single-shooting problem."""

    problem: SingleShootingProblem
    bundle: SingleShootingBundle
    name: str
    simplification: int | str | None = None

    def to_function(self, name: str | None = None) -> Function:
        """Expand this staged joint kernel into a regular symbolic ``Function``."""
        function = self.problem._expanded_joint_function(
            self.bundle,
            name=name or self.name,
        )
        if self.simplification is None:
            return function
        return function.simplify(max_effort=self.simplification, name=function.name)

    @property
    def nodes(self):
        """Return dependency nodes for shared helper discovery."""
        return self.to_function().nodes

    @property
    def input_names(self) -> tuple[str, ...]:
        """Return the exposed input names."""
        return self.problem.input_names

    @property
    def output_names(self) -> tuple[str, ...]:
        """Return the exposed output names."""
        return _single_shooting_bundle_output_names(self.problem, self.bundle)

    def generate_rust(
        self,
        *,
        config=None,
        function_name: str | None = None,
        backend_mode: str = "std",
        scalar_type: str = "f64",
        math_library: str | None = None,
    ):
        """Generate compact Rust for the staged joint kernel."""
        from .rust_codegen import generate_rust

        return generate_rust(
            self,
            config=config,
            function_name=function_name,
            backend_mode=backend_mode,
            scalar_type=scalar_type,
            math_library=math_library,
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
        math_library: str | None = None,
    ):
        """Create a Rust crate containing the staged joint kernel."""
        from .rust_codegen import create_rust_project

        return create_rust_project(
            self,
            path,
            config=config,
            crate_name=crate_name,
            function_name=function_name,
            backend_mode=backend_mode,
            scalar_type=scalar_type,
            math_library=math_library,
        )


@dataclass(frozen=True, slots=True)
class SingleShootingProblem:
    """Deterministic single-shooting optimal-control problem."""

    name: str
    horizon: int
    dynamics: Function
    stage_cost: Function
    terminal_cost: Function
    initial_state_name: str = "x0"
    control_sequence_name: str = "U"
    parameter_name: str = "p"
    simplification: int | str | None = None

    def __post_init__(self) -> None:
        """Validate stage and cost signatures."""
        if self.horizon <= 0:
            raise ValueError("horizon must be a positive integer")
        _validate_single_shooting_dynamics(self.dynamics)
        _validate_single_shooting_stage_cost(self.stage_cost)
        _validate_single_shooting_terminal_cost(self.terminal_cost)
        _validate_single_shooting_shapes(self.dynamics, self.stage_cost, self.terminal_cost)

    @property
    def state_size(self) -> int:
        """Return the state dimension."""
        return _single_shooting_arg_size(self.dynamics.inputs[0])

    @property
    def control_size(self) -> int:
        """Return the per-stage control dimension."""
        return _single_shooting_arg_size(self.dynamics.inputs[1])

    @property
    def parameter_size(self) -> int:
        """Return the shared parameter-vector dimension."""
        return _single_shooting_arg_size(self.dynamics.inputs[2])

    @property
    def input_names(self) -> tuple[str, ...]:
        """Return the exposed runtime input names."""
        return (
            self.initial_state_name,
            self.control_sequence_name,
            self.parameter_name,
        )

    @property
    def output_names(self) -> tuple[str, ...]:
        """Return the default primal output names."""
        return ("cost",)

    @property
    def inputs(self) -> tuple[FunctionArg, ...]:
        """Return compiled symbolic inputs."""
        return self._compiled_inputs()

    @property
    def outputs(self) -> tuple[FunctionArg, ...]:
        """Return compiled symbolic outputs for the primal cost kernel."""
        return self.to_function().outputs

    @property
    def nodes(self):
        """Return dependency nodes for shared helper discovery."""
        return self.to_function().nodes

    def primal(
        self,
        *,
        include_states: bool = False,
        name: str | None = None,
    ) -> SingleShootingProblem | SingleShootingPrimalFunction:
        """Return a staged primal kernel source."""
        if not include_states and name is None:
            return self
        return SingleShootingPrimalFunction(
            problem=self,
            name=name or _single_shooting_primal_name(self.name, include_states),
            include_states=include_states,
            simplification=self.simplification,
        )

    def gradient(
        self,
        *,
        include_states: bool = False,
        name: str | None = None,
    ) -> SingleShootingGradientFunction:
        """Return a staged gradient kernel source."""
        return SingleShootingGradientFunction(
            problem=self,
            name=name or f"{self.name}_gradient_{self.control_sequence_name}",
            include_states=include_states,
            simplification=self.simplification,
        )

    def hvp(
        self,
        *,
        include_states: bool = False,
        name: str | None = None,
    ) -> SingleShootingHvpFunction:
        """Return a staged Hessian-vector-product kernel source."""
        return SingleShootingHvpFunction(
            problem=self,
            name=name or f"{self.name}_hvp_{self.control_sequence_name}",
            include_states=include_states,
            simplification=self.simplification,
        )

    def joint(
        self,
        bundle: SingleShootingBundle,
        *,
        name: str | None = None,
    ) -> SingleShootingJointFunction:
        """Return a staged joint kernel source."""
        _validate_single_shooting_bundle(bundle)
        return SingleShootingJointFunction(
            problem=self,
            bundle=bundle,
            name=name or _single_shooting_joint_name(self.name, bundle, self.control_sequence_name),
            simplification=self.simplification,
        )

    def to_function(
        self,
        *,
        include_states: bool = False,
        name: str | None = None,
    ) -> Function:
        """Expand the total-cost kernel into a regular symbolic ``Function``."""
        x0, U, p = self._compiled_inputs()
        current_state: FunctionArg = x0
        rollout_states: list[FunctionArg] = [current_state]
        total_cost = SX.const(0.0)

        for stage_index in range(self.horizon):
            u_t = _slice_packed_sequence(U, stage_index, self.control_size, self.dynamics.inputs[1])
            total_cost = total_cost + _extract_scalar_output(
                self.stage_cost(current_state, u_t, p)
            )
            current_state = _extract_single_output(self.dynamics(current_state, u_t, p))
            rollout_states.append(current_state)

        total_cost = total_cost + _extract_scalar_output(self.terminal_cost(current_state, p))
        outputs: list[FunctionArg] = [total_cost]
        output_names = ["cost"]
        if include_states:
            outputs.append(_flatten_rollout_states(rollout_states))
            output_names.append("x_traj")

        function = Function(
            name or self.name,
            (x0, U, p),
            outputs,
            input_names=self.input_names,
            output_names=tuple(output_names),
        )
        if self.simplification is None:
            return function
        return function.simplify(max_effort=self.simplification, name=function.name)

    def _expanded_gradient_function(
        self,
        *,
        include_states: bool,
        name: str,
    ) -> Function:
        """Expand the staged gradient kernel into a symbolic ``Function``."""
        cost_function = self.to_function(include_states=False, name=f"{name}_cost")
        gradient_function = cost_function.gradient(1, name=f"{name}_grad")
        outputs: list[FunctionArg] = [gradient_function.outputs[0]]
        output_names = [f"gradient_{self.control_sequence_name}"]
        if include_states:
            outputs.append(self.to_function(include_states=True, name=f"{name}_states").outputs[1])
            output_names.append("x_traj")
        return Function(
            name,
            gradient_function.inputs,
            outputs,
            input_names=gradient_function.input_names,
            output_names=tuple(output_names),
        )

    def _expanded_hvp_function(
        self,
        *,
        include_states: bool,
        name: str,
    ) -> Function:
        """Expand the staged HVP kernel into a symbolic ``Function``."""
        cost_function = self.to_function(include_states=False, name=f"{name}_cost")
        hvp_function = cost_function.hvp(1, name=f"{name}_hvp")
        outputs: list[FunctionArg] = [hvp_function.outputs[0]]
        output_names = [f"hvp_{self.control_sequence_name}"]
        if include_states:
            outputs.append(self.to_function(include_states=True, name=f"{name}_states").outputs[1])
            output_names.append("x_traj")
        return Function(
            name,
            hvp_function.inputs,
            outputs,
            input_names=hvp_function.input_names,
            output_names=tuple(output_names),
        )

    def _expanded_joint_function(
        self,
        bundle: SingleShootingBundle,
        *,
        name: str,
    ) -> Function:
        """Expand a staged joint kernel into a symbolic ``Function``."""
        _validate_single_shooting_bundle(bundle)
        cost_function = self.to_function(include_states=False, name=f"{name}_cost")
        gradient_function = cost_function.gradient(1, name=f"{name}_grad")
        outputs: list[FunctionArg] = []
        output_names: list[str] = []
        if bundle.include_cost:
            outputs.append(cost_function.outputs[0])
            output_names.append("cost")
        if bundle.include_gradient:
            outputs.append(gradient_function.outputs[0])
            output_names.append(f"gradient_{self.control_sequence_name}")
        if bundle.include_states:
            outputs.append(self.to_function(include_states=True, name=f"{name}_states").outputs[1])
            output_names.append("x_traj")
        return Function(
            name,
            cost_function.inputs,
            outputs,
            input_names=cost_function.input_names,
            output_names=tuple(output_names),
        )

    def _compiled_inputs(self) -> tuple[FunctionArg, SXVector, FunctionArg]:
        """Return symbolic runtime inputs for ``x0``, ``U``, and ``p``."""
        x0 = _make_symbolic_like(self.dynamics.inputs[0], self.initial_state_name)
        p = _make_symbolic_like(self.dynamics.inputs[2], self.parameter_name)
        U = SXVector.sym(self.control_sequence_name, self.horizon * self.control_size)
        return x0, U, p

    def generate_rust(
        self,
        *,
        config=None,
        function_name: str | None = None,
        backend_mode: str = "std",
        scalar_type: str = "f64",
        math_library: str | None = None,
    ):
        """Generate compact Rust for the primal total-cost kernel."""
        from .rust_codegen import generate_rust

        return generate_rust(
            self,
            config=config,
            function_name=function_name,
            backend_mode=backend_mode,
            scalar_type=scalar_type,
            math_library=math_library,
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
        math_library: str | None = None,
    ):
        """Create a Rust crate containing the total-cost kernel."""
        from .rust_codegen import create_rust_project

        return create_rust_project(
            self,
            path,
            config=config,
            crate_name=crate_name,
            function_name=function_name,
            backend_mode=backend_mode,
            scalar_type=scalar_type,
            math_library=math_library,
        )


def _single_shooting_arg_size(value: FunctionArg) -> int:
    """Return the flattened scalar dimension of ``value``."""
    if isinstance(value, SX):
        return 1
    return len(value)


def _same_single_shooting_shape(left: FunctionArg, right: FunctionArg) -> bool:
    """Return ``True`` when two symbolic values share the same shape."""
    if isinstance(left, SX) and isinstance(right, SX):
        return True
    if isinstance(left, SXVector) and isinstance(right, SXVector):
        return len(left) == len(right)
    return False


def _make_symbolic_like(value: FunctionArg, base_name: str) -> FunctionArg:
    """Create a fresh symbolic input with the same shape as ``value``."""
    if isinstance(value, SX):
        return SX.sym(base_name)
    return SXVector.sym(base_name, len(value))


def _slice_packed_sequence(
    sequence: SXVector,
    stage_index: int,
    block_size: int,
    formal: FunctionArg,
) -> FunctionArg:
    """Return one stage block from a packed control sequence."""
    start = stage_index * block_size
    if isinstance(formal, SX):
        return sequence[start]
    return SXVector(sequence.elements[start : start + block_size])


def _extract_single_output(value: object) -> FunctionArg:
    """Normalize a single-output function call result."""
    if isinstance(value, tuple):
        if len(value) != 1:
            raise ValueError("expected a single function output")
        return value[0]
    if isinstance(value, (SX, SXVector)):
        return value
    raise TypeError("single-shooting stages must return symbolic outputs")


def _extract_scalar_output(value: object) -> SX:
    """Normalize a single scalar output."""
    output = _extract_single_output(value)
    if not isinstance(output, SX):
        raise ValueError("single-shooting cost functions must return scalar outputs")
    return output


def _flatten_rollout_states(states: list[FunctionArg]) -> SXVector:
    """Flatten a rollout state sequence into one packed vector."""
    scalars: list[SX] = []
    for state in states:
        if isinstance(state, SX):
            scalars.append(state)
        else:
            scalars.extend(state.elements)
    return SXVector(tuple(scalars))


def _single_shooting_primal_name(base_name: str, include_states: bool) -> str:
    """Return the default primal wrapper name."""
    if include_states:
        return f"{base_name}_with_states"
    return base_name


def _single_shooting_joint_name(
    base_name: str,
    bundle: SingleShootingBundle,
    control_sequence_name: str,
) -> str:
    """Return the default joint-kernel name."""
    labels: list[str] = []
    if bundle.include_cost:
        labels.append("cost")
    if bundle.include_gradient:
        labels.append(f"gradient_{control_sequence_name}")
    if bundle.include_states:
        labels.append("states")
    return f"{base_name}_{'_'.join(labels)}"


def _single_shooting_bundle_output_names(
    problem: SingleShootingProblem,
    bundle: SingleShootingBundle,
) -> tuple[str, ...]:
    """Return output names for a joint bundle."""
    names: list[str] = []
    if bundle.include_cost:
        names.append("cost")
    if bundle.include_gradient:
        names.append(f"gradient_{problem.control_sequence_name}")
    if bundle.include_states:
        names.append("x_traj")
    return tuple(names)


def _validate_single_shooting_bundle(bundle: SingleShootingBundle) -> None:
    """Validate a joint single-shooting bundle."""
    if not (bundle.include_cost or bundle.include_gradient):
        raise ValueError("SingleShootingBundle must request at least cost or gradient")
    if sum((bundle.include_cost, bundle.include_gradient, bundle.include_states)) < 2:
        raise ValueError("joint single-shooting kernels require at least two requested outputs")


def _validate_single_shooting_dynamics(function: Function) -> None:
    """Validate the dynamics function signature."""
    if len(function.inputs) != 3:
        raise ValueError("SingleShootingProblem dynamics must accept (x, u, p)")
    if len(function.outputs) != 1:
        raise ValueError("SingleShootingProblem dynamics must return exactly one output")
    if not _same_single_shooting_shape(function.inputs[0], function.outputs[0]):
        raise ValueError("SingleShootingProblem dynamics must return the next state with the same shape as x")


def _validate_single_shooting_stage_cost(function: Function) -> None:
    """Validate the stage-cost function signature."""
    if len(function.inputs) != 3:
        raise ValueError("SingleShootingProblem stage_cost must accept (x, u, p)")
    if len(function.outputs) != 1 or not isinstance(function.outputs[0], SX):
        raise ValueError("SingleShootingProblem stage_cost must return exactly one scalar output")


def _validate_single_shooting_terminal_cost(function: Function) -> None:
    """Validate the terminal-cost function signature."""
    if len(function.inputs) != 2:
        raise ValueError("SingleShootingProblem terminal_cost must accept (x, p)")
    if len(function.outputs) != 1 or not isinstance(function.outputs[0], SX):
        raise ValueError("SingleShootingProblem terminal_cost must return exactly one scalar output")


def _validate_single_shooting_shapes(
    dynamics: Function,
    stage_cost: Function,
    terminal_cost: Function,
) -> None:
    """Validate that all stage functions agree on state, control, and parameter shapes."""
    if not _same_single_shooting_shape(dynamics.inputs[0], stage_cost.inputs[0]):
        raise ValueError("stage_cost x input must have the same shape as dynamics x")
    if not _same_single_shooting_shape(dynamics.inputs[1], stage_cost.inputs[1]):
        raise ValueError("stage_cost u input must have the same shape as dynamics u")
    if not _same_single_shooting_shape(dynamics.inputs[2], stage_cost.inputs[2]):
        raise ValueError("stage_cost p input must have the same shape as dynamics p")
    if not _same_single_shooting_shape(dynamics.outputs[0], terminal_cost.inputs[0]):
        raise ValueError("terminal_cost x input must have the same shape as the dynamics state")
    if not _same_single_shooting_shape(dynamics.inputs[2], terminal_cost.inputs[1]):
        raise ValueError("terminal_cost p input must have the same shape as dynamics p")
