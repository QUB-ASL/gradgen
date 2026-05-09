"""Loop-structured single-shooting optimal-control abstractions."""

from __future__ import annotations

from dataclasses import dataclass, replace

from .function import Function
from ._staged import _create_rust_project
from ._staged import _generate_rust
from ._staged import _simplify_function
from .sx import SX, SXVector

FunctionArg = SX | SXVector


@dataclass(frozen=True, slots=True)
class SingleShootingBundle:
    """Requested outputs for a joint single-shooting kernel."""

    include_cost: bool = False
    include_gradient: bool = False
    include_hvp: bool = False
    include_states: bool = False

    def add_cost(self) -> SingleShootingBundle:
        """Include the total cost in the joint kernel outputs."""
        return replace(self, include_cost=True)

    def add_gradient(self) -> SingleShootingBundle:
        """Include the gradient with respect to the packed control sequence."""
        return replace(self, include_gradient=True)

    def add_hvp(self) -> SingleShootingBundle:
        """Include the HVP with respect to the packed control sequence."""
        return replace(self, include_hvp=True)

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
        """Expand this staged kernel into a symbolic ``Function``."""
        function = self.problem.to_function(
            include_states=self.include_states,
            name=name or self.name,
        )
        return _simplify_function(function, self.simplification)

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
        return _simplify_function(gradient_function, self.simplification)

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
class SingleShootingHvpFunction:
    """HVP kernel for a single-shooting optimal-control problem."""

    problem: SingleShootingProblem
    name: str
    include_states: bool = False
    simplification: int | str | None = None

    def to_function(self, name: str | None = None) -> Function:
        """Expand this staged HVP kernel into a symbolic ``Function``."""
        hvp_function = self.problem._expanded_hvp_function(
            include_states=self.include_states,
            name=name or self.name,
        )
        return _simplify_function(hvp_function, self.simplification)

    @property
    def nodes(self):
        """Return dependency nodes for shared helper discovery."""
        return self.to_function().nodes

    @property
    def input_names(self) -> tuple[str, ...]:
        """Return the exposed input names."""
        return (
            *self.problem.input_names,
            f"v_{self.problem.control_sequence_name}",
        )

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
    ):
        """Generate compact Rust for the staged HVP kernel."""
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
        """Create a Rust crate containing the staged HVP kernel."""
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
class SingleShootingJointFunction:
    """Joint cost/gradient/HVP/state kernel for a single-shooting problem."""

    problem: SingleShootingProblem
    bundle: SingleShootingBundle
    name: str
    simplification: int | str | None = None

    def to_function(self, name: str | None = None) -> Function:
        """Expand this staged joint kernel into a symbolic ``Function``."""
        function = self.problem._expanded_joint_function(
            self.bundle,
            name=name or self.name,
        )
        return _simplify_function(function, self.simplification)

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
class SingleShootingProblem:
    """Deterministic single-shooting optimal-control problem.

    The problem represents a fixed-horizon rollout with dynamics
    ``x_next = f(x, u, p)`` and total cost
    ``sum_k ell(x_k, u_k, p) + V_f(x_N, p)``. Optional vector-valued
    penalty residuals may be supplied to augment this cost as
    ``c / 2 * ||q(x_k, u_k, p)||_2^2`` at each stage and
    ``c / 2 * ||q_N(x_N, p)||_2^2`` at the terminal state.

    Args:
        name: Name used for expanded functions and generated kernels.
        horizon: Optional positive rollout horizon. It may be supplied in the
            constructor or later with :meth:`with_horizon`.
        dynamics: Optional function accepting ``(x, u, p)`` and returning the
            next state with the same shape as ``x``. It may be supplied in the
            constructor or later with :meth:`with_dynamics`.
        stage_cost: Optional scalar function accepting ``(x, u, p)`` and
            returning ``ell(x, u, p)``. It may be supplied in the constructor
            or later with :meth:`with_stage_cost`.
        terminal_cost: Optional scalar function accepting ``(x, p)`` and
            returning ``V_f(x, p)``. It may be supplied in the constructor or
            later with :meth:`with_terminal_cost`.
        initial_state_name: Runtime input name for the initial state.
        control_sequence_name: Runtime input name for the packed controls.
        parameter_name: Runtime input name for the shared parameters.
        simplification: Optional simplification effort applied to expanded
            functions and derivative helper kernels.
        stage_penalty: Optional vector-valued residual function accepting
            ``(x, u, p)`` and returning ``q(x, u, p)``.
        terminal_penalty: Optional vector-valued residual function accepting
            ``(x, p)`` and returning ``q_N(x, p)``.
        penalty_weight: Optional scalar ``c`` multiplying both squared
            residual norms. Pass a numeric value to bake ``c`` into generated
            code, or pass an ``SX`` symbol such as ``SX.sym("c")`` to expose
            ``c`` as a runtime scalar input.
    """

    name: str
    horizon: int | None = None
    dynamics: Function | None = None
    stage_cost: Function | None = None
    terminal_cost: Function | None = None
    initial_state_name: str = "x0"
    control_sequence_name: str = "U"
    parameter_name: str = "p"
    simplification: int | str | None = None
    stage_penalty: Function | None = None
    terminal_penalty: Function | None = None
    penalty_weight: float | SX | None = None

    def __post_init__(self) -> None:
        """Validate stage and cost signatures."""
        if self.horizon is not None and self.horizon <= 0:
            raise ValueError("horizon must be a positive integer")
        if self.dynamics is not None:
            _validate_single_shooting_dynamics(self.dynamics)
        if self.stage_cost is not None:
            _validate_single_shooting_stage_cost(self.stage_cost)
        if self.terminal_cost is not None:
            _validate_single_shooting_terminal_cost(self.terminal_cost)
        if self._has_complete_core():
            _validate_single_shooting_penalty_configuration(
                self.stage_penalty,
                self.terminal_penalty,
                self.penalty_weight,
            )
        if self.stage_penalty is not None:
            _validate_single_shooting_stage_penalty(self.stage_penalty)
        if self.terminal_penalty is not None:
            _validate_single_shooting_terminal_penalty(
                self.terminal_penalty
            )
        if self._has_complete_core():
            _validate_single_shooting_shapes(
                self._dynamics(),
                self._stage_cost(),
                self._terminal_cost(),
                self.stage_penalty,
                self.terminal_penalty,
            )

    def with_horizon(self, horizon: int) -> SingleShootingProblem:
        """Return a copy configured with a positive rollout horizon.

        Args:
            horizon: Number of control intervals in the single-shooting
                rollout. It must be a positive integer.

        Returns:
            A new :class:`SingleShootingProblem` with the requested horizon.
        """
        return replace(self, horizon=horizon)

    def with_dynamics(self, dynamics: Function) -> SingleShootingProblem:
        """Return a copy configured with the dynamics function.

        Args:
            dynamics: Function accepting ``(x, u, p)`` and returning the next
                state with the same shape as ``x``.

        Returns:
            A new :class:`SingleShootingProblem` using ``dynamics``.
        """
        return replace(self, dynamics=dynamics)

    def with_stage_cost(self, stage_cost: Function) -> SingleShootingProblem:
        """Return a copy configured with the scalar stage cost.

        Args:
            stage_cost: Function accepting ``(x, u, p)`` and returning one
                scalar output.

        Returns:
            A new :class:`SingleShootingProblem` using ``stage_cost``.
        """
        return replace(self, stage_cost=stage_cost)

    def with_terminal_cost(
        self, terminal_cost: Function
    ) -> SingleShootingProblem:
        """Return a copy configured with the scalar terminal cost.

        Args:
            terminal_cost: Function accepting ``(x, p)`` and returning one
                scalar output.

        Returns:
            A new :class:`SingleShootingProblem` using ``terminal_cost``.
        """
        return replace(self, terminal_cost=terminal_cost)

    def with_costs(
        self,
        stage_cost: Function,
        terminal_cost: Function,
    ) -> SingleShootingProblem:
        """Return a copy configured with stage and terminal costs.

        Args:
            stage_cost: Scalar stage cost accepting ``(x, u, p)``.
            terminal_cost: Scalar terminal cost accepting ``(x, p)``.

        Returns:
            A new :class:`SingleShootingProblem` using both costs.
        """
        return replace(
            self,
            stage_cost=stage_cost,
            terminal_cost=terminal_cost,
        )

    def with_penalties(
        self,
        stage_penalty: Function,
        terminal_penalty: Function,
        penalty_weight: float | SX,
    ) -> SingleShootingProblem:
        """Return a copy configured with residual penalties.

        Args:
            stage_penalty: Vector or scalar residual accepting ``(x, u, p)``.
            terminal_penalty: Vector or scalar residual accepting ``(x, p)``.
            penalty_weight: Numeric penalty weight, or an ``SX`` symbol such
                as ``SX.sym("c")`` to expose the weight as a runtime input.

        Returns:
            A new :class:`SingleShootingProblem` with residual penalties.
        """
        if penalty_weight is None:
            raise ValueError("penalty_weight must be provided")
        return replace(
            self,
            stage_penalty=stage_penalty,
            terminal_penalty=terminal_penalty,
            penalty_weight=penalty_weight,
        )

    def with_input_names(
        self,
        *,
        initial_state_name: str | None = None,
        control_sequence_name: str | None = None,
        parameter_name: str | None = None,
    ) -> SingleShootingProblem:
        """Return a copy with customized generated input names.

        Args:
            initial_state_name: Optional runtime input name for the initial
                state.
            control_sequence_name: Optional runtime input name for the packed
                control sequence.
            parameter_name: Optional runtime input name for shared
                parameters.

        Returns:
            A new :class:`SingleShootingProblem` with updated input names.
        """
        return replace(
            self,
            initial_state_name=(
                self.initial_state_name
                if initial_state_name is None else initial_state_name
            ),
            control_sequence_name=(
                self.control_sequence_name
                if control_sequence_name is None else control_sequence_name
            ),
            parameter_name=(
                self.parameter_name if parameter_name is None
                else parameter_name
            ),
        )

    def with_simplification(
        self, simplification: int | str | None
    ) -> SingleShootingProblem:
        """Return a copy with the requested simplification effort.

        Args:
            simplification: Simplification effort used for expanded functions
                and derivative helper kernels.

        Returns:
            A new :class:`SingleShootingProblem` with the simplification
            setting.
        """
        return replace(self, simplification=simplification)

    @property
    def state_size(self) -> int:
        """Return the state dimension."""
        return _single_shooting_arg_size(self._dynamics().inputs[0])

    @property
    def control_size(self) -> int:
        """Return the per-stage control dimension."""
        return _single_shooting_arg_size(self._dynamics().inputs[1])

    @property
    def parameter_size(self) -> int:
        """Return the shared parameter-vector dimension."""
        return _single_shooting_arg_size(self._dynamics().inputs[2])

    @property
    def has_runtime_penalty_weight(self) -> bool:
        """Return whether ``c`` is exposed as a runtime scalar input."""
        return isinstance(self.penalty_weight, SX)

    @property
    def penalty_weight_name(self) -> str | None:
        """Return the runtime input name for symbolic ``c``."""
        if isinstance(self.penalty_weight, SX):
            return self.penalty_weight.name or "c"
        return None

    @property
    def input_names(self) -> tuple[str, ...]:
        """Return the exposed runtime input names."""
        names = [
            self.initial_state_name,
            self.control_sequence_name,
            self.parameter_name,
        ]
        if self.has_runtime_penalty_weight:
            penalty_weight_name = self.penalty_weight_name
            if penalty_weight_name is not None:
                names.append(penalty_weight_name)
        return tuple(names)

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

    def _has_complete_core(self) -> bool:
        """Return whether the required problem definition is present."""
        return (
            self.horizon is not None
            and self.dynamics is not None
            and self.stage_cost is not None
            and self.terminal_cost is not None
        )

    def _require_complete(self) -> None:
        """Raise a helpful error when a builder-style problem is incomplete."""
        missing: list[str] = []
        if self.horizon is None:
            missing.append("horizon")
        if self.dynamics is None:
            missing.append("dynamics")
        if self.stage_cost is None:
            missing.append("stage_cost")
        if self.terminal_cost is None:
            missing.append("terminal_cost")
        if missing:
            joined = ", ".join(missing)
            raise ValueError(
                "SingleShootingProblem is incomplete; configure "
                f"{joined} before expanding or generating code"
            )

    def _dynamics(self) -> Function:
        """Return configured dynamics after checking completeness."""
        self._require_complete()
        assert self.dynamics is not None
        return self.dynamics

    def _stage_cost(self) -> Function:
        """Return configured stage cost after checking completeness."""
        self._require_complete()
        assert self.stage_cost is not None
        return self.stage_cost

    def _terminal_cost(self) -> Function:
        """Return configured terminal cost after checking completeness."""
        self._require_complete()
        assert self.terminal_cost is not None
        return self.terminal_cost

    def _horizon(self) -> int:
        """Return configured horizon after checking completeness."""
        self._require_complete()
        assert self.horizon is not None
        return self.horizon

    def primal(
        self,
        *,
        include_states: bool = False,
        name: str | None = None,
    ) -> SingleShootingProblem | SingleShootingPrimalFunction:
        """Return a staged primal kernel source."""
        self._require_complete()
        if not include_states and name is None:
            return self
        return SingleShootingPrimalFunction(
            problem=self,
            name=name
            or _single_shooting_primal_name(self.name, include_states),
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
        self._require_complete()
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
        self._require_complete()
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
        self._require_complete()
        _validate_single_shooting_bundle(bundle)
        return SingleShootingJointFunction(
            problem=self,
            bundle=bundle,
            name=name
            or _single_shooting_joint_name(
                self.name, bundle, self.control_sequence_name
            ),
            simplification=self.simplification,
        )

    def to_function(
        self,
        *,
        include_states: bool = False,
        name: str | None = None,
    ) -> Function:
        """Expand the total-cost kernel into a symbolic ``Function``."""
        self._require_complete()
        dynamics = self._dynamics()
        horizon = self._horizon()
        compiled_inputs = self._compiled_inputs()
        x0, U, p = compiled_inputs[:3]
        penalty_weight = (
            compiled_inputs[3]
            if self.has_runtime_penalty_weight
            else None
        )
        current_state: FunctionArg = x0
        rollout_states: list[FunctionArg] = [current_state]
        total_cost = SX.const(0.0)

        for stage_index in range(horizon):
            u_t = _slice_packed_sequence(
                U, stage_index, self.control_size, dynamics.inputs[1]
            )
            total_cost = total_cost + self._stage_total_cost(
                current_state, u_t, p, penalty_weight
            )
            current_state = _extract_single_output(
                dynamics(current_state, u_t, p)
            )
            rollout_states.append(current_state)

        total_cost = total_cost + self._terminal_total_cost(
            current_state, p, penalty_weight
        )
        outputs: list[FunctionArg] = [total_cost]
        output_names = ["cost"]
        if include_states:
            outputs.append(_flatten_rollout_states(rollout_states))
            output_names.append("x_traj")

        function = Function(
            name or self.name,
            compiled_inputs,
            outputs,
            input_names=self.input_names,
            output_names=tuple(output_names),
            single_shooting_problem=self,
            single_shooting_include_states=include_states,
        )
        return _simplify_function(function, self.simplification)

    def _expanded_gradient_function(
        self,
        *,
        include_states: bool,
        name: str,
    ) -> Function:
        """Expand the staged gradient kernel into a symbolic ``Function``."""
        cost_function = self.to_function(
            include_states=False, name=f"{name}_cost"
        )
        gradient_function = cost_function.gradient(1, name=f"{name}_grad")
        outputs: list[FunctionArg] = [gradient_function.outputs[0]]
        output_names = [f"gradient_{self.control_sequence_name}"]
        if include_states:
            outputs.append(
                self.to_function(
                    include_states=True, name=f"{name}_states"
                ).outputs[1]
            )
            output_names.append("x_traj")
        return Function(
            name,
            gradient_function.inputs,
            outputs,
            input_names=gradient_function.input_names,
            output_names=tuple(output_names),
            single_shooting_problem=self,
            single_shooting_include_states=include_states,
        )

    def _expanded_hvp_function(
        self,
        *,
        include_states: bool,
        name: str,
    ) -> Function:
        """Expand the staged HVP kernel into a symbolic ``Function``."""
        cost_function = self.to_function(
            include_states=False, name=f"{name}_cost"
        )
        hvp_function = cost_function.hvp(1, name=f"{name}_hvp")
        outputs: list[FunctionArg] = [hvp_function.outputs[0]]
        output_names = [f"hvp_{self.control_sequence_name}"]
        if include_states:
            outputs.append(
                self.to_function(
                    include_states=True, name=f"{name}_states"
                ).outputs[1]
            )
            output_names.append("x_traj")
        return Function(
            name,
            hvp_function.inputs,
            outputs,
            input_names=hvp_function.input_names,
            output_names=tuple(output_names),
            single_shooting_problem=self,
            single_shooting_include_states=include_states,
        )

    def _expanded_joint_function(
        self,
        bundle: SingleShootingBundle,
        *,
        name: str,
    ) -> Function:
        """Expand a staged joint kernel into a symbolic ``Function``."""
        _validate_single_shooting_bundle(bundle)
        cost_function = self.to_function(
            include_states=False, name=f"{name}_cost"
        )
        gradient_function = cost_function.gradient(1, name=f"{name}_grad")
        hvp_function = (
            cost_function.hvp(1, name=f"{name}_hvp")
            if bundle.include_hvp
            else None
        )
        outputs: list[FunctionArg] = []
        output_names: list[str] = []
        if bundle.include_cost:
            outputs.append(cost_function.outputs[0])
            output_names.append("cost")
        if bundle.include_gradient:
            outputs.append(gradient_function.outputs[0])
            output_names.append(f"gradient_{self.control_sequence_name}")
        if bundle.include_hvp:
            assert hvp_function is not None
            outputs.append(hvp_function.outputs[0])
            output_names.append(f"hvp_{self.control_sequence_name}")
        if bundle.include_states:
            outputs.append(
                self.to_function(
                    include_states=True, name=f"{name}_states"
                ).outputs[1]
            )
            output_names.append("x_traj")
        inputs = (
            hvp_function.inputs
            if hvp_function is not None
            else cost_function.inputs
        )
        input_names = (
            hvp_function.input_names
            if hvp_function is not None
            else cost_function.input_names
        )
        return Function(
            name,
            inputs,
            outputs,
            input_names=input_names,
            output_names=tuple(output_names),
            single_shooting_problem=self,
            single_shooting_include_states=bundle.include_states,
        )

    def _compiled_inputs(self) -> tuple[FunctionArg, ...]:
        """Return symbolic runtime inputs."""
        dynamics = self._dynamics()
        horizon = self._horizon()
        x0 = _make_symbolic_like(
            dynamics.inputs[0], self.initial_state_name
        )
        p = _make_symbolic_like(dynamics.inputs[2], self.parameter_name)
        U = SXVector.sym(
            self.control_sequence_name, horizon * self.control_size
        )
        if self.has_runtime_penalty_weight:
            return x0, U, p, SX.sym(self.penalty_weight_name or "c")
        return x0, U, p

    def stage_total_cost_function(self) -> Function:
        """Return the scalar stage cost including residual penalties.

        The returned function has the same ``(x, u, p)`` signature as
        :attr:`stage_cost` when ``penalty_weight`` is numeric, and
        ``(x, u, p, c)`` when ``penalty_weight`` is symbolic. When
        :attr:`stage_penalty` is present its output is squared, summed,
        multiplied by ``penalty_weight / 2``, and added to the base stage
        cost.

        Returns:
            A scalar :class:`~gradgen.function.Function` for the effective
            stage contribution used by primal, gradient, HVP, and Rust
            code-generation paths.
        """
        stage_cost = self._stage_cost()
        x, u, p = stage_cost.inputs
        penalty_weight = self._helper_penalty_weight_symbol()
        inputs = (x, u, p)
        input_names = stage_cost.input_names
        if penalty_weight is not None:
            inputs = (*inputs, penalty_weight)
            input_names = (*input_names, self.penalty_weight_name or "c")
        return Function(
            f"{stage_cost.name}_with_penalty",
            inputs,
            (self._stage_total_cost(x, u, p, penalty_weight),),
            input_names=input_names,
            output_names=stage_cost.output_names,
            single_shooting_problem=self,
        )

    def terminal_total_cost_function(self) -> Function:
        """Return the scalar terminal cost including residual penalties.

        The returned function has the same ``(x, p)`` signature as
        :attr:`terminal_cost` when ``penalty_weight`` is numeric, and
        ``(x, p, c)`` when ``penalty_weight`` is symbolic. When
        :attr:`terminal_penalty` is present its output is squared, summed,
        multiplied by ``penalty_weight / 2``, and added to the base terminal
        cost.

        Returns:
            A scalar :class:`~gradgen.function.Function` for the effective
            terminal contribution used by primal, gradient, HVP, and Rust
            code-generation paths.
        """
        terminal_cost = self._terminal_cost()
        x, p = terminal_cost.inputs
        penalty_weight = self._helper_penalty_weight_symbol()
        inputs = (x, p)
        input_names = terminal_cost.input_names
        if penalty_weight is not None:
            inputs = (*inputs, penalty_weight)
            input_names = (*input_names, self.penalty_weight_name or "c")
        return Function(
            f"{terminal_cost.name}_with_penalty",
            inputs,
            (self._terminal_total_cost(x, p, penalty_weight),),
            input_names=input_names,
            output_names=terminal_cost.output_names,
            single_shooting_problem=self,
        )

    def _stage_total_cost(
        self,
        x: FunctionArg,
        u: FunctionArg,
        p: FunctionArg,
        penalty_weight: SX | None = None,
    ) -> SX:
        """Return the symbolic stage cost including residual penalties."""
        cost = _extract_scalar_output(self._stage_cost()(x, u, p))
        if self.stage_penalty is None:
            return cost
        return cost + self._weighted_squared_norm(
            _extract_single_output(self.stage_penalty(x, u, p)),
            penalty_weight,
        )

    def _terminal_total_cost(
        self,
        x: FunctionArg,
        p: FunctionArg,
        penalty_weight: SX | None = None,
    ) -> SX:
        """Return the symbolic terminal cost including residual penalties."""
        cost = _extract_scalar_output(self._terminal_cost()(x, p))
        if self.terminal_penalty is None:
            return cost
        return cost + self._weighted_squared_norm(
            _extract_single_output(self.terminal_penalty(x, p)),
            penalty_weight,
        )

    def _weighted_squared_norm(
        self,
        residual: FunctionArg,
        penalty_weight: SX | None = None,
    ) -> SX:
        """Return ``penalty_weight / 2 * ||residual||_2^2``."""
        if self.penalty_weight is None:
            raise ValueError("penalty_weight must be provided")
        if penalty_weight is not None:
            weight = penalty_weight
        elif isinstance(self.penalty_weight, SX):
            weight = self.penalty_weight
        else:
            weight = SX.const(float(self.penalty_weight))
        return (SX.const(0.5) * weight) * _squared_norm(residual)

    def _helper_penalty_weight_symbol(self) -> SX | None:
        """Return the helper input symbol for runtime penalty weights."""
        if not self.has_runtime_penalty_weight:
            return None
        return SX.sym(self.penalty_weight_name or "c")

    def generate_rust(
        self,
        *,
        config=None,
        function_name: str | None = None,
        backend_mode: str = "std",
        scalar_type: str = "f64",
    ):
        """Generate compact Rust for the primal total-cost kernel."""
        self._require_complete()
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
        """Create a Rust crate containing the total-cost kernel."""
        self._require_complete()
        return _create_rust_project(
            self,
            path,
            config=config,
            crate_name=crate_name,
            function_name=function_name,
            backend_mode=backend_mode,
            scalar_type=scalar_type,
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
    return SXVector(sequence.elements[start: start + block_size])


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
        raise ValueError(
            "single-shooting cost functions must return scalar outputs"
        )
    return output


def _squared_norm(value: FunctionArg) -> SX:
    """Return the sum of squares for a scalar or vector symbolic value."""
    if isinstance(value, SX):
        return value * value
    total = SX.const(0.0)
    for element in value.elements:
        total = total + element * element
    return total


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
    if bundle.include_hvp:
        labels.append(f"hvp_{control_sequence_name}")
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
    if bundle.include_hvp:
        names.append(f"hvp_{problem.control_sequence_name}")
    if bundle.include_states:
        names.append("x_traj")
    return tuple(names)


def _validate_single_shooting_bundle(bundle: SingleShootingBundle) -> None:
    """Validate a joint single-shooting bundle."""
    if not (
        bundle.include_cost or bundle.include_gradient or bundle.include_hvp
    ):
        raise ValueError(
            "SingleShootingBundle must request at least cost, gradient, or HVP"
        )
    if (
        sum(
            (
                bundle.include_cost,
                bundle.include_gradient,
                bundle.include_hvp,
                bundle.include_states,
            )
        )
        < 2
    ):
        raise ValueError(
            "joint single-shooting kernels require at least two "
            "requested outputs"
        )


def _validate_single_shooting_dynamics(function: Function) -> None:
    """Validate the dynamics function signature."""
    if len(function.inputs) != 3:
        raise ValueError(
            "SingleShootingProblem dynamics must accept (x, u, p)"
        )
    if len(function.outputs) != 1:
        raise ValueError(
            "SingleShootingProblem dynamics must return exactly one output"
        )
    if not _same_single_shooting_shape(
        function.inputs[0], function.outputs[0]
    ):
        raise ValueError(
            "SingleShootingProblem dynamics must return the next state "
            "with the same shape as x"
        )


def _validate_single_shooting_stage_cost(function: Function) -> None:
    """Validate the stage-cost function signature."""
    if len(function.inputs) != 3:
        raise ValueError(
            "SingleShootingProblem stage_cost must accept (x, u, p)"
        )
    if len(function.outputs) != 1 or not isinstance(function.outputs[0], SX):
        raise ValueError(
            "SingleShootingProblem stage_cost must return exactly one "
            "scalar output"
        )


def _validate_single_shooting_terminal_cost(function: Function) -> None:
    """Validate the terminal-cost function signature."""
    if len(function.inputs) != 2:
        raise ValueError(
            "SingleShootingProblem terminal_cost must accept (x, p)"
        )
    if len(function.outputs) != 1 or not isinstance(function.outputs[0], SX):
        raise ValueError(
            "SingleShootingProblem terminal_cost must return exactly one "
            "scalar output"
        )


def _validate_single_shooting_penalty_configuration(
    stage_penalty: Function | None,
    terminal_penalty: Function | None,
    penalty_weight: float | SX | None,
) -> None:
    """Validate optional residual penalty fields are supplied together."""
    has_any_penalty = stage_penalty is not None or terminal_penalty is not None
    if has_any_penalty and (
        stage_penalty is None
        or terminal_penalty is None
        or penalty_weight is None
    ):
        raise ValueError(
            "SingleShootingProblem penalties require stage_penalty, "
            "terminal_penalty, and penalty_weight"
        )
    if isinstance(penalty_weight, SXVector):
        raise TypeError("penalty_weight must be a scalar")


def _validate_single_shooting_stage_penalty(function: Function) -> None:
    """Validate the stage residual signature."""
    if len(function.inputs) != 3:
        raise ValueError(
            "SingleShootingProblem stage_penalty must accept (x, u, p)"
        )
    if len(function.outputs) != 1:
        raise ValueError(
            "SingleShootingProblem stage_penalty must return exactly one "
            "residual output"
        )
    if not isinstance(function.outputs[0], (SX, SXVector)):
        raise ValueError(
            "SingleShootingProblem stage_penalty must return a scalar or "
            "vector symbolic residual"
        )


def _validate_single_shooting_terminal_penalty(function: Function) -> None:
    """Validate the terminal residual signature."""
    if len(function.inputs) != 2:
        raise ValueError(
            "SingleShootingProblem terminal_penalty must accept (x, p)"
        )
    if len(function.outputs) != 1:
        raise ValueError(
            "SingleShootingProblem terminal_penalty must return exactly one "
            "residual output"
        )
    if not isinstance(function.outputs[0], (SX, SXVector)):
        raise ValueError(
            "SingleShootingProblem terminal_penalty must return a scalar or "
            "vector symbolic residual"
        )


def _validate_single_shooting_shapes(
    dynamics: Function,
    stage_cost: Function,
    terminal_cost: Function,
    stage_penalty: Function | None = None,
    terminal_penalty: Function | None = None,
) -> None:
    """Validate stage functions agree on state, control, and parameters."""
    if not _same_single_shooting_shape(
        dynamics.inputs[0], stage_cost.inputs[0]
    ):
        raise ValueError(
            "stage_cost x input must have the same shape as dynamics x"
        )
    if not _same_single_shooting_shape(
        dynamics.inputs[1], stage_cost.inputs[1]
    ):
        raise ValueError(
            "stage_cost u input must have the same shape as dynamics u"
        )
    if not _same_single_shooting_shape(
        dynamics.inputs[2], stage_cost.inputs[2]
    ):
        raise ValueError(
            "stage_cost p input must have the same shape as dynamics p"
        )
    if not _same_single_shooting_shape(
        dynamics.outputs[0], terminal_cost.inputs[0]
    ):
        raise ValueError(
            "terminal_cost x input must have the same shape as the "
            "dynamics state"
        )
    if not _same_single_shooting_shape(
        dynamics.inputs[2], terminal_cost.inputs[1]
    ):
        raise ValueError(
            "terminal_cost p input must have the same shape as dynamics p"
        )
    if stage_penalty is not None:
        if not _same_single_shooting_shape(
            dynamics.inputs[0], stage_penalty.inputs[0]
        ):
            raise ValueError(
                "stage_penalty x input must have the same shape as "
                "dynamics x"
            )
        if not _same_single_shooting_shape(
            dynamics.inputs[1], stage_penalty.inputs[1]
        ):
            raise ValueError(
                "stage_penalty u input must have the same shape as "
                "dynamics u"
            )
        if not _same_single_shooting_shape(
            dynamics.inputs[2], stage_penalty.inputs[2]
        ):
            raise ValueError(
                "stage_penalty p input must have the same shape as "
                "dynamics p"
            )
    if terminal_penalty is not None:
        if not _same_single_shooting_shape(
            dynamics.outputs[0], terminal_penalty.inputs[0]
        ):
            raise ValueError(
                "terminal_penalty x input must have the same shape as the "
                "dynamics state"
            )
        if not _same_single_shooting_shape(
            dynamics.inputs[2], terminal_penalty.inputs[1]
        ):
            raise ValueError(
                "terminal_penalty p input must have the same shape as "
                "dynamics p"
            )
