import unittest

from gradgen import (
    CodeGenerationBuilder,
    Function,
    RustBackendConfig,
    SX,
    SXVector,
    SingleShootingBundle,
    SingleShootingProblem,
)


def _build_reference_problem(*, horizon: int = 3) -> SingleShootingProblem:
    x = SXVector.sym("x", 2)
    u = SXVector.sym("u", 1)
    p = SXVector.sym("p", 2)

    dynamics = Function(
        "dynamics",
        [x, u, p],
        [SXVector((x[0] + p[0] * x[1] + u[0], x[1] + p[1] * u[0] - 0.5 * x[0]))],
        input_names=["x", "u", "p"],
        output_names=["x_next"],
    )
    stage_cost = Function(
        "stage_cost",
        [x, u, p],
        [x[0] * x[0] + 2.0 * x[1] * x[1] + 0.3 * u[0] * u[0] + p[0] * u[0]],
        input_names=["x", "u", "p"],
        output_names=["ell"],
    )
    terminal_cost = Function(
        "terminal_cost",
        [x, p],
        [3.0 * x[0] * x[0] + 0.5 * x[1] * x[1] + p[1] * x[0]],
        input_names=["x", "p"],
        output_names=["vf"],
    )
    return SingleShootingProblem(
        name="mpc_cost",
        horizon=horizon,
        dynamics=dynamics,
        stage_cost=stage_cost,
        terminal_cost=terminal_cost,
        initial_state_name="x0",
        control_sequence_name="U",
        parameter_name="p",
    )


def _manual_rollout(x0: list[float], U: list[float], p: list[float], horizon: int) -> tuple[float, list[float]]:
    current = [float(x0[0]), float(x0[1])]
    packed_states = [current[0], current[1]]
    total_cost = 0.0

    for stage_index in range(horizon):
        u_t = float(U[stage_index])
        total_cost += current[0] * current[0] + 2.0 * current[1] * current[1] + 0.3 * u_t * u_t + p[0] * u_t
        next_state = [
            current[0] + p[0] * current[1] + u_t,
            current[1] + p[1] * u_t - 0.5 * current[0],
        ]
        current = next_state
        packed_states.extend(current)

    total_cost += 3.0 * current[0] * current[0] + 0.5 * current[1] * current[1] + p[1] * current[0]
    return total_cost, packed_states


def _finite_difference_gradient(
    x0: list[float],
    U: list[float],
    p: list[float],
    horizon: int,
    *,
    epsilon: float = 1e-6,
) -> list[float]:
    gradient: list[float] = []
    for control_index in range(len(U)):
        forward = list(U)
        backward = list(U)
        forward[control_index] += epsilon
        backward[control_index] -= epsilon
        forward_cost, _ = _manual_rollout(x0, forward, p, horizon)
        backward_cost, _ = _manual_rollout(x0, backward, p, horizon)
        gradient.append((forward_cost - backward_cost) / (2.0 * epsilon))
    return gradient


def _finite_difference_hvp(
    x0: list[float],
    U: list[float],
    p: list[float],
    v_U: list[float],
    horizon: int,
    *,
    epsilon: float = 1e-4,
) -> list[float]:
    forward_controls = [
        control + epsilon * direction for control, direction in zip(U, v_U)
    ]
    backward_controls = [
        control - epsilon * direction for control, direction in zip(U, v_U)
    ]
    forward_gradient = _finite_difference_gradient(x0, forward_controls, p, horizon, epsilon=epsilon)
    backward_gradient = _finite_difference_gradient(x0, backward_controls, p, horizon, epsilon=epsilon)
    return [
        (forward_value - backward_value) / (2.0 * epsilon)
        for forward_value, backward_value in zip(forward_gradient, backward_gradient)
    ]


def _build_scalar_reference_problem(*, horizon: int = 2) -> SingleShootingProblem:
    x = SX.sym("x")
    u = SX.sym("u")
    p = SX.sym("p")

    dynamics = Function(
        "scalar_dynamics",
        [x, u, p],
        [0.5 * x + p * u],
        input_names=["x", "u", "p"],
        output_names=["x_next"],
    )
    stage_cost = Function(
        "scalar_stage_cost",
        [x, u, p],
        [x * x + 2.0 * u * u + p * x * u],
        input_names=["x", "u", "p"],
        output_names=["ell"],
    )
    terminal_cost = Function(
        "scalar_terminal_cost",
        [x, p],
        [3.0 * x * x + p * x],
        input_names=["x", "p"],
        output_names=["vf"],
    )
    return SingleShootingProblem(
        name="scalar_cost",
        horizon=horizon,
        dynamics=dynamics,
        stage_cost=stage_cost,
        terminal_cost=terminal_cost,
        initial_state_name="x0",
        control_sequence_name="U",
        parameter_name="p",
    )


def _manual_scalar_rollout(x0: float, U: list[float], p: float, horizon: int) -> tuple[float, list[float]]:
    current = float(x0)
    packed_states = [current]
    total_cost = 0.0

    for stage_index in range(horizon):
        u_t = float(U[stage_index])
        total_cost += current * current + 2.0 * u_t * u_t + p * current * u_t
        current = 0.5 * current + p * u_t
        packed_states.append(current)

    total_cost += 3.0 * current * current + p * current
    return total_cost, packed_states


def _finite_difference_scalar_gradient(
    x0: float,
    U: list[float],
    p: float,
    horizon: int,
    *,
    epsilon: float = 1e-6,
) -> list[float]:
    gradient: list[float] = []
    for control_index in range(len(U)):
        forward = list(U)
        backward = list(U)
        forward[control_index] += epsilon
        backward[control_index] -= epsilon
        forward_cost, _ = _manual_scalar_rollout(x0, forward, p, horizon)
        backward_cost, _ = _manual_scalar_rollout(x0, backward, p, horizon)
        gradient.append((forward_cost - backward_cost) / (2.0 * epsilon))
    return gradient


class SingleShootingProblemTests(unittest.TestCase):
    def test_problem_expands_cost_gradient_and_rollout_states(self) -> None:
        problem = _build_reference_problem()
        x0 = [1.0, -0.5]
        U = [0.2, -0.1, 0.3]
        p = [0.4, -1.2]

        expected_cost, expected_states = _manual_rollout(x0, U, p, problem.horizon)
        expected_gradient = _finite_difference_gradient(x0, U, p, problem.horizon)

        cost_function = problem.to_function()
        self.assertEqual(cost_function.input_names, ("x0", "U", "p"))
        self.assertEqual(cost_function.output_names, ("cost",))
        self.assertAlmostEqual(cost_function(x0, U, p), expected_cost, places=10)

        cost_and_states = problem.to_function(include_states=True)
        actual_cost, actual_states = cost_and_states(x0, U, p)
        self.assertAlmostEqual(actual_cost, expected_cost, places=10)
        self.assertEqual(tuple(expected_states), actual_states)

        gradient_function = problem.gradient().to_function()
        actual_gradient = gradient_function(x0, U, p)
        self.assertEqual(len(actual_gradient), len(expected_gradient))
        for actual_value, expected_value in zip(actual_gradient, expected_gradient):
            self.assertAlmostEqual(actual_value, expected_value, places=5)

        gradient_and_states = problem.gradient(include_states=True).to_function()
        actual_gradient, actual_states = gradient_and_states(x0, U, p)
        self.assertEqual(tuple(expected_states), actual_states)
        for actual_value, expected_value in zip(actual_gradient, expected_gradient):
            self.assertAlmostEqual(actual_value, expected_value, places=5)

        joint_function = (
            problem.joint(
                SingleShootingBundle()
                .add_cost()
                .add_gradient()
                .add_rollout_states()
            )
            .to_function()
        )
        actual_cost, actual_gradient, actual_states = joint_function(x0, U, p)
        self.assertAlmostEqual(actual_cost, expected_cost, places=10)
        self.assertEqual(tuple(expected_states), actual_states)
        for actual_value, expected_value in zip(actual_gradient, expected_gradient):
            self.assertAlmostEqual(actual_value, expected_value, places=5)

    def test_problem_expands_hvp_and_rollout_states(self) -> None:
        # This checks the second-order path against a finite-difference
        # derivative of the control-sequence gradient, which is independent of
        # the single-shooting HVP implementation itself.
        problem = _build_reference_problem()
        x0 = [1.0, -0.5]
        U = [0.2, -0.1, 0.3]
        p = [0.4, -1.2]
        v_U = [0.5, -1.0, 0.25]

        _, expected_states = _manual_rollout(x0, U, p, problem.horizon)
        expected_hvp = _finite_difference_hvp(x0, U, p, v_U, problem.horizon)

        hvp_and_states = problem.hvp(include_states=True).to_function()
        actual_hvp, actual_states = hvp_and_states(x0, U, p, v_U)
        self.assertEqual(tuple(expected_states), actual_states)
        self.assertEqual(len(actual_hvp), len(expected_hvp))
        for actual_value, expected_value in zip(actual_hvp, expected_hvp):
            self.assertAlmostEqual(actual_value, expected_value, places=4)

    def test_validation_rejects_mismatched_stage_signatures(self) -> None:
        x = SXVector.sym("x", 2)
        u = SXVector.sym("u", 1)
        p = SXVector.sym("p", 2)

        dynamics = Function(
            "dynamics",
            [x, u, p],
            [SXVector((x[0] + u[0], x[1] + p[0]))],
            input_names=["x", "u", "p"],
            output_names=["x_next"],
        )
        bad_stage_cost = Function(
            "bad_stage_cost",
            [x, SXVector.sym("u2", 2), p],
            [x[0] + x[1]],
            input_names=["x", "u2", "p"],
            output_names=["ell"],
        )
        terminal_cost = Function(
            "terminal_cost",
            [x, p],
            [x[0]],
            input_names=["x", "p"],
            output_names=["vf"],
        )

        with self.assertRaises(ValueError):
            SingleShootingProblem(
                name="broken",
                horizon=3,
                dynamics=dynamics,
                stage_cost=bad_stage_cost,
                terminal_cost=terminal_cost,
            )

    def test_scalar_problem_supports_scalar_state_control_and_parameter(self) -> None:
        # This exercises the SX-only single-shooting path rather than the
        # vector path. It helps catch shape bugs in packed control slicing,
        # scalar rollout flattening, and gradient expansion for horizon data.
        problem = _build_scalar_reference_problem(horizon=2)
        x0 = 1.25
        U = [0.4, -0.2]
        p = -0.75

        expected_cost, expected_states = _manual_scalar_rollout(x0, U, p, problem.horizon)
        expected_gradient = _finite_difference_scalar_gradient(x0, U, p, problem.horizon)

        self.assertEqual(problem.state_size, 1)
        self.assertEqual(problem.control_size, 1)
        self.assertEqual(problem.parameter_size, 1)

        cost_and_states = problem.to_function(include_states=True)
        actual_cost, actual_states = cost_and_states(x0, U, p)
        self.assertAlmostEqual(actual_cost, expected_cost, places=10)
        self.assertEqual(tuple(expected_states), actual_states)

        gradient_and_states = problem.gradient(include_states=True).to_function()
        actual_gradient, actual_states = gradient_and_states(x0, U, p)
        self.assertEqual(tuple(expected_states), actual_states)
        self.assertEqual(len(actual_gradient), len(expected_gradient))
        for actual_value, expected_value in zip(actual_gradient, expected_gradient):
            self.assertAlmostEqual(actual_value, expected_value, places=5)

    def test_validation_rejects_nonpositive_horizon_and_bad_terminal_output(self) -> None:
        x = SXVector.sym("x", 2)
        u = SXVector.sym("u", 1)
        p = SXVector.sym("p", 2)

        dynamics = Function(
            "dynamics",
            [x, u, p],
            [SXVector((x[0] + u[0], x[1] + p[0]))],
            input_names=["x", "u", "p"],
            output_names=["x_next"],
        )
        stage_cost = Function(
            "stage_cost",
            [x, u, p],
            [x[0] * x[0] + x[1]],
            input_names=["x", "u", "p"],
            output_names=["ell"],
        )
        bad_terminal_cost = Function(
            "bad_terminal_cost",
            [x, p],
            [SXVector((x[0], x[1]))],
            input_names=["x", "p"],
            output_names=["vf"],
        )

        with self.assertRaises(ValueError):
            SingleShootingProblem(
                name="bad_horizon",
                horizon=0,
                dynamics=dynamics,
                stage_cost=stage_cost,
                terminal_cost=Function(
                    "terminal_cost",
                    [x, p],
                    [x[0]],
                    input_names=["x", "p"],
                    output_names=["vf"],
                ),
            )

        with self.assertRaises(ValueError):
            SingleShootingProblem(
                name="bad_terminal",
                horizon=2,
                dynamics=dynamics,
                stage_cost=stage_cost,
                terminal_cost=bad_terminal_cost,
            )

    def test_joint_bundle_requires_multiple_requested_outputs(self) -> None:
        bundle = SingleShootingBundle().add_cost()
        problem = _build_reference_problem()

        with self.assertRaises(ValueError):
            problem.joint(bundle)

        with self.assertRaises(ValueError):
            problem.joint(SingleShootingBundle().add_rollout_states())

    def test_builder_accepts_single_shooting_problem(self) -> None:
        problem = _build_reference_problem()
        builder = (
            CodeGenerationBuilder()
            .with_backend_config(RustBackendConfig().with_crate_name("single_shooting"))
            .for_function(problem)
            .add_primal(include_states=True)
            .add_gradient(include_states=True)
            .add_hvp(include_states=True)
            .add_joint(
                SingleShootingBundle()
                .add_cost()
                .add_gradient()
                .add_rollout_states()
            )
            .done()
        )

        resolved = builder._resolved_function_specs()
        self.assertEqual(len(resolved), 1)


if __name__ == "__main__":
    unittest.main()
