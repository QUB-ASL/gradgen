import json
from datetime import datetime
import contextlib
import subprocess
import re
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import multiprocessing
import traceback
import sys
import venv

import gradgen._rust_codegen.codegen as rust_codegen_module
import gradgen.single_shooting as single_shooting_module
from gradgen._rust_codegen.project_support import _gradgen_version
from gradgen import (
    CodeGenerationBuilder,
    ComposedFunction,
    Function,
    FunctionBundle,
    RustBackendConfig,
    SX,
    SXVector,
    SingleShootingBundle,
    SingleShootingProblem,
    bilinear_form,
    clear_registered_elementary_functions,
    create_multi_function_rust_project,
    create_rust_derivative_bundle,
    create_rust_project,
    derivative,
    map_function,
    matvec,
    quadform,
    register_elementary_function,
    zip_function,
)


class RustCodegenTests(unittest.TestCase):
    def tearDown(self) -> None:
        clear_registered_elementary_functions()

    @staticmethod
    def _run_cargo(
        project_dir: Path, *args: str
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["cargo", *args],
            cwd=project_dir,
            check=True,
            capture_output=True,
            text=True,
        )

    @classmethod
    def _run_cargo_clippy_clean(
        cls, project_dir: Path
    ) -> subprocess.CompletedProcess[str]:
        return cls._run_cargo(
            project_dir, "clippy", "--quiet", "--", "-D", "warnings"
        )

    @staticmethod
    def _venv_python(venv_dir: Path) -> Path:
        if sys.platform == "win32":
            return venv_dir / "Scripts" / "python.exe"
        return venv_dir / "bin" / "python"

    @staticmethod
    def _append_rust_test(project_dir: Path, test_source: str) -> None:
        lib_rs = project_dir / "src" / "lib.rs"
        with lib_rs.open("a", encoding="utf-8") as handle:
            handle.write("\n")
            handle.write(test_source)

    @staticmethod
    def _normalize_inputs(inputs: object) -> tuple[object, ...]:
        if isinstance(inputs, tuple):
            return inputs
        return (inputs,)

    @staticmethod
    def _flatten_runtime_output(
        function: Function, result: object
    ) -> list[float]:
        def flatten_one(declaration: object, value: object) -> list[float]:
            if isinstance(declaration, SX):
                return [float(value)]
            return [float(item) for item in value]

        if len(function.outputs) == 1:
            return flatten_one(function.outputs[0], result)

        flattened: list[float] = []
        for declaration, value in zip(function.outputs, result):
            flattened.extend(flatten_one(declaration, value))
        return flattened

    @staticmethod
    def _rust_array_literal(values: list[float], scalar_type: str) -> str:
        return (
            "["
            + ", ".join(
                f"{repr(float(value))}_{scalar_type}" for value in values
            )
            + "]"
        )

    @classmethod
    def _append_reference_test(
        cls,
        project_dir: Path,
        function: Function,
        *,
        function_name: str,
        inputs: object,
        test_name: str,
        config: RustBackendConfig | None = None,
        tolerance: float = 1e-12,
        workspace_size_override: int | None = None,
    ) -> None:
        codegen = function.generate_rust(
            config=config, function_name=function_name
        )
        numeric_inputs = cls._normalize_inputs(inputs)
        expected = cls._flatten_runtime_output(
            function, function(*numeric_inputs)
        )
        rust_tolerance = float(tolerance)
        scalar_type = codegen.scalar_type

        input_binding_lines: list[str] = []
        parameter_names: list[str] = []
        for index, values in enumerate(numeric_inputs):
            name = function.input_names[index]
            parameter_names.append(name)
            if isinstance(values, (list, tuple)):
                rust_values = [float(item) for item in values]
            else:
                rust_values = [float(values)]
            input_binding_lines.append(
                f"        let {name} = {cls._rust_array_literal(
                    rust_values, scalar_type)};"
            )

        output_binding_lines: list[str] = []
        output_assertion_lines: list[str] = []
        expected_offset = 0
        for index, size in enumerate(codegen.output_sizes):
            output_name = function.output_names[index]
            expected_slice = expected[expected_offset:expected_offset + size]
            expected_offset += size
            output_binding_lines.append(
                f"        let mut {output_name} = [0.0_{scalar_type}; {size}];"
            )
            output_assertion_lines.append(
                "        assert_close_slice("
                f"&{output_name}, "
                f"&{cls._rust_array_literal(expected_slice, scalar_type)}, "
                f"{rust_tolerance}_{scalar_type},"
                ");"
            )

        parameter_list = ", ".join(
            [
                *[f"&{name}" for name in parameter_names],
                *[f"&mut {name}" for name in function.output_names],
                "&mut work",
            ]
        )

        workspace_size = (
            codegen.workspace_size
            if workspace_size_override is None
            else workspace_size_override
        )

        cls._append_rust_test(
            project_dir,
            f"""
#[cfg(test)]
mod tests {{
    use super::*;

    fn assert_close_slice(
        actual: &[{scalar_type}],
        expected: &[{scalar_type}],
        tolerance: {scalar_type},
    ) {{
        assert_eq!(actual.len(), expected.len());
        for (actual_value, expected_value) in
            actual.iter().zip(expected.iter())
        {{
            assert!(
                (actual_value - expected_value).abs() <= tolerance,
                "expected {{expected_value}}, got {{actual_value}}"
            );
        }}
    }}

    #[test]
    fn {test_name}() {{
{chr(10).join(input_binding_lines)}
{chr(10).join(output_binding_lines)}
        let mut work = [0.0_{scalar_type}; {workspace_size}];
        {function_name}({parameter_list});
{chr(10).join(output_assertion_lines)}
    }}
}}
""".lstrip(),
        )

    @staticmethod
    def _build_single_shooting_problem(
        *, horizon: int = 3
    ) -> SingleShootingProblem:
        x = SXVector.sym("x", 2)
        u = SXVector.sym("u", 1)
        p = SXVector.sym("p", 2)
        dynamics = Function(
            "dynamics",
            [x, u, p],
            [
                SXVector(
                    (
                        x[0] + p[0] * x[1] + u[0],
                        x[1] + p[1] * u[0] - 0.5 * x[0],
                    )
                )
            ],
            input_names=["x", "u", "p"],
            output_names=["x_next"],
        )
        stage_cost = Function(
            "stage_cost",
            [x, u, p],
            [
                x[0] * x[0]
                + 2.0 * x[1] * x[1]
                + 0.3 * u[0] * u[0]
                + p[0] * u[0]
            ],
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

    @staticmethod
    def _manual_single_shooting_rollout(
        x0: list[float],
        U: list[float],
        p: list[float],
        horizon: int,
    ) -> tuple[float, list[float]]:
        current = [float(x0[0]), float(x0[1])]
        packed_states = [current[0], current[1]]
        total_cost = 0.0
        for stage_index in range(horizon):
            u_t = float(U[stage_index])
            total_cost += (
                current[0] * current[0]
                + 2.0 * current[1] * current[1]
                + 0.3 * u_t * u_t
                + p[0] * u_t
            )
            next_state = [
                current[0] + p[0] * current[1] + u_t,
                current[1] + p[1] * u_t - 0.5 * current[0],
            ]
            current = next_state
            packed_states.extend(current)
        total_cost += (
            3.0 * current[0] * current[0]
            + 0.5 * current[1] * current[1]
            + p[1] * current[0]
        )
        return total_cost, packed_states

    @classmethod
    def _manual_single_shooting_gradient(
        cls,
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
            forward_cost, _ = cls._manual_single_shooting_rollout(
                x0, forward, p, horizon
            )
            backward_cost, _ = cls._manual_single_shooting_rollout(
                x0, backward, p, horizon
            )
            gradient.append((forward_cost - backward_cost) / (2.0 * epsilon))
        return gradient

    @classmethod
    def _manual_single_shooting_hvp(
        cls,
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
        forward_gradient = cls._manual_single_shooting_gradient(
            x0, forward_controls, p, horizon, epsilon=epsilon
        )
        backward_gradient = cls._manual_single_shooting_gradient(
            x0, backward_controls, p, horizon, epsilon=epsilon
        )
        return [
            (forward_value - backward_value) / (2.0 * epsilon)
            for forward_value, backward_value in zip(
                forward_gradient, backward_gradient
            )
        ]

    @staticmethod
    def _build_multi_control_single_shooting_problem(
        *, horizon: int = 2
    ) -> SingleShootingProblem:
        x = SXVector.sym("x", 2)
        u = SXVector.sym("u", 2)
        p = SXVector.sym("p", 2)
        dynamics = Function(
            "multi_control_dynamics",
            [x, u, p],
            [
                SXVector(
                    (
                        0.8 * x[0] + p[0] * u[0] + u[1],
                        x[1] + u[0] - p[1] * u[1] + 0.25 * x[0],
                    )
                )
            ],
            input_names=["x", "u", "p"],
            output_names=["x_next"],
        )
        stage_cost = Function(
            "multi_control_stage_cost",
            [x, u, p],
            [
                1.2 * x[0] * x[0]
                + 0.5 * x[1] * x[1]
                + 0.2 * u[0] * u[0]
                + 0.3 * u[1] * u[1]
                + p[0] * u[0] * u[1]
                + 0.1 * x[0] * u[1]
            ],
            input_names=["x", "u", "p"],
            output_names=["ell"],
        )
        terminal_cost = Function(
            "multi_control_terminal_cost",
            [x, p],
            [2.0 * x[0] * x[0] + 1.5 * x[1] * x[1] + p[1] * x[1]],
            input_names=["x", "p"],
            output_names=["vf"],
        )
        return SingleShootingProblem(
            name="multi_control_cost",
            horizon=horizon,
            dynamics=dynamics,
            stage_cost=stage_cost,
            terminal_cost=terminal_cost,
            initial_state_name="x0",
            control_sequence_name="u_seq",
            parameter_name="p",
        )

    @staticmethod
    def _manual_multi_control_single_shooting_rollout(
        x0: list[float],
        U: list[float],
        p: list[float],
        horizon: int,
    ) -> tuple[float, list[float]]:
        current = [float(x0[0]), float(x0[1])]
        packed_states = [current[0], current[1]]
        total_cost = 0.0
        for stage_index in range(horizon):
            u0 = float(U[2 * stage_index])
            u1 = float(U[(2 * stage_index) + 1])
            total_cost += (
                1.2 * current[0] * current[0]
                + 0.5 * current[1] * current[1]
                + 0.2 * u0 * u0
                + 0.3 * u1 * u1
                + p[0] * u0 * u1
                + 0.1 * current[0] * u1
            )
            next_state = [
                0.8 * current[0] + p[0] * u0 + u1,
                current[1] + u0 - p[1] * u1 + 0.25 * current[0],
            ]
            current = next_state
            packed_states.extend(current)
        total_cost += (
            2.0 * current[0] * current[0]
            + 1.5 * current[1] * current[1]
            + p[1] * current[1]
        )
        return total_cost, packed_states

    @classmethod
    def _manual_multi_control_single_shooting_gradient(
        cls,
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
            forward_cost, _ = (
                cls._manual_multi_control_single_shooting_rollout(
                    x0, forward, p, horizon
                )
            )
            backward_cost, _ = (
                cls._manual_multi_control_single_shooting_rollout(
                    x0, backward, p, horizon
                )
            )
            gradient.append((forward_cost - backward_cost) / (2.0 * epsilon))
        return gradient

    def test_generates_scalar_function_with_slice_abi(self) -> None:
        x = SX.sym("x")
        f = Function(
            "square_plus_one",
            [x],
            [x * x + 1],
            input_names=["x"],
            output_names=["y"],
        )

        result = f.generate_rust()

        self.assertEqual(result.function_name, "square_plus_one")
        self.assertEqual(result.input_sizes, (1,))
        self.assertEqual(result.output_sizes, (1,))
        self.assertIn('"x",', result.source)
        self.assertIn('"y",', result.source)
        self.assertIn("pub struct FunctionMetadata {", result.source)
        self.assertIn(
            "pub fn square_plus_one_meta() -> FunctionMetadata {",
            result.source,
        )
        self.assertIn("workspace_size: 1,", result.source)
        self.assertIn("/// Arguments:", result.source)
        self.assertIn("/// - `x`:", result.source)
        self.assertIn(
            "///   input slice for the declared argument `x`", result.source
        )
        self.assertIn("///   Expected length: 1.", result.source)
        self.assertIn("/// - `y`:", result.source)
        self.assertIn(
            "///   primal output slice for the declared result `y`",
            result.source,
        )
        self.assertIn("///   Expected length: 1.", result.source)
        self.assertIn(
            "workspace slice used to store intermediate values",
            result.source,
        )
        self.assertIn(
            "Expected length: at least 1.",
            result.source,
        )
        self.assertIn(
            "pub fn square_plus_one(",
            result.source,
        )
        self.assertIn(
            'WorkspaceTooSmall("work expected at least 1")',
            result.source,
        )
        self.assertIn(
            'InputTooSmall("x expected length 1")',
            result.source,
        )
        self.assertIn(
            'OutputTooSmall("y expected length 1")',
            result.source,
        )
        self.assertIn("work[0] = x[0] * x[0];", result.source)
        self.assertIn("work[0] += 1.0_f64;", result.source)
        self.assertIn("y[0] = work[0];", result.source)

    def test_generates_vector_function_with_deterministic_workspace_layout(
        self,
    ) -> None:
        x = SXVector.sym("x", 2)
        y = SXVector.sym("y", 2)
        f = Function(
            "f",
            [x, y],
            [x.dot(y), x + y],
            input_names=["x", "y"],
            output_names=["dot", "sum"],
        )

        result = f.generate_rust(function_name="kernel")

        self.assertEqual(result.function_name, "kernel")
        self.assertEqual(result.input_sizes, (2, 2))
        self.assertEqual(result.output_sizes, (1, 2))
        self.assertIn('"x",', result.source)
        self.assertIn('"y",', result.source)
        self.assertIn('"dot",', result.source)
        self.assertIn('"sum",', result.source)
        self.assertIn("pub struct FunctionMetadata {", result.source)
        self.assertIn(
            "pub fn kernel_meta() -> FunctionMetadata {", result.source
        )
        self.assertIn("workspace_size: 3,", result.source)
        self.assertIn("pub fn kernel(", result.source)
        self.assertIn(
            'WorkspaceTooSmall("work expected at least 3")',
            result.source,
        )
        self.assertIn("work[0] = x[0] * y[0];", result.source)
        self.assertIn("work[1] = x[1] * y[1];", result.source)
        self.assertIn("work[0] += work[1];", result.source)
        self.assertIn("dot[0] = work[0];", result.source)

    def test_single_shooting_codegen_matches_manual_reference(self) -> None:
        problem = self._build_single_shooting_problem(horizon=3)
        x0 = [1.0, -0.5]
        U = [0.2, -0.1, 0.3]
        v_U = [0.5, -1.0, 0.25]
        p = [0.4, -1.2]
        expected_cost, expected_states = self._manual_single_shooting_rollout(
            x0, U, p, problem.horizon
        )
        expected_gradient = self._manual_single_shooting_gradient(
            x0, U, p, problem.horizon
        )
        expected_hvp = self._manual_single_shooting_hvp(
            x0, U, p, v_U, problem.horizon
        )

        builder = (
            CodeGenerationBuilder()
            .with_backend_config(
                RustBackendConfig().with_crate_name("single_shooting_kernel")
            )
            .for_function(problem)
            .add_primal(include_states=True)
            .add_gradient(include_states=True)
            .add_hvp(include_states=True)
            .add_joint(
                SingleShootingBundle()
                .add_cost()
                .add_gradient()
                .add_hvp()
                .add_rollout_states()
            )
            .with_simplification("medium")
            .done()
        )

        with TemporaryDirectory() as tmpdir:
            project = builder.build(Path(tmpdir) / "single_shooting_kernel")
            primal_codegen, gradient_codegen, hvp_codegen, joint_codegen = (
                project.codegens
            )
            lib_text = project.lib_rs.read_text(encoding="utf-8")

            self.assertIn("for stage_index in 0..3 {", lib_text)
            self.assertIn("for stage_index in (1..3).rev() {", lib_text)
            self.assertIn(
                "packed control-sequence slice laid out stage-major",
                lib_text,
            )
            self.assertIn(
                "shared parameter slice used at every stage",
                lib_text,
            )
            self.assertNotIn(
                "(stage_index * 1)..((stage_index + 1) * 1)", lib_text
            )
            self.assertNotIn("0..(0 + 1)", lib_text)
            self.assertNotIn("(0 * 1)..((0 + 1) * 1)", lib_text)

            expected_cost_literal = self._rust_array_literal(
                [expected_cost], "f64"
            )
            expected_states_literal = self._rust_array_literal(
                expected_states, "f64"
            )
            expected_gradient_literal = self._rust_array_literal(
                expected_gradient, "f64"
            )
            expected_hvp_literal = self._rust_array_literal(
                expected_hvp, "f64"
            )

            self._append_rust_test(
                project.project_dir,
                f"""
#[cfg(test)]
mod single_shooting_runtime_tests {{
    use super::*;

    fn assert_close_slice(
        actual: &[f64],
        expected: &[f64],
        tolerance: f64,
    ) {{
        assert_eq!(actual.len(), expected.len());
        for (actual_value, expected_value) in
            actual.iter().zip(expected.iter())
        {{
            assert!(
                (actual_value - expected_value).abs() <= tolerance,
                "expected {{expected_value}}, got {{actual_value}}"
            );
        }}
    }}

    #[test]
    fn matches_manual_reference() {{
        let x0 = {self._rust_array_literal(x0, "f64")};
        let U = {self._rust_array_literal(U, "f64")};
        let v_U = {self._rust_array_literal(v_U, "f64")};
        let p = {self._rust_array_literal(p, "f64")};

        let mut cost = [0.0_f64; 1];
        let mut x_traj = [0.0_f64; 8];
        let mut primal_work = [0.0_f64; {primal_codegen.workspace_size}];
        {primal_codegen.function_name}(
            &x0,
            &U,
            &p,
            &mut cost,
            &mut x_traj,
            &mut primal_work,
        );
        assert_close_slice(
            &cost,
            &{expected_cost_literal},
            1e-10_f64,
        );
        assert_close_slice(
            &x_traj,
            &{expected_states_literal},
            1e-10_f64,
        );

        let mut gradient_U = [0.0_f64; 3];
        let mut grad_states = [0.0_f64; 8];
        let mut gradient_work = [0.0_f64; {gradient_codegen.workspace_size}];
        {gradient_codegen.function_name}(
            &x0,
            &U,
            &p,
            &mut gradient_U,
            &mut grad_states,
            &mut gradient_work,
        );
        assert_close_slice(
            &gradient_U,
            &{expected_gradient_literal},
            1e-5_f64,
        );
        assert_close_slice(
            &grad_states,
            &{expected_states_literal},
            1e-10_f64,
        );

        let mut hvp_U = [0.0_f64; 3];
        let mut hvp_states = [0.0_f64; 8];
        let mut hvp_work = [0.0_f64; {hvp_codegen.workspace_size}];
        {hvp_codegen.function_name}(
            &x0,
            &U,
            &p,
            &v_U,
            &mut hvp_U,
            &mut hvp_states,
            &mut hvp_work,
        );
        assert_close_slice(&hvp_U, &{expected_hvp_literal}, 2e-4_f64);
        assert_close_slice(
            &hvp_states,
            &{expected_states_literal},
            1e-10_f64,
        );

        let mut joint_cost = [0.0_f64; 1];
        let mut joint_gradient_U = [0.0_f64; 3];
        let mut joint_hvp_U = [0.0_f64; 3];
        let mut joint_states = [0.0_f64; 8];
        let mut joint_work = [0.0_f64; {joint_codegen.workspace_size}];
        {joint_codegen.function_name}(
            &x0,
            &U,
            &p,
            &v_U,
            &mut joint_cost,
            &mut joint_gradient_U,
            &mut joint_hvp_U,
            &mut joint_states,
            &mut joint_work,
        );
        assert_close_slice(&joint_cost, &{expected_cost_literal}, 1e-10_f64);
        assert_close_slice(
            &joint_gradient_U,
            &{expected_gradient_literal},
            1e-5_f64,
        );
        assert_close_slice(&joint_hvp_U, &{expected_hvp_literal}, 2e-4_f64);
        assert_close_slice(
            &joint_states,
            &{expected_states_literal},
            1e-10_f64,
        );
    }}
}}
""".lstrip(),
            )

            completed = self._run_cargo(project.project_dir, "test", "--quiet")
            self.assertEqual(completed.returncode, 0)

    @staticmethod
    def _run_build_very_large_ss_problem(queue: multiprocessing.Queue) -> None:
        try:
            problem = RustCodegenTests._build_single_shooting_problem(
                horizon=20
            )
            builder = (
                CodeGenerationBuilder()
                .with_backend_config(
                    RustBackendConfig().with_crate_name(
                        "single_shooting_no_expand"
                    )
                )
                .for_function(problem)
                .add_primal(include_states=True)
                .add_gradient(include_states=True)
                .add_hvp(include_states=True)
                .add_joint(
                    SingleShootingBundle()
                    .add_cost()
                    .add_gradient()
                    .add_hvp()
                    .add_rollout_states()
                )
                .with_simplification("medium")
                .done()
            )

            with TemporaryDirectory() as tmpdir:
                builder.build(tmpdir)

            queue.put(None)
        except Exception:
            queue.put(traceback.format_exc())

    def test_single_shooting_large_horizon(self) -> None:
        # A timeout of 15s is of course too much, but we just want to make sure
        # that it doesn't take forever to generate Rust code. Code generation
        # should be almost instantaneous
        timeout_seconds = 15
        queue = multiprocessing.Queue()
        proc = multiprocessing.Process(
            target=self._run_build_very_large_ss_problem, args=(queue,)
        )
        proc.start()
        proc.join(timeout_seconds)

        if proc.is_alive():
            proc.terminate()
            proc.join()
            self.fail(
                (
                    "builder.build(tmpdir) exceeded "
                    f"timeout of {timeout_seconds:.1f}s"
                )
            )

        if proc.exitcode != 0:
            self.fail(f"Build subprocess exited with code {proc.exitcode}")

        error = queue.get() if not queue.empty() else None
        if error is not None:
            self.fail(f"builder.build(tmpdir) raised an exception:\n{error}")

    def test_single_shooting_builder_keeps_staged_sources(self) -> None:
        problem = self._build_single_shooting_problem(horizon=20)
        builder = (
            CodeGenerationBuilder()
            .with_backend_config(
                RustBackendConfig().with_crate_name(
                    "single_shooting_no_expand"
                )
            )
            .for_function(problem)
            .add_primal(include_states=True)
            .add_gradient(include_states=True)
            .add_hvp(include_states=True)
            .add_joint(
                SingleShootingBundle()
                .add_cost()
                .add_gradient()
                .add_hvp()
                .add_rollout_states()
            )
            .done()
        )

        with TemporaryDirectory() as tmpdir:
            with (
                patch.object(
                    single_shooting_module.SingleShootingPrimalFunction,
                    "to_function",
                    side_effect=AssertionError(
                        "staged primal source should not be expanded"
                    ),
                ),
                patch.object(
                    single_shooting_module.SingleShootingGradientFunction,
                    "to_function",
                    side_effect=AssertionError(
                        "staged gradient source should not be expanded"
                    ),
                ),
                patch.object(
                    single_shooting_module.SingleShootingHvpFunction,
                    "to_function",
                    side_effect=AssertionError(
                        "staged hvp source should not be expanded during build"
                    ),
                ),
                patch.object(
                    single_shooting_module.SingleShootingJointFunction,
                    "to_function",
                    side_effect=AssertionError(
                        "staged joint source should not be expanded"
                    ),
                ),
            ):
                project = builder.build(
                    Path(tmpdir) / "single_shooting_no_expand"
                )

        self.assertEqual(len(project.codegens), 4)

    def test_single_shooting_horizon_one_joint(self) -> None:
        problem = self._build_single_shooting_problem(horizon=1)
        x0 = [0.7, -0.3]
        U = [0.25]
        p = [0.4, -1.2]
        expected_cost, expected_states = self._manual_single_shooting_rollout(
            x0, U, p, problem.horizon
        )
        expected_gradient = self._manual_single_shooting_gradient(
            x0, U, p, problem.horizon
        )

        builder = (
            CodeGenerationBuilder()
            .with_backend_config(
                RustBackendConfig().with_crate_name("single_shooting_h1")
            )
            .for_function(problem)
            .add_gradient(include_states=True)
            .add_joint(SingleShootingBundle().add_cost().add_rollout_states())
            .with_simplification("medium")
            .done()
        )

        with TemporaryDirectory() as tmpdir:
            project = builder.build(Path(tmpdir) / "single_shooting_h1")
            gradient_codegen, joint_codegen = project.codegens
            lib_text = project.lib_rs.read_text(encoding="utf-8")

            self.assertIn("for stage_index in 0..1 {", lib_text)
            self.assertIn("for stage_index in (1..1).rev() {", lib_text)
            self.assertNotIn(
                "(stage_index * 1)..((stage_index + 1) * 1)", lib_text
            )
            self.assertNotIn("0..(0 + 1)", lib_text)
            self.assertNotIn("(0 * 1)..((0 + 1) * 1)", lib_text)

            expected_cost_literal = self._rust_array_literal(
                [expected_cost], "f64"
            )
            expected_states_literal = self._rust_array_literal(
                expected_states, "f64"
            )
            expected_gradient_literal = self._rust_array_literal(
                expected_gradient, "f64"
            )

            self._append_rust_test(
                project.project_dir,
                f"""
#[cfg(test)]
mod single_shooting_horizon_one_tests {{
    use super::*;

    fn assert_close_slice(
        actual: &[f64],
        expected: &[f64],
        tolerance: f64,
    ) {{
        assert_eq!(actual.len(), expected.len());
        for (actual_value, expected_value) in
            actual.iter().zip(expected.iter())
        {{
            assert!(
                (actual_value - expected_value).abs() <= tolerance,
                "expected {{expected_value}}, got {{actual_value}}"
            );
        }}
    }}

    #[test]
    fn matches_horizon_one_reference() {{
        let x0 = {self._rust_array_literal(x0, "f64")};
        let U = {self._rust_array_literal(U, "f64")};
        let p = {self._rust_array_literal(p, "f64")};

        let mut gradient_U = [0.0_f64; 1];
        let mut grad_states = [0.0_f64; 4];
        let mut gradient_work = [0.0_f64; {gradient_codegen.workspace_size}];
        {gradient_codegen.function_name}(
            &x0,
            &U,
            &p,
            &mut gradient_U,
            &mut grad_states,
            &mut gradient_work,
        );
        assert_close_slice(
            &gradient_U,
            &{expected_gradient_literal},
            1e-5_f64,
        );
        assert_close_slice(
            &grad_states,
            &{expected_states_literal},
            1e-10_f64,
        );

        let mut joint_cost = [0.0_f64; 1];
        let mut joint_states = [0.0_f64; 4];
        let mut joint_work = [0.0_f64; {joint_codegen.workspace_size}];
        {joint_codegen.function_name}(
            &x0,
            &U,
            &p,
            &mut joint_cost,
            &mut joint_states,
            &mut joint_work,
        );
        assert_close_slice(&joint_cost, &{expected_cost_literal}, 1e-10_f64);
        assert_close_slice(
            &joint_states,
            &{expected_states_literal},
            1e-10_f64,
        );
    }}
}}
""".lstrip(),
            )

            completed = self._run_cargo(project.project_dir, "test", "--quiet")
            self.assertEqual(completed.returncode, 0)

    def test_single_shooting_multi_control_blocks(self) -> None:
        problem = self._build_multi_control_single_shooting_problem(horizon=2)
        x0 = [0.5, -1.0]
        U = [0.2, -0.1, 0.3, 0.4]
        p = [0.6, -0.4]
        expected_cost, expected_states = (
            self._manual_multi_control_single_shooting_rollout(
                x0, U, p, problem.horizon
            )
        )
        expected_gradient = (
            self._manual_multi_control_single_shooting_gradient(
                x0, U, p, problem.horizon
            )
        )

        builder = (
            CodeGenerationBuilder()
            .with_backend_config(
                RustBackendConfig().with_crate_name("single_shooting_multi_u")
            )
            .for_function(problem)
            .add_primal(include_states=True)
            .add_gradient(include_states=True)
            .with_simplification("medium")
            .done()
        )

        with TemporaryDirectory() as tmpdir:
            project = builder.build(Path(tmpdir) / "single_shooting_multi_u")
            primal_codegen, gradient_codegen = project.codegens
            lib_text = project.lib_rs.read_text(encoding="utf-8")

            expected_cost_literal = self._rust_array_literal(
                [expected_cost], "f64"
            )
            expected_states_literal = self._rust_array_literal(
                expected_states, "f64"
            )
            expected_gradient_literal = self._rust_array_literal(
                expected_gradient, "f64"
            )

            self.assertIn(
                ("&u_seq[(stage_index * 2).." "((stage_index + 1) * 2)]"),
                lib_text,
            )
            self.assertIn(
                (
                    "&mut gradient_u_seq[(stage_index * 2).."
                    "((stage_index + 1) * 2)]"
                ),
                lib_text,
            )

            self._append_rust_test(
                project.project_dir,
                f"""
#[cfg(test)]
mod single_shooting_multi_u_tests {{
    use super::*;

    fn assert_close_slice(
        actual: &[f64],
        expected: &[f64],
        tolerance: f64,
    ) {{
        assert_eq!(actual.len(), expected.len());
        for (actual_value, expected_value) in
            actual.iter().zip(expected.iter())
        {{
            assert!(
                (actual_value - expected_value).abs() <= tolerance,
                "expected {{expected_value}}, got {{actual_value}}"
            );
        }}
    }}

    #[test]
    fn matches_multi_input_control_reference() {{
        let x0 = {self._rust_array_literal(x0, "f64")};
        let u_seq = {self._rust_array_literal(U, "f64")};
        let p = {self._rust_array_literal(p, "f64")};

        let mut cost = [0.0_f64; 1];
        let mut x_traj = [0.0_f64; 6];
        let mut primal_work = [0.0_f64; {primal_codegen.workspace_size}];
        {primal_codegen.function_name}(
            &x0,
            &u_seq,
            &p,
            &mut cost,
            &mut x_traj,
            &mut primal_work,
        );
        assert_close_slice(
            &cost,
            &{expected_cost_literal},
            1e-10_f64,
        );
        assert_close_slice(
            &x_traj,
            &{expected_states_literal},
            1e-10_f64,
        );

        let mut gradient_u_seq = [0.0_f64; 4];
        let mut grad_states = [0.0_f64; 6];
        let mut gradient_work = [0.0_f64; {gradient_codegen.workspace_size}];
        {gradient_codegen.function_name}(
            &x0,
            &u_seq,
            &p,
            &mut gradient_u_seq,
            &mut grad_states,
            &mut gradient_work,
        );
        assert_close_slice(
            &gradient_u_seq,
            &{expected_gradient_literal},
            1e-5_f64,
        );
        assert_close_slice(
            &grad_states,
            &{expected_states_literal},
            1e-10_f64,
        );
    }}
}}
""".lstrip(),
            )

            completed = self._run_cargo(project.project_dir, "test", "--quiet")
            self.assertEqual(completed.returncode, 0)

    def test_include_states_rejected_for_non_ss_sources(self) -> None:
        x = SXVector.sym("x", 2)
        f = Function(
            "energy", [x], [x.norm2sq()], input_names=["x"], output_names=["y"]
        )

        with self.assertRaises(ValueError):
            with TemporaryDirectory() as tmpdir:
                (
                    CodeGenerationBuilder()
                    .with_backend_config(
                        RustBackendConfig().with_crate_name("bad_states")
                    )
                    .for_function(f)
                    .add_primal(include_states=True)
                    .done()
                    .build(Path(tmpdir) / "unused")
                )

        state = SXVector.sym("state", 2)
        p = SXVector.sym("p", 2)
        pf = SXVector.sym("pf", 1)
        g = Function(
            "g",
            [state, p],
            [SXVector((state[0] + p[0], state[1] + p[1]))],
            input_names=["state", "p"],
            output_names=["next_state"],
        )
        h = Function(
            "h",
            [state, pf],
            [state[0] + pf[0]],
            input_names=["state", "pf"],
            output_names=["y"],
        )
        composed = (
            ComposedFunction("demo", x)
            .then(g, p=[1.0, 2.0])
            .finish(h, p=[3.0])
        )

        with self.assertRaises(ValueError):
            with TemporaryDirectory() as tmpdir:
                (
                    CodeGenerationBuilder()
                    .with_backend_config(
                        RustBackendConfig().with_crate_name(
                            "bad_composed_states"
                        )
                    )
                    .for_function(composed)
                    .add_gradient(include_states=True)
                    .done()
                    .build(Path(tmpdir) / "unused")
                )

    def test_create_rust_project_writes_metadata_json(self) -> None:
        x = SXVector.sym("x", 2)
        f = Function(
            "energy", [x], [x.norm2sq()], input_names=["x"], output_names=["y"]
        )

        with TemporaryDirectory() as tmpdir:
            project = create_rust_project(f, Path(tmpdir) / "energy_kernel")
            metadata = json.loads(
                project.metadata_json.read_text(encoding="utf-8")
            )

            self.assertEqual(metadata["crate_name"], "energy")
            self.assertEqual(metadata["gradgen_version"], _gradgen_version())
            self.assertIsNotNone(
                datetime.fromisoformat(
                    metadata["created_at"].replace("Z", "+00:00")
                )
            )
            self.assertEqual(
                metadata["functions"],
                [
                    {
                        "function_name": "energy",
                        "workspace_size": 1,
                        "input_names": ["x"],
                        "input_sizes": [2],
                        "output_names": ["y"],
                        "output_sizes": [1],
                    }
                ],
            )

    def test_create_rust_project_attempts_cargo_fmt_without_failing_on_error(
        self,
    ) -> None:
        x = SXVector.sym("x", 2)
        f = Function(
            "energy", [x], [x.norm2sq()], input_names=["x"], output_names=["y"]
        )

        with TemporaryDirectory() as tmpdir:
            with patch(
                "gradgen._rust_codegen.project_support.shutil.which",
                return_value="/usr/bin/cargo",
            ):
                with patch(
                    "gradgen._rust_codegen.project_support.subprocess.run",
                    side_effect=subprocess.CalledProcessError(
                        1, ["cargo", "fmt"], stderr="rustfmt missing"
                    ),
                ) as mocked_run:
                    project = create_rust_project(
                        f, Path(tmpdir) / "energy_kernel"
                    )
                    self.assertTrue(project.lib_rs.exists())

        mocked_run.assert_called_once_with(
            ["cargo", "fmt"],
            cwd=project.project_dir,
            check=True,
            capture_output=True,
            text=True,
        )

    def test_python_interface_scaffolding(self) -> None:
        x = SXVector.sym("x", 2)
        w = SXVector.sym("w", 1)
        f = Function(
            "energy",
            [x, w],
            [x.norm2sq() + w[0], x[0] + x[1]],
            input_names=["x", "w"],
            output_names=["cost", "state"],
        )

        config = RustBackendConfig().with_enable_python_interface(True)

        with TemporaryDirectory() as tmpdir:
            project = create_rust_project(
                f, Path(tmpdir) / "energy_kernel", config=config
            )
            cargo_text = project.cargo_toml.read_text(encoding="utf-8")
            lib_text = project.lib_rs.read_text(encoding="utf-8")
            readme_text = project.readme.read_text(encoding="utf-8")
            wrapper = project.python_interface

            self.assertIsNotNone(wrapper)
            assert wrapper is not None

            wrapper_cargo = wrapper.cargo_toml.read_text(encoding="utf-8")
            wrapper_lib = wrapper.lib_rs.read_text(encoding="utf-8")
            wrapper_pyproject = wrapper.pyproject.read_text(encoding="utf-8")
            wrapper_readme = wrapper.readme.read_text(encoding="utf-8")

            self.assertNotIn("[dependencies.pyo3]", cargo_text)
            self.assertNotIn('crate-type = ["cdylib", "rlib"]', cargo_text)
            self.assertFalse((project.project_dir / "pyproject.toml").exists())
            self.assertIn("This crate stays pure Rust", readme_text)
            self.assertIn("separate PyO3 wrapper crate", readme_text)
            self.assertIn("cargo build", readme_text)
            self.assertIn("[dependencies.pyo3]", wrapper_cargo)
            self.assertIn('version = "0.28.2"', wrapper_cargo)
            self.assertIn(
                'energy = { path = "../energy_kernel" }', wrapper_cargo
            )
            self.assertIn('crate-type = ["cdylib", "rlib"]', wrapper_cargo)
            self.assertIn("#[pyclass]", wrapper_lib)
            self.assertIn("struct Workspace", wrapper_lib)
            self.assertIn("all_functions", wrapper_lib)
            self.assertIn("function_info", wrapper_lib)
            self.assertIn("#[pymodule]", wrapper_lib)
            self.assertIn("workspace_for_function", wrapper_lib)
            self.assertIn("call(", wrapper_lib)
            self.assertIn('#[pyfunction(name = "energy"', wrapper_lib)
            self.assertIn("pyerr_from_gradgen_error", wrapper_lib)
            self.assertIn("PyValueError::new_err", wrapper_lib)
            self.assertIn("cost", lib_text)
            self.assertIn("state", lib_text)
            self.assertIn("maturin", wrapper_pyproject)
            self.assertIn('module-name = "energy"', wrapper_pyproject)
            self.assertIn('version = "0.1.0"', wrapper_pyproject)
            self.assertIn("Python Interface Wrapper", wrapper_readme)
            self.assertIn("__version__", wrapper_lib)
            self.assertIn("__all__", wrapper_lib)
            self.assertIn("__getattr__", wrapper_lib)

    def test_python_interface_version_bumps(self) -> None:
        x = SXVector.sym("x", 2)
        w = SXVector.sym("w", 1)
        f = Function(
            "energy",
            [x, w],
            [x.norm2sq() + w[0], x[0] + x[1]],
            input_names=["x", "w"],
            output_names=["cost", "state"],
        )

        config = (
            RustBackendConfig()
            .with_crate_name("blah")
            .with_enable_python_interface(True)
            .with_build_python_interface(False)
        )

        with TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir) / "energy_kernel"

            first_project = create_rust_project(f, project_dir, config=config)
            first_wrapper = first_project.python_interface
            assert first_wrapper is not None
            first_pyproject = first_wrapper.pyproject.read_text(
                encoding="utf-8"
            )

            second_project = create_rust_project(f, project_dir, config=config)
            second_wrapper = second_project.python_interface
            assert second_wrapper is not None
            second_pyproject = second_wrapper.pyproject.read_text(
                encoding="utf-8"
            )

            self.assertIn('version = "0.1.0"', first_pyproject)
            self.assertIn('version = "0.2.0"', second_pyproject)

    def test_create_rust_project_can_build_generated_crate(self) -> None:
        x = SXVector.sym("x", 2)
        w = SXVector.sym("w", 1)
        f = Function(
            "energy",
            [x, w],
            [x.norm2sq() + w[0], x[0] + x[1]],
            input_names=["x", "w"],
            output_names=["cost", "state"],
        )

        config = RustBackendConfig().with_build_crate(True)

        with (
            TemporaryDirectory() as tmpdir,
            patch(
                "gradgen._rust_codegen.project._run_cargo_build",
            ) as run_cargo_build,
        ):
            project = create_rust_project(
                f, Path(tmpdir) / "energy_kernel", config=config
            )

            self.assertIsNotNone(project)
            run_cargo_build.assert_called_once_with(
                Path(tmpdir).resolve() / "energy_kernel"
            )

    def test_create_rust_project_raises_when_cargo_missing_for_build(
        self,
    ) -> None:
        x = SXVector.sym("x", 2)
        f = Function(
            "energy", [x], [x.norm2sq()], input_names=["x"], output_names=["y"]
        )

        with (
            TemporaryDirectory() as tmpdir,
            patch(
                "gradgen._rust_codegen.project_support.shutil.which",
                return_value=None,
            ),
        ):
            with self.assertRaisesRegex(
                RuntimeError, "cargo is required to build"
            ):
                create_rust_project(
                    f,
                    Path(tmpdir) / "energy_kernel",
                    config=RustBackendConfig().with_build_crate(True),
                )

    def test_create_rust_project_with_python_interface_builds(self) -> None:
        x = SXVector.sym("x", 2)
        w = SXVector.sym("w", 1)
        f = Function(
            "energy",
            [x, w],
            [x.norm2sq() + w[0], x[0] + x[1]],
            input_names=["x", "w"],
            output_names=["cost", "state"],
        )

        config = RustBackendConfig().with_enable_python_interface(True)

        with TemporaryDirectory() as tmpdir:
            project = create_rust_project(
                f, Path(tmpdir) / "energy_kernel", config=config
            )
            completed = self._run_cargo(project.project_dir, "check")
            self.assertEqual(completed.returncode, 0)
            self.assertIsNotNone(project.python_interface)
            wrapper = project.python_interface
            assert wrapper is not None
            completed_wrapper = self._run_cargo(wrapper.project_dir, "check")
            self.assertEqual(completed_wrapper.returncode, 0)

    def test_create_rust_project_can_skip_python_interface_build(self) -> None:
        x = SXVector.sym("x", 2)
        w = SXVector.sym("w", 1)
        f = Function(
            "energy",
            [x, w],
            [x.norm2sq() + w[0], x[0] + x[1]],
            input_names=["x", "w"],
            output_names=["cost", "state"],
        )

        config = (
            RustBackendConfig()
            .with_enable_python_interface(True)
            .with_build_python_interface(False)
        )

        with TemporaryDirectory() as tmpdir:
            project = create_rust_project(
                f, Path(tmpdir) / "energy_kernel", config=config
            )
            self.assertIsNotNone(project.python_interface)
            assert project.python_interface is not None
            self.assertTrue(project.python_interface.project_dir.is_dir())

    def test_builder_build_uses_parent_directory_and_crate_name(self) -> None:
        x = SXVector.sym("x", 2)
        f = Function(
            "energy", [x], [x.norm2sq()], input_names=["x"], output_names=["y"]
        )

        builder = (
            CodeGenerationBuilder()
            .with_backend_config(RustBackendConfig().with_crate_name("abc"))
            .for_function(f)
            .add_primal()
            .done()
        )

        with TemporaryDirectory() as tmpdir:
            project = builder.build(Path(tmpdir) / "my_crates")
            root = Path(tmpdir).resolve()
            self.assertEqual(project.project_dir, root / "my_crates" / "abc")

    def test_builder_build_defaults_to_current_directory(self) -> None:
        x = SXVector.sym("x", 2)
        f = Function(
            "energy", [x], [x.norm2sq()], input_names=["x"], output_names=["y"]
        )

        builder = (
            CodeGenerationBuilder()
            .with_backend_config(RustBackendConfig().with_crate_name("abc"))
            .for_function(f)
            .add_primal()
            .done()
        )

        with TemporaryDirectory() as tmpdir, contextlib.chdir(tmpdir):
            project = builder.build()
            self.assertEqual(
                project.project_dir, Path(tmpdir).resolve() / "abc"
            )

    def test_builder_build_can_build_generated_crate(self) -> None:
        x = SXVector.sym("x", 2)
        f = Function(
            "energy", [x], [x.norm2sq()], input_names=["x"], output_names=["y"]
        )

        builder = (
            CodeGenerationBuilder()
            .with_backend_config(
                RustBackendConfig()
                .with_crate_name("abc")
                .with_build_crate(True)
            )
            .for_function(f)
            .add_primal()
            .done()
        )

        with (
            TemporaryDirectory() as tmpdir,
            patch.object(
                rust_codegen_module,
                "_run_cargo_build",
            ) as run_cargo_build,
        ):
            builder.build(Path(tmpdir) / "my_crates")
            run_cargo_build.assert_called_once_with(
                Path(tmpdir).resolve() / "my_crates" / "abc"
            )

    def test_python_interface_is_importable(self) -> None:
        x = SXVector.sym("x", 2)
        w = SXVector.sym("w", 1)
        f = Function(
            "energy",
            [x, w],
            [x.norm2sq() + w[0], x[0] + x[1]],
            input_names=["x", "w"],
            output_names=["cost", "state"],
        )

        config = (
            RustBackendConfig()
            .with_crate_name("immediate_import_demo")
            .with_enable_python_interface(True)
        )

        with TemporaryDirectory() as tmpdir:
            project = create_rust_project(
                f, Path(tmpdir) / "energy_kernel", config=config
            )
            assert project.python_interface is not None

            scratch_dir = Path(tmpdir) / "scratch"
            scratch_dir.mkdir()

            completed = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "import immediate_import_demo\n"
                        "print(immediate_import_demo.__version__)\n"
                        "print(immediate_import_demo.__all__)\n"
                        "print(immediate_import_demo.all_functions())\n"
                    ),
                ],
                cwd=scratch_dir,
                check=True,
                capture_output=True,
                text=True,
            )

            self.assertIn("0.1.0", completed.stdout)
            self.assertIn("__version__", completed.stdout)
            self.assertIn("Workspace", completed.stdout)
            self.assertIn("all_functions", completed.stdout)
            self.assertIn("energy", completed.stdout)

    def test_single_output_python_interface_returns_dictionary(self) -> None:
        x = SXVector.sym("x", 2)
        w = SXVector.sym("w", 1)
        f = Function(
            "magic",
            [x, w],
            [x.sin().norm2sq() * w[0]],
            input_names=["x", "w"],
            output_names=["a"],
        )

        config = (
            RustBackendConfig()
            .with_crate_name("blah")
            .with_enable_python_interface(True)
        )

        with TemporaryDirectory() as tmpdir:
            project = create_rust_project(
                f, Path(tmpdir) / "magic_kernel", config=config
            )
            self._run_cargo(project.project_dir, "check")

            assert project.python_interface is not None
            wrapper = project.python_interface
            self._run_cargo(wrapper.project_dir, "check")

            venv_dir = Path(tmpdir) / "venv"
            venv.EnvBuilder(with_pip=True).create(venv_dir)
            python_bin = self._venv_python(venv_dir)

            subprocess.run(
                [
                    str(python_bin),
                    "-m",
                    "pip",
                    "install",
                    "-e",
                    str(wrapper.project_dir),
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            completed = subprocess.run(
                [
                    str(python_bin),
                    "-c",
                    (
                        "import blah\n"
                        "ws = blah.workspace_for_function('magic')\n"
                        "result = blah.magic([1.0, 2.0], [5.0], ws)\n"
                        "print(result)\n"
                    ),
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            self.assertIn("'a':", completed.stdout)

    def test_function_bundle_supports_chainable_updates(self) -> None:
        bundle = (
            FunctionBundle()
            .add_f()
            .add_jf(wrt=[0, 2])
            .add_hessian(wrt=1)
            .add_hvp(wrt=(0, 1))
        )

        self.assertEqual(
            [(item.kind, item.wrt_indices) for item in bundle.items],
            [("f", None), ("jf", (0, 2)), ("hessian", (1,)), ("hvp", (0, 1))],
        )

    def test_code_generation_builder_supports_simplification_setting(
        self,
    ) -> None:
        x = SX.sym("x")
        f = Function("f", [x], [x * x], input_names=["x"], output_names=["y"])

        builder = (
            CodeGenerationBuilder(f)
            .add_primal()
            .add_jacobian()
            .add_hvp()
            .add_joint(FunctionBundle().add_f().add_jf(wrt=0).add_hvp(wrt=0))
            .with_simplification("medium")
        )

        with TemporaryDirectory() as tmpdir:
            project = builder.build(Path(tmpdir) / "simplified_builder")
            lib_text = project.lib_rs.read_text(encoding="utf-8")

            self.assertIn("work[0] = x[0] * x[0];", lib_text)
            self.assertIn("work[0] = 2.0_f64 * x[0];", lib_text)
            self.assertIn("work[0] = 2.0_f64 * v_x[0];", lib_text)
            self.assertIn("work[1] = 2.0_f64 * x[0];", lib_text)
            self.assertIn("work[2] = 2.0_f64 * v_x[0];", lib_text)

    def test_code_generation_builder_supports_vjp_generation(self) -> None:
        x = SXVector.sym("x", 2)
        f = Function(
            "G",
            [x],
            [SXVector((x[0] + x[1], x[0] * x[1], x[1].sin()))],
            input_names=["x"],
            output_names=["y"],
        )

        builder = (
            CodeGenerationBuilder(f)
            .add_primal()
            .add_jacobian()
            .add_vjp()
            .with_simplification("medium")
        )

        with TemporaryDirectory() as tmpdir:
            project = builder.build(Path(tmpdir) / "vjp_builder")
            lib_text = project.lib_rs.read_text(encoding="utf-8")

            self.assertIn("pub fn vjp_builder_G_f(", lib_text)
            self.assertIn("pub fn vjp_builder_G_jf(", lib_text)
            self.assertIn("pub fn vjp_builder_G_vjp(", lib_text)
            self.assertIn("cotangent_y", lib_text)
            self.assertNotIn("y_row0", lib_text)

    def test_composed_function_generates_loop_based_primal_kernel(
        self,
    ) -> None:
        x = SXVector.sym("x", 2)
        state = SXVector.sym("state", 2)
        p = SXVector.sym("p", 2)
        pf = SXVector.sym("pf", 1)

        g = Function(
            "G",
            [state, p],
            [SXVector((state[0] + p[0], state[1] * p[1]))],
            input_names=["state", "p"],
            output_names=["next_state"],
        )
        h = Function(
            "h",
            [state, pf],
            [state[0] + state[1] + pf[0]],
            input_names=["state", "pf"],
            output_names=["y"],
        )
        composed = (
            ComposedFunction("repeat_demo", x)
            .repeat(g, params=[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
            .finish(h, p=[7.0])
        )

        with TemporaryDirectory() as tmpdir:
            project = create_rust_project(
                composed, Path(tmpdir) / "repeat_demo"
            )
            lib_text = project.lib_rs.read_text(encoding="utf-8")

            self.assertIn(
                (
                    "pub fn repeat_demo(x: &[f64], y: &mut [f64], "
                    "work: &mut [f64]) -> Result<(), GradgenError> "
                ),
                lib_text,
            )
            self.assertIn("for repeat_index in 0..3 {", lib_text)
            self.assertNotIn("parameters: &[f64]", lib_text)

            self._append_reference_test(
                project.project_dir,
                composed.to_function(),
                function_name=project.codegen.function_name,
                inputs=([1.0, 2.0],),
                test_name="evaluates_composed_primal_reference",
                workspace_size_override=project.codegen.workspace_size,
            )

            completed = self._run_cargo(project.project_dir, "test", "--quiet")
            self.assertEqual(completed.returncode, 0)

    def test_composed_primal_mixed_params(self) -> None:
        x = SXVector.sym("x", 2)
        state = SXVector.sym("state", 2)
        p1 = SXVector.sym("p1", 2)
        p2 = SXVector.sym("p2", 2)

        g = Function(
            "g",
            [state, p1],
            [SXVector((state[0] + p1[0], state[1] * p1[1]))],
            input_names=["state", "p"],
            output_names=["next_state"],
        )
        h = Function(
            "h",
            [state, SXVector.sym("pf", 0)],
            [state[0] - state[1]],
            input_names=["state", "pf"],
            output_names=["y"],
        )

        composed = (
            ComposedFunction("mixed_primal", x)
            .then(g, p=[10.0, 20.0])
            .repeat(g, params=[p1, p2])
            .finish(h, p=[])
        )

        with TemporaryDirectory() as tmpdir:
            project = create_rust_project(
                composed, Path(tmpdir) / "mixed_primal"
            )
            lib_text = project.lib_rs.read_text(encoding="utf-8")

            self.assertIn("parameters: &[f64]", lib_text)
            self.assertIn("&[10.0_f64, 20.0_f64]", lib_text)
            self.assertIn("for repeat_index in 0..2 {", lib_text)
            self.assertIn("repeat_index * 2", lib_text)
            self.assertIn("(repeat_index + 1) * 2", lib_text)

            self._append_reference_test(
                project.project_dir,
                composed.to_function(),
                function_name=project.codegen.function_name,
                inputs=([1.0, 2.0], [3.0, 4.0, 5.0, 6.0]),
                test_name="evaluates_mixed_fixed_and_symbolic_stages",
                workspace_size_override=project.codegen.workspace_size,
            )

            completed = self._run_cargo(project.project_dir, "test", "--quiet")
            self.assertEqual(completed.returncode, 0)

    def test_composed_primal_offsets(self) -> None:
        x = SXVector.sym("x", 2)
        state = SXVector.sym("state", 2)
        p_stage = SXVector.sym("p_stage", 2)
        p_terminal = SXVector.sym("p_terminal", 1)

        g = Function(
            "g",
            [state, p_stage],
            [SXVector((state[0] + p_stage[0], state[1] + p_stage[1]))],
            input_names=["state", "p"],
            output_names=["next_state"],
        )
        h = Function(
            "h",
            [state, p_terminal],
            [state[0] * state[1] + p_terminal[0]],
            input_names=["state", "pf"],
            output_names=["y"],
        )

        composed = (
            ComposedFunction("offset_demo", x)
            .then(g, p=p_stage)
            .finish(h, p=p_terminal)
        )

        with TemporaryDirectory() as tmpdir:
            project = create_rust_project(
                composed, Path(tmpdir) / "offset_demo"
            )
            lib_text = project.lib_rs.read_text(encoding="utf-8")

            self.assertIn("parameters: &[f64]", lib_text)
            self.assertIn("parameters[0..2]", lib_text)
            self.assertIn("parameters[2..3]", lib_text)
            self.assertNotIn("for repeat_index", lib_text)

            self._append_reference_test(
                project.project_dir,
                composed.to_function(),
                function_name=project.codegen.function_name,
                inputs=([2.0, 3.0], [5.0, 7.0, 11.0]),
                test_name="evaluates_symbolic_stage_and_terminal_offsets",
                workspace_size_override=project.codegen.workspace_size,
            )

            completed = self._run_cargo(project.project_dir, "test", "--quiet")
            self.assertEqual(completed.returncode, 0)

    def test_composed_gradient_codegen_packs_symbolic_parameters(self) -> None:
        x = SXVector.sym("x", 2)
        state = SXVector.sym("state", 2)
        p = SXVector.sym("p", 2)
        pf = SXVector.sym("pf", 1)

        g = Function(
            "G",
            [state, p],
            [SXVector((state[0] + p[0], state[1] * p[1]))],
            input_names=["state", "p"],
            output_names=["next_state"],
        )
        h = Function(
            "h",
            [state, pf],
            [state[0] + state[1] + pf[0]],
            input_names=["state", "pf"],
            output_names=["y"],
        )
        composed = (
            ComposedFunction("packed_demo", x)
            .then(g, p=p)
            .repeat(g, params=[p, p])
            .finish(h, p=pf)
        )
        gradient = composed.gradient()

        with TemporaryDirectory() as tmpdir:
            project = create_rust_project(
                gradient, Path(tmpdir) / "packed_demo_grad"
            )
            lib_text = project.lib_rs.read_text(encoding="utf-8")

            self.assertIn("parameters: &[f64]", lib_text)
            self.assertIn("for repeat_index in (0..2).rev() {", lib_text)
            self.assertIn("_vjp(", lib_text)

            self._append_reference_test(
                project.project_dir,
                gradient.to_function(),
                function_name=project.codegen.function_name,
                inputs=([1.0, 2.0], [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]),
                test_name="evaluates_composed_gradient_reference",
                workspace_size_override=project.codegen.workspace_size,
            )

            completed = self._run_cargo(project.project_dir, "test", "--quiet")
            self.assertEqual(completed.returncode, 0)

    def test_code_generation_builder_accepts_composed_sources_directly(
        self,
    ) -> None:
        x = SXVector.sym("x", 2)
        state = SXVector.sym("state", 2)
        p = SXVector.sym("p", 2)
        pf = SXVector.sym("pf", 1)

        g = Function(
            "G",
            [state, p],
            [SXVector((state[0] + p[0], state[1] * p[1]))],
            input_names=["state", "p"],
            output_names=["next_state"],
        )
        h = Function(
            "h",
            [state, pf],
            [state[0] + state[1] + pf[0]],
            input_names=["state", "pf"],
            output_names=["y"],
        )
        composed = (
            ComposedFunction("loop_demo", x)
            .repeat(g, params=[p, p, p])
            .finish(h, p=pf)
        )

        builder = (
            CodeGenerationBuilder()
            .with_backend_config(
                RustBackendConfig().with_crate_name("loop_demo")
            )
            .for_function(composed)
            .add_primal()
            .add_gradient()
            .done()
        )

        with TemporaryDirectory() as tmpdir:
            project = builder.build(Path(tmpdir) / "loop_demo")
            lib_text = project.lib_rs.read_text(encoding="utf-8")

            self.assertIn("pub fn loop_demo_loop_demo_f(", lib_text)
            self.assertIn("pub fn loop_demo_loop_demo_grad_x(", lib_text)
            self.assertIn("for repeat_index in 0..3 {", lib_text)
            self.assertIn("for repeat_index in (0..3).rev() {", lib_text)
            self.assertEqual(
                len(
                    re.findall(
                        r"^fn loop_demo_loop_demo_repeat_0_[A-Za-z0-9_]+\(",
                        lib_text,
                        flags=re.MULTILINE,
                    )
                ),
                2,
            )
            self.assertNotIn("loop_demo_loop_demo_f_repeat_0_", lib_text)

            primal_function = composed.to_function()
            gradient_function = composed.gradient().to_function()
            primal_expected = self._flatten_runtime_output(
                primal_function,
                primal_function(
                    [1.0, 2.0], [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
                ),
            )
            gradient_expected = self._flatten_runtime_output(
                gradient_function,
                gradient_function(
                    [1.0, 2.0], [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
                ),
            )
            primal_expected_literal = self._rust_array_literal(
                primal_expected, "f64"
            )
            gradient_expected_literal = self._rust_array_literal(
                gradient_expected, "f64"
            )
            self._append_rust_test(
                project.project_dir,
                f"""
#[cfg(test)]
mod tests {{
    use super::*;

    fn assert_close_slice(
        actual: &[f64],
        expected: &[f64],
        tolerance: f64,
    ) {{
        assert_eq!(actual.len(), expected.len());
        for (actual_value, expected_value) in
            actual.iter().zip(expected.iter())
        {{
            assert!(
                (actual_value - expected_value).abs() <= tolerance,
                "expected {{expected_value}}, got {{actual_value}}"
            );
        }}
    }}

    #[test]
    fn evaluates_composed_builder_reference() {{
        let x = [1.0_f64, 2.0_f64];
        let parameters = [
            3.0_f64,
            4.0_f64,
            5.0_f64,
            6.0_f64,
            7.0_f64,
            8.0_f64,
            9.0_f64,
        ];
        let mut primal_y = [0.0_f64; 1];
        let mut primal_work = [0.0_f64; {project.codegens[0].workspace_size}];
        {project.codegens[0].function_name}(
            &x,
            &parameters,
            &mut primal_y,
            &mut primal_work,
        );
        assert_close_slice(&primal_y, &{primal_expected_literal}, 1e-12_f64);

        let mut gradient_y = [0.0_f64; 2];
        let mut gradient_work = [
            0.0_f64;
            {project.codegens[1].workspace_size}
        ];
        {project.codegens[1].function_name}(
            &x,
            &parameters,
            &mut gradient_y,
            &mut gradient_work,
        );
        assert_close_slice(
            &gradient_y,
            &{gradient_expected_literal},
            1e-12_f64,
        );
    }}
}}
""".lstrip(),
            )

            completed = self._run_cargo(project.project_dir, "test", "--quiet")
            self.assertEqual(completed.returncode, 0)

    def test_code_generation_builder_supports_multiple_source_functions(
        self,
    ) -> None:
        x = SX.sym("x")
        u = SX.sym("u")
        f = Function("f", [x], [x * x], input_names=["x"], output_names=["y"])
        g = Function(
            "g", [u], [u.sin()], input_names=["u"], output_names=["z"]
        )

        builder = (
            CodeGenerationBuilder()
            .with_backend_config(
                RustBackendConfig().with_crate_name("multi_demo")
            )
            .for_function(f)
            .add_primal()
            .add_jacobian()
            .with_simplification("medium")
            .done()
            .for_function(g)
            .add_primal()
            .add_jacobian()
            .done()
        )

        with TemporaryDirectory() as tmpdir:
            project = builder.build(Path(tmpdir) / "multi_demo")
            lib_text = project.lib_rs.read_text(encoding="utf-8")
            metadata = json.loads(
                project.metadata_json.read_text(encoding="utf-8")
            )

            self.assertIn("pub fn multi_demo_f_f(", lib_text)
            self.assertIn("pub fn multi_demo_f_jf(", lib_text)
            self.assertIn("pub fn multi_demo_g_f(", lib_text)
            self.assertIn("pub fn multi_demo_g_jf(", lib_text)
            self.assertEqual(
                tuple(codegen.function_name for codegen in project.codegens),
                (
                    "multi_demo_f_f",
                    "multi_demo_f_jf",
                    "multi_demo_g_f",
                    "multi_demo_g_jf",
                ),
            )
            self.assertEqual(metadata["crate_name"], "multi_demo")
            self.assertEqual(metadata["gradgen_version"], _gradgen_version())
            self.assertIsNotNone(
                datetime.fromisoformat(
                    metadata["created_at"].replace("Z", "+00:00")
                )
            )
            self.assertEqual(
                [entry["function_name"] for entry in metadata["functions"]],
                [
                    "multi_demo_f_f",
                    "multi_demo_f_jf",
                    "multi_demo_g_f",
                    "multi_demo_g_jf",
                ],
            )

    def test_builder_requires_selected_function(self) -> None:
        with self.assertRaises(ValueError):
            CodeGenerationBuilder().add_primal()

    def test_builder_rejects_function_override(self) -> None:
        x = SX.sym("x")
        u = SX.sym("u")
        f = Function("f", [x], [x * x], input_names=["x"], output_names=["y"])
        g = Function(
            "g", [u], [u.sin()], input_names=["u"], output_names=["z"]
        )
        builder = (
            CodeGenerationBuilder()
            .with_backend_config(
                RustBackendConfig()
                .with_crate_name("multi_demo")
                .with_function_name("shared_name")
            )
            .for_function(f)
            .add_primal()
            .done()
            .for_function(g)
            .add_primal()
            .done()
        )

        with self.assertRaises(ValueError):
            with TemporaryDirectory() as tmpdir:
                builder.build(Path(tmpdir) / "unused")

    def test_code_generation_builder_rejects_invalid_function_bundle(
        self,
    ) -> None:
        x = SX.sym("x")
        f = Function("f", [x], [x * x], input_names=["x"], output_names=["y"])

        with self.assertRaises(ValueError):
            with TemporaryDirectory() as tmpdir:
                CodeGenerationBuilder(f).add_joint(
                    FunctionBundle().add_f()
                ).build(Path(tmpdir) / "unused")

        with self.assertRaises(IndexError):
            with TemporaryDirectory() as tmpdir:
                CodeGenerationBuilder(f).add_joint(
                    FunctionBundle().add_f().add_jf(wrt=3)
                ).build(Path(tmpdir) / "unused")

    def test_scoped_function_builder_requires_done_before_building_parent(
        self,
    ) -> None:
        x = SX.sym("x")
        f = Function("f", [x], [x * x], input_names=["x"], output_names=["y"])
        scoped = CodeGenerationBuilder().for_function(f).add_primal()

        with self.assertRaises(AttributeError):
            with TemporaryDirectory() as tmpdir:
                scoped.build(
                    Path(tmpdir) / "unused"
                )  # type: ignore[attr-defined]

    def test_scoped_function_builder_rejects_done_without_requests(
        self,
    ) -> None:
        x = SX.sym("x")
        f = Function("f", [x], [x * x], input_names=["x"], output_names=["y"])

        with self.assertRaises(ValueError):
            CodeGenerationBuilder().for_function(f).done()

    def test_builder_supports_callback_style(self) -> None:
        x = SX.sym("x")
        f = Function("f", [x], [x * x], input_names=["x"], output_names=["y"])

        builder = CodeGenerationBuilder().for_function(
            f, lambda b: b.add_primal()
        )

        with TemporaryDirectory() as tmpdir:
            project = builder.build(Path(tmpdir) / "callback_style")
            lib_text = project.lib_rs.read_text(encoding="utf-8")

            self.assertIn("pub fn callback_style_f_f(", lib_text)

    def test_backend_config_supports_scalar_type_updates(self) -> None:
        config = RustBackendConfig().with_scalar_type("f32")

        self.assertEqual(config.scalar_type, "f32")

    def test_backend_config_rejects_invalid_scalar_type(self) -> None:
        with self.assertRaises(ValueError):
            RustBackendConfig().with_scalar_type("f16")

    def test_backend_config_rejects_invalid_backend_mode(self) -> None:
        with self.assertRaises(ValueError):
            RustBackendConfig().with_backend_mode("gpu")

    def test_backend_config_rejects_invalid_crate_name(self) -> None:
        with self.assertRaises(ValueError):
            RustBackendConfig().with_crate_name("my-kernel")

        with self.assertRaises(ValueError):
            RustBackendConfig().with_crate_name("123kernel")

        with self.assertRaises(ValueError):
            RustBackendConfig().with_crate_name("fn")

    def test_backend_config_rejects_invalid_function_name(self) -> None:
        with self.assertRaises(ValueError):
            RustBackendConfig().with_function_name("123kernel")

        with self.assertRaises(ValueError):
            RustBackendConfig().with_function_name("fn")

    def test_generate_rust_accepts_backend_config(self) -> None:
        x = SX.sym("x")
        f = Function(
            "f", [x], [x.sin()], input_names=["x"], output_names=["y"]
        )
        config = (
            RustBackendConfig()
            .with_backend_mode("no_std")
            .with_function_name("eval_kernel")
            .with_emit_metadata_helpers(False)
        )

        result = f.generate_rust(config=config)

        self.assertEqual(result.function_name, "eval_kernel")
        self.assertEqual(result.backend_mode, "no_std")
        self.assertEqual(result.scalar_type, "f64")
        self.assertEqual(result.math_library, "libm")
        self.assertIn("#![no_std]", result.source)
        self.assertIn("pub fn eval_kernel(", result.source)
        self.assertNotIn("WORK_SIZE", result.source)
        self.assertNotIn("pub fn eval_kernel_work_size()", result.source)
        self.assertNotIn("pub fn eval_kernel_input_names()", result.source)
        self.assertNotIn("pub fn eval_kernel_output_names()", result.source)

    def test_generate_rust_no_longer_accepts_math_library_keyword(
        self,
    ) -> None:
        x = SX.sym("x")
        f = Function(
            "f", [x], [x.sin()], input_names=["x"], output_names=["y"]
        )

        with self.assertRaises(TypeError):
            f.generate_rust(backend_mode="no_std", math_library="micromath")

    def test_create_rust_project_accepts_backend_config(self) -> None:
        x = SX.sym("x")
        f = Function("f", [x], [x * x], input_names=["x"], output_names=["y"])
        config = (
            RustBackendConfig()
            .with_crate_name("my_kernel")
            .with_function_name("eval_kernel")
        )

        with TemporaryDirectory() as tmpdir:
            project = f.create_rust_project(
                Path(tmpdir) / "custom_project", config=config
            )

            cargo_text = project.cargo_toml.read_text(encoding="utf-8")
            lib_text = project.lib_rs.read_text(encoding="utf-8")

            self.assertIn('name = "my_kernel"', cargo_text)
            self.assertIn("pub fn eval_kernel(", lib_text)
            self.assertEqual(project.codegen.function_name, "eval_kernel")

    def test_workspace_size_is_bounded_by_number_of_non_leaf_nodes(
        self,
    ) -> None:
        x = SX.sym("x")
        expr = (x * x + 1) + (x * x + 1) * (x * x + 1)
        f = Function("f", [x], [expr])

        result = f.generate_rust()

        self.assertLessEqual(
            result.workspace_size,
            len(
                [
                    node
                    for node in f.nodes
                    if node.op not in {"symbol", "const"}
                ]
            ),
        )

    def test_workspace_slots_are_reused(self) -> None:
        x = SX.sym("x")
        a = x + 1
        b = a * 2
        c = x + 3
        expr = b + c
        f = Function("f", [x], [expr])

        result = f.generate_rust()

        self.assertLess(
            result.workspace_size,
            len(
                [
                    node
                    for node in f.nodes
                    if node.op not in {"symbol", "const"}
                ]
            ),
        )

    def test_f32_codegen_uses_f32_slice_abi_and_literals(self) -> None:
        x = SX.sym("x")
        f = Function(
            "square_plus_one",
            [x],
            [x * x + 1],
            input_names=["x"],
            output_names=["y"],
        )

        result = f.generate_rust(scalar_type="f32")

        self.assertEqual(result.scalar_type, "f32")
        self.assertIn(
            (
                "pub fn square_plus_one(x: &[f32], y: &mut [f32], "
                "work: &mut [f32]) -> Result<(), GradgenError> "
            ),
            result.source,
        )
        self.assertIn(
            (
                "if work.is_empty() { "
                "return Err(GradgenError::WorkspaceTooSmall("
                '"work expected at least 1")); };'
            ),
            result.source,
        )
        self.assertIn("work[0] += 1.0_f32;", result.source)

    def test_no_std_f32_codegen_uses_libm_f32_entry_points(self) -> None:
        x = SX.sym("x")
        expr = x.sin() + x.cos() + x.exp() + x.log() + x.sqrt() + (x**2)
        f = Function("f", [x], [expr], input_names=["x"], output_names=["y"])

        result = f.generate_rust(backend_mode="no_std", scalar_type="f32")

        self.assertEqual(result.scalar_type, "f32")
        self.assertIn("#![no_std]", result.source)
        self.assertIn("libm::sinf(", result.source)
        self.assertIn("libm::cosf(", result.source)
        self.assertIn("libm::expf(", result.source)
        self.assertIn("libm::logf(", result.source)
        self.assertIn("libm::sqrtf(", result.source)
        self.assertIn("x[0] * x[0]", result.source)
        self.assertNotIn("libm::powf(", result.source)

    def test_generated_code_supports_extended_math_methods(self) -> None:
        x = SX.sym("x")
        expr = (
            x.tan()
            + x.asin()
            + x.acos()
            + x.atan()
            + x.sinh()
            + x.cosh()
            + x.tanh()
            + x.expm1()
            + x.log1p()
            + x.abs()
        )
        f = Function("f", [x], [expr], input_names=["x"], output_names=["y"])

        result = f.generate_rust()

        self.assertIn(".tan()", result.source)
        self.assertIn(".asin()", result.source)
        self.assertIn(".acos()", result.source)
        self.assertIn(".atan()", result.source)
        self.assertIn(".sinh()", result.source)
        self.assertIn(".cosh()", result.source)
        self.assertIn(".tanh()", result.source)
        self.assertIn(".exp_m1()", result.source)
        self.assertIn(".ln_1p()", result.source)
        self.assertIn(".abs()", result.source)

    def test_generated_code_supports_additional_elementary_math_methods(
        self,
    ) -> None:
        x = SX.sym("x")
        y = SX.sym("y")
        expr = (
            x.asinh()
            + x.acosh()
            + x.atanh()
            + x.cbrt()
            + x.erf()
            + x.erfc()
            + x.floor()
            + x.ceil()
            + x.round()
            + x.trunc()
            + x.fract()
            + x.signum()
            + x.atan2(y)
            + x.hypot(y)
            + x.minimum(y)
        )
        f = Function(
            "f", [x, y], [expr], input_names=["x", "y"], output_names=["z"]
        )

        result = f.generate_rust()

        self.assertIn(".asinh()", result.source)
        self.assertIn(".acosh()", result.source)
        self.assertIn(".atanh()", result.source)
        self.assertIn(".cbrt()", result.source)
        self.assertIn("erf(", result.source)
        self.assertIn("erfc(", result.source)
        self.assertIn(".floor()", result.source)
        self.assertIn(".ceil()", result.source)
        self.assertIn(".round()", result.source)
        self.assertIn(".trunc()", result.source)
        self.assertIn(".fract()", result.source)
        self.assertIn(".signum()", result.source)
        self.assertIn(".atan2(", result.source)
        self.assertIn(".hypot(", result.source)
        self.assertIn(".min(", result.source)

    def test_no_std_codegen_supports_extended_libm_functions(self) -> None:
        x = SX.sym("x")
        expr = (
            x.tan()
            + x.asin()
            + x.acos()
            + x.atan()
            + x.sinh()
            + x.cosh()
            + x.tanh()
            + x.expm1()
            + x.log1p()
            + x.abs()
        )
        f = Function("f", [x], [expr], input_names=["x"], output_names=["y"])

        result = f.generate_rust(backend_mode="no_std")

        self.assertIn("libm::tan(", result.source)
        self.assertIn("libm::asin(", result.source)
        self.assertIn("libm::acos(", result.source)
        self.assertIn("libm::atan(", result.source)
        self.assertIn("libm::sinh(", result.source)
        self.assertIn("libm::cosh(", result.source)
        self.assertIn("libm::tanh(", result.source)
        self.assertIn("libm::expm1(", result.source)
        self.assertIn("libm::log1p(", result.source)
        self.assertIn("libm::fabs(", result.source)

    def test_generated_code_supports_vector_norms(self) -> None:
        x = SXVector.sym("x", 3)
        f = Function(
            "f",
            [x],
            [
                x.norm1(),
                x.norm2(),
                x.norm2sq(),
                x.norm_inf(),
                x.norm_p(3),
                x.norm_p_to_p(3),
            ],
            input_names=["x"],
            output_names=["n1", "n2", "n2sq", "ni", "np", "npp"],
        )

        result = f.generate_rust()

        self.assertIn("fn norm1(values: &[f64]) -> f64 {", result.source)
        self.assertIn("fn norm2(values: &[f64]) -> f64 {", result.source)
        self.assertIn("fn norm2sq(values: &[f64]) -> f64 {", result.source)
        self.assertIn("fn norm_inf(values: &[f64]) -> f64 {", result.source)
        self.assertIn(
            "fn norm_p(values: &[f64], p: f64) -> f64 {", result.source
        )
        self.assertIn(
            "fn norm_p_to_p(values: &[f64], p: f64) -> f64 {", result.source
        )
        self.assertIn("norm1(x)", result.source)
        self.assertIn("norm2(x)", result.source)
        self.assertIn("norm2sq(x)", result.source)
        self.assertIn("norm_inf(x)", result.source)
        self.assertIn("norm_p(x, 3.0_f64)", result.source)
        self.assertIn("norm_p_to_p(x, 3.0_f64)", result.source)

    def test_generated_code_supports_vector_reductions(self) -> None:
        x = SXVector.sym("x", 3)
        f = Function(
            "f",
            [x],
            [x.sum(), x.prod(), x.max(), x.min(), x.mean()],
            input_names=["x"],
            output_names=["sum", "prod", "max", "min", "mean"],
        )

        result = f.generate_rust()

        self.assertIn("fn vec_sum(values: &[f64]) -> f64 {", result.source)
        self.assertIn("fn vec_prod(values: &[f64]) -> f64 {", result.source)
        self.assertIn("fn vec_max(values: &[f64]) -> f64 {", result.source)
        self.assertIn("fn vec_min(values: &[f64]) -> f64 {", result.source)
        self.assertIn("fn vec_mean(values: &[f64]) -> f64 {", result.source)
        self.assertIn("vec_sum(x)", result.source)
        self.assertIn("vec_prod(x)", result.source)
        self.assertIn("vec_max(x)", result.source)
        self.assertIn("vec_min(x)", result.source)
        self.assertIn("vec_mean(x)", result.source)

    def test_generated_code_supports_constant_matrix_helpers(self) -> None:
        x = SXVector.sym("x", 2)
        y = SXVector.sym("y", 2)
        matrix = [[2.0, 1.0], [1.0, 3.0]]
        f = Function(
            "f",
            [x, y],
            [
                matvec(matrix, x),
                quadform(matrix, x),
                bilinear_form(x, matrix, y),
            ],
            input_names=["x", "y"],
            output_names=["mx", "qx", "bxy"],
        )

        result = f.generate_rust()

        self.assertIn(
            (
                "fn matvec_component(matrix: &[f64], rows: usize, "
                "cols: usize, row: usize, x: &[f64]) -> f64 {"
            ),
            result.source,
        )
        self.assertIn(
            (
                "fn matvec(matrix: &[f64], rows: usize, cols: usize, "
                "x: &[f64], y: &mut [f64]) {"
            ),
            result.source,
        )
        self.assertIn(
            "fn quadform(matrix: &[f64], size: usize, x: &[f64]) -> f64 {",
            result.source,
        )
        self.assertIn(
            (
                "fn bilinear_form(x: &[f64], matrix: &[f64], rows: usize, "
                "cols: usize, y: &[f64]) -> f64 {"
            ),
            result.source,
        )
        self.assertIn(
            "matvec(&[2.0_f64, 1.0_f64, 1.0_f64, 3.0_f64], 2, 2, x, mx);",
            result.source,
        )
        self.assertIn(
            "work[0] = quadform(&[2.0_f64, 2.0_f64, 0.0_f64, 3.0_f64], 2, x);",
            result.source,
        )
        self.assertIn(
            (
                "work[1] = bilinear_form(x, &[2.0_f64, 1.0_f64, "
                "1.0_f64, 3.0_f64], 2, 2, y);"
            ),
            result.source,
        )

    def test_generated_code_supports_f32_constant_matrix_helpers(self) -> None:
        x = SXVector.sym("x", 2)
        y = SXVector.sym("y", 2)
        matrix = [[2.0, 1.0], [1.0, 3.0]]
        f = Function(
            "f",
            [x, y],
            [
                matvec(matrix, x),
                quadform(matrix, x),
                bilinear_form(x, matrix, y),
            ],
            input_names=["x", "y"],
            output_names=["mx", "qx", "bxy"],
        )

        result = f.generate_rust(scalar_type="f32")

        self.assertIn(
            (
                "fn matvec_component(matrix: &[f32], rows: usize, "
                "cols: usize, row: usize, x: &[f32]) -> f32 {"
            ),
            result.source,
        )
        self.assertIn(
            (
                "fn matvec(matrix: &[f32], rows: usize, cols: usize, "
                "x: &[f32], y: &mut [f32]) {"
            ),
            result.source,
        )
        self.assertIn(
            "fn quadform(matrix: &[f32], size: usize, x: &[f32]) -> f32 {",
            result.source,
        )
        self.assertIn(
            (
                "fn bilinear_form(x: &[f32], matrix: &[f32], rows: usize, "
                "cols: usize, y: &[f32]) -> f32 {"
            ),
            result.source,
        )
        self.assertIn(
            "matvec(&[2.0_f32, 1.0_f32, 1.0_f32, 3.0_f32], 2, 2, x, mx);",
            result.source,
        )

    def test_generated_code_uses_matvec_helpers_for_quadratic_form_derivatives(
        self,
    ) -> None:
        x = SXVector.sym("x", 2)
        matrix = [[2.0, 1.0], [1.0, 3.0]]
        f = Function(
            "f",
            [x],
            [quadform(matrix, x)],
            input_names=["x"],
            output_names=["y"],
        )

        gradient_result = f.gradient(0).generate_rust()
        hvp_result = f.hvp(0).generate_rust()

        self.assertIn(
            "matvec(&[4.0_f64, 2.0_f64, 2.0_f64, 6.0_f64], 2, 2, x, y);",
            gradient_result.source,
        )
        self.assertIn(
            "matvec(&[4.0_f64, 2.0_f64, 2.0_f64, 6.0_f64], 2, 2, v_x, y);",
            hvp_result.source,
        )

    def test_multi_function_project_emits_norm2_helper_once(self) -> None:
        x = SXVector.sym("x", 3)
        f1 = Function(
            "f1", [x], [x.norm2()], input_names=["x"], output_names=["y1"]
        )
        f2 = Function(
            "f2", [x], [x.norm2()], input_names=["x"], output_names=["y2"]
        )

        with TemporaryDirectory() as tmpdir:
            project = create_multi_function_rust_project(
                (f1, f2), Path(tmpdir) / "norm2_bundle"
            )
            lib_text = project.lib_rs.read_text(encoding="utf-8")

            self.assertEqual(
                lib_text.count("fn norm2(values: &[f64]) -> f64 {"), 1
            )

    def test_multi_function_project_emits_matrix_helpers_once(self) -> None:
        x = SXVector.sym("x", 2)
        y = SXVector.sym("y", 2)
        matrix = [[2.0, 1.0], [1.0, 3.0]]
        f1 = Function(
            "f1",
            [x],
            [quadform(matrix, x)],
            input_names=["x"],
            output_names=["y1"],
        )
        f2 = Function(
            "f2",
            [x, y],
            [matvec(matrix, x), bilinear_form(x, matrix, y)],
            input_names=["x", "y"],
            output_names=["mx", "bxy"],
        )

        with TemporaryDirectory() as tmpdir:
            project = create_multi_function_rust_project(
                (f1, f2), Path(tmpdir) / "matrix_bundle"
            )
            lib_text = project.lib_rs.read_text(encoding="utf-8")

            self.assertEqual(
                lib_text.count(
                    (
                        "fn matvec_component(matrix: &[f64], rows: usize, "
                        "cols: usize, row: usize, x: &[f64]) -> f64 {"
                    )
                ),
                1,
            )
            self.assertEqual(
                lib_text.count(
                    (
                        "fn matvec(matrix: &[f64], rows: usize, "
                        "cols: usize, x: &[f64], y: &mut [f64]) {"
                    )
                ),
                1,
            )
            self.assertEqual(
                lib_text.count(
                    "fn quadform(matrix: &[f64], size: usize, x: &[f64]) "
                    "-> f64 {"
                ),
                1,
            )
            self.assertEqual(
                lib_text.count(
                    (
                        "fn bilinear_form(x: &[f64], matrix: &[f64], "
                        "rows: usize, cols: usize, y: &[f64]) -> f64 {"
                    )
                ),
                1,
            )

    def test_generated_code_reuses_shared_dag_nodes(self) -> None:
        x = SX.sym("x")
        z = (x * x) + 1
        f = Function(
            "f", [x], [z + z * z], input_names=["x"], output_names=["y"]
        )

        result = f.generate_rust()

        self.assertEqual(result.source.count("x[0] * x[0]"), 1)
        self.assertIn("work[0] = x[0] * x[0];", result.source)
        self.assertIn("work[0] += 1.0_f64;", result.source)
        self.assertIn("work[1] = work[0] * work[0];", result.source)
        self.assertIn("work[0] += work[1];", result.source)

    def test_generated_code_uses_rust_math_methods(self) -> None:
        x = SX.sym("x")
        expr = x.sin() + x.cos() + x.exp() + x.log() + x.sqrt() + (x**2)
        f = Function("f", [x], [expr], input_names=["x"], output_names=["y"])

        result = f.generate_rust()

        self.assertIn(".sin()", result.source)
        self.assertIn(".cos()", result.source)
        self.assertIn(".exp()", result.source)
        self.assertIn(".ln()", result.source)
        self.assertIn(".sqrt()", result.source)
        self.assertIn("x[0] * x[0]", result.source)
        self.assertNotIn(".powf(2.0_f64)", result.source)

    def test_square_power_is_lowered_to_multiplication_across_backends(
        self,
    ) -> None:
        x = SX.sym("x")
        f = Function("f", [x], [x**2], input_names=["x"], output_names=["y"])

        std_f64 = f.generate_rust(backend_mode="std", scalar_type="f64")
        no_std_f64 = f.generate_rust(backend_mode="no_std", scalar_type="f64")
        std_f32 = f.generate_rust(backend_mode="std", scalar_type="f32")
        no_std_f32 = f.generate_rust(backend_mode="no_std", scalar_type="f32")

        for result in (std_f64, no_std_f64, std_f32, no_std_f32):
            self.assertIn("x[0] * x[0]", result.source)
            self.assertNotIn("powf(", result.source)
            self.assertNotIn("libm::pow(", result.source)
            self.assertNotIn("libm::powf(", result.source)

    def test_non_square_power_keeps_pow_calls(self) -> None:
        x = SX.sym("x")
        f = Function("f", [x], [x**3], input_names=["x"], output_names=["y"])

        std_result = f.generate_rust(backend_mode="std", scalar_type="f64")
        no_std_result = f.generate_rust(
            backend_mode="no_std", scalar_type="f64"
        )

        self.assertIn(".powf(3.0_f64)", std_result.source)
        self.assertIn("libm::pow(x[0], 3.0_f64)", no_std_result.source)

    def test_only_exact_square_constant_is_lowered(self) -> None:
        x = SX.sym("x")
        almost_two = Function(
            "f", [x], [x**2.0000001], input_names=["x"], output_names=["y"]
        )

        std_result = almost_two.generate_rust(
            backend_mode="std", scalar_type="f64"
        )
        no_std_result = almost_two.generate_rust(
            backend_mode="no_std", scalar_type="f64"
        )

        self.assertIn(".powf(2.0000001_f64)", std_result.source)
        self.assertIn("libm::pow(x[0], 2.0000001_f64)", no_std_result.source)
        self.assertNotIn("x[0] * x[0]", std_result.source)

    def test_no_std_codegen_uses_libm_and_no_std_crate_attr(self) -> None:
        x = SX.sym("x")
        expr = x.sin() + x.cos() + x.exp() + x.log() + x.sqrt() + (x**2)
        f = Function("f", [x], [expr], input_names=["x"], output_names=["y"])

        result = f.generate_rust(backend_mode="no_std")

        self.assertEqual(result.backend_mode, "no_std")
        self.assertIn("#![no_std]", result.source)
        self.assertIn("libm::sin(", result.source)
        self.assertIn("libm::cos(", result.source)
        self.assertIn("libm::exp(", result.source)
        self.assertIn("libm::log(", result.source)
        self.assertIn("libm::sqrt(", result.source)
        self.assertIn("x[0] * x[0]", result.source)
        self.assertNotIn("libm::pow(", result.source)

    def test_no_std_codegen_supports_sine_explicitly(self) -> None:
        x = SX.sym("x")
        f = Function(
            "sine_only", [x], [x.sin()], input_names=["x"], output_names=["y"]
        )

        result = f.generate_rust(backend_mode="no_std")

        self.assertIn("#![no_std]", result.source)
        self.assertIn("libm::sin(x[0])", result.source)

    def test_no_std_codegen_uses_libm_namespace(self) -> None:
        x = SX.sym("x")
        f = Function(
            "custom_math",
            [x],
            [x.sin() + x.cos()],
            input_names=["x"],
            output_names=["y"],
        )

        result = f.generate_rust(backend_mode="no_std")

        self.assertEqual(result.backend_mode, "no_std")
        self.assertEqual(result.math_library, "libm")
        self.assertIn("#![no_std]", result.source)
        self.assertIn("libm::sin(x[0])", result.source)
        self.assertIn("libm::cos(x[0])", result.source)
        self.assertNotIn('libm = "0.2"', result.source)

    def test_function_level_codegen_works_for_derived_functions(self) -> None:
        x = SX.sym("x")
        df = Function(
            "df",
            [x],
            [derivative(x * x, x)],
            input_names=["x"],
            output_names=["dx"],
        )

        result = df.generate_rust()

        self.assertIn("pub fn df(", result.source)
        self.assertIn("dx: &mut [f64]", result.source)

    def test_create_rust_project_writes_expected_files(self) -> None:
        x = SX.sym("x")
        f = Function(
            "square_plus_one",
            [x],
            [x * x + 1],
            input_names=["x"],
            output_names=["y"],
        )

        with TemporaryDirectory() as tmpdir:
            project = f.create_rust_project(Path(tmpdir) / "generated_kernel")

            self.assertTrue(project.project_dir.is_dir())
            self.assertTrue(project.cargo_toml.is_file())
            self.assertTrue(project.readme.is_file())
            self.assertTrue(project.lib_rs.is_file())

            cargo_text = project.cargo_toml.read_text(encoding="utf-8")
            readme_text = project.readme.read_text(encoding="utf-8")
            lib_text = project.lib_rs.read_text(encoding="utf-8")

            self.assertIn("[package]", cargo_text)
            self.assertIn('name = "square_plus_one"', cargo_text)
            self.assertIn("# square_plus_one", readme_text)
            self.assertIn("cargo build", readme_text)
            self.assertIn(
                "workspace, input, and output dimensions", readme_text
            )
            self.assertIn(
                "pub fn square_plus_one_meta() -> FunctionMetadata {", lib_text
            )
            self.assertIn("pub fn square_plus_one(", lib_text)

            completed = self._run_cargo(
                project.project_dir, "build", "--quiet"
            )
            self.assertEqual(completed.returncode, 0)

    def test_module_level_project_creation_supports_custom_names(self) -> None:
        x = SX.sym("x")
        f = Function("f", [x], [x * x], input_names=["x"], output_names=["y"])

        with TemporaryDirectory() as tmpdir:
            project = create_rust_project(
                f,
                Path(tmpdir) / "custom_project",
                crate_name="my_kernel",
                function_name="eval_kernel",
            )

            cargo_text = project.cargo_toml.read_text(encoding="utf-8")
            lib_text = project.lib_rs.read_text(encoding="utf-8")

            self.assertIn('name = "my_kernel"', cargo_text)
            self.assertIn("pub fn eval_kernel(", lib_text)
            self.assertEqual(project.codegen.function_name, "eval_kernel")

            completed = self._run_cargo(
                project.project_dir, "build", "--quiet"
            )
            self.assertEqual(completed.returncode, 0)

    def test_no_std_project_builds(self) -> None:
        x = SX.sym("x")
        f = Function(
            "trig_kernel",
            [x],
            [x.sin() + x.cos()],
            input_names=["x"],
            output_names=["y"],
        )

        with TemporaryDirectory() as tmpdir:
            project = f.create_rust_project(
                Path(tmpdir) / "trig_kernel",
                backend_mode="no_std",
            )

            cargo_text = project.cargo_toml.read_text(encoding="utf-8")
            readme_text = project.readme.read_text(encoding="utf-8")
            lib_text = project.lib_rs.read_text(encoding="utf-8")

            self.assertIn('libm = "0.2"', cargo_text)
            self.assertIn("Backend mode: `no_std`", readme_text)
            self.assertIn("uses `libm`", readme_text)
            self.assertIn("#![forbid(unsafe_code)]", lib_text)
            self.assertIn("#![no_std]", lib_text)

            try:
                completed = self._run_cargo(
                    project.project_dir, "build", "--quiet"
                )
            except subprocess.CalledProcessError as exc:
                if "Could not resolve host: index.crates.io" in exc.stderr:
                    self.skipTest(
                        (
                            "cargo could not fetch libm in the "
                            "offline test environment"
                        )
                    )
                raise
            self.assertEqual(completed.returncode, 0)

    def test_invalid_backend_mode_is_rejected(self) -> None:
        x = SX.sym("x")
        f = Function("f", [x], [x], input_names=["x"], output_names=["y"])

        with self.assertRaises(ValueError):
            f.generate_rust(backend_mode="gpu")

    def test_generated_rust_project_runs_numeric_smoke_test(self) -> None:
        x = SX.sym("x")
        f = Function(
            "square_plus_one",
            [x],
            [x.sin() + x.cos() + x.exp() + x.log() + x.sqrt() + (x**2)],
            input_names=["x"],
            output_names=["y"],
        )

        with TemporaryDirectory() as tmpdir:
            project = f.create_rust_project(Path(tmpdir) / "runtime_kernel")
            self._append_reference_test(
                project.project_dir,
                f,
                function_name="square_plus_one",
                inputs=4.0,
                test_name="evaluates_against_python_reference",
            )

            completed = self._run_cargo(project.project_dir, "test", "--quiet")
            self.assertEqual(completed.returncode, 0)

    def test_generated_rust_project_is_clippy_clean(self) -> None:
        x = SXVector.sym("x", 3)
        u = SXVector.sym("u", 1)
        f = Function(
            "energy",
            [x, u],
            [x.norm2sq() + u[0] * x[0].sin() + x[1] * x[2]],
            input_names=["x", "u"],
            output_names=["y"],
        )

        with TemporaryDirectory() as tmpdir:
            project = f.create_rust_project(Path(tmpdir) / "energy_kernel")

            completed = self._run_cargo_clippy_clean(project.project_dir)
            self.assertEqual(completed.returncode, 0)

    def test_generated_rust_project_runs_vector_numeric_smoke_test(
        self,
    ) -> None:
        x = SXVector.sym("x", 2)
        f = Function(
            "dot_and_shift",
            [x],
            [x.dot(x), x + SXVector((1 * x[0] * 0 + 1, 1 * x[1] * 0 + 1))],
            input_names=["x"],
            output_names=["dot", "shift"],
        )

        with TemporaryDirectory() as tmpdir:
            project = f.create_rust_project(Path(tmpdir) / "vector_kernel")
            self._append_reference_test(
                project.project_dir,
                f,
                function_name="dot_and_shift",
                inputs=([2.0, 3.0],),
                test_name="evaluates_vector_outputs_against_python_reference",
            )

            completed = self._run_cargo(project.project_dir, "test", "--quiet")
            self.assertEqual(completed.returncode, 0)

    def test_multi_function_generated_rust_project_is_clippy_clean(
        self,
    ) -> None:
        x = SXVector.sym("x", 2)
        u = SXVector.sym("u", 1)
        f1 = Function(
            "energy",
            [x, u],
            [x.norm2sq() + u[0] * x[0]],
            input_names=["x", "u"],
            output_names=["y"],
        )
        f2 = Function(
            "coupling",
            [x, u],
            [x[0] * x[1] + u[0].exp()],
            input_names=["x", "u"],
            output_names=["z"],
        )

        with TemporaryDirectory() as tmpdir:
            project = create_multi_function_rust_project(
                (f1, f2),
                Path(tmpdir) / "bundle_kernel",
            )

            completed = self._run_cargo_clippy_clean(project.project_dir)
            self.assertEqual(completed.returncode, 0)

    def test_generated_rust_project_runs_f32_reference_test(self) -> None:
        x = SX.sym("x")
        f = Function(
            "square_plus_one",
            [x],
            [x.sin() + x * x + 1],
            input_names=["x"],
            output_names=["y"],
        )

        with TemporaryDirectory() as tmpdir:
            project = f.create_rust_project(
                Path(tmpdir) / "f32_kernel", scalar_type="f32"
            )
            self._append_reference_test(
                project.project_dir,
                f,
                function_name=project.codegen.function_name,
                inputs=1.25,
                test_name="evaluates_f32_kernel_against_python_reference",
                config=RustBackendConfig().with_scalar_type("f32"),
                tolerance=1e-5,
            )

            completed = self._run_cargo(project.project_dir, "test", "--quiet")
            self.assertEqual(completed.returncode, 0)

    def test_generated_rust_project_runs_unary_math(self) -> None:
        x = SX.sym("x")
        f = Function(
            "extended_math",
            [x],
            [
                x.tan()
                + x.asin()
                + x.acos()
                + x.atan()
                + x.sinh()
                + x.cosh()
                + x.tanh()
                + x.expm1()
                + x.log1p()
                + x.abs()
            ],
            input_names=["x"],
            output_names=["y"],
        )

        with TemporaryDirectory() as tmpdir:
            project = f.create_rust_project(
                Path(tmpdir) / "extended_math_kernel"
            )
            self._append_reference_test(
                project.project_dir,
                f,
                function_name=project.codegen.function_name,
                inputs=0.2,
                test_name="extended_unary_math_pyref",
                tolerance=1e-12,
            )

            completed = self._run_cargo(project.project_dir, "test", "--quiet")
            self.assertEqual(completed.returncode, 0)

    def test_generated_rust_project_runs_norm_reference_test(self) -> None:
        x = SXVector.sym("x", 3)
        f = Function(
            "norm_kernel",
            [x],
            [
                x.norm1(),
                x.norm2(),
                x.norm2sq(),
                x.norm_inf(),
                x.norm_p(3),
                x.norm_p_to_p(3),
            ],
            input_names=["x"],
            output_names=["n1", "n2", "n2sq", "ni", "np", "npp"],
        )

        with TemporaryDirectory() as tmpdir:
            project = f.create_rust_project(Path(tmpdir) / "norm_kernel")
            self._append_reference_test(
                project.project_dir,
                f,
                function_name=project.codegen.function_name,
                inputs=([3.0, -4.0, 1.0],),
                test_name="evaluates_norm_helpers_against_python_reference",
                tolerance=1e-12,
            )

            completed = self._run_cargo(project.project_dir, "test", "--quiet")
            self.assertEqual(completed.returncode, 0)

    def test_generated_rust_project_runs_reduction_reference_test(
        self,
    ) -> None:
        x = SXVector.sym("x", 3)
        f = Function(
            "reduction_kernel",
            [x],
            [x.sum(), x.prod(), x.max(), x.min(), x.mean()],
            input_names=["x"],
            output_names=["sum", "prod", "max", "min", "mean"],
        )

        with TemporaryDirectory() as tmpdir:
            project = f.create_rust_project(Path(tmpdir) / "reduction_kernel")
            self._append_reference_test(
                project.project_dir,
                f,
                function_name=project.codegen.function_name,
                inputs=([3.0, -4.0, 1.0],),
                test_name="reduction_helpers_pyref",
                tolerance=1e-12,
            )

            completed = self._run_cargo(project.project_dir, "test", "--quiet")
            self.assertEqual(completed.returncode, 0)

    def test_generated_rust_project_runs_matrix_helpers(self) -> None:
        x = SXVector.sym("x", 2)
        y = SXVector.sym("y", 2)
        matrix = [[2.0, 1.0], [1.0, 3.0]]
        f = Function(
            "matrix_kernel",
            [x, y],
            [
                matvec(matrix, x),
                quadform(matrix, x),
                bilinear_form(x, matrix, y),
            ],
            input_names=["x", "y"],
            output_names=["mx", "qx", "bxy"],
        )

        with TemporaryDirectory() as tmpdir:
            project = f.create_rust_project(Path(tmpdir) / "matrix_kernel")
            self._append_reference_test(
                project.project_dir,
                f,
                function_name=project.codegen.function_name,
                inputs=([1.0, 2.0], [3.0, 4.0]),
                test_name="const_matrix_helpers_pyref",
                tolerance=1e-12,
            )

            completed = self._run_cargo(project.project_dir, "test", "--quiet")
            self.assertEqual(completed.returncode, 0)

    def test_generated_rust_project_runs_non_square_matvec(self) -> None:
        x = SXVector.sym("x", 3)
        matrix = [[2.0, -1.0, 0.5], [1.5, 0.0, 4.0]]
        f = Function(
            "rectangular_matvec_kernel",
            [x],
            [matvec(matrix, x)],
            input_names=["x"],
            output_names=["y"],
        )

        with TemporaryDirectory() as tmpdir:
            project = f.create_rust_project(
                Path(tmpdir) / "rectangular_matvec_kernel"
            )
            self._append_reference_test(
                project.project_dir,
                f,
                function_name=project.codegen.function_name,
                inputs=([1.0, -2.0, 3.0],),
                test_name="rectangular_matvec_pyref",
                tolerance=1e-12,
            )

            completed = self._run_cargo(project.project_dir, "test", "--quiet")
            self.assertEqual(completed.returncode, 0)

    def test_generated_f32_matrix_helpers(self) -> None:
        x = SXVector.sym("x", 2)
        y = SXVector.sym("y", 2)
        matrix = [[2.0, 1.0], [1.0, 3.0]]
        f = Function(
            "matrix_kernel_f32",
            [x, y],
            [
                matvec(matrix, x),
                quadform(matrix, x),
                bilinear_form(x, matrix, y),
            ],
            input_names=["x", "y"],
            output_names=["mx", "qx", "bxy"],
        )
        config = RustBackendConfig().with_scalar_type("f32")

        with TemporaryDirectory() as tmpdir:
            project = f.create_rust_project(
                Path(tmpdir) / "matrix_kernel_f32", config=config
            )
            self._append_reference_test(
                project.project_dir,
                f,
                function_name=project.codegen.function_name,
                inputs=([1.0, 2.0], [3.0, 4.0]),
                test_name="const_matrix_helpers_pyref_f32",
                config=config,
                tolerance=1e-5,
            )

            completed = self._run_cargo(project.project_dir, "test", "--quiet")
            self.assertEqual(completed.returncode, 0)

    def test_generated_no_std_matrix_helpers(self) -> None:
        x = SXVector.sym("x", 2)
        y = SXVector.sym("y", 2)
        matrix = [[2.0, 1.0], [1.0, 3.0]]
        f = Function(
            "matrix_kernel",
            [x, y],
            [
                matvec(matrix, x),
                quadform(matrix, x),
                bilinear_form(x, matrix, y),
            ],
            input_names=["x", "y"],
            output_names=["mx", "qx", "bxy"],
        )
        config = RustBackendConfig().with_backend_mode("no_std")

        with TemporaryDirectory() as tmpdir:
            project = f.create_rust_project(
                Path(tmpdir) / "matrix_kernel_no_std", config=config
            )
            self._append_reference_test(
                project.project_dir,
                f,
                function_name=project.codegen.function_name,
                inputs=([1.0, 2.0], [3.0, 4.0]),
                test_name="const_matrix_helpers_pyref_no_std",
                config=config,
                tolerance=1e-12,
            )

            try:
                completed = self._run_cargo(
                    project.project_dir, "test", "--quiet"
                )
            except subprocess.CalledProcessError as exc:
                if (
                    "Could not resolve host" in exc.stderr
                    or "failed to get `libm` as a dependency" in exc.stderr
                ):
                    self.skipTest(
                        (
                            "no_std runtime test requires fetching "
                            "libm from crates.io"
                        )
                    )
                raise
            self.assertEqual(completed.returncode, 0)

    def test_generated_rust_project_runs_jacobian_reference_test(self) -> None:
        x = SXVector.sym("x", 2)
        jac = Function(
            "f", [x], [x.dot(x)], input_names=["x"], output_names=["y"]
        ).jacobian(0)

        with TemporaryDirectory() as tmpdir:
            project = jac.create_rust_project(Path(tmpdir) / "jacobian_kernel")
            self._append_reference_test(
                project.project_dir,
                jac,
                function_name=project.codegen.function_name,
                inputs=([2.0, 3.0],),
                test_name="evaluates_jacobian_against_python_reference",
            )
            completed = self._run_cargo(project.project_dir, "test", "--quiet")
            self.assertEqual(completed.returncode, 0)

    def test_generated_rust_project_runs_vector_jacobian_reference_test(
        self,
    ) -> None:
        x = SXVector.sym("x", 2)
        jac = Function(
            "G",
            [x],
            [SXVector((x[0] + x[1], x[0] * x[1], x[1].sin()))],
            input_names=["x"],
            output_names=["y"],
        ).jacobian(0)

        with TemporaryDirectory() as tmpdir:
            project = jac.create_rust_project(
                Path(tmpdir) / "vector_jacobian_kernel"
            )
            lib_text = project.lib_rs.read_text(encoding="utf-8")
            self.assertIn("pub fn G_jacobian_x(", lib_text)
            self.assertIn("jacobian_y: &mut [f64]", lib_text)
            self.assertIn(
                (
                    "output slice receiving the Jacobian block "
                    "for declared result `y`"
                ),
                lib_text,
            )
            self.assertNotIn("y_row0", lib_text)

            self._append_reference_test(
                project.project_dir,
                jac,
                function_name=project.codegen.function_name,
                inputs=([3.0, 4.0],),
                test_name="evaluates_vector_jacobian_against_python_reference",
            )
            completed = self._run_cargo(project.project_dir, "test", "--quiet")
            self.assertEqual(completed.returncode, 0)

    def test_generated_rust_project_runs_vjp_reference_test(self) -> None:
        x = SXVector.sym("x", 2)
        reverse = Function(
            "G",
            [x],
            [SXVector((x[0] + x[1], x[0] * x[1], x[1].sin()))],
            input_names=["x"],
            output_names=["y"],
        ).vjp(wrt_index=0)

        with TemporaryDirectory() as tmpdir:
            project = reverse.create_rust_project(Path(tmpdir) / "vjp_kernel")
            self._append_reference_test(
                project.project_dir,
                reverse,
                function_name=project.codegen.function_name,
                inputs=([3.0, 4.0], [2.0, -1.0, 5.0]),
                test_name="evaluates_vjp_against_python_reference",
            )
            completed = self._run_cargo(project.project_dir, "test", "--quiet")
            self.assertEqual(completed.returncode, 0)

    def test_generated_rust_project_runs_hessian_reference_test(self) -> None:
        x = SXVector.sym("x", 2)
        f = Function(
            "f",
            [x],
            [(x[0] * x[0]) + (x[0] * x[1]) + (x[1] * x[1])],
            input_names=["x"],
            output_names=["y"],
        ).hessian(0)

        with TemporaryDirectory() as tmpdir:
            project = f.create_rust_project(Path(tmpdir) / "hessian_kernel")
            self._append_reference_test(
                project.project_dir,
                f,
                function_name=project.codegen.function_name,
                inputs=([2.0, 3.0],),
                test_name="evaluates_hessian_against_python_reference",
            )
            completed = self._run_cargo(project.project_dir, "test", "--quiet")
            self.assertEqual(completed.returncode, 0)

    def test_generated_rust_project_runs_custom_vector_hessian_reference_test(
        self,
    ) -> None:
        weighted_sqnorm = register_elementary_function(
            name="weighted_sqnorm_runtime",
            input_dimension=2,
            parameter_dimension=2,
            parameter_defaults=[1.0, 1.0],
            eval_python=lambda x, w: w[0] * x[0] * x[0] + w[1] * x[1] * x[1],
            jacobian=lambda x, w: SXVector((2 * w[0] * x[0], 2 * w[1] * x[1])),
            hessian=lambda x, w: (
                SXVector((2 * w[0], SX.const(0.0))),
                SXVector((SX.const(0.0), 2 * w[1])),
            ),
            rust_primal="""
fn weighted_sqnorm_runtime(
    x: &[{{ scalar_type }}],
    w: &[{{ scalar_type }}],
) -> {{ scalar_type }} {
    w[0] * x[0] * x[0] + w[1] * x[1] * x[1]
}
""",
            rust_hessian="""
fn weighted_sqnorm_runtime_hessian(
    x: &[{{ scalar_type }}],
    w: &[{{ scalar_type }}],
    out: &mut [{{ scalar_type }}],
) {
    let _ = x;
    out[0] = 2.0_{{ scalar_type }} * w[0];
    out[1] = 0.0_{{ scalar_type }};
    out[2] = 0.0_{{ scalar_type }};
    out[3] = 2.0_{{ scalar_type }} * w[1];
}
""",
        )

        x = SXVector.sym("x", 2)
        f = Function(
            "f",
            [x],
            [weighted_sqnorm(x, w=[2.0, 3.0])],
            input_names=["x"],
            output_names=["y"],
        ).hessian(0)

        with TemporaryDirectory() as tmpdir:
            project = f.create_rust_project(
                Path(tmpdir) / "custom_hessian_kernel"
            )
            self._append_reference_test(
                project.project_dir,
                f,
                function_name=project.codegen.function_name,
                inputs=([1.0, 2.0],),
                test_name="custom_vector_hessian_pyref",
            )
            completed = self._run_cargo(project.project_dir, "test", "--quiet")
            self.assertEqual(completed.returncode, 0)

    def test_generated_rust_project_runs_workspace_reuse_reference_test(
        self,
    ) -> None:
        x = SX.sym("x")
        z = (x * x) + 1
        f = Function(
            "reuse_kernel",
            [x],
            [z + z * z],
            input_names=["x"],
            output_names=["y"],
        )

        with TemporaryDirectory() as tmpdir:
            project = f.create_rust_project(Path(tmpdir) / "reuse_kernel")
            self._append_reference_test(
                project.project_dir,
                f,
                function_name=project.codegen.function_name,
                inputs=2.5,
                test_name="reuse_heavy_kernel_pyref",
                tolerance=1e-12,
            )

            completed = self._run_cargo(project.project_dir, "test", "--quiet")
            self.assertEqual(completed.returncode, 0)

    def test_custom_vector_derivative_helpers_skip_dead_component_workspace(
        self,
    ) -> None:
        weighted_sqnorm = register_elementary_function(
            name="weighted_sqnorm_no_dead_work",
            input_dimension=2,
            parameter_dimension=2,
            eval_python=lambda x, w: w[0] * x[0] * x[0] + w[1] * x[1] * x[1],
            jacobian=lambda x, w: [2 * w[0] * x[0], 2 * w[1] * x[1]],
            hessian=lambda x, w: [
                [2 * w[0], 0.0],
                [0.0, 2 * w[1]],
            ],
            hvp=lambda x, v, w: [2 * w[0] * v[0], 2 * w[1] * v[1]],
            rust_primal="""
fn weighted_sqnorm_no_dead_work(
    x: &[{{ scalar_type }}],
    w: &[{{ scalar_type }}],
) -> {{ scalar_type }} {
    w[0] * x[0] * x[0] + w[1] * x[1] * x[1]
}
""",
            rust_jacobian="""
fn weighted_sqnorm_no_dead_work_jacobian(
    x: &[{{ scalar_type }}],
    w: &[{{ scalar_type }}],
    out: &mut [{{ scalar_type }}],
) {
    out[0] = 2.0_{{ scalar_type }} * w[0] * x[0];
    out[1] = 2.0_{{ scalar_type }} * w[1] * x[1];
}
""",
            rust_hessian="""
fn weighted_sqnorm_no_dead_work_hessian(
    x: &[{{ scalar_type }}],
    w: &[{{ scalar_type }}],
    out: &mut [{{ scalar_type }}],
) {
    let _ = x;
    out[0] = 2.0_{{ scalar_type }} * w[0];
    out[1] = 0.0_{{ scalar_type }};
    out[2] = 0.0_{{ scalar_type }};
    out[3] = 2.0_{{ scalar_type }} * w[1];
}
""",
            rust_hvp="""
fn weighted_sqnorm_no_dead_work_hvp(
    x: &[{{ scalar_type }}],
    v_x: &[{{ scalar_type }}],
    w: &[{{ scalar_type }}],
    out: &mut [{{ scalar_type }}],
) {
    let _ = x;
    out[0] = 2.0_{{ scalar_type }} * w[0] * v_x[0];
    out[1] = 2.0_{{ scalar_type }} * w[1] * v_x[1];
}
""",
        )

        x = SXVector.sym("x", 2)
        function = Function(
            "custom_energy",
            [x],
            [weighted_sqnorm(x, w=[2.0, 3.0])],
            input_names=["x"],
            output_names=["y"],
        )

        gradient_codegen = (
            function.gradient(0)
            .simplify("medium", name="custom_energy_grad")
            .generate_rust(backend_mode="no_std")
        )
        hessian_codegen = (
            function.hessian(0)
            .simplify("medium", name="custom_energy_hessian")
            .generate_rust(backend_mode="no_std")
        )
        hvp_codegen = (
            function.hvp(0)
            .simplify("medium", name="custom_energy_hvp")
            .generate_rust(backend_mode="no_std")
        )

        self.assertEqual(gradient_codegen.workspace_size, 0)
        self.assertIn(
            "weighted_sqnorm_no_dead_work_jacobian(",
            gradient_codegen.source,
        )
        self.assertNotIn("work[0] =", gradient_codegen.source)

        self.assertEqual(hessian_codegen.workspace_size, 0)
        self.assertIn(
            "weighted_sqnorm_no_dead_work_hessian(x, &[2.0_f64, 3.0_f64], y);",
            hessian_codegen.source,
        )
        self.assertNotIn("work[0] =", hessian_codegen.source)

        self.assertEqual(hvp_codegen.workspace_size, 0)
        self.assertIn(
            "weighted_sqnorm_no_dead_work_hvp(",
            hvp_codegen.source,
        )
        self.assertNotIn("work[0] =", hvp_codegen.source)

    def test_generated_rust_project_builds_for_simplified_function(
        self,
    ) -> None:
        x = SX.sym("x")
        f = Function(
            "f",
            [x],
            [derivative(x * x, x)],
            input_names=["x"],
            output_names=["dx"],
        ).simplify(max_effort="medium")

        with TemporaryDirectory() as tmpdir:
            project = f.create_rust_project(Path(tmpdir) / "simplified_kernel")
            completed = self._run_cargo(
                project.project_dir, "build", "--quiet"
            )
            self.assertEqual(completed.returncode, 0)

    def test_zero_workspace_function_codegen_builds(self) -> None:
        x = SX.sym("x")
        f = Function(
            "identity", [x], [x], input_names=["x"], output_names=["y"]
        )

        result = f.generate_rust()

        self.assertEqual(result.workspace_size, 0)
        self.assertIn("workspace_size: 0,", result.source)
        self.assertIn(
            (
                "pub fn identity(x: &[f64], y: &mut [f64], "
                "_work: &mut [f64]) -> Result<(), GradgenError> "
            ),
            result.source,
        )
        self.assertNotIn("assert!(work.len() >= 0);", result.source)
        self.assertNotIn(
            "pub fn identity(x: &[f64], y: &mut [f64], work: &mut [f64]) {",
            result.source,
        )

        with TemporaryDirectory() as tmpdir:
            project = f.create_rust_project(Path(tmpdir) / "identity_kernel")
            completed = self._run_cargo(
                project.project_dir, "build", "--quiet"
            )
            self.assertEqual(completed.returncode, 0)

    def test_no_std_project_runs_reference_test(self) -> None:
        x = SX.sym("x")
        f = Function(
            "trig_kernel",
            [x],
            [x.sin() + x.cos()],
            input_names=["x"],
            output_names=["y"],
        )

        with TemporaryDirectory() as tmpdir:
            project = f.create_rust_project(
                Path(tmpdir) / "trig_kernel",
                backend_mode="no_std",
            )
            self._append_reference_test(
                project.project_dir,
                f,
                function_name=project.codegen.function_name,
                inputs=0.25,
                test_name="evaluates_no_std_kernel_against_python_reference",
                tolerance=1e-12,
            )
            try:
                completed = self._run_cargo(
                    project.project_dir, "test", "--quiet"
                )
            except subprocess.CalledProcessError as exc:
                if "Could not resolve host: index.crates.io" in exc.stderr:
                    self.skipTest(
                        (
                            "cargo could not fetch libm in the "
                            "offline test environment"
                        )
                    )
                raise
            self.assertEqual(completed.returncode, 0)

    def test_no_std_f32_project_builds(self) -> None:
        x = SX.sym("x")
        f = Function(
            "trig_kernel",
            [x],
            [x.sin() + x.cos()],
            input_names=["x"],
            output_names=["y"],
        )

        with TemporaryDirectory() as tmpdir:
            project = f.create_rust_project(
                Path(tmpdir) / "trig_kernel_f32",
                backend_mode="no_std",
                scalar_type="f32",
            )

            cargo_text = project.cargo_toml.read_text(encoding="utf-8")
            readme_text = project.readme.read_text(encoding="utf-8")
            lib_text = project.lib_rs.read_text(encoding="utf-8")

            self.assertIn('libm = "0.2"', cargo_text)
            self.assertIn("Scalar type: `f32`", readme_text)
            self.assertIn("libm::sinf(", lib_text)
            self.assertIn(
                (
                    "pub fn trig_kernel(x: &[f32], y: &mut [f32], "
                    "work: &mut [f32]) -> Result<(), GradgenError> "
                ),
                lib_text,
            )

            try:
                completed = self._run_cargo(
                    project.project_dir, "build", "--quiet"
                )
            except subprocess.CalledProcessError as exc:
                if "Could not resolve host: index.crates.io" in exc.stderr:
                    self.skipTest(
                        (
                            "cargo could not fetch libm in the "
                            "offline test environment"
                        )
                    )
                raise
            self.assertEqual(completed.returncode, 0)

    def test_no_std_f32_project_runs_reference_test(self) -> None:
        x = SX.sym("x")
        f = Function(
            "trig_kernel",
            [x],
            [x.sin() + x.cos()],
            input_names=["x"],
            output_names=["y"],
        )

        with TemporaryDirectory() as tmpdir:
            project = f.create_rust_project(
                Path(tmpdir) / "trig_kernel_f32_runtime",
                backend_mode="no_std",
                scalar_type="f32",
            )
            self._append_reference_test(
                project.project_dir,
                f,
                function_name=project.codegen.function_name,
                inputs=0.25,
                test_name="no_std_f32_kernel_pyref",
                config=RustBackendConfig()
                .with_backend_mode("no_std")
                .with_scalar_type("f32"),
                tolerance=1e-5,
            )
            try:
                completed = self._run_cargo(
                    project.project_dir, "test", "--quiet"
                )
            except subprocess.CalledProcessError as exc:
                if "Could not resolve host: index.crates.io" in exc.stderr:
                    self.skipTest(
                        (
                            "cargo could not fetch libm in the "
                            "offline test environment"
                        )
                    )
                raise
            self.assertEqual(completed.returncode, 0)

    def test_generated_rust_project_builds_without_metadata_helpers(
        self,
    ) -> None:
        x = SX.sym("x")
        f = Function(
            "square_plus_one",
            [x],
            [x * x + 1],
            input_names=["x"],
            output_names=["y"],
        )
        config = RustBackendConfig().with_emit_metadata_helpers(False)

        with TemporaryDirectory() as tmpdir:
            project = f.create_rust_project(
                Path(tmpdir) / "helperless_kernel",
                config=config,
            )
            self.assertNotIn(
                "WORK_SIZE", project.lib_rs.read_text(encoding="utf-8")
            )
            self._append_rust_test(
                project.project_dir,
                """
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn evaluates_without_metadata_helpers() {
        let x = [3.0_f64];
        let mut y = [0.0_f64];
        let mut work = [0.0_f64; 2];
        square_plus_one(&x, &mut y, &mut work);
        assert_eq!(y[0], 10.0_f64);
    }
}
""".lstrip(),
            )

            completed = self._run_cargo(project.project_dir, "test", "--quiet")
            self.assertEqual(completed.returncode, 0)

    def test_joint_kernel_argument_docs(self) -> None:
        x = SXVector.sym("x", 3)
        u = SXVector.sym("u", 2)
        f = Function(
            "f",
            [x, u],
            [x.dot(x) + u[0]],
            input_names=["x", "u"],
            output_names=["y"],
        )
        builder = CodeGenerationBuilder(f).add_joint(
            FunctionBundle().add_f().add_jf(wrt=0).add_hvp(wrt=0)
        )

        with TemporaryDirectory() as tmpdir:
            project = builder.build(Path(tmpdir) / "my_kernel")
            lib_text = project.lib_rs.read_text(encoding="utf-8")

            self.assertIn("/// - `x`:", lib_text)
            self.assertIn(
                "///   input slice for the declared argument `x`", lib_text
            )
            self.assertIn("///   Expected length: 3.", lib_text)
            self.assertIn("/// - `u`:", lib_text)
            self.assertIn(
                "///   input slice for the declared argument `u`", lib_text
            )
            self.assertIn("///   Expected length: 2.", lib_text)
            self.assertIn("/// - `v_x`:", lib_text)
            self.assertIn(
                (
                    "///   tangent or direction input associated with "
                    "declared argument `x`;"
                ),
                lib_text,
            )
            self.assertIn(
                "///   use this slice when forming Hessian-vector-product or",
                lib_text,
            )
            self.assertIn("///   derivative terms", lib_text)
            self.assertIn("///   Expected length: 3.", lib_text)
            self.assertIn("/// - `y`:", lib_text)
            self.assertIn(
                "///   primal output slice for the declared result `y`",
                lib_text,
            )
            self.assertIn("///   Expected length: 1.", lib_text)
            self.assertIn("/// - `jacobian_y`:", lib_text)
            self.assertIn(
                (
                    "///   output slice receiving the Jacobian block "
                    "for declared result `y`"
                ),
                lib_text,
            )
            self.assertIn("///   Expected length: 3.", lib_text)
            self.assertIn("/// - `hvp_y`:", lib_text)
            self.assertIn(
                (
                    "///   output slice receiving the "
                    "Hessian-vector product for declared"
                ),
                lib_text,
            )
            self.assertIn("///   result `y`", lib_text)
            self.assertIn("///   Expected length: 3.", lib_text)

    def test_project_exposes_name_helpers(self) -> None:
        x = SXVector.sym("x", 2)
        y = SX.sym("y")
        f = Function(
            "named_kernel",
            [x, y],
            [x.dot(x), y],
            input_names=["state vector", "gain"],
            output_names=["energy", "gain out"],
        )

        with TemporaryDirectory() as tmpdir:
            project = f.create_rust_project(Path(tmpdir) / "named_kernel")
            self._append_rust_test(
                project.project_dir,
                """
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exposes_declared_argument_names() {
    }

    #[test]
    fn exposes_metadata_struct() {
        let meta = named_kernel_meta();
        assert_eq!(meta.function_name, "named_kernel");
        assert_eq!(meta.input_names, ["state vector", "gain"]);
        assert_eq!(meta.input_sizes, [2, 1]);
        assert_eq!(meta.output_names, ["energy", "gain out"]);
        assert_eq!(meta.output_sizes, [1, 1]);
    }
}
""".lstrip(),
            )

            completed = self._run_cargo(project.project_dir, "test", "--quiet")
            self.assertEqual(completed.returncode, 0)

    def test_codegen_sanitizes_names_for_rust(self) -> None:
        x = SX.sym("x")
        f = Function(
            "my function",
            [x],
            [x * x],
            input_names=["1-input"],
            output_names=["out value"],
        )

        result = f.generate_rust()

        self.assertIn("pub fn my_function(", result.source)
        self.assertIn("_1_input: &[f64]", result.source)
        self.assertIn("out_value: &mut [f64]", result.source)

    def test_generate_rust_rejects_name_collisions(self) -> None:
        x = SX.sym("x")
        y = SX.sym("y")
        f = Function(
            "f",
            [x, y],
            [x + y],
            input_names=["x-value", "x_value"],
            output_names=["sum"],
        )

        with self.assertRaisesRegex(
            ValueError, "both map to the Rust identifier"
        ):
            f.generate_rust()

    def test_generate_rust_rejects_collision_with_work_argument(self) -> None:
        x = SX.sym("x")
        f = Function(
            "f",
            [x],
            [x],
            input_names=["x"],
            output_names=["work"],
        )

        with self.assertRaisesRegex(
            ValueError, "both map to the Rust identifier 'work'"
        ):
            f.generate_rust()

    def test_create_rust_derivative_bundle_writes_projects(self) -> None:
        x = SXVector.sym("x", 2)
        f = Function(
            "f", [x], [x.dot(x)], input_names=["x"], output_names=["y"]
        )

        with TemporaryDirectory() as tmpdir:
            bundle = f.create_rust_derivative_bundle(
                Path(tmpdir) / "bundle",
                simplify_derivatives="high",
            )

            self.assertTrue(bundle.bundle_dir.is_dir())
            self.assertIsNotNone(bundle.primal)
            self.assertEqual(len(bundle.jacobians), 1)
            self.assertEqual(len(bundle.hessians), 1)
            self.assertTrue(
                (bundle.bundle_dir / "primal" / "Cargo.toml").is_file()
            )
            self.assertTrue(
                (bundle.bundle_dir / "f_jacobian_x" / "Cargo.toml").is_file()
            )
            self.assertTrue(
                (bundle.bundle_dir / "f_hessian_x" / "Cargo.toml").is_file()
            )

            self.assertEqual(
                self._run_cargo(
                    bundle.primal.project_dir, "build", "--quiet"
                ).returncode,
                0,
            )
            self.assertEqual(
                self._run_cargo(
                    bundle.jacobians[0].project_dir, "build", "--quiet"
                ).returncode,
                0,
            )
            self.assertEqual(
                self._run_cargo(
                    bundle.hessians[0].project_dir, "build", "--quiet"
                ).returncode,
                0,
            )

    def test_module_level_create_rust_derivative_bundle_works(self) -> None:
        x = SX.sym("x")
        f = Function("f", [x], [x * x], input_names=["x"], output_names=["y"])

        with TemporaryDirectory() as tmpdir:
            bundle = create_rust_derivative_bundle(
                f,
                Path(tmpdir) / "bundle",
                include_hessians=False,
            )

            self.assertIsNotNone(bundle.primal)
            self.assertEqual(len(bundle.jacobians), 1)
            self.assertEqual(len(bundle.hessians), 0)

    def test_builder_creates_multi_function_crate(self) -> None:
        x = SXVector.sym("x", 2)
        f = Function(
            "f",
            [x],
            [x[0] * x[0] + x[0] * x[1] + x[1] * x[1]],
            input_names=["x"],
            output_names=["y"],
        )
        builder = (
            CodeGenerationBuilder(f).add_primal().add_gradient().add_hvp()
        )

        with TemporaryDirectory() as tmpdir:
            project = builder.build(Path(tmpdir) / "single_crate")

            self.assertTrue(project.project_dir.is_dir())
            self.assertEqual(len(project.codegens), 3)
            lib_text = project.lib_rs.read_text(encoding="utf-8")
            self.assertIn("pub fn single_crate_f_f(", lib_text)
            self.assertIn("pub fn single_crate_f_grad(", lib_text)
            self.assertIn("pub fn single_crate_f_hvp(", lib_text)

            self._append_rust_test(
                project.project_dir,
                """
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn evaluates_primal_gradient_and_hvp() {
        let x = [3.0_f64, 4.0_f64];
        let v_x = [1.0_f64, 2.0_f64];

        let mut y = [0.0_f64];
        let mut y_grad = [0.0_f64, 0.0_f64];
        let mut y_hvp = [0.0_f64, 0.0_f64];

        let mut work_f = vec![0.0_f64; single_crate_f_f_meta().workspace_size];
        let mut work_grad = vec![
            0.0_f64;
            single_crate_f_grad_meta().workspace_size
        ];
        let mut work_hvp = vec![
            0.0_f64;
            single_crate_f_hvp_meta().workspace_size
        ];

        single_crate_f_f(&x, &mut y, &mut work_f);
        single_crate_f_grad(&x, &mut y_grad, &mut work_grad);
        single_crate_f_hvp(&x, &v_x, &mut y_hvp, &mut work_hvp);

        assert_eq!(y[0], 37.0_f64);
        assert_eq!(y_grad, [10.0_f64, 11.0_f64]);
        assert_eq!(y_hvp, [4.0_f64, 5.0_f64]);
    }
}
""".lstrip(),
            )

            completed = self._run_cargo(project.project_dir, "test", "--quiet")
            self.assertEqual(completed.returncode, 0)

    def test_code_generation_builder_supports_joint_requests(self) -> None:
        x = SXVector.sym("x", 2)
        f = Function(
            "f",
            [x],
            [x[0] * x[0] + x[0] * x[1] + x[1] * x[1]],
            input_names=["x"],
            output_names=["y"],
        )
        builder = CodeGenerationBuilder(f).add_joint(
            FunctionBundle().add_f().add_jf(wrt=0)
        )

        with TemporaryDirectory() as tmpdir:
            project = builder.build(Path(tmpdir) / "joint")

            self.assertTrue(project.project_dir.is_dir())
            self.assertEqual(len(project.codegens), 1)
            lib_text = project.lib_rs.read_text(encoding="utf-8")
            self.assertIn("pub fn joint_f_f_jf(", lib_text)

            self._append_rust_test(
                project.project_dir,
                """
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn evaluates_joint_primal_and_jacobian() {
        let x = [3.0_f64, 4.0_f64];
        let mut y = [0.0_f64];
        let mut jacobian_y = [0.0_f64, 0.0_f64];
        let mut work = vec![0.0_f64; joint_f_f_jf_meta().workspace_size];

        joint_f_f_jf(&x, &mut y, &mut jacobian_y, &mut work);

        assert_eq!(y[0], 37.0_f64);
        assert_eq!(jacobian_y, [10.0_f64, 11.0_f64]);
    }
}
""".lstrip(),
            )

            completed = self._run_cargo(project.project_dir, "test", "--quiet")
            self.assertEqual(completed.returncode, 0)

    def test_builder_supports_joint_jacobian_hvp(self) -> None:
        x = SXVector.sym("x", 2)
        f = Function(
            "f",
            [x],
            [x[0] * x[0] + x[0] * x[1] + x[1] * x[1]],
            input_names=["x"],
            output_names=["y"],
        )
        builder = CodeGenerationBuilder(f).add_joint(
            FunctionBundle().add_f().add_jf(wrt=0).add_hvp(wrt=0)
        )

        with TemporaryDirectory() as tmpdir:
            project = builder.build(Path(tmpdir) / "joint_f_jf_hvp")

            lib_text = project.lib_rs.read_text(encoding="utf-8")
            self.assertIn("pub fn joint_f_jf_hvp_f_f_jf_hvp(", lib_text)

            self._append_rust_test(
                project.project_dir,
                """
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn evaluates_joint_primal_jacobian_and_hvp() {
        let x = [3.0_f64, 4.0_f64];
        let v_x = [1.0_f64, 2.0_f64];
        let mut y = [0.0_f64];
        let mut jacobian_y = [0.0_f64, 0.0_f64];
        let mut hvp_y = [0.0_f64, 0.0_f64];
        let mut work = vec![
            0.0_f64;
            joint_f_jf_hvp_f_f_jf_hvp_meta().workspace_size
        ];

        joint_f_jf_hvp_f_f_jf_hvp(
            &x,
            &v_x,
            &mut y,
            &mut jacobian_y,
            &mut hvp_y,
            &mut work,
        );

        assert_eq!(y[0], 37.0_f64);
        assert_eq!(jacobian_y, [10.0_f64, 11.0_f64]);
        assert_eq!(hvp_y, [4.0_f64, 5.0_f64]);
    }
}
""".lstrip(),
            )

            completed = self._run_cargo(project.project_dir, "test", "--quiet")
            self.assertEqual(completed.returncode, 0)

    def test_builder_supports_joint_hessian(self) -> None:
        x = SXVector.sym("x", 2)
        f = Function(
            "joint_hessian",
            [x],
            [(x[0] * x[0]) + (x[0] * x[1]) + (x[1] * x[1])],
            input_names=["x"],
            output_names=["y"],
        )
        builder = CodeGenerationBuilder(f).add_joint(
            FunctionBundle().add_f().add_hessian(wrt=0)
        )

        with TemporaryDirectory() as tmpdir:
            project = builder.build(Path(tmpdir) / "joint_hessian")
            lib_text = project.lib_rs.read_text(encoding="utf-8")

            self.assertIn(
                "pub fn joint_hessian_joint_hessian_f_hessian(", lib_text
            )
            self.assertIn("hessian_y: &mut [f64]", lib_text)
            self.assertIn("hessian_y expected length 4", lib_text)

            self._append_rust_test(
                project.project_dir,
                """
#[cfg(test)]
mod joint_hessian_tests {
    use super::*;

    #[test]
    fn evaluates_joint_primal_and_flat_hessian() {
        let x = [2.0_f64, 3.0_f64];
        let mut y = [0.0_f64; 1];
        let mut hessian_y = [0.0_f64; 4];
        let mut work = vec![
            0.0_f64;
            joint_hessian_joint_hessian_f_hessian_meta().workspace_size
        ];

        joint_hessian_joint_hessian_f_hessian(
            &x,
            &mut y,
            &mut hessian_y,
            &mut work,
        );

        assert_eq!(y, [19.0_f64]);
        assert_eq!(hessian_y, [2.0_f64, 1.0_f64, 1.0_f64, 2.0_f64]);
    }
}
""".lstrip(),
            )

            completed = self._run_cargo(project.project_dir, "test", "--quiet")
            self.assertEqual(completed.returncode, 0)

    def test_builder_expands_function_bundle(self) -> None:
        x = SX.sym("x")
        y = SX.sym("y")
        f = Function(
            "f",
            [x, y],
            [x * x + x * y + y * y],
            input_names=["x", "y"],
            output_names=["out"],
        )
        builder = CodeGenerationBuilder(f).add_joint(
            FunctionBundle().add_f().add_jf(wrt=[0, 1])
        )

        with TemporaryDirectory() as tmpdir:
            project = builder.build(Path(tmpdir) / "multi_joint")
            lib_text = project.lib_rs.read_text(encoding="utf-8")

            self.assertIn("pub fn multi_joint_f_f_jf_x(", lib_text)
            self.assertIn("pub fn multi_joint_f_f_jf_y(", lib_text)

    def test_builder_supports_no_std_f32(self) -> None:
        x = SX.sym("x")
        f = Function(
            "f", [x], [x * x + 1], input_names=["x"], output_names=["y"]
        )
        builder = (
            CodeGenerationBuilder(f)
            .with_backend_config(
                RustBackendConfig()
                .with_crate_name("my_kernel")
                .with_backend_mode("no_std")
                .with_scalar_type("f32")
            )
            .add_primal()
            .add_gradient()
            .add_hvp()
        )

        with TemporaryDirectory() as tmpdir:
            project = builder.build(Path(tmpdir) / "builder_no_std_f32")

            lib_text = project.lib_rs.read_text(encoding="utf-8")
            cargo_text = project.cargo_toml.read_text(encoding="utf-8")

            self.assertIn("#![no_std]", lib_text)
            self.assertIn(
                (
                    "pub fn my_kernel_f_f(x: &[f32], y: &mut [f32], "
                    "work: &mut [f32]) -> Result<(), GradgenError> "
                ),
                lib_text,
            )
            self.assertIn(
                (
                    "pub fn my_kernel_f_grad(x: &[f32], y: &mut [f32], "
                    "work: &mut [f32]) -> Result<(), GradgenError> "
                ),
                lib_text,
            )
            self.assertIn("pub fn my_kernel_f_hvp", lib_text)
            self.assertIn("-> Result<(), GradgenError>", lib_text)
            self.assertIn('libm = "0.2"', cargo_text)
            self.assertIn('name = "my_kernel"', cargo_text)
            self.assertIn("#![forbid(unsafe_code)]", lib_text)

            try:
                completed = self._run_cargo(
                    project.project_dir, "build", "--quiet"
                )
            except subprocess.CalledProcessError as exc:
                if "Could not resolve host: index.crates.io" in exc.stderr:
                    self.skipTest(
                        (
                            "cargo could not fetch libm in the "
                            "offline test environment"
                        )
                    )
                raise
            self.assertEqual(completed.returncode, 0)

    def test_builder_supports_matrix_helpers(self) -> None:
        x = SXVector.sym("x", 2)
        y = SXVector.sym("y", 2)
        matrix = [[2.0, 1.0], [1.0, 3.0]]
        f = Function(
            "f",
            [x, y],
            [quadform(matrix, x)],
            input_names=["x", "y"],
            output_names=["y_out"],
        )
        builder = (
            CodeGenerationBuilder(f)
            .with_backend_config(
                RustBackendConfig().with_crate_name("matrix_builder")
            )
            .add_primal()
            .add_gradient()
            .add_hvp()
        )

        with TemporaryDirectory() as tmpdir:
            project = builder.build(Path(tmpdir) / "matrix_builder")
            lib_text = project.lib_rs.read_text(encoding="utf-8")

            self.assertIn(
                (
                    "fn matvec_component(matrix: &[f64], rows: usize, "
                    "cols: usize, row: usize, x: &[f64]) -> f64 {"
                ),
                lib_text,
            )
            self.assertIn(
                (
                    "fn matvec(matrix: &[f64], rows: usize, cols: usize, "
                    "x: &[f64], y: &mut [f64]) {"
                ),
                lib_text,
            )
            self.assertIn(
                "fn quadform(matrix: &[f64], size: usize, x: &[f64]) -> f64 {",
                lib_text,
            )
            self.assertIn("pub fn matrix_builder_f_f(", lib_text)
            self.assertIn("pub fn matrix_builder_f_grad_x(", lib_text)
            self.assertIn("pub fn matrix_builder_f_grad_y(", lib_text)
            self.assertIn("pub fn matrix_builder_f_hvp_x(", lib_text)
            self.assertIn("pub fn matrix_builder_f_hvp_y(", lib_text)

    def test_map_function_matches_reference(self) -> None:
        x = SXVector.sym("x", 2)
        g = Function(
            "g",
            [x],
            [SXVector((x[0] + 2.0 * x[1], x[0] * x[1]))],
            input_names=["x"],
            output_names=["y"],
        )
        mapped = map_function(
            g, 3, input_name="x_seq", name="g_map", simplification="medium"
        )
        expanded = mapped.to_function()
        inputs = ([1.0, 2.0, -1.0, 3.0, 0.5, -2.0],)

        with TemporaryDirectory() as tmpdir:
            project = create_rust_project(
                mapped, Path(tmpdir) / "g_map_kernel"
            )
            self.assertIn(
                "for stage_index in 0..3",
                project.lib_rs.read_text(encoding="utf-8"),
            )
            self._append_reference_test(
                project.project_dir,
                expanded,
                function_name=project.codegen.function_name,
                inputs=inputs,
                test_name="mapped_primal_matches_reference",
                workspace_size_override=project.codegen.workspace_size,
            )
            completed = self._run_cargo(project.project_dir, "test", "--quiet")
            self.assertEqual(completed.returncode, 0)

    def test_map_function_jacobian_matches_reference(self) -> None:
        x = SXVector.sym("x", 2)
        g = Function(
            "g",
            [x],
            [SXVector((x[0] + x[1], x[0] * x[1]))],
            input_names=["x"],
            output_names=["y"],
        )
        mapped = map_function(
            g, 2, input_name="x_seq", name="g_map", simplification="medium"
        )
        mapped_jacobian = mapped.jacobian(0, name="g_map_jacobian_x_seq")
        expanded = mapped_jacobian.to_function()
        inputs = ([2.0, 3.0, -1.0, 4.0],)

        with TemporaryDirectory() as tmpdir:
            project = create_rust_project(
                mapped_jacobian, Path(tmpdir) / "g_map_jacobian_kernel"
            )
            lib_text = project.lib_rs.read_text(encoding="utf-8")
            self.assertIn("for stage_index in 0..2", lib_text)
            self.assertIn("jacobian_y.fill(0.0_f64);", lib_text)
            self._append_reference_test(
                project.project_dir,
                expanded,
                function_name=project.codegen.function_name,
                inputs=inputs,
                test_name="mapped_jacobian_matches_reference",
                workspace_size_override=project.codegen.workspace_size,
            )
            completed = self._run_cargo(project.project_dir, "test", "--quiet")
            self.assertEqual(completed.returncode, 0)

    def test_zip_function_matches_reference(self) -> None:
        x = SXVector.sym("x", 2)
        y = SXVector.sym("y", 2)
        g = Function(
            "pairwise",
            [x, y],
            [SXVector((x[0] + y[1], x[1] * y[0] + x[0]))],
            input_names=["x", "y"],
            output_names=["z"],
        )
        zipped = zip_function(
            g,
            3,
            input_names=("x_seq", "y_seq"),
            name="pairwise_zip",
            simplification="medium",
        )
        expanded = zipped.to_function()
        inputs = (
            [1.0, 2.0, -1.0, 3.0, 0.5, -2.0],
            [4.0, -1.0, 2.0, 5.0, -3.0, 1.5],
        )

        with TemporaryDirectory() as tmpdir:
            project = create_rust_project(
                zipped, Path(tmpdir) / "pairwise_zip_kernel"
            )
            self.assertIn(
                "for stage_index in 0..3",
                project.lib_rs.read_text(encoding="utf-8"),
            )
            self._append_reference_test(
                project.project_dir,
                expanded,
                function_name=project.codegen.function_name,
                inputs=inputs,
                test_name="zip_primal_matches_reference",
                workspace_size_override=project.codegen.workspace_size,
            )
            completed = self._run_cargo(project.project_dir, "test", "--quiet")
            self.assertEqual(completed.returncode, 0)

    def test_zip_function_jacobian_matches_reference(self) -> None:
        x = SXVector.sym("x", 2)
        y = SXVector.sym("y", 1)
        g = Function(
            "pairwise_scalar",
            [x, y],
            [x[0] * y[0] + x[1]],
            input_names=["x", "y"],
            output_names=["out"],
        )
        zipped = zip_function(
            g,
            3,
            input_names=("x_seq", "y_seq"),
            name="pairwise_zip",
            simplification="medium",
        )
        zipped_jacobian = zipped.jacobian(
            1, name="pairwise_zip_jacobian_y_seq"
        )
        expanded = zipped_jacobian.to_function()
        inputs = (
            [2.0, 1.0, -1.0, 4.0, 0.5, -2.0],
            [3.0, -2.0, 1.5],
        )

        with TemporaryDirectory() as tmpdir:
            builder = (
                CodeGenerationBuilder()
                .with_backend_config(
                    RustBackendConfig().with_crate_name("pairwise_zip_kernel")
                )
                .for_function(zipped)
                .add_primal()
                .add_jacobian()
                .done()
            )
            project = builder.build(Path(tmpdir) / "pairwise_zip_kernel")
            lib_text = project.lib_rs.read_text(encoding="utf-8")
            self.assertIn(
                "pub fn pairwise_zip_kernel_pairwise_zip_f(", lib_text
            )
            self.assertIn(
                "pub fn pairwise_zip_kernel_pairwise_zip_jf_y_seq(", lib_text
            )
            self.assertIn("for stage_index in 0..3", lib_text)
            self._append_reference_test(
                project.project_dir,
                expanded,
                function_name="pairwise_zip_kernel_pairwise_zip_jf_y_seq",
                inputs=inputs,
                test_name="zip_jacobian_matches_reference",
                workspace_size_override=next(
                    codegen.workspace_size
                    for codegen in project.codegens
                    if codegen.function_name
                    == "pairwise_zip_kernel_pairwise_zip_jf_y_seq"
                ),
            )
            completed = self._run_cargo(project.project_dir, "test", "--quiet")
            self.assertEqual(completed.returncode, 0)


if __name__ == "__main__":
    unittest.main()
