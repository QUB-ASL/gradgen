import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import sympy as sp

from gradgen import CodeGenerationBuilder, ComposedFunction, Function, RustBackendConfig, SXVector


class IntegrationTests(unittest.TestCase):
    @staticmethod
    def _run_cargo(project_dir: Path, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["cargo", *args],
            cwd=project_dir,
            check=True,
            capture_output=True,
            text=True,
        )

    @staticmethod
    def _rust_array_literal(values: list[float], scalar_type: str) -> str:
        return "[" + ", ".join(f"{repr(float(value))}_{scalar_type}" for value in values) + "]"

    @staticmethod
    def _flatten_sympy_matrix(matrix: sp.Matrix) -> list[float]:
        return [float(value) for value in matrix]

    @staticmethod
    def _append_rust_test(project_dir: Path, test_source: str) -> None:
        lib_rs = project_dir / "src" / "lib.rs"
        with lib_rs.open("a", encoding="utf-8") as handle:
            handle.write("\n")
            handle.write(test_source)

    def test_scalar_function_matches_sympy_gradient_and_hessian(self) -> None:
        x_symbols = sp.symbols("x0:5", real=True)
        sympy_expr = (
            sp.exp(x_symbols[0] * x_symbols[1])
            + sp.sin(x_symbols[2]) * sp.log(1 + x_symbols[3] ** 2)
            + sp.sqrt(x_symbols[4] + 5)
            + x_symbols[0] * x_symbols[3] * x_symbols[4]
        )
        sympy_gradient = sp.Matrix([sp.diff(sympy_expr, symbol) for symbol in x_symbols])
        sympy_hessian = sp.hessian(sympy_expr, x_symbols)

        numeric_point = [0.5, -0.75, 0.3, 1.2, 2.0]
        substitutions = dict(zip(x_symbols, numeric_point))
        expected_primal = float(sympy_expr.subs(substitutions).evalf())
        expected_gradient = self._flatten_sympy_matrix(sympy_gradient.subs(substitutions).evalf())
        expected_hessian = self._flatten_sympy_matrix(sympy_hessian.subs(substitutions).evalf())

        x = SXVector.sym("x", 5)
        expr = (
            (x[0] * x[1]).exp()
            + x[2].sin() * (1 + x[3] * x[3]).log()
            + (x[4] + 5).sqrt()
            + x[0] * x[3] * x[4]
        )
        function = Function("energy", [x], [expr], input_names=["x"], output_names=["y"])

        builder = (
            CodeGenerationBuilder(function)
            .with_backend_config(RustBackendConfig().with_crate_name("sympy_scalar"))
            .add_primal()
            .add_gradient()
            .add_hessian()
            .with_simplification("medium")
        )

        with TemporaryDirectory() as tmpdir:
            project = builder.build(Path(tmpdir) / "sympy_scalar")
            primal_codegen, gradient_codegen, hessian_codegen = project.codegens

            self._append_rust_test(
                project.project_dir,
                f"""
#[cfg(test)]
mod integration_sympy_scalar {{
    use super::*;

    fn assert_close_slice(actual: &[f64], expected: &[f64], tolerance: f64) {{
        assert_eq!(actual.len(), expected.len());
        for (actual_value, expected_value) in actual.iter().zip(expected.iter()) {{
            assert!(
                (actual_value - expected_value).abs() <= tolerance,
                "expected {{expected_value}}, got {{actual_value}}"
            );
        }}
    }}

    #[test]
    fn matches_sympy_reference_values() {{
        let x = {self._rust_array_literal(numeric_point, "f64")};
        let mut primal_y = [0.0_f64; 1];
        let mut primal_work = [0.0_f64; {primal_codegen.workspace_size}];
        {primal_codegen.function_name}(&x, &mut primal_y, &mut primal_work);
        assert_close_slice(&primal_y, &{self._rust_array_literal([expected_primal], "f64")}, 1e-10_f64);

        let mut gradient_y = [0.0_f64; 5];
        let mut gradient_work = [0.0_f64; {gradient_codegen.workspace_size}];
        {gradient_codegen.function_name}(&x, &mut gradient_y, &mut gradient_work);
        assert_close_slice(&gradient_y, &{self._rust_array_literal(expected_gradient, "f64")}, 1e-10_f64);

        let mut hessian_y = [0.0_f64; 25];
        let mut hessian_work = [0.0_f64; {hessian_codegen.workspace_size}];
        {hessian_codegen.function_name}(&x, &mut hessian_y, &mut hessian_work);
        assert_close_slice(&hessian_y, &{self._rust_array_literal(expected_hessian, "f64")}, 1e-10_f64);
    }}
}}
""".lstrip(),
            )

            completed = self._run_cargo(project.project_dir, "test", "--quiet")
            self.assertEqual(completed.returncode, 0)

    def test_vector_function_matches_sympy_jacobian_and_vjp(self) -> None:
        x_symbols = sp.symbols("x0:4", real=True)
        sympy_outputs = sp.Matrix(
            [
                x_symbols[0] * x_symbols[1] + sp.sin(x_symbols[2]),
                sp.exp(x_symbols[1] - x_symbols[3]) + x_symbols[0] ** 2,
                sp.log(1 + x_symbols[0] ** 2 + x_symbols[2] ** 2) + x_symbols[3],
            ]
        )
        sympy_jacobian = sympy_outputs.jacobian(x_symbols)

        numeric_point = [0.7, -1.1, 0.4, 0.2]
        cotangent = [1.5, -0.5, 2.0]
        substitutions = dict(zip(x_symbols, numeric_point))
        expected_primal = self._flatten_sympy_matrix(sympy_outputs.subs(substitutions).evalf())
        expected_jacobian = self._flatten_sympy_matrix(sympy_jacobian.subs(substitutions).evalf())
        expected_vjp = self._flatten_sympy_matrix(
            (sympy_jacobian.subs(substitutions).T * sp.Matrix(cotangent)).evalf()
        )

        x = SXVector.sym("x", 4)
        outputs = SXVector(
            (
                x[0] * x[1] + x[2].sin(),
                (x[1] - x[3]).exp() + x[0] * x[0],
                (1 + x[0] * x[0] + x[2] * x[2]).log() + x[3],
            )
        )
        function = Function("g", [x], [outputs], input_names=["x"], output_names=["y"])

        builder = (
            CodeGenerationBuilder(function)
            .with_backend_config(RustBackendConfig().with_crate_name("sympy_vector"))
            .add_primal()
            .add_jacobian()
            .add_vjp()
            .with_simplification("medium")
        )

        with TemporaryDirectory() as tmpdir:
            project = builder.build(Path(tmpdir) / "sympy_vector")
            primal_codegen, jacobian_codegen, vjp_codegen = project.codegens

            self._append_rust_test(
                project.project_dir,
                f"""
#[cfg(test)]
mod integration_sympy_vector {{
    use super::*;

    fn assert_close_slice(actual: &[f64], expected: &[f64], tolerance: f64) {{
        assert_eq!(actual.len(), expected.len());
        for (actual_value, expected_value) in actual.iter().zip(expected.iter()) {{
            assert!(
                (actual_value - expected_value).abs() <= tolerance,
                "expected {{expected_value}}, got {{actual_value}}"
            );
        }}
    }}

    #[test]
    fn matches_sympy_reference_values() {{
        let x = {self._rust_array_literal(numeric_point, "f64")};
        let cotangent_y = {self._rust_array_literal(cotangent, "f64")};
        let mut primal_y = [0.0_f64; 3];
        let mut primal_work = [0.0_f64; {primal_codegen.workspace_size}];
        {primal_codegen.function_name}(&x, &mut primal_y, &mut primal_work);
        assert_close_slice(&primal_y, &{self._rust_array_literal(expected_primal, "f64")}, 1e-10_f64);

        let mut jacobian_y = [0.0_f64; 12];
        let mut jacobian_work = [0.0_f64; {jacobian_codegen.workspace_size}];
        {jacobian_codegen.function_name}(&x, &mut jacobian_y, &mut jacobian_work);
        assert_close_slice(&jacobian_y, &{self._rust_array_literal(expected_jacobian, "f64")}, 1e-10_f64);

        let mut vjp_y = [0.0_f64; 4];
        let mut vjp_work = [0.0_f64; {vjp_codegen.workspace_size}];
        {vjp_codegen.function_name}(&x, &cotangent_y, &mut vjp_y, &mut vjp_work);
        assert_close_slice(&vjp_y, &{self._rust_array_literal(expected_vjp, "f64")}, 1e-10_f64);
    }}
}}
""".lstrip(),
            )

            completed = self._run_cargo(project.project_dir, "test", "--quiet")
            self.assertEqual(completed.returncode, 0)

    def test_composed_function_matches_sympy_primal_and_gradient(self) -> None:
        x0, x1 = sp.symbols("x0 x1", real=True)
        p_symbols = sp.symbols("p0:8", real=True)
        pf0, pf1 = sp.symbols("pf0 pf1", real=True)

        state = sp.Matrix([x0, x1])
        for repeat_index in range(4):
            p0 = p_symbols[2 * repeat_index]
            p1 = p_symbols[2 * repeat_index + 1]
            state = sp.Matrix(
                [
                    sp.Float("0.7") * state[0] + p0 * state[1] + sp.sin(p1),
                    sp.Float("0.2") * state[1] + p1 * state[0] + p0**2,
                ]
            )
        sympy_expr = state[0] ** 2 + pf0 * state[1] + sp.exp(state[0] - pf1)
        sympy_gradient = sp.Matrix([sp.diff(sympy_expr, x0), sp.diff(sympy_expr, x1)])

        x_values = [0.25, -0.4]
        parameter_values = [0.6, -0.2, -0.3, 0.5, 0.9, -0.7, 0.1, 0.8, 0.75, -0.1]
        substitutions = {
            x0: x_values[0],
            x1: x_values[1],
            **{symbol: value for symbol, value in zip(p_symbols, parameter_values[:8])},
            pf0: parameter_values[8],
            pf1: parameter_values[9],
        }
        expected_primal = float(sympy_expr.subs(substitutions).evalf())
        expected_gradient = self._flatten_sympy_matrix(sympy_gradient.subs(substitutions).evalf())

        x = SXVector.sym("x", 2)
        state_vector = SXVector.sym("state", 2)
        p = SXVector.sym("p", 2)
        pf = SXVector.sym("pf", 2)

        g = Function(
            "g",
            [state_vector, p],
            [
                SXVector(
                    (
                        0.7 * state_vector[0] + p[0] * state_vector[1] + p[1].sin(),
                        0.2 * state_vector[1] + p[1] * state_vector[0] + p[0] * p[0],
                    )
                )
            ],
            input_names=["state", "p"],
            output_names=["next_state"],
        )
        h = Function(
            "h",
            [state_vector, pf],
            [state_vector[0] * state_vector[0] + pf[0] * state_vector[1] + (state_vector[0] - pf[1]).exp()],
            input_names=["state", "pf"],
            output_names=["y"],
        )
        composed = ComposedFunction("sympy_composed", x).repeat(g, params=[p, p, p, p]).finish(h, p=pf)

        builder = (
            CodeGenerationBuilder()
            .with_backend_config(RustBackendConfig().with_crate_name("sympy_composed"))
            .for_function(composed)
            .add_primal()
            .add_gradient()
            .with_simplification("medium")
            .done()
        )

        with TemporaryDirectory() as tmpdir:
            project = builder.build(Path(tmpdir) / "sympy_composed")
            primal_codegen, gradient_codegen = project.codegens

            self._append_rust_test(
                project.project_dir,
                f"""
#[cfg(test)]
mod integration_sympy_composed {{
    use super::*;

    fn assert_close_slice(actual: &[f64], expected: &[f64], tolerance: f64) {{
        assert_eq!(actual.len(), expected.len());
        for (actual_value, expected_value) in actual.iter().zip(expected.iter()) {{
            assert!(
                (actual_value - expected_value).abs() <= tolerance,
                "expected {{expected_value}}, got {{actual_value}}"
            );
        }}
    }}

    #[test]
    fn matches_sympy_reference_values() {{
        let x = {self._rust_array_literal(x_values, "f64")};
        let parameters = {self._rust_array_literal(parameter_values, "f64")};
        let mut primal_y = [0.0_f64; 1];
        let mut primal_work = [0.0_f64; {primal_codegen.workspace_size}];
        {primal_codegen.function_name}(&x, &parameters, &mut primal_y, &mut primal_work);
        assert_close_slice(&primal_y, &{self._rust_array_literal([expected_primal], "f64")}, 1e-10_f64);

        let mut gradient_y = [0.0_f64; 2];
        let mut gradient_work = [0.0_f64; {gradient_codegen.workspace_size}];
        {gradient_codegen.function_name}(&x, &parameters, &mut gradient_y, &mut gradient_work);
        assert_close_slice(&gradient_y, &{self._rust_array_literal(expected_gradient, "f64")}, 1e-10_f64);
    }}
}}
""".lstrip(),
            )

            completed = self._run_cargo(project.project_dir, "test", "--quiet")
            self.assertEqual(completed.returncode, 0)
