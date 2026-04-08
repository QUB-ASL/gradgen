import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import sympy as sp

from gradgen import (
    CodeGenerationBuilder,
    ComposedFunction,
    FunctionComposer,
    Function,
    FunctionBundle,
    RustBackendConfig,
    SX,
    SXVector,
    SingleShootingBundle,
    SingleShootingProblem,
    map_function,
    reduce_function,
    zip_function,
)


class IntegrationTests(unittest.TestCase):
    @staticmethod
    def _run_cargo(project_dir: Path, *args: str) \
            -> subprocess.CompletedProcess[str]:
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
        sympy_gradient = sp.Matrix([sp.diff(sympy_expr, symbol)
                                    for symbol in x_symbols])
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

    def test_composed_pipeline_emits_norm2sq_helper(self) -> None:
        x = SXVector.sym("x", 2)
        mapped_function = Function(
            "u_map",
            [x],
            [x.norm2sq().sin()],
            input_names=["x"],
            output_names=["y"],
        )
        mapped = map_function(
            mapped_function,
            5,
            input_name="x_seq",
            name="mapped_seq",
        )

        a = SX.sym("a")
        y = SX.sym("y")
        reduced_function = Function(
            "h",
            [a, y],
            [a + y],
            input_names=["a", "y"],
            output_names=["s"],
        )
        reduced = reduce_function(
            reduced_function,
            5,
            accumulator_input_name="acc",
            input_name="y_seq",
            output_name="acc_final",
            name="summation",
        )

        b = SX.sym("b")
        s = SX.sym("s")
        post = Function(
            "post",
            [b, s],
            [b * s**3],
            input_names=["b", "s"],
            output_names=["z"],
        )

        comp = (
            FunctionComposer(mapped)
            .feed_into(reduced, arg="y_seq")
            .feed_into(post, arg="s")
            .compose(name="comp")
        )

        builder = (
            CodeGenerationBuilder()
            .with_backend_config(
                RustBackendConfig()
                .with_crate_name("compozer")
                .with_enable_python_interface()
                .with_build_python_interface()
            )
            .for_function(comp)
            .add_primal()
            .done()
        )

        with TemporaryDirectory() as tmpdir:
            project = builder.build(Path(tmpdir) / "compozer")
            self.assertIsNotNone(project.python_interface)
            lib_text = project.lib_rs.read_text(encoding="utf-8")
            self.assertIn("fn norm2sq(", lib_text)

    def test_nested_composed_pipeline_shares_module_prelude_once(self) -> None:
        x = SXVector.sym("x", 2)
        mapped_function = Function(
            "u_map",
            [x],
            [x.norm2sq().sin()],
            input_names=["x"],
            output_names=["y"],
        )
        mapped = map_function(
            mapped_function,
            5,
            input_name="x_seq",
            name="mapped_seq",
        )

        a = SX.sym("a")
        y = SX.sym("y")
        reduced_function = Function(
            "h",
            [a, y],
            [a + y],
            input_names=["a", "y"],
            output_names=["s"],
        )
        reduced = reduce_function(
            reduced_function,
            5,
            accumulator_input_name="acc",
            input_name="y_seq",
            output_name="acc_final",
            name="summation",
        )

        b = SX.sym("b")
        s = SX.sym("s")
        post = Function(
            "post",
            [b, s],
            [b * s**3],
            input_names=["b", "s"],
            output_names=["z"],
        )

        comp = (
            FunctionComposer(mapped)
            .feed_into(reduced, arg="y_seq")
            .feed_into(post, arg="s")
            .compose(name="comp")
        )

        ss = SX.sym("ss")
        j = Function(
            "jjj",
            [ss],
            [ss + 1],
            input_names=["ss"],
            output_names=["zz"],
        )
        comp2 = (
            FunctionComposer(comp)
            .feed_into(j, arg="ss")
            .compose(name="comp2")
        )

        builder = (
            CodeGenerationBuilder()
            .with_backend_config(
                RustBackendConfig()
                .with_crate_name("compozer")
                .with_enable_python_interface()
                .with_build_python_interface()
            )
            .for_function(comp2)
            .add_primal()
            .done()
        )

        with TemporaryDirectory() as tmpdir:
            project = builder.build(Path(tmpdir) / "compozer")
            self.assertIsNotNone(project.python_interface)
            lib_text = project.lib_rs.read_text(encoding="utf-8")
            self.assertEqual(
                lib_text.count("pub enum GradgenError {"),
                1,
            )
            self.assertEqual(
                lib_text.count("pub struct FunctionMetadata {"),
                1,
            )

    def test_named_gradient_builder_matches_sympy_blocks(self) -> None:
        x0, x1, p = sp.symbols("x0 x1 p", real=True)
        sympy_expr = x0 * x0 + x1 * p + sp.sin(x0 + p) + x1 * x1 * x1
        sympy_dx = sp.Matrix([sp.diff(sympy_expr, x0),
                              sp.diff(sympy_expr, x1)])
        sympy_dp = sp.Matrix([sp.diff(sympy_expr, p)])

        numeric_x = [0.4, -1.2]
        numeric_p = [0.75]
        substitutions = {
            x0: numeric_x[0],
            x1: numeric_x[1],
            p: numeric_p[0],
        }
        expected_dx = self._flatten_sympy_matrix(
            sympy_dx.subs(substitutions).evalf()
        )
        expected_dp = self._flatten_sympy_matrix(
            sympy_dp.subs(substitutions).evalf()
        )

        x = SXVector.sym("x", 2)
        p_symbol = SXVector.sym("p", 1)
        function = Function(
            "named_gradient",
            [x, p_symbol[0]],
            [
                x[0] * x[0]
                + x[1] * p_symbol[0]
                + (x[0] + p_symbol[0]).sin()
                + x[1] * x[1] * x[1]
            ],
            input_names=["x", "p"],
            output_names=["y"],
        )

        builder = (
            CodeGenerationBuilder(function)
            .with_backend_config(
                RustBackendConfig().with_crate_name("named_gradient")
            )
            .add_gradient(wrt=["x", "p"])
            .with_simplification("medium")
        )

        with TemporaryDirectory() as tmpdir:
            project = builder.build(Path(tmpdir) / "named_gradient")
            gradient_x_codegen, gradient_p_codegen = project.codegens
            gradient_x_workspace = gradient_x_codegen.workspace_size
            gradient_p_workspace = gradient_p_codegen.workspace_size

            self._append_rust_test(
                project.project_dir,
                f"""
#[cfg(test)]
mod integration_named_gradient_bundle {{
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
    fn matches_sympy_reference_blocks() {{
        let x = {self._rust_array_literal(numeric_x, "f64")};
        let p = {self._rust_array_literal(numeric_p, "f64")};

        let mut gradient_x = [0.0_f64; 2];
        let mut gradient_x_work = [0.0_f64; {gradient_x_workspace}];
        {gradient_x_codegen.function_name}(
            &x,
            &p,
            &mut gradient_x,
            &mut gradient_x_work,
        );
        assert_close_slice(
            &gradient_x,
            &{self._rust_array_literal(expected_dx, "f64")},
            1e-10_f64,
        );

        let mut gradient_p = [0.0_f64; 1];
        let mut gradient_p_work = [0.0_f64; {gradient_p_workspace}];
        {gradient_p_codegen.function_name}(
            &x,
            &p,
            &mut gradient_p,
            &mut gradient_p_work,
        );
        assert_close_slice(
            &gradient_p,
            &{self._rust_array_literal(expected_dp, "f64")},
            1e-10_f64,
        );
    }}
}}
""".lstrip(),
            )

            completed = self._run_cargo(project.project_dir, "test", "--quiet")
            self.assertEqual(completed.returncode, 0)

    def test_map_zip_pipeline_matches_sympy_primal_and_jacobian(self) -> None:
        # Build a SymPy reference for a 4-stage map->zip pipeline.
        # Each stage uses two state entries z=(z0,z1) and one exogenous scalar w.
        # The map stage emits a nonlinear dynamic term plus a constant term,
        # and the zip stage combines them into the final per-stage output.
        z_symbols = sp.symbols("z0:8", real=True)
        w_symbols = sp.symbols("w0:4", real=True)

        sympy_outputs: list[sp.Expr] = []
        for stage_index in range(4):
            z0 = z_symbols[2 * stage_index]
            z1 = z_symbols[(2 * stage_index) + 1]
            w = w_symbols[stage_index]
            mapped_dynamic = z0**2 + sp.sin(z1)
            mapped_constant = sp.Float("2.5")
            sympy_outputs.append(sp.exp(mapped_dynamic + w) + mapped_constant * w + 1)

        sympy_output_vector = sp.Matrix(sympy_outputs)
        sympy_jacobian = sympy_output_vector.jacobian(z_symbols)

        z_values = [0.2, -0.3, 0.4, 0.1, -0.5, 0.25, 0.3, -0.2]
        w_values = [0.05, -0.1, 0.2, -0.15]
        substitutions = {
            **{symbol: value for symbol, value in zip(z_symbols, z_values)},
            **{symbol: value for symbol, value in zip(w_symbols, w_values)},
        }
        expected_primal = self._flatten_sympy_matrix(
            sympy_output_vector.subs(substitutions).evalf())
        expected_jacobian = self._flatten_sympy_matrix(
            sympy_jacobian.subs(substitutions).evalf())

        # Symbolically define and expand the map stage over 4 stages.
        # The second output is effectively constant but still symbolic.
        z_stage = SXVector.sym("z", 2)
        map_kernel = Function(
            "map_kernel",
            [z_stage],
            [z_stage[0] * z_stage[0] + z_stage[1].sin(), z_stage[0] * 0 + 2.5],
            input_names=["z"],
            output_names=["dynamic", "constant"],
        )
        mapped = map_function(map_kernel, 4,
                              input_name="z_seq", name="mapped_stage") \
            .to_function()

        # Define and expand the zip stage, consuming mapped outputs and w.
        dynamic_stage = SXVector.sym("dynamic", 1)
        constant_stage = SXVector.sym("constant", 1)
        w_stage = SXVector.sym("w", 1)
        zip_kernel = Function(
            "zip_kernel",
            [dynamic_stage, constant_stage, w_stage],
            [
                (dynamic_stage[0] + w_stage[0]).exp()
                + constant_stage[0] * w_stage[0]
                + 1.0
            ],
            input_names=["dynamic", "constant", "w"],
            output_names=["y"],
        )
        batched = zip_function(
            zip_kernel,
            4,
            input_names=["dynamic_seq", "constant_seq", "w_seq"],
            name="batched_stage",
        ).to_function()

        # Compose map + zip into one function with packed sequence inputs.
        z_seq = SXVector.sym("z_seq", 8)
        w_seq = SXVector.sym("w_seq", 4)
        mapped_dynamic_seq, mapped_constant_seq = mapped(z_seq)
        pipeline_output = batched(
            mapped_dynamic_seq, mapped_constant_seq, w_seq)
        pipeline = Function(
            "map_zip_pipeline",
            [z_seq, w_seq],
            [pipeline_output],
            input_names=["z_seq", "w_seq"],
            output_names=["y_seq"],
        )

        # Generate Rust for primal and Jacobian, then verify numeric outputs
        # against the SymPy reference end-to-end through cargo test.
        builder = (
            CodeGenerationBuilder(pipeline)
            .with_backend_config(
                RustBackendConfig().with_crate_name("sympy_map_zip_pipeline")
            )
            .add_primal()
            .add_jacobian()
            .with_simplification("medium")
        )

        with TemporaryDirectory() as tmpdir:
            project = builder.build(Path(tmpdir) / "sympy_map_zip_pipeline")
            primal_codegen = next(
                codegen
                for codegen in project.codegens
                if codegen.function_name.endswith("_f")
            )
            jacobian_codegen = next(
                codegen
                for codegen in project.codegens
                if codegen.function_name.endswith("_jf_z_seq")
            )

            self._append_rust_test(
                project.project_dir,
                f"""
#[cfg(test)]
mod integration_sympy_map_zip_pipeline {{
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
        let z_seq = {self._rust_array_literal(z_values, "f64")};
        let w_seq = {self._rust_array_literal(w_values, "f64")};

        let mut primal_y_seq = [0.0_f64; 4];
        let mut primal_work = [0.0_f64; {primal_codegen.workspace_size}];
        {primal_codegen.function_name}(&z_seq, &w_seq, &mut primal_y_seq, &mut primal_work);
        assert_close_slice(&primal_y_seq, &{self._rust_array_literal(expected_primal, "f64")}, 1e-10_f64);

        let mut jacobian_y_seq = [0.0_f64; 32];
        let mut jacobian_work = [0.0_f64; {jacobian_codegen.workspace_size}];
        {jacobian_codegen.function_name}(&z_seq, &w_seq, &mut jacobian_y_seq, &mut jacobian_work);
        assert_close_slice(&jacobian_y_seq, &{self._rust_array_literal(expected_jacobian, "f64")}, 1e-10_f64);
    }}
}}
""".lstrip(),
            )

            completed = self._run_cargo(project.project_dir, "test", "--quiet")
            self.assertEqual(completed.returncode, 0)

    def test_composed_function_matches_sympy_primal_and_jacobian(self) -> None:
        x0, x1 = sp.symbols("x0 x1", real=True)
        p_symbols = sp.symbols("p0:8", real=True)

        state = sp.Matrix([x0, x1])
        for repeat_index in range(4):
            p0 = p_symbols[2 * repeat_index]
            p1 = p_symbols[2 * repeat_index + 1]
            state = sp.Matrix(
                [
                    sp.Float("0.7") * state[0] ** 2
                    + p0 * state[1]
                    + sp.sin(p1),
                    sp.Float("0.2") * state[1] ** 2
                    + p1 * state[0]
                    + p0**2,
                ]
            )
        sympy_jacobian = state.jacobian(sp.Matrix([x0, x1]))

        x_values = [0.25, -0.4]
        parameter_values = [0.6, -0.2, -0.3, 0.5, 0.9, -0.7, 0.1, 0.8]
        substitutions = {
            x0: x_values[0],
            x1: x_values[1],
            **{
                symbol: value
                for symbol, value in zip(p_symbols, parameter_values)
            },
        }
        expected_primal = self._flatten_sympy_matrix(
            state.subs(substitutions).evalf()
        )
        expected_jacobian = self._flatten_sympy_matrix(
            sympy_jacobian.subs(substitutions).evalf()
        )

        x = SXVector.sym("x", 2)
        state_vector = SXVector.sym("state", 2)
        p = SXVector.sym("p", 2)

        g = Function(
            "g",
            [state_vector, p],
                [
                    SXVector(
                        (
                            0.7 * state_vector[0] * state_vector[0]
                            + p[0] * state_vector[1]
                            + p[1].sin(),
                            0.2 * state_vector[1] * state_vector[1]
                            + p[1] * state_vector[0]
                            + p[0] * p[0],
                        )
                    )
            ],
            input_names=["state", "p"],
            output_names=["next_state"],
        )
        composed = ComposedFunction("sympy_composed", x).repeat(
            g,
            params=[p, p, p, p],
        ).finish()

        builder = (
            CodeGenerationBuilder()
            .with_backend_config(
                RustBackendConfig().with_crate_name("sympy_composed")
            )
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
        let mut primal_y = [0.0_f64; 2];
        let mut primal_work = [0.0_f64; {primal_codegen.workspace_size}];
        {primal_codegen.function_name}(
            &x,
            &parameters,
            &mut primal_y,
            &mut primal_work,
        );
        assert_close_slice(
            &primal_y,
            &{self._rust_array_literal(expected_primal, "f64")},
            1e-10_f64,
        );

        let mut gradient_y = [0.0_f64; 4];
        let mut gradient_work = [0.0_f64; {gradient_codegen.workspace_size}];
        {gradient_codegen.function_name}(
            &x,
            &parameters,
            &mut gradient_y,
            &mut gradient_work,
        );
            assert_close_slice(
                &gradient_y,
                &{self._rust_array_literal(expected_jacobian, "f64")},
                1e-10_f64,
            );
        }}
}}
""".lstrip(),
            )

            completed = self._run_cargo(project.project_dir, "test", "--quiet")
            self.assertEqual(completed.returncode, 0)

            joint_builder = (
                CodeGenerationBuilder()
                .with_backend_config(
                    RustBackendConfig().with_crate_name("sympy_composed")
                )
                .for_function(composed)
                .add_primal()
                .add_gradient()
                .add_joint(FunctionBundle().add_f().add_jf(wrt=0))
                .with_simplification("medium")
                .done()
            )
            joint_project = joint_builder.build(
                Path(tmpdir) / "sympy_composed_joint"
            )
            joint_codegen = joint_project.codegens[2]

            self._append_rust_test(
                joint_project.project_dir,
                f"""
#[cfg(test)]
mod integration_sympy_composed_joint {{
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
        let mut y = [0.0_f64; {joint_codegen.output_sizes[0]}];
        let mut jacobian_y = [0.0_f64; {joint_codegen.output_sizes[1]}];
        let mut work = [0.0_f64; {joint_codegen.workspace_size}];
        {joint_codegen.function_name}(
            &x,
            &parameters,
            &mut y,
            &mut jacobian_y,
            &mut work,
        );
        assert_close_slice(
            &y,
            &{self._rust_array_literal(expected_primal, "f64")},
            1e-10_f64,
        );
        assert_close_slice(
            &jacobian_y,
            &{self._rust_array_literal(expected_jacobian, "f64")},
            1e-10_f64,
        );
    }}
}}
""".lstrip(),
            )

            completed = self._run_cargo(joint_project.project_dir, "test", "--quiet")
            self.assertEqual(completed.returncode, 0)

    def test_single_shooting_problem_matches_sympy_cost_gradient_and_states(self) -> None:
        sx0, sx1 = sp.symbols("x0:2", real=True)
        u_symbols = sp.symbols("u0:6", real=True)
        p0, p1 = sp.symbols("p0 p1", real=True)

        state = sp.Matrix([sx0, sx1])
        packed_states = [state[0], state[1]]
        total_cost = sp.Integer(0)
        for stage_index in range(3):
            u0 = u_symbols[2 * stage_index]
            u1 = u_symbols[(2 * stage_index) + 1]
            total_cost += (
                state[0] ** 2
                + sp.Rational(3, 5) * state[1] ** 2
                + sp.Rational(1, 5) * u0**2
                + sp.Rational(2, 5) * u1**2
                + p0 * state[0] * u1
                + p1 * u0 * state[1]
            )
            state = sp.Matrix(
                [
                    state[0] + p0 * u0 + sp.sin(u1),
                    sp.Rational(3, 5) * state[1] + p1 * u1 + state[0] * u0,
                ]
            )
            packed_states.extend([state[0], state[1]])

        total_cost += (
            sp.Rational(3, 2) * state[0] ** 2
            + sp.Rational(5, 4) * state[1] ** 2
            + p0 * state[0]
            + p1 * state[1]
        )
        gradient = sp.Matrix([sp.diff(total_cost, symbol) for symbol in u_symbols])
        sympy_states = sp.Matrix(packed_states)

        x0_values = [0.25, -0.4]
        u_values = [0.2, -0.1, 0.3, 0.15, -0.25, 0.4]
        p_values = [0.7, -0.5]
        substitutions = {
            sx0: x0_values[0],
            sx1: x0_values[1],
            **{symbol: value for symbol, value in zip(u_symbols, u_values)},
            p0: p_values[0],
            p1: p_values[1],
        }
        expected_cost = float(total_cost.subs(substitutions).evalf())
        expected_gradient = self._flatten_sympy_matrix(gradient.subs(substitutions).evalf())
        expected_states = self._flatten_sympy_matrix(sympy_states.subs(substitutions).evalf())

        x = SXVector.sym("x", 2)
        u = SXVector.sym("u", 2)
        p = SXVector.sym("p", 2)

        dynamics = Function(
            "dynamics",
            [x, u, p],
            [SXVector((x[0] + p[0] * u[0] + u[1].sin(), 0.6 * x[1] + p[1] * u[1] + x[0] * u[0]))],
            input_names=["x", "u", "p"],
            output_names=["x_next"],
        )
        stage_cost = Function(
            "stage_cost",
            [x, u, p],
            [
                x[0] * x[0]
                + 0.6 * x[1] * x[1]
                + 0.2 * u[0] * u[0]
                + 0.4 * u[1] * u[1]
                + p[0] * x[0] * u[1]
                + p[1] * u[0] * x[1]
            ],
            input_names=["x", "u", "p"],
            output_names=["ell"],
        )
        terminal_cost = Function(
            "terminal_cost",
            [x, p],
            [1.5 * x[0] * x[0] + 1.25 * x[1] * x[1] + p[0] * x[0] + p[1] * x[1]],
            input_names=["x", "p"],
            output_names=["vf"],
        )
        problem = SingleShootingProblem(
            name="sympy_single_shooting",
            horizon=3,
            dynamics=dynamics,
            stage_cost=stage_cost,
            terminal_cost=terminal_cost,
            initial_state_name="x0",
            control_sequence_name="u_seq",
            parameter_name="p",
        )

        builder = (
            CodeGenerationBuilder()
            .with_backend_config(RustBackendConfig().with_crate_name("sympy_single_shooting"))
            .for_function(problem)
            .add_joint(
                SingleShootingBundle()
                .add_cost()
                .add_gradient()
                .add_rollout_states()
            )
            .with_simplification("medium")
            .done()
        )

        with TemporaryDirectory() as tmpdir:
            project = builder.build(Path(tmpdir) / "sympy_single_shooting")
            (joint_codegen,) = project.codegens

            self._append_rust_test(
                project.project_dir,
                f"""
#[cfg(test)]
mod integration_sympy_single_shooting {{
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
        let x0 = {self._rust_array_literal(x0_values, "f64")};
        let u_seq = {self._rust_array_literal(u_values, "f64")};
        let p = {self._rust_array_literal(p_values, "f64")};

        let mut cost = [0.0_f64; 1];
        let mut gradient_u_seq = [0.0_f64; 6];
        let mut x_traj = [0.0_f64; 8];
        let mut work = [0.0_f64; {joint_codegen.workspace_size}];
        {joint_codegen.function_name}(&x0, &u_seq, &p, &mut cost, &mut gradient_u_seq, &mut x_traj, &mut work);

        assert_close_slice(&cost, &{self._rust_array_literal([expected_cost], "f64")}, 1e-10_f64);
        assert_close_slice(&gradient_u_seq, &{self._rust_array_literal(expected_gradient, "f64")}, 1e-10_f64);
        assert_close_slice(&x_traj, &{self._rust_array_literal(expected_states, "f64")}, 1e-10_f64);
    }}
}}
""".lstrip(),
            )

            completed = self._run_cargo(project.project_dir, "test", "--quiet")
            self.assertEqual(completed.returncode, 0)

    def test_single_shooting_matches_sympy_cost_gradient_hvp_and_states(self) \
            -> None:
        sx0, sx1 = sp.symbols("x0:2", real=True)
        u_symbols = sp.symbols("u0:6", real=True)
        v_symbols = sp.symbols("v0:6", real=True)
        p0, p1 = sp.symbols("p0 p1", real=True)

        state = sp.Matrix([sx0, sx1])
        packed_states = [state[0], state[1]]
        total_cost = sp.Integer(0)
        for stage_index in range(3):
            u0 = u_symbols[2 * stage_index]
            u1 = u_symbols[(2 * stage_index) + 1]
            total_cost += (
                sp.Rational(4, 5) * state[0] ** 2
                + sp.Rational(7, 10) * state[1] ** 2
                + sp.Rational(3, 10) * u0**2
                + sp.Rational(2, 5) * u1**2
                + p0 * state[0] * u1
                + p1 * state[1] * u0
                + sp.sin(u0) * state[0]
                + sp.cos(u1) * state[1]
            )
            state = sp.Matrix(
                [
                    state[0] + p0 * u0 + sp.sin(u1) + sp.Rational(1, 10) * state[1] * u0,
                    sp.Rational(11, 20) * state[1] + p1 * u1 + state[0] * u0 + sp.cos(u0),
                ]
            )
            packed_states.extend([state[0], state[1]])

        total_cost += (
            sp.Rational(9, 5) * state[0] ** 2
            + sp.Rational(6, 5) * state[1] ** 2
            + p0 * state[0]
            + p1 * state[1]
            + sp.sin(state[0] - state[1])
        )

        gradient = sp.Matrix([sp.diff(total_cost, symbol) for symbol in u_symbols])
        hessian = sp.hessian(total_cost, u_symbols)
        hvp = hessian * sp.Matrix(v_symbols)
        sympy_states = sp.Matrix(packed_states)

        x0_values = [0.1, -0.35]
        u_values = [0.15, -0.2, 0.25, 0.3, -0.1, 0.4]
        v_values = [0.5, -0.25, 0.75, -0.6, 0.2, 0.45]
        p_values = [0.6, -0.4]
        substitutions = {
            sx0: x0_values[0],
            sx1: x0_values[1],
            **{symbol: value for symbol, value in zip(u_symbols, u_values)},
            **{symbol: value for symbol, value in zip(v_symbols, v_values)},
            p0: p_values[0],
            p1: p_values[1],
        }
        expected_cost = float(total_cost.subs(substitutions).evalf())
        expected_gradient = self._flatten_sympy_matrix(gradient.subs(substitutions).evalf())
        expected_hvp = self._flatten_sympy_matrix(hvp.subs(substitutions).evalf())
        expected_states = self._flatten_sympy_matrix(sympy_states.subs(substitutions).evalf())

        x = SXVector.sym("x", 2)
        u = SXVector.sym("u", 2)
        p = SXVector.sym("p", 2)

        dynamics = Function(
            "dynamics",
            [x, u, p],
            [
                SXVector(
                    (
                        x[0] + p[0] * u[0] + u[1].sin() + 0.1 * x[1] * u[0],
                        0.55 * x[1] + p[1] * u[1] + x[0] * u[0] + u[0].cos(),
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
                0.8 * x[0] * x[0]
                + 0.7 * x[1] * x[1]
                + 0.3 * u[0] * u[0]
                + 0.4 * u[1] * u[1]
                + p[0] * x[0] * u[1]
                + p[1] * x[1] * u[0]
                + u[0].sin() * x[0]
                + u[1].cos() * x[1]
            ],
            input_names=["x", "u", "p"],
            output_names=["ell"],
        )
        terminal_cost = Function(
            "terminal_cost",
            [x, p],
            [1.8 * x[0] * x[0] + 1.2 * x[1] * x[1] + p[0] * x[0] + p[1] * x[1] + (x[0] - x[1]).sin()],
            input_names=["x", "p"],
            output_names=["vf"],
        )
        problem = SingleShootingProblem(
            name="sympy_single_shooting_joint_hvp",
            horizon=3,
            dynamics=dynamics,
            stage_cost=stage_cost,
            terminal_cost=terminal_cost,
            initial_state_name="x0",
            control_sequence_name="u_seq",
            parameter_name="p",
        )

        builder = (
            CodeGenerationBuilder()
            .with_backend_config(RustBackendConfig().with_crate_name("sympy_single_shooting_joint_hvp"))
            .for_function(problem)
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
            project = builder.build(Path(tmpdir) / "sympy_single_shooting_joint_hvp")
            (joint_codegen,) = project.codegens

            self._append_rust_test(
                project.project_dir,
                f"""
#[cfg(test)]
mod integration_sympy_single_shooting_joint_hvp {{
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
        let x0 = {self._rust_array_literal(x0_values, "f64")};
        let u_seq = {self._rust_array_literal(u_values, "f64")};
        let v_u_seq = {self._rust_array_literal(v_values, "f64")};
        let p = {self._rust_array_literal(p_values, "f64")};

        let mut cost = [0.0_f64; 1];
        let mut gradient_u_seq = [0.0_f64; 6];
        let mut hvp_u_seq = [0.0_f64; 6];
        let mut x_traj = [0.0_f64; 8];
        let mut work = [0.0_f64; {joint_codegen.workspace_size}];
        {joint_codegen.function_name}(
            &x0,
            &u_seq,
            &p,
            &v_u_seq,
            &mut cost,
            &mut gradient_u_seq,
            &mut hvp_u_seq,
            &mut x_traj,
            &mut work,
        );

        assert_close_slice(&cost, &{self._rust_array_literal([expected_cost], "f64")}, 1e-10_f64);
        assert_close_slice(&gradient_u_seq, &{self._rust_array_literal(expected_gradient, "f64")}, 1e-9_f64);
        assert_close_slice(&hvp_u_seq, &{self._rust_array_literal(expected_hvp, "f64")}, 1e-8_f64);
        assert_close_slice(&x_traj, &{self._rust_array_literal(expected_states, "f64")}, 1e-10_f64);
    }}
}}
""".lstrip(),
            )

            completed = self._run_cargo(project.project_dir, "test", "--quiet")
            self.assertEqual(completed.returncode, 0)
