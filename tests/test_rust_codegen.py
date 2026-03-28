import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from gradgen import Function, SX, SXVector, create_rust_project, derivative


class RustCodegenTests(unittest.TestCase):
    def test_generates_scalar_function_with_slice_abi(self) -> None:
        x = SX.sym("x")
        f = Function("square_plus_one", [x], [x * x + 1], input_names=["x"], output_names=["y"])

        result = f.generate_rust()

        self.assertEqual(result.function_name, "square_plus_one")
        self.assertEqual(result.input_sizes, (1,))
        self.assertEqual(result.output_sizes, (1,))
        self.assertIn("pub fn square_plus_one(x: &[f64], y: &mut [f64], work: &mut [f64]) {", result.source)
        self.assertIn("assert_eq!(x.len(), 1);", result.source)
        self.assertIn("assert_eq!(y.len(), 1);", result.source)
        self.assertIn("work[0] = x[0] * x[0];", result.source)
        self.assertIn("work[1] = 1.0 + work[0];", result.source)
        self.assertIn("y[0] = work[1];", result.source)

    def test_generates_vector_function_with_deterministic_workspace_layout(self) -> None:
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
        self.assertIn(
            "pub fn kernel(x: &[f64], y: &[f64], dot: &mut [f64], sum: &mut [f64], work: &mut [f64]) {",
            result.source,
        )
        self.assertIn("assert!(work.len() >= 5);", result.source)
        self.assertIn("work[0] = x[0] * y[0];", result.source)
        self.assertIn("work[1] = x[1] * y[1];", result.source)
        self.assertIn("work[2] = work[0] + work[1];", result.source)
        self.assertIn("dot[0] = work[2];", result.source)

    def test_workspace_size_matches_number_of_non_leaf_nodes(self) -> None:
        x = SX.sym("x")
        expr = (x * x + 1) + (x * x + 1) * (x * x + 1)
        f = Function("f", [x], [expr])

        result = f.generate_rust()

        self.assertEqual(result.workspace_size, len([node for node in f.nodes if node.op not in {"symbol", "const"}]))

    def test_generated_code_reuses_shared_dag_nodes(self) -> None:
        x = SX.sym("x")
        z = (x * x) + 1
        f = Function("f", [x], [z + z * z], input_names=["x"], output_names=["y"])

        result = f.generate_rust()

        self.assertEqual(result.source.count("x[0] * x[0]"), 1)
        self.assertIn("work[0] = x[0] * x[0];", result.source)
        self.assertIn("work[1] = 1.0 + work[0];", result.source)
        self.assertIn("work[2] = work[1] * work[1];", result.source)
        self.assertIn("work[3] = work[1] + work[2];", result.source)

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
        self.assertIn(".powf(2.0)", result.source)

    def test_function_level_codegen_works_for_derived_functions(self) -> None:
        x = SX.sym("x")
        df = Function("df", [x], [derivative(x * x, x)], input_names=["x"], output_names=["dx"])

        result = df.generate_rust()

        self.assertIn("pub fn df(", result.source)
        self.assertIn("dx: &mut [f64]", result.source)

    def test_create_rust_project_writes_expected_files(self) -> None:
        x = SX.sym("x")
        f = Function("square_plus_one", [x], [x * x + 1], input_names=["x"], output_names=["y"])

        with TemporaryDirectory() as tmpdir:
            project = f.create_rust_project(Path(tmpdir) / "generated_kernel")

            self.assertTrue(project.project_dir.is_dir())
            self.assertTrue(project.cargo_toml.is_file())
            self.assertTrue(project.readme.is_file())
            self.assertTrue(project.lib_rs.is_file())

            cargo_text = project.cargo_toml.read_text(encoding="utf-8")
            readme_text = project.readme.read_text(encoding="utf-8")
            lib_text = project.lib_rs.read_text(encoding="utf-8")

            self.assertIn('[package]', cargo_text)
            self.assertIn('name = "square_plus_one"', cargo_text)
            self.assertIn("# square_plus_one", readme_text)
            self.assertIn("cargo build", readme_text)
            self.assertIn("pub fn square_plus_one(", lib_text)

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


if __name__ == "__main__":
    unittest.main()
