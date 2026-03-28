import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from gradgen import Function, SX, SXVector, create_rust_project, derivative


class RustCodegenTests(unittest.TestCase):
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
    def _append_rust_test(project_dir: Path, test_source: str) -> None:
        lib_rs = project_dir / "src" / "lib.rs"
        with lib_rs.open("a", encoding="utf-8") as handle:
            handle.write("\n")
            handle.write(test_source)

    def test_generates_scalar_function_with_slice_abi(self) -> None:
        x = SX.sym("x")
        f = Function("square_plus_one", [x], [x * x + 1], input_names=["x"], output_names=["y"])

        result = f.generate_rust()

        self.assertEqual(result.function_name, "square_plus_one")
        self.assertEqual(result.input_sizes, (1,))
        self.assertEqual(result.output_sizes, (1,))
        self.assertIn("/// Workspace length required by [`square_plus_one`].", result.source)
        self.assertIn("pub const SQUARE_PLUS_ONE_WORK_SIZE: usize = 2;", result.source)
        self.assertIn("pub fn square_plus_one_work_size() -> usize {", result.source)
        self.assertIn("pub fn square_plus_one_num_inputs() -> usize {", result.source)
        self.assertIn("pub fn square_plus_one_input_0_size() -> usize {", result.source)
        self.assertIn("pub fn square_plus_one_num_outputs() -> usize {", result.source)
        self.assertIn("pub fn square_plus_one_output_0_size() -> usize {", result.source)
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
        self.assertIn("pub const KERNEL_WORK_SIZE: usize = 5;", result.source)
        self.assertIn("pub fn kernel_input_0_size() -> usize {", result.source)
        self.assertIn("pub fn kernel_input_1_size() -> usize {", result.source)
        self.assertIn("pub fn kernel_output_0_size() -> usize {", result.source)
        self.assertIn("pub fn kernel_output_1_size() -> usize {", result.source)
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
        self.assertIn("libm::pow(", result.source)

    def test_no_std_codegen_supports_sine_explicitly(self) -> None:
        x = SX.sym("x")
        f = Function("sine_only", [x], [x.sin()], input_names=["x"], output_names=["y"])

        result = f.generate_rust(backend_mode="no_std")

        self.assertIn("#![no_std]", result.source)
        self.assertIn("libm::sin(x[0])", result.source)

    def test_no_std_codegen_supports_custom_math_library_namespace(self) -> None:
        x = SX.sym("x")
        f = Function("custom_math", [x], [x.sin() + x.cos()], input_names=["x"], output_names=["y"])

        result = f.generate_rust(backend_mode="no_std", math_library="xyz")

        self.assertEqual(result.backend_mode, "no_std")
        self.assertEqual(result.math_library, "xyz")
        self.assertIn("#![no_std]", result.source)
        self.assertIn("xyz::sin(x[0])", result.source)
        self.assertIn("xyz::cos(x[0])", result.source)
        self.assertNotIn('libm = "0.2"', result.source)

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
            self.assertIn("workspace, input, and output dimensions", readme_text)
            self.assertIn("pub const SQUARE_PLUS_ONE_WORK_SIZE: usize = 2;", lib_text)
            self.assertIn("pub fn square_plus_one(", lib_text)

            completed = self._run_cargo(project.project_dir, "build", "--quiet")
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

            completed = self._run_cargo(project.project_dir, "build", "--quiet")
            self.assertEqual(completed.returncode, 0)

    def test_no_std_project_builds(self) -> None:
        x = SX.sym("x")
        f = Function("trig_kernel", [x], [x.sin() + x.cos()], input_names=["x"], output_names=["y"])

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
            self.assertIn("#![no_std]", lib_text)

            try:
                completed = self._run_cargo(project.project_dir, "build", "--quiet")
            except subprocess.CalledProcessError as exc:
                if "Could not resolve host: index.crates.io" in exc.stderr:
                    self.skipTest("cargo could not fetch libm in the offline test environment")
                raise
            self.assertEqual(completed.returncode, 0)

    def test_no_std_project_uses_custom_math_library_metadata(self) -> None:
        x = SX.sym("x")
        f = Function("custom_math", [x], [x.sin()], input_names=["x"], output_names=["y"])

        with TemporaryDirectory() as tmpdir:
            project = f.create_rust_project(
                Path(tmpdir) / "custom_math",
                backend_mode="no_std",
                math_library="xyz",
            )

            cargo_text = project.cargo_toml.read_text(encoding="utf-8")
            readme_text = project.readme.read_text(encoding="utf-8")
            lib_text = project.lib_rs.read_text(encoding="utf-8")

            self.assertNotIn('libm = "0.2"', cargo_text)
            self.assertIn("Math library namespace: `xyz`", readme_text)
            self.assertIn("uses `xyz`", readme_text)
            self.assertIn("xyz::sin(x[0])", lib_text)

    def test_no_std_project_builds_with_micromath_shim(self) -> None:
        x = SX.sym("x")
        f = Function("micro_kernel", [x], [x.sin() + x.cos()], input_names=["x"], output_names=["y"])

        with TemporaryDirectory() as tmpdir:
            project = f.create_rust_project(
                Path(tmpdir) / "micro_kernel",
                backend_mode="no_std",
                math_library="crate::math",
            )

            cargo_toml = project.cargo_toml.read_text(encoding="utf-8")
            cargo_toml += 'micromath = "2"\n'
            project.cargo_toml.write_text(cargo_toml, encoding="utf-8")

            shim = """
mod math {
    use micromath::F32Ext;

    pub fn sin(x: f64) -> f64 {
        (x as f32).sin() as f64
    }

    pub fn cos(x: f64) -> f64 {
        (x as f32).cos() as f64
    }
}

"""
            lib_text = project.lib_rs.read_text(encoding="utf-8")
            if lib_text.startswith("#![no_std]\n"):
                lib_text = lib_text.replace("#![no_std]\n", f"#![no_std]\n\n{shim}", 1)
            else:
                lib_text = shim + lib_text
            project.lib_rs.write_text(lib_text, encoding="utf-8")

            try:
                completed = self._run_cargo(project.project_dir, "build", "--quiet")
            except subprocess.CalledProcessError as exc:
                if "Could not resolve host: index.crates.io" in exc.stderr:
                    self.skipTest("cargo could not fetch micromath in the offline test environment")
                raise
            self.assertEqual(completed.returncode, 0)

    def test_invalid_backend_mode_is_rejected(self) -> None:
        x = SX.sym("x")
        f = Function("f", [x], [x], input_names=["x"], output_names=["y"])

        with self.assertRaises(ValueError):
            f.generate_rust(backend_mode="gpu")

    def test_generated_rust_project_runs_numeric_smoke_test(self) -> None:
        x = SX.sym("x")
        f = Function("square_plus_one", [x], [x * x + 1], input_names=["x"], output_names=["y"])

        with TemporaryDirectory() as tmpdir:
            project = f.create_rust_project(Path(tmpdir) / "runtime_kernel")
            self._append_rust_test(
                project.project_dir,
                """
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn evaluates_square_plus_one() {
        let x = [3.0_f64];
        let mut y = [0.0_f64];
        let mut work = [0.0_f64; SQUARE_PLUS_ONE_WORK_SIZE];
        square_plus_one(&x, &mut y, &mut work);
        assert_eq!(y[0], 10.0);
    }
}
""".lstrip(),
            )

            completed = self._run_cargo(project.project_dir, "test", "--quiet")
            self.assertEqual(completed.returncode, 0)

    def test_generated_rust_project_runs_vector_numeric_smoke_test(self) -> None:
        x = SXVector.sym("x", 2)
        f = Function("dot_and_shift", [x], [x.dot(x), x + SXVector((1 * x[0] * 0 + 1, 1 * x[1] * 0 + 1))], input_names=["x"], output_names=["dot", "shift"])

        with TemporaryDirectory() as tmpdir:
            project = f.create_rust_project(Path(tmpdir) / "vector_kernel")
            self._append_rust_test(
                project.project_dir,
                """
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn evaluates_vector_outputs() {
        let x = [2.0_f64, 3.0_f64];
        let mut dot = [0.0_f64];
        let mut shift = [0.0_f64, 0.0_f64];
        let mut work = [0.0_f64; DOT_AND_SHIFT_WORK_SIZE];
        dot_and_shift(&x, &mut dot, &mut shift, &mut work);
        assert_eq!(dot[0], 13.0);
        assert_eq!(shift, [3.0, 4.0]);
    }
}
""".lstrip(),
            )

            completed = self._run_cargo(project.project_dir, "test", "--quiet")
            self.assertEqual(completed.returncode, 0)

    def test_generated_rust_project_builds_for_jacobian_function(self) -> None:
        x = SXVector.sym("x", 2)
        jac = Function("f", [x], [x.dot(x)], input_names=["x"], output_names=["y"]).jacobian(0)

        with TemporaryDirectory() as tmpdir:
            project = jac.create_rust_project(Path(tmpdir) / "jacobian_kernel")
            completed = self._run_cargo(project.project_dir, "build", "--quiet")
            self.assertEqual(completed.returncode, 0)

    def test_generated_rust_project_builds_for_hessian_function(self) -> None:
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
            completed = self._run_cargo(project.project_dir, "build", "--quiet")
            self.assertEqual(completed.returncode, 0)

    def test_generated_rust_project_builds_for_simplified_function(self) -> None:
        x = SX.sym("x")
        f = Function("f", [x], [derivative(x * x, x)], input_names=["x"], output_names=["dx"]).simplify(max_effort="medium")

        with TemporaryDirectory() as tmpdir:
            project = f.create_rust_project(Path(tmpdir) / "simplified_kernel")
            completed = self._run_cargo(project.project_dir, "build", "--quiet")
            self.assertEqual(completed.returncode, 0)

    def test_zero_workspace_function_codegen_builds(self) -> None:
        x = SX.sym("x")
        f = Function("identity", [x], [x], input_names=["x"], output_names=["y"])

        result = f.generate_rust()

        self.assertEqual(result.workspace_size, 0)
        self.assertIn("pub const IDENTITY_WORK_SIZE: usize = 0;", result.source)

        with TemporaryDirectory() as tmpdir:
            project = f.create_rust_project(Path(tmpdir) / "identity_kernel")
            completed = self._run_cargo(project.project_dir, "build", "--quiet")
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

        result = f.generate_rust(function_name="123 kernel")

        self.assertIn("pub fn _123_kernel(", result.source)
        self.assertIn("_1_input: &[f64]", result.source)
        self.assertIn("out_value: &mut [f64]", result.source)


if __name__ == "__main__":
    unittest.main()
