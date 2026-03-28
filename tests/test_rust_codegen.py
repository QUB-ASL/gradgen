import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from gradgen import (
    CodeGenerationBuilder,
    Function,
    RustBackendConfig,
    SX,
    SXVector,
    create_rust_derivative_bundle,
    create_rust_project,
    derivative,
)


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

    @staticmethod
    def _normalize_inputs(inputs: object) -> tuple[object, ...]:
        if isinstance(inputs, tuple):
            return inputs
        return (inputs,)

    @staticmethod
    def _flatten_runtime_output(function: Function, result: object) -> list[float]:
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
        return "[" + ", ".join(f"{repr(float(value))}_{scalar_type}" for value in values) + "]"

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
    ) -> None:
        codegen = function.generate_rust(config=config, function_name=function_name)
        numeric_inputs = cls._normalize_inputs(inputs)
        expected = cls._flatten_runtime_output(function, function(*numeric_inputs))
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
                f"        let {name} = {cls._rust_array_literal(rust_values, scalar_type)};"
            )

        output_binding_lines: list[str] = []
        output_assertion_lines: list[str] = []
        expected_offset = 0
        for index, size in enumerate(codegen.output_sizes):
            output_name = function.output_names[index]
            expected_slice = expected[expected_offset : expected_offset + size]
            expected_offset += size
            output_binding_lines.append(
                f"        let mut {output_name} = [0.0_{scalar_type}; {size}];"
            )
            output_assertion_lines.append(
                f"        assert_close_slice(&{output_name}, &{cls._rust_array_literal(expected_slice, scalar_type)}, {rust_tolerance}_{scalar_type});"
            )

        parameter_list = ", ".join(
            [
                *[f"&{name}" for name in parameter_names],
                *[f"&mut {name}" for name in function.output_names],
                "&mut work",
            ]
        )

        cls._append_rust_test(
            project_dir,
            f"""
#[cfg(test)]
mod tests {{
    use super::*;

    fn assert_close_slice(actual: &[{scalar_type}], expected: &[{scalar_type}], tolerance: {scalar_type}) {{
        assert_eq!(actual.len(), expected.len());
        for (actual_value, expected_value) in actual.iter().zip(expected.iter()) {{
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
        let mut work = [0.0_{scalar_type}; {codegen.workspace_size}];
        {function_name}({parameter_list});
{chr(10).join(output_assertion_lines)}
    }}
}}
""".lstrip(),
        )

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
        self.assertIn('pub const SQUARE_PLUS_ONE_INPUT_NAMES: &[&str] = &[', result.source)
        self.assertIn('"x",', result.source)
        self.assertIn("pub fn square_plus_one_input_names() -> &'static [&'static str] {", result.source)
        self.assertIn("pub fn square_plus_one_input_0_size() -> usize {", result.source)
        self.assertIn("pub fn square_plus_one_num_outputs() -> usize {", result.source)
        self.assertIn('pub const SQUARE_PLUS_ONE_OUTPUT_NAMES: &[&str] = &[', result.source)
        self.assertIn('"y",', result.source)
        self.assertIn("pub fn square_plus_one_output_names() -> &'static [&'static str] {", result.source)
        self.assertIn("pub fn square_plus_one_output_0_size() -> usize {", result.source)
        self.assertIn("pub fn square_plus_one(x: &[f64], y: &mut [f64], work: &mut [f64]) {", result.source)
        self.assertIn("assert_eq!(x.len(), 1);", result.source)
        self.assertIn("assert_eq!(y.len(), 1);", result.source)
        self.assertIn("work[0] = x[0] * x[0];", result.source)
        self.assertIn("work[1] = 1.0_f64 + work[0];", result.source)
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
        self.assertIn('pub const KERNEL_INPUT_NAMES: &[&str] = &[', result.source)
        self.assertIn('"x",', result.source)
        self.assertIn('"y",', result.source)
        self.assertIn("pub fn kernel_input_0_size() -> usize {", result.source)
        self.assertIn("pub fn kernel_input_1_size() -> usize {", result.source)
        self.assertIn('pub const KERNEL_OUTPUT_NAMES: &[&str] = &[', result.source)
        self.assertIn('"dot",', result.source)
        self.assertIn('"sum",', result.source)
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

    def test_backend_config_supports_chainable_updates(self) -> None:
        config = (
            RustBackendConfig()
            .with_backend_mode("no_std")
            .with_math_lib("libm")
            .with_crate_name("my_kernel")
            .with_function_name("eval_kernel")
            .with_emit_metadata_helpers(False)
        )

        self.assertEqual(config.backend_mode, "no_std")
        self.assertEqual(config.scalar_type, "f64")
        self.assertEqual(config.math_library, "libm")
        self.assertEqual(config.crate_name, "my_kernel")
        self.assertEqual(config.function_name, "eval_kernel")
        self.assertFalse(config.emit_metadata_helpers)

    def test_code_generation_builder_supports_simplification_setting(self) -> None:
        x = SX.sym("x")
        f = Function("f", [x], [x * x], input_names=["x"], output_names=["y"])

        builder = (
            CodeGenerationBuilder(f)
            .add_primal()
            .add_jacobian()
            .add_hvp()
            .add_joint(("f", "jf", "hvp"))
            .with_simplification("medium")
        )

        with TemporaryDirectory() as tmpdir:
            project = builder.build(Path(tmpdir) / "simplified_builder")
            lib_text = project.lib_rs.read_text(encoding="utf-8")

            self.assertIn("work[0] = x[0].powf(2.0_f64);", lib_text)
            self.assertIn("work[0] = 2.0_f64 * x[0];", lib_text)
            self.assertIn("work[0] = 2.0_f64 * v_x[0];", lib_text)
            self.assertIn("work[1] = 2.0_f64 * x[0];", lib_text)
            self.assertIn("work[2] = 2.0_f64 * v_x[0];", lib_text)

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

    def test_generate_rust_accepts_backend_config(self) -> None:
        x = SX.sym("x")
        f = Function("f", [x], [x.sin()], input_names=["x"], output_names=["y"])
        config = (
            RustBackendConfig()
            .with_backend_mode("no_std")
            .with_math_lib("libm")
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

    def test_create_rust_project_accepts_backend_config(self) -> None:
        x = SX.sym("x")
        f = Function("f", [x], [x * x], input_names=["x"], output_names=["y"])
        config = (
            RustBackendConfig()
            .with_crate_name("my_kernel")
            .with_function_name("eval_kernel")
        )

        with TemporaryDirectory() as tmpdir:
            project = f.create_rust_project(Path(tmpdir) / "custom_project", config=config)

            cargo_text = project.cargo_toml.read_text(encoding="utf-8")
            lib_text = project.lib_rs.read_text(encoding="utf-8")

            self.assertIn('name = "my_kernel"', cargo_text)
            self.assertIn("pub fn eval_kernel(", lib_text)
            self.assertEqual(project.codegen.function_name, "eval_kernel")

    def test_workspace_size_matches_number_of_non_leaf_nodes(self) -> None:
        x = SX.sym("x")
        expr = (x * x + 1) + (x * x + 1) * (x * x + 1)
        f = Function("f", [x], [expr])

        result = f.generate_rust()

        self.assertEqual(result.workspace_size, len([node for node in f.nodes if node.op not in {"symbol", "const"}]))

    def test_f32_codegen_uses_f32_slice_abi_and_literals(self) -> None:
        x = SX.sym("x")
        f = Function("square_plus_one", [x], [x * x + 1], input_names=["x"], output_names=["y"])

        result = f.generate_rust(scalar_type="f32")

        self.assertEqual(result.scalar_type, "f32")
        self.assertIn("pub fn square_plus_one(x: &[f32], y: &mut [f32], work: &mut [f32]) {", result.source)
        self.assertIn("work[1] = 1.0_f32 + work[0];", result.source)

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
        self.assertIn("libm::powf(", result.source)

    def test_generated_code_reuses_shared_dag_nodes(self) -> None:
        x = SX.sym("x")
        z = (x * x) + 1
        f = Function("f", [x], [z + z * z], input_names=["x"], output_names=["y"])

        result = f.generate_rust()

        self.assertEqual(result.source.count("x[0] * x[0]"), 1)
        self.assertIn("work[0] = x[0] * x[0];", result.source)
        self.assertIn("work[1] = 1.0_f64 + work[0];", result.source)
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
        self.assertIn(".powf(2.0_f64)", result.source)

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

    def test_generated_rust_project_runs_vector_numeric_smoke_test(self) -> None:
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

    def test_generated_rust_project_runs_f32_reference_test(self) -> None:
        x = SX.sym("x")
        f = Function("square_plus_one", [x], [x.sin() + x * x + 1], input_names=["x"], output_names=["y"])

        with TemporaryDirectory() as tmpdir:
            project = f.create_rust_project(Path(tmpdir) / "f32_kernel", scalar_type="f32")
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

    def test_generated_rust_project_runs_jacobian_reference_test(self) -> None:
        x = SXVector.sym("x", 2)
        jac = Function("f", [x], [x.dot(x)], input_names=["x"], output_names=["y"]).jacobian(0)

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

    def test_no_std_project_runs_reference_test(self) -> None:
        x = SX.sym("x")
        f = Function("trig_kernel", [x], [x.sin() + x.cos()], input_names=["x"], output_names=["y"])

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
                completed = self._run_cargo(project.project_dir, "test", "--quiet")
            except subprocess.CalledProcessError as exc:
                if "Could not resolve host: index.crates.io" in exc.stderr:
                    self.skipTest("cargo could not fetch libm in the offline test environment")
                raise
            self.assertEqual(completed.returncode, 0)

    def test_no_std_f32_project_builds(self) -> None:
        x = SX.sym("x")
        f = Function("trig_kernel", [x], [x.sin() + x.cos()], input_names=["x"], output_names=["y"])

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
            self.assertIn("pub fn trig_kernel(x: &[f32], y: &mut [f32], work: &mut [f32]) {", lib_text)

            try:
                completed = self._run_cargo(project.project_dir, "build", "--quiet")
            except subprocess.CalledProcessError as exc:
                if "Could not resolve host: index.crates.io" in exc.stderr:
                    self.skipTest("cargo could not fetch libm in the offline test environment")
                raise
            self.assertEqual(completed.returncode, 0)

    def test_no_std_f32_project_runs_reference_test(self) -> None:
        x = SX.sym("x")
        f = Function("trig_kernel", [x], [x.sin() + x.cos()], input_names=["x"], output_names=["y"])

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
                test_name="evaluates_no_std_f32_kernel_against_python_reference",
                config=RustBackendConfig().with_backend_mode("no_std").with_scalar_type("f32"),
                tolerance=1e-5,
            )
            try:
                completed = self._run_cargo(project.project_dir, "test", "--quiet")
            except subprocess.CalledProcessError as exc:
                if "Could not resolve host: index.crates.io" in exc.stderr:
                    self.skipTest("cargo could not fetch libm in the offline test environment")
                raise
            self.assertEqual(completed.returncode, 0)

    def test_generated_rust_project_builds_without_metadata_helpers(self) -> None:
        x = SX.sym("x")
        f = Function("square_plus_one", [x], [x * x + 1], input_names=["x"], output_names=["y"])
        config = RustBackendConfig().with_emit_metadata_helpers(False)

        with TemporaryDirectory() as tmpdir:
            project = f.create_rust_project(
                Path(tmpdir) / "helperless_kernel",
                config=config,
            )
            self.assertNotIn("WORK_SIZE", project.lib_rs.read_text(encoding="utf-8"))
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

    def test_generated_rust_project_exposes_input_and_output_name_helpers(self) -> None:
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
        assert_eq!(named_kernel_input_names(), ["state vector", "gain"]);
        assert_eq!(named_kernel_output_names(), ["energy", "gain out"]);
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

        result = f.generate_rust(function_name="123 kernel")

        self.assertIn("pub fn _123_kernel(", result.source)
        self.assertIn("_1_input: &[f64]", result.source)
        self.assertIn("out_value: &mut [f64]", result.source)

    def test_create_rust_derivative_bundle_writes_expected_projects(self) -> None:
        x = SXVector.sym("x", 2)
        f = Function("f", [x], [x.dot(x)], input_names=["x"], output_names=["y"])

        with TemporaryDirectory() as tmpdir:
            bundle = f.create_rust_derivative_bundle(
                Path(tmpdir) / "bundle",
                simplify_derivatives="high",
            )

            self.assertTrue(bundle.bundle_dir.is_dir())
            self.assertIsNotNone(bundle.primal)
            self.assertEqual(len(bundle.jacobians), 1)
            self.assertEqual(len(bundle.hessians), 1)
            self.assertTrue((bundle.bundle_dir / "primal" / "Cargo.toml").is_file())
            self.assertTrue((bundle.bundle_dir / "f_jacobian_x" / "Cargo.toml").is_file())
            self.assertTrue((bundle.bundle_dir / "f_hessian_x" / "Cargo.toml").is_file())

            self.assertEqual(self._run_cargo(bundle.primal.project_dir, "build", "--quiet").returncode, 0)
            self.assertEqual(self._run_cargo(bundle.jacobians[0].project_dir, "build", "--quiet").returncode, 0)
            self.assertEqual(self._run_cargo(bundle.hessians[0].project_dir, "build", "--quiet").returncode, 0)

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

    def test_code_generation_builder_creates_single_multi_function_crate(self) -> None:
        x = SXVector.sym("x", 2)
        f = Function(
            "f",
            [x],
            [x[0] * x[0] + x[0] * x[1] + x[1] * x[1]],
            input_names=["x"],
            output_names=["y"],
        )
        builder = (
            CodeGenerationBuilder(f)
            .add_primal()
            .add_gradient()
            .add_hvp()
        )

        with TemporaryDirectory() as tmpdir:
            project = builder.build(Path(tmpdir) / "single_crate")

            self.assertTrue(project.project_dir.is_dir())
            self.assertEqual(len(project.codegens), 3)
            lib_text = project.lib_rs.read_text(encoding="utf-8")
            self.assertIn("pub fn f(", lib_text)
            self.assertIn("pub fn f_gradient_x(", lib_text)
            self.assertIn("pub fn f_hvp_x(", lib_text)

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

        let mut work_f = [0.0_f64; F_WORK_SIZE];
        let mut work_grad = [0.0_f64; F_GRADIENT_X_WORK_SIZE];
        let mut work_hvp = [0.0_f64; F_HVP_X_WORK_SIZE];

        f(&x, &mut y, &mut work_f);
        f_gradient_x(&x, &mut y_grad, &mut work_grad);
        f_hvp_x(&x, &v_x, &mut y_hvp, &mut work_hvp);

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
        builder = CodeGenerationBuilder(f).add_joint(("f", "jf"))

        with TemporaryDirectory() as tmpdir:
            project = builder.build(Path(tmpdir) / "joint")

            self.assertTrue(project.project_dir.is_dir())
            self.assertEqual(len(project.codegens), 1)
            lib_text = project.lib_rs.read_text(encoding="utf-8")
            self.assertIn("pub fn f_joint_f_jf_x(", lib_text)

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
        let mut work = [0.0_f64; F_JOINT_F_JF_X_WORK_SIZE];

        f_joint_f_jf_x(&x, &mut y, &mut jacobian_y, &mut work);

        assert_eq!(y[0], 37.0_f64);
        assert_eq!(jacobian_y, [10.0_f64, 11.0_f64]);
    }
}
""".lstrip(),
            )

            completed = self._run_cargo(project.project_dir, "test", "--quiet")
            self.assertEqual(completed.returncode, 0)

    def test_code_generation_builder_supports_joint_primal_jacobian_and_hvp(self) -> None:
        x = SXVector.sym("x", 2)
        f = Function(
            "f",
            [x],
            [x[0] * x[0] + x[0] * x[1] + x[1] * x[1]],
            input_names=["x"],
            output_names=["y"],
        )
        builder = CodeGenerationBuilder(f).add_joint(("f", "jf", "hvp"))

        with TemporaryDirectory() as tmpdir:
            project = builder.build(Path(tmpdir) / "joint_f_jf_hvp")

            lib_text = project.lib_rs.read_text(encoding="utf-8")
            self.assertIn("pub fn f_joint_f_jf_hvp_x(", lib_text)

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
        let mut work = [0.0_f64; F_JOINT_F_JF_HVP_X_WORK_SIZE];

        f_joint_f_jf_hvp_x(&x, &v_x, &mut y, &mut jacobian_y, &mut hvp_y, &mut work);

        assert_eq!(y[0], 37.0_f64);
        assert_eq!(jacobian_y, [10.0_f64, 11.0_f64]);
        assert_eq!(hvp_y, [4.0_f64, 5.0_f64]);
    }
}
""".lstrip(),
            )

            completed = self._run_cargo(project.project_dir, "test", "--quiet")
            self.assertEqual(completed.returncode, 0)

    def test_code_generation_builder_rejects_invalid_joint_requests(self) -> None:
        x = SX.sym("x")
        f = Function("f", [x], [x * x], input_names=["x"], output_names=["y"])

        with self.assertRaises(ValueError):
            CodeGenerationBuilder(f).add_joint(("f",)).build("/tmp/unused")

        with self.assertRaises(ValueError):
            CodeGenerationBuilder(f).add_joint(("f", "f")).build("/tmp/unused")

    def test_code_generation_builder_supports_no_std_f32_backend_config(self) -> None:
        x = SX.sym("x")
        f = Function("f", [x], [x * x + 1], input_names=["x"], output_names=["y"])
        builder = (
            CodeGenerationBuilder(f)
            .with_backend_config(
                RustBackendConfig()
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
            self.assertIn("pub fn f(x: &[f32], y: &mut [f32], work: &mut [f32]) {", lib_text)
            self.assertIn("pub fn f_gradient_x(x: &[f32], y: &mut [f32], work: &mut [f32]) {", lib_text)
            self.assertIn("pub fn f_hvp_x(x: &[f32], v_x: &[f32], y: &mut [f32], work: &mut [f32]) {", lib_text)
            self.assertIn('libm = "0.2"', cargo_text)

            try:
                completed = self._run_cargo(project.project_dir, "build", "--quiet")
            except subprocess.CalledProcessError as exc:
                if "Could not resolve host: index.crates.io" in exc.stderr:
                    self.skipTest("cargo could not fetch libm in the offline test environment")
                raise
            self.assertEqual(completed.returncode, 0)


if __name__ == "__main__":
    unittest.main()
