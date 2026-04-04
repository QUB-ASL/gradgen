import unittest
from pathlib import Path

from gradgen._rust_codegen.config import RustBackendMode, RustScalarType
from gradgen._rust_codegen.models import RustCodegenResult, RustProjectResult, _ArgSpec


class RustModelTests(unittest.TestCase):
    def test_codegen_result_dataclass(self) -> None:
        result = RustCodegenResult(
            source="pub fn demo() {}\n",
            python_name="demo",
            function_name="demo",
            workspace_size=3,
            input_names=("x",),
            input_sizes=(1,),
            output_names=("y",),
            output_sizes=(1,),
            backend_mode="std",
            scalar_type="f64",
            math_library=None,
        )
        self.assertEqual(result.python_name, "demo")
        self.assertEqual(result.backend_mode, "std")

    def test_arg_spec_records_rendered_metadata(self) -> None:
        spec = _ArgSpec("x", "x", '"x"', "input slice", 2)
        self.assertEqual(spec.raw_name, "x")
        self.assertEqual(spec.size, 2)

    def test_project_result_dataclass(self) -> None:
        codegen = RustCodegenResult(
            source="pub fn demo() {}\n",
            python_name="demo",
            function_name="demo",
            workspace_size=0,
            input_names=(),
            input_sizes=(),
            output_names=(),
            output_sizes=(),
            backend_mode="std",
            scalar_type="f64",
            math_library=None,
        )
        project = RustProjectResult(
            project_dir=Path("."),
            cargo_toml=Path("Cargo.toml"),
            readme=Path("README.md"),
            metadata_json=Path("metadata.json"),
            lib_rs=Path("src/lib.rs"),
            codegen=codegen,
        )
        self.assertEqual(project.codegen.function_name, "demo")
