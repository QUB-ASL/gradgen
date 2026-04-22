import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from gradgen._rust_codegen.project_support import (
    _derive_python_function_name,
    _gradgen_version,
    _metadata_created_at,
    _next_python_interface_version,
    _maybe_simplify_derivative_function,
    _private_helper_section_key,
    _render_multi_function_lib,
)
from gradgen._rust_codegen.config import RustBackendConfig
from gradgen._rust_codegen.models import RustCodegenResult
from gradgen.function import Function
from gradgen.sx import SXVector


class ProjectSupportTests(unittest.TestCase):
    def test_python_function_name_derivation_strips_crate_prefix(self) -> None:
        self.assertEqual(_derive_python_function_name("abc_demo_f", "abc"), "demo")

    def test_metadata_created_at_looks_like_iso_timestamp(self) -> None:
        self.assertIn("T", _metadata_created_at())

    def test_private_helper_section_key_skips_public_functions(self) -> None:
        self.assertIsNone(_private_helper_section_key("pub fn demo() {}"))

    def test_next_python_interface_version_starts_at_point_one(self) -> None:
        with TemporaryDirectory() as tmpdir:
            self.assertEqual(_next_python_interface_version(Path(tmpdir)), "0.1.0")

    def test_gradgen_version_is_available(self) -> None:
        self.assertIsInstance(_gradgen_version(), str)

    def test_maybe_simplify_derivative_function_returns_original_when_disabled(self) -> None:
        x = SXVector.sym("x", 1)
        function = Function(
            "demo",
            [x],
            [x[0] + 1.0],
            input_names=["x"],
            output_names=["y"],
        )
        self.assertIs(_maybe_simplify_derivative_function(function, None), function)

    def test_render_multi_function_lib_deduplicates_private_helpers(self) -> None:
        codegen_a = RustCodegenResult(
            source="#![no_std]\n\nfn helper() {}\n\npub fn a() {}\n",
            python_name="a",
            function_name="a",
            workspace_size=0,
            input_names=(),
            input_sizes=(),
            output_names=(),
            output_sizes=(),
            backend_mode="no_std",
            scalar_type="f64",
            math_library="libm",
        )
        codegen_b = RustCodegenResult(
            source="#![no_std]\n\nfn helper() {}\n\npub fn b() {}\n",
            python_name="b",
            function_name="b",
            workspace_size=0,
            input_names=(),
            input_sizes=(),
            output_names=(),
            output_sizes=(),
            backend_mode="no_std",
            scalar_type="f64",
            math_library="libm",
        )

        rendered = _render_multi_function_lib(
            (codegen_a, codegen_b),
            RustBackendConfig(backend_mode="no_std"),
        )
        self.assertTrue(rendered.startswith("#![no_std]"))
        self.assertEqual(rendered.count("fn helper() {}"), 1)

    def test_render_multi_function_lib_deduplicates_custom_header(self) -> None:
        codegen_a = RustCodegenResult(
            source="use smallvec::{smallvec, SmallVec};\n\npub fn a() {}\n",
            python_name="a",
            function_name="a",
            workspace_size=0,
            input_names=(),
            input_sizes=(),
            output_names=(),
            output_sizes=(),
            backend_mode="std",
            scalar_type="f64",
            math_library=None,
        )
        codegen_b = RustCodegenResult(
            source="use smallvec::{smallvec, SmallVec};\n\npub fn b() {}\n",
            python_name="b",
            function_name="b",
            workspace_size=0,
            input_names=(),
            input_sizes=(),
            output_names=(),
            output_sizes=(),
            backend_mode="std",
            scalar_type="f64",
            math_library=None,
        )

        rendered = _render_multi_function_lib(
            (codegen_a, codegen_b),
            RustBackendConfig(header="use smallvec::{smallvec, SmallVec};"),
        )
        self.assertEqual(rendered.count("use smallvec::{smallvec, SmallVec};"), 1)

    def test_render_multi_function_lib_deduplicates_multisection_header(
        self,
    ) -> None:
        header = "use smallvec::{smallvec, SmallVec};\n\nfn helper() {}"
        codegen_a = RustCodegenResult(
            source=(
                "#![forbid(unsafe_code)]\n\n"
                "use smallvec::{smallvec, SmallVec};\n\n"
                "fn helper() {}\n\n"
                "pub fn a() {}\n"
            ),
            python_name="a",
            function_name="a",
            workspace_size=0,
            input_names=(),
            input_sizes=(),
            output_names=(),
            output_sizes=(),
            backend_mode="std",
            scalar_type="f64",
            math_library=None,
        )
        codegen_b = RustCodegenResult(
            source=(
                "#![forbid(unsafe_code)]\n\n"
                "use smallvec::{smallvec, SmallVec};\n\n"
                "fn helper() {}\n\n"
                "pub fn b() {}\n"
            ),
            python_name="b",
            function_name="b",
            workspace_size=0,
            input_names=(),
            input_sizes=(),
            output_names=(),
            output_sizes=(),
            backend_mode="std",
            scalar_type="f64",
            math_library=None,
        )

        rendered = _render_multi_function_lib(
            (codegen_a, codegen_b),
            RustBackendConfig(header=header),
        )
        self.assertEqual(rendered.count("use smallvec::{smallvec, SmallVec};"), 1)
        self.assertEqual(rendered.count("fn helper() {}"), 1)
