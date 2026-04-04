import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from gradgen._rust_codegen.project_support import (
    _derive_python_function_name,
    _gradgen_version,
    _metadata_created_at,
    _next_python_interface_version,
    _private_helper_section_key,
)


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
