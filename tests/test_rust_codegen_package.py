import unittest

import gradgen._rust_codegen as rust_codegen_package


class RustCodegenPackageTests(unittest.TestCase):
    def test_package_reexports_core_builder_types(self) -> None:
        self.assertTrue(hasattr(rust_codegen_package, "CodeGenerationBuilder"))
        self.assertTrue(hasattr(rust_codegen_package, "FunctionBundle"))
