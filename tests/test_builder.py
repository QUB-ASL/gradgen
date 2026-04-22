import unittest

from gradgen._rust_codegen.builder import CodeGenerationBuilder


class BuilderTests(unittest.TestCase):
    def test_builder_is_instantiable(self) -> None:
        builder = CodeGenerationBuilder()
        self.assertIsNotNone(builder)

    def test_builder_has_no_additional_dependencies_helper(self) -> None:
        self.assertFalse(hasattr(CodeGenerationBuilder(), "with_additional_dependencies"))
