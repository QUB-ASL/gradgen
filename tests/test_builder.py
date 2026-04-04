import unittest

from gradgen._rust_codegen.builder import CodeGenerationBuilder


class BuilderTests(unittest.TestCase):
    def test_builder_is_instantiable(self) -> None:
        builder = CodeGenerationBuilder()
        self.assertIsNotNone(builder)
