import unittest

from gradgen._rust_codegen.builder import CodeGenerationBuilder
from gradgen._rust_codegen.config import RustBackendConfig


class BuilderTests(unittest.TestCase):
    def test_builder_is_instantiable(self) -> None:
        builder = CodeGenerationBuilder()
        self.assertIsNotNone(builder)

    def test_builder_can_store_additional_dependencies(self) -> None:
        builder = (
            CodeGenerationBuilder()
            .with_backend_config(RustBackendConfig().with_crate_name("demo"))
            .with_additional_dependencies(
                ["serde", ("smallvec", "1.13")]
            )
        )
        assert builder.config is not None
        self.assertEqual(
            builder.config.additional_dependencies,
            (("serde", None), ("smallvec", "1.13")),
        )
