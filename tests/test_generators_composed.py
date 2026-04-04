import unittest

from gradgen._rust_codegen.generators import composed as composed_generators


class GeneratorComposedTests(unittest.TestCase):
    def test_module_exports_primary_entrypoints(self) -> None:
        self.assertTrue(hasattr(composed_generators, "_generate_composed_primal_rust"))
        self.assertTrue(hasattr(composed_generators, "_generate_composed_gradient_rust"))
