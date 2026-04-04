import unittest

from gradgen._rust_codegen.generators import single_shooting as single_shooting_generators


class GeneratorSingleShootingTests(unittest.TestCase):
    def test_module_exports_primary_entrypoints(self) -> None:
        self.assertTrue(hasattr(single_shooting_generators, "_generate_single_shooting_primal_rust"))
        self.assertTrue(hasattr(single_shooting_generators, "_generate_single_shooting_gradient_rust"))
        self.assertTrue(hasattr(single_shooting_generators, "_generate_single_shooting_hvp_rust"))
