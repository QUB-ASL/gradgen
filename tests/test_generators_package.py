import unittest

from gradgen._rust_codegen import generators


class GeneratorsPackageTests(unittest.TestCase):
    def test_package_reexports_family_generators(self) -> None:
        self.assertTrue(hasattr(generators, "_generate_composed_primal_rust"))
        self.assertTrue(hasattr(generators, "_generate_zipped_primal_rust"))
        self.assertTrue(hasattr(generators, "_generate_single_shooting_primal_rust"))
