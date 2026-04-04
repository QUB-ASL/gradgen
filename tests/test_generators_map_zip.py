import unittest

from gradgen._rust_codegen.generators import map_zip as map_zip_generators


class GeneratorMapZipTests(unittest.TestCase):
    def test_module_exports_primary_entrypoints(self) -> None:
        self.assertTrue(hasattr(map_zip_generators, "_generate_zipped_primal_rust"))
        self.assertTrue(hasattr(map_zip_generators, "_generate_zipped_jacobian_rust"))
        self.assertTrue(hasattr(map_zip_generators, "_generate_reduced_primal_rust"))
