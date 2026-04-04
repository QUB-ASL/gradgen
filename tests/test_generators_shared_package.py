import unittest

from gradgen._rust_codegen.generators import shared as shared_generators


class GeneratorsSharedPackageTests(unittest.TestCase):
    def test_package_reexports_shared_helper_groups(self) -> None:
        self.assertTrue(hasattr(shared_generators, "_build_directional_derivative_function"))
        self.assertTrue(hasattr(shared_generators, "_build_single_shooting_helpers"))
