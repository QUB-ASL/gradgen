import unittest

from gradgen._custom_elementary import registry


class CustomRegistryTests(unittest.TestCase):
    def test_registry_module_exposes_register_and_clear(self) -> None:
        self.assertTrue(hasattr(registry, "register_elementary_function"))
        self.assertTrue(hasattr(registry, "clear_registered_elementary_functions"))
