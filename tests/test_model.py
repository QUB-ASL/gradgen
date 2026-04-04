import unittest

from gradgen._custom_elementary import model


class CustomModelTests(unittest.TestCase):
    def test_model_module_exports_registered_function_dataclass(self) -> None:
        self.assertTrue(hasattr(model, "RegisteredElementaryFunction"))
