import unittest

from gradgen._custom_elementary import callbacks


class CustomCallbacksTests(unittest.TestCase):
    def test_callbacks_module_exposes_eval_helpers(self) -> None:
        self.assertTrue(hasattr(callbacks, "invoke_custom_callback"))
        self.assertTrue(hasattr(callbacks, "validate_registered_function"))
