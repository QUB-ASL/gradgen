import unittest

from gradgen._rust_codegen.config import RustBackendConfig


class RustBackendConfigTests(unittest.TestCase):
    def test_defaults_match_expected_generation_behavior(self) -> None:
        config = RustBackendConfig()
        self.assertEqual(config.backend_mode, "std")
        self.assertEqual(config.scalar_type, "f64")
        self.assertIsNone(config.crate_name)
        self.assertIsNone(config.function_name)
        self.assertTrue(config.emit_metadata_helpers)
        self.assertFalse(config.enable_python_interface)
        self.assertTrue(config.build_python_interface)
        self.assertFalse(config.build_crate)

    def test_enable_python_interface_defaults_to_true(self) -> None:
        config = RustBackendConfig().with_enable_python_interface()
        self.assertTrue(config.enable_python_interface)

    def test_build_flags_can_be_toggled(self) -> None:
        config = RustBackendConfig().with_build_python_interface(False).with_build_crate(True)
        self.assertFalse(config.build_python_interface)
        self.assertTrue(config.build_crate)
