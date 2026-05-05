import unittest

from gradgen._rust_codegen.config import RustBackendConfig


class RustBackendConfigTests(unittest.TestCase):
    def test_defaults_match_expected_generation_behavior(self) -> None:
        config = RustBackendConfig()
        self.assertEqual(config.backend_mode, "std")
        self.assertEqual(config.scalar_type, "f64")
        self.assertIsNone(config.crate_name)
        self.assertIsNone(config.function_name)
        self.assertIsNone(config.header)
        self.assertTrue(config.emit_metadata_helpers)
        self.assertFalse(config.enable_python_interface)
        self.assertTrue(config.build_python_interface)
        self.assertFalse(config.build_crate)
        self.assertEqual(config.build_profile, "release")
        self.assertFalse(config.prefer_direct_output_sinks)

    def test_enable_python_interface_defaults_to_true(self) -> None:
        config = RustBackendConfig().with_enable_python_interface()
        self.assertTrue(config.enable_python_interface)

    def test_build_flags_can_be_toggled(self) -> None:
        config = RustBackendConfig() \
            .with_build_python_interface(False) \
            .with_build_crate(True) \
            .with_build_profile("debug")
        self.assertFalse(config.build_python_interface)
        self.assertTrue(config.build_crate)
        self.assertEqual(config.build_profile, "debug")

    def test_build_profile_must_be_supported(self) -> None:
        with self.assertRaises(ValueError):
            RustBackendConfig().with_build_profile("fast")

    def test_additional_dependencies_are_normalized(self) -> None:
        config = RustBackendConfig().with_additional_dependencies(
            ["serde", ("smallvec", "1.13")]
        )
        self.assertEqual(
            config.additional_dependencies,
            (("serde", None), ("smallvec", "1.13")),
        )

    def test_header_can_be_set(self) -> None:
        config = RustBackendConfig().with_header(
            "use smallvec::{smallvec};"
        )
        self.assertEqual(config.header, "use smallvec::{smallvec};")

    def test_direct_output_sinks_can_be_enabled(self) -> None:
        config = RustBackendConfig().with_prefer_direct_output_sinks()
        self.assertTrue(config.prefer_direct_output_sinks)
