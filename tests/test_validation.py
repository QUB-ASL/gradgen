import unittest
from types import SimpleNamespace

from gradgen._rust_codegen.validation import (
    resolve_backend_config,
    validate_backend_mode,
    validate_crate_name,
    validate_generated_argument_names,
    validate_scalar_type,
)


class ValidationTests(unittest.TestCase):
    def test_validate_backend_mode_accepts_std_and_no_std(self) -> None:
        validate_backend_mode("std")
        validate_backend_mode("no_std")

    def test_validate_scalar_type_rejects_unknown_types(self) -> None:
        with self.assertRaises(ValueError):
            validate_scalar_type("f16")

    def test_validate_crate_name_rejects_invalid_identifiers(self) -> None:
        with self.assertRaises(ValueError):
            validate_crate_name("not-a-valid-name")

    def test_validate_gen_rejects_duplicates(self) -> None:
        input_specs = (SimpleNamespace(raw_name="x", rust_name="x"),)
        output_specs = (SimpleNamespace(raw_name="y", rust_name="x"),)
        with self.assertRaises(ValueError):
            validate_generated_argument_names(input_specs, output_specs)

    def test_resolve_backend_config_sets_overrides(self) -> None:
        config = resolve_backend_config(
            None,
            crate_name="abc",
            function_name="demo",
            backend_mode="no_std",
            scalar_type="f32",
        )
        self.assertEqual(config.crate_name, "abc")
        self.assertEqual(config.function_name, "demo")
        self.assertEqual(config.backend_mode, "no_std")
        self.assertEqual(config.scalar_type, "f32")
