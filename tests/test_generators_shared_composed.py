import unittest

from gradgen._rust_codegen.generators.shared.composed import (
    _build_composed_input_specs,
    _compose_composed_helper_base_name,
    _compose_offset_expr,
)


class SharedComposedTests(unittest.TestCase):
    def test_offset_expr_omits_zero_offset(self) -> None:
        self.assertEqual(_compose_offset_expr(0, "idx"), "idx")

    def test_helper_base_name_uses_crate_prefix(self) -> None:
        self.assertIn("crate", _compose_composed_helper_base_name("crate", "demo"))

    def test_input_specs_include_parameter_slice_when_needed(self) -> None:
        specs = _build_composed_input_specs("x", 2, "p", 3)
        self.assertEqual(len(specs), 2)
