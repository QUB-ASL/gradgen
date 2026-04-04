import unittest
from types import SimpleNamespace

from gradgen._rust_codegen.generators.shared.single_shooting import (
    _build_single_shooting_input_specs,
    _build_single_shooting_output_specs,
    _compose_single_shooting_helper_base_name,
    _emit_single_shooting_control_slice,
    _emit_single_shooting_stage_range,
)


class SharedSingleShootingTests(unittest.TestCase):
    def _problem(self) -> SimpleNamespace:
        return SimpleNamespace(
            initial_state_name="x0",
            control_sequence_name="U",
            parameter_name="p",
            state_size=2,
            control_size=1,
            parameter_size=3,
            horizon=4,
        )

    def test_helper_base_name_prefixes_crate_name(self) -> None:
        self.assertIn("crate", _compose_single_shooting_helper_base_name("crate", "problem"))

    def test_input_specs_include_optional_hvp(self) -> None:
        specs = _build_single_shooting_input_specs(self._problem(), include_hvp=True)
        self.assertEqual(len(specs), 4)

    def test_output_specs_can_include_rollout_states(self) -> None:
        specs = _build_single_shooting_output_specs(
            self._problem(),
            include_cost=True,
            include_gradient=True,
            include_hvp=False,
            include_states=True,
        )
        self.assertEqual([spec.raw_name for spec in specs], ["cost", "gradient_U", "x_traj"])

    def test_slice_helpers_format_ranges(self) -> None:
        self.assertEqual(_emit_single_shooting_control_slice("U", "i", 1), "&U[i..(i + 1)]")
        self.assertEqual(_emit_single_shooting_stage_range("i", 2), "(i * 2)..((i + 1) * 2)")
