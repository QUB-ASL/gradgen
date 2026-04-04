import unittest
from types import SimpleNamespace

from gradgen._rust_codegen.rendering.custom import (
    _build_custom_vector_hessian_wrapper_lines,
    _build_custom_vector_hvp_wrapper_lines,
    _build_custom_vector_jacobian_wrapper_lines,
    _is_passthrough_one,
    _is_passthrough_zero,
    _match_passthrough_custom_vector_hessian_entry,
    _match_passthrough_matvec_component,
)
from gradgen.sx import SX, SXNode


class RenderingCustomTests(unittest.TestCase):
    def test_wrapper_line_helpers_use_expected_names(self) -> None:
        spec = SimpleNamespace(name="demo", vector_dim=3)

        jacobian = "\n".join(_build_custom_vector_jacobian_wrapper_lines(spec, "f64"))
        hvp = "\n".join(_build_custom_vector_hvp_wrapper_lines(spec, "f64"))
        hessian = "\n".join(_build_custom_vector_hessian_wrapper_lines(spec, "f64"))

        self.assertIn("demo_jacobian_component", jacobian)
        self.assertIn("demo_hvp_component", hvp)
        self.assertIn("demo_hessian_entry", hessian)

    def test_passthrough_detection_helpers_accept_trivial_wrappers(self) -> None:
        matvec = SX(SXNode.make("matvec_component", ()))
        hessian_entry = SX(SXNode.make("custom_vector_hessian_entry", ()))

        self.assertIsNotNone(_match_passthrough_matvec_component(matvec))
        self.assertIsNotNone(_match_passthrough_custom_vector_hessian_entry(hessian_entry))
        self.assertTrue(_is_passthrough_zero(SX.const(0.0)))
        self.assertTrue(_is_passthrough_one(SX.const(1.0)))
