import unittest
from types import SimpleNamespace

from gradgen import (
    clear_registered_elementary_functions,
    register_elementary_function,
)
from gradgen._rust_codegen.rendering.custom import (
    _build_custom_vector_hessian_wrapper_lines,
    _build_custom_vector_hvp_wrapper_lines,
    _build_custom_vector_jacobian_wrapper_lines,
    _match_custom_vector_derivative_output,
    _is_passthrough_one,
    _is_passthrough_zero,
    _match_passthrough_custom_vector_hessian_entry,
    _match_passthrough_matvec_component,
)
from gradgen.sx import SX, SXNode, SXVector


class RenderingCustomTests(unittest.TestCase):
    def tearDown(self) -> None:
        clear_registered_elementary_functions()

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

    def test_jacobian_output_matcher_accepts_trivial_wrappers(self) -> None:
        register_elementary_function(
            name="demo",
            input_dimension=2,
            jacobian=lambda value: value,
        )
        x0 = SX.sym("x0")
        x1 = SX.sym("x1")
        component0 = SX(
            SXNode.make(
                "custom_vector_jacobian_component",
                (SX.const(0.0).node, x0.node, x1.node),
                name="demo",
            )
        )
        component1 = SX(
            SXNode.make(
                "custom_vector_jacobian_component",
                (SX.const(1.0).node, x0.node, x1.node),
                name="demo",
            )
        )
        wrapped = SXVector(
            (
                (SX.const(1.0) * component0) + SX.const(0.0),
                SX.const(0.0) + (component1 * SX.const(1.0)),
            )
        )

        matched = _match_custom_vector_derivative_output(
            wrapped,
            derivative_kind="jacobian",
            scalar_bindings={x0.node: "x[0]", x1.node: "x[1]"},
            workspace_map={},
            backend_mode="std",
            scalar_type="f64",
            math_library=None,
        )

        self.assertIsNotNone(matched)
        self.assertEqual(matched[0], "demo")
        self.assertEqual(matched[1], "x")
