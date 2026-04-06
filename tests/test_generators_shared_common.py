import unittest

from gradgen import Function, SXVector
from gradgen._rust_codegen.generators.shared.common \
    import _build_directional_derivative_function


class SharedCommonTests(unittest.TestCase):
    def test_directional_derivative_function_is_constructed(self) -> None:
        x = SXVector.sym("x", 1)
        f = Function(
            "demo", [x], [x[0] * x[0]], 
            input_names=["x"], output_names=["y"])
        jvp = _build_directional_derivative_function(
            f,
            active_indices=(0,),
            tangent_names=("t",),
            name="demo_jvp",
        )
        self.assertEqual(jvp.input_names, ("x", "t"))
