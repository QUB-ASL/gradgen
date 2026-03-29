import math
import unittest

from gradgen.function import Function
from gradgen.sx import SX, SXVector


class FunctionTests(unittest.TestCase):
    def test_function_uses_default_names(self) -> None:
        x = SX.sym("x")
        y = SX.sym("y")
        f = Function("f", [x], [x + y])

        self.assertEqual(f.input_names, ("i0",))
        self.assertEqual(f.output_names, ("o0",))

    def test_function_preserves_custom_names(self) -> None:
        x = SX.sym("x")
        y = SX.sym("y")
        f = Function(
            "f",
            [x, y],
            [x + y],
            input_names=["x_in", "y_in"],
            output_names=["sum_out"],
        )

        self.assertEqual(f.input_names, ("x_in", "y_in"))
        self.assertEqual(f.output_names, ("sum_out",))

    def test_function_validates_name_counts_and_uniqueness(self) -> None:
        x = SX.sym("x")

        with self.assertRaises(ValueError):
            Function("f", [x], [x], input_names=["a", "b"])

        with self.assertRaises(ValueError):
            Function("f", [x], [x], input_names=["dup", "dup"])

        with self.assertRaises(ValueError):
            Function("f", [x], [x], output_names=["a", "b"])

    def test_function_requires_inputs_and_outputs(self) -> None:
        x = SX.sym("x")

        with self.assertRaises(ValueError):
            Function("f", [], [x])

        with self.assertRaises(ValueError):
            Function("f", [x], [])

    def test_function_inputs_must_be_unique_symbols(self) -> None:
        x = SX.sym("x")

        with self.assertRaises(ValueError):
            Function("f", [x + 1], [x])

        with self.assertRaises(ValueError):
            Function("f", [x, x], [x])

        with self.assertRaises(ValueError):
            Function("f", [SXVector((x, x))], [x])

    def test_function_flattens_scalar_and_vector_arguments(self) -> None:
        x = SX.sym("x")
        y = SXVector.sym("y", 2)
        f = Function("f", [x, y], [y, x])

        self.assertEqual([item.name for item in f.flat_inputs], ["x", "y_0", "y_1"])
        self.assertEqual([item.name for item in f.flat_outputs], ["y_0", "y_1", "x"])

    def test_function_nodes_are_topologically_sorted(self) -> None:
        x = SX.sym("x")
        y = SX.sym("y")
        expr = (x + y).sin()
        f = Function("f", [x, y], [expr])

        self.assertEqual([node.op for node in f.nodes], ["symbol", "symbol", "add", "sin"])

    def test_function_repr_is_informative(self) -> None:
        x = SX.sym("x")
        f = Function("f", [x], [x], input_names=["x_in"], output_names=["x_out"])

        self.assertEqual(
            repr(f),
            "Function(name='f', input_names=('x_in',), output_names=('x_out',))",
        )

    def test_function_symbolic_call_returns_substituted_expression(self) -> None:
        x = SX.sym("x")
        y = SX.sym("y")
        f = Function("f", [x], [x + y])

        z = SX.sym("z")
        result = f(z)

        self.assertIsInstance(result, SX)
        self.assertEqual(result.op, "add")
        self.assertEqual({arg.name for arg in result.args}, {"z", "y"})

    def test_function_symbolic_call_preserves_symbol_metadata(self) -> None:
        x = SX.sym("x", metadata={"domain": "real"})
        y = SX.sym("y", metadata={"domain": "complex"})
        f = Function("f", [x], [x + y])

        z = SX.sym("z", metadata={"domain": "real"})
        result = f(z)

        self.assertIsInstance(result, SX)
        result_args = {arg.name: arg for arg in result.args if arg.name is not None}
        self.assertEqual(result_args["z"].metadata, {"domain": "real"})
        self.assertEqual(result_args["y"].metadata, {"domain": "complex"})

    def test_function_numeric_scalar_call_returns_float(self) -> None:
        x = SX.sym("x")
        f = Function("f", [x], [(x * x) + 1])

        result = f(3.0)

        self.assertEqual(result, 10.0)

    def test_function_numeric_call_supports_extended_unary_math(self) -> None:
        x = SX.sym("x")
        expr = (
            x.tan()
            + x.asin()
            + x.acos()
            + x.atan()
            + x.sinh()
            + x.cosh()
            + x.tanh()
            + x.expm1()
            + x.log1p()
            + x.abs()
        )
        f = Function("f", [x], [expr])

        value = 0.2
        result = f(value)

        expected = (
            math.tan(value)
            + math.asin(value)
            + math.acos(value)
            + math.atan(value)
            + math.sinh(value)
            + math.cosh(value)
            + math.tanh(value)
            + math.expm1(value)
            + math.log1p(value)
            + abs(value)
        )
        self.assertAlmostEqual(result, expected)

    def test_function_numeric_vector_call_supports_norms(self) -> None:
        x = SXVector.sym("x", 3)
        f = Function("f", [x], [x.norm1(), x.norm2(), x.norm2sq(), x.norm_inf(), x.norm_p(3), x.norm_p_to_p(3)])

        result = f([3.0, -4.0, 1.0])

        self.assertEqual(result[0], 8.0)
        self.assertAlmostEqual(result[1], math.sqrt(26.0))
        self.assertEqual(result[2], 26.0)
        self.assertEqual(result[3], 4.0)
        self.assertAlmostEqual(result[4], 92.0 ** (1.0 / 3.0))
        self.assertEqual(result[5], 92.0)

    def test_function_joint_returns_combined_primal_and_jacobian_outputs(self) -> None:
        x = SXVector.sym("x", 2)
        f = Function(
            "f",
            [x],
            [x[0] * x[0] + x[0] * x[1] + x[1] * x[1]],
            input_names=["x"],
            output_names=["y"],
        )

        joint = f.joint(("f", "jf"), 0, simplify_joint="high")

        self.assertEqual(joint.name, "f_joint_f_jf_x")
        self.assertEqual(joint.output_names, ("y", "jacobian_y"))
        self.assertEqual(joint([3.0, 4.0]), (37.0, (10.0, 11.0)))

    def test_function_joint_supports_primal_and_hvp(self) -> None:
        x = SXVector.sym("x", 2)
        f = Function(
            "f",
            [x],
            [x[0] * x[0] + x[0] * x[1] + x[1] * x[1]],
            input_names=["x"],
            output_names=["y"],
        )

        joint = f.joint(("f", "hvp"), 0, simplify_joint="high")

        self.assertEqual(joint.name, "f_joint_f_hvp_x")
        self.assertEqual(joint.input_names, ("x", "v_x"))
        self.assertEqual(joint.output_names, ("y", "hvp_y"))
        self.assertEqual(joint([3.0, 4.0], [1.0, 2.0]), (37.0, (4.0, 5.0)))

    def test_function_joint_validates_components(self) -> None:
        x = SX.sym("x")
        f = Function("f", [x], [x * x], input_names=["x"], output_names=["y"])

        with self.assertRaises(ValueError):
            f.joint(("f",))

        with self.assertRaises(ValueError):
            f.joint(("f", "f"))

        with self.assertRaises(ValueError):
            f.joint(("f", "banana"))

    def test_function_numeric_multi_input_multi_output_call_returns_tuple(self) -> None:
        x = SX.sym("x")
        y = SX.sym("y")
        f = Function("f", [x, y], [x + y, x * y])

        result = f(2.0, 3.0)

        self.assertEqual(result, (5.0, 6.0))

    def test_function_numeric_vector_call_returns_numeric_tuple(self) -> None:
        x = SXVector.sym("x", 2)
        f = Function("f", [x], [x, x.dot(x)])

        result = f([2.0, 3.0])

        self.assertEqual(result, ((2.0, 3.0), 13.0))

    def test_function_accepts_symbolic_vector_arguments(self) -> None:
        x = SXVector.sym("x", 2)
        y = SXVector.sym("y", 2)
        f = Function("f", [x], [x + y])

        z = SXVector.sym("z", 2)
        result = f(z)

        self.assertIsInstance(result, SXVector)
        self.assertEqual(len(result), 2)
        self.assertEqual({arg.name for arg in result[0].args}, {"z_0", "y_0"})

    def test_function_validates_call_argument_count(self) -> None:
        x = SX.sym("x")
        f = Function("f", [x], [x])

        with self.assertRaises(ValueError):
            f()

    def test_function_validates_vector_argument_lengths(self) -> None:
        x = SXVector.sym("x", 2)
        f = Function("f", [x], [x])

        with self.assertRaises(ValueError):
            f([1.0])

    def test_function_rejects_invalid_argument_shapes(self) -> None:
        x = SX.sym("x")
        y = SXVector.sym("y", 2)
        f_scalar = Function("f_scalar", [x], [x])
        f_vector = Function("f_vector", [y], [y])

        with self.assertRaises(TypeError):
            f_scalar([1.0])

        with self.assertRaises(TypeError):
            f_vector(1.0)

    def test_function_numeric_call_supports_unary_math(self) -> None:
        x = SX.sym("x")
        f = Function("f", [x], [x.sin() + x.cos() + x.exp()])

        result = f(0.5)

        self.assertAlmostEqual(result, math.sin(0.5) + math.cos(0.5) + math.exp(0.5))


if __name__ == "__main__":
    unittest.main()
