import math
import random
import unittest

from gradgen.function import Function
from gradgen.sx import SX, SXVector, bilinear_form, matvec, quadform


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

    def test_function_can_use_sliced_vector_views_from_packed_input(self) -> None:
        z = SXVector.sym("z", 4)
        x_view = z[0:3]
        u_view = z[3:4]
        f_expr = x_view.norm2() * u_view + x_view.norm2sq()
        f = Function("f", [z], [f_expr], input_names=["z"], output_names=["y"])

        result = f([3.0, 4.0, 1.0, 2.0])

        self.assertAlmostEqual(result, 2.0 * math.sqrt(26.0) + 26.0)

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

    def test_function_numeric_call_supports_additional_elementary_math(self) -> None:
        x = SX.sym("x")
        y = SX.sym("y")
        expr = (
            x.asinh()
            + x.cbrt()
            + x.erf()
            + x.erfc()
            + x.floor()
            + x.ceil()
            + x.round()
            + x.trunc()
            + x.fract()
            + x.signum()
            + x.hypot(y)
            + x.atan2(y)
            + x.minimum(y)
        )
        f = Function("f", [x, y], [expr])

        x_value = 1.25
        y_value = 2.0
        result = f(x_value, y_value)

        expected = (
            math.asinh(x_value)
            + math.copysign(abs(x_value) ** (1.0 / 3.0), x_value)
            + math.erf(x_value)
            + math.erfc(x_value)
            + math.floor(x_value)
            + math.ceil(x_value)
            + math.floor(x_value + 0.5)
            + math.trunc(x_value)
            + (x_value - math.trunc(x_value))
            + 1.0
            + math.hypot(x_value, y_value)
            + math.atan2(x_value, y_value)
            + min(x_value, y_value)
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

    def test_function_numeric_vector_call_supports_reductions(self) -> None:
        x = SXVector.sym("x", 3)
        f = Function("f", [x], [x.sum(), x.prod(), x.max(), x.min(), x.mean()])

        result = f([3.0, -4.0, 1.0])

        self.assertEqual(result[0], 0.0)
        self.assertEqual(result[1], -12.0)
        self.assertEqual(result[2], 3.0)
        self.assertEqual(result[3], -4.0)
        self.assertEqual(result[4], 0.0)

    def test_function_numeric_call_supports_constant_matrix_helpers(self) -> None:
        x = SXVector.sym("x", 2)
        y = SXVector.sym("y", 2)
        matrix = [[2.0, 1.0], [1.0, 3.0]]
        f = Function(
            "f",
            [x, y],
            [matvec(matrix, x), quadform(matrix, x), bilinear_form(x, matrix, y)],
        )

        result = f([1.0, 2.0], [3.0, 4.0])

        self.assertEqual(result[0], (4.0, 7.0))
        self.assertEqual(result[1], 18.0)
        self.assertEqual(result[2], 40.0)

    def test_function_numeric_call_supports_randomized_small_constant_matrix_helpers(self) -> None:
        rng = random.Random(1234)

        for size in (1, 2, 3):
            matrix = [[rng.uniform(-2.0, 2.0) for _ in range(size)] for _ in range(size)]
            x_value = [rng.uniform(-2.0, 2.0) for _ in range(size)]
            y_value = [rng.uniform(-2.0, 2.0) for _ in range(size)]

            x = SXVector.sym(f"x_{size}", size)
            y = SXVector.sym(f"y_{size}", size)
            f = Function(
                f"f_{size}",
                [x, y],
                [matvec(matrix, x), quadform(matrix, x), bilinear_form(x, matrix, y)],
            )

            result = f(x_value, y_value)

            expected_matvec = tuple(
                sum(matrix[row][col] * x_value[col] for col in range(size)) for row in range(size)
            )
            expected_quadform = sum(
                matrix[row][col] * x_value[row] * x_value[col]
                for row in range(size)
                for col in range(size)
            )
            expected_bilinear = sum(
                matrix[row][col] * x_value[row] * y_value[col]
                for row in range(size)
                for col in range(size)
            )

            self.assertEqual(len(result[0]), size)
            for actual, expected in zip(result[0], expected_matvec):
                self.assertAlmostEqual(actual, expected)
            self.assertAlmostEqual(result[1], expected_quadform)
            self.assertAlmostEqual(result[2], expected_bilinear)

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
