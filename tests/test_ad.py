import math
import unittest

from gradgen import Function, SX, SXVector, derivative, gradient, hessian, jacobian, jvp, vjp


class ForwardADTests(unittest.TestCase):
    def test_derivative_of_symbol_is_one_by_default(self) -> None:
        x = SX.sym("x")

        result = derivative(x, x)

        self.assertEqual(result.op, "const")
        self.assertEqual(result.value, 1.0)

    def test_derivative_of_independent_symbol_is_zero(self) -> None:
        x = SX.sym("x")
        y = SX.sym("y")

        result = derivative(y, x)

        self.assertEqual(result.op, "const")
        self.assertEqual(result.value, 0.0)

    def test_forward_rules_for_basic_scalar_ops(self) -> None:
        x = SX.sym("x")

        add_eval = Function("d_add", [x], [derivative(x + x, x)])
        sub_eval = Function("d_sub", [x], [derivative(x - x, x)])

        self.assertEqual(add_eval(3.0), 2.0)
        self.assertEqual(sub_eval(3.0), 0.0)

        square = derivative(x * x, x)
        self.assertEqual(square.op, "add")

        quotient = derivative(x / x, x)
        self.assertEqual(quotient.op, "div")

    def test_forward_rules_for_unary_ops_match_numeric_evaluation(self) -> None:
        x = SX.sym("x")
        expr = x.sin() + x.cos() + x.exp() + x.log() + x.sqrt()
        derivative_expr = derivative(expr, x)
        evaluator = Function("df", [x], [derivative_expr])

        value = 1.7
        result = evaluator(value)
        expected = (
            math.cos(value)
            - math.sin(value)
            + math.exp(value)
            + (1.0 / value)
            + (1.0 / (2.0 * math.sqrt(value)))
        )

        self.assertAlmostEqual(result, expected)

    def test_derivative_of_power_matches_expected_form(self) -> None:
        x = SX.sym("x")
        expr = derivative(x**3, x)
        evaluator = Function("df", [x], [expr])

        self.assertAlmostEqual(evaluator(2.0), 12.0)

    def test_jvp_requires_explicit_vector_seed(self) -> None:
        x = SXVector.sym("x", 2)

        with self.assertRaises(ValueError):
            jvp(x.dot(x), x)

    def test_vector_jvp_returns_directional_derivative(self) -> None:
        x = SXVector.sym("x", 2)
        expr = x.dot(x)
        directional = jvp(expr, x, [1.0, 0.0])
        evaluator = Function("df", [x], [directional])

        self.assertEqual(evaluator([3.0, 4.0]), 6.0)

    def test_vector_output_jvp_preserves_vector_shape(self) -> None:
        x = SXVector.sym("x", 2)
        expr = x.sin()
        directional = jvp(expr, x, [1.0, 2.0])
        evaluator = Function("df", [x], [directional])

        result = evaluator([0.0, 0.0])

        self.assertEqual(result, (1.0, 2.0))

    def test_function_jvp_builds_directional_derivative_function(self) -> None:
        x = SX.sym("x")
        y = SX.sym("y")
        f = Function("f", [x, y], [x * y + x.sin()])

        df_dx = f.jvp(1.0, 0.0, name="df_dx")
        df_dy = f.jvp(0.0, 1.0, name="df_dy")

        self.assertEqual(df_dx.name, "df_dx")
        self.assertAlmostEqual(df_dx(2.0, 5.0), 5.0 + math.cos(2.0))
        self.assertAlmostEqual(df_dy(2.0, 5.0), 2.0)

    def test_function_jvp_supports_vector_inputs(self) -> None:
        x = SXVector.sym("x", 2)
        f = Function("f", [x], [x.dot(x), x.sin()])
        directional = f.jvp([1.0, 0.0])

        result = directional([3.0, 4.0])

        self.assertEqual(result[0], 6.0)
        self.assertEqual(result[1], (math.cos(3.0), 0.0))

    def test_function_jvp_validates_tangent_argument_count(self) -> None:
        x = SX.sym("x")
        f = Function("f", [x], [x])

        with self.assertRaises(ValueError):
            f.jvp()


class ReverseADTests(unittest.TestCase):
    def test_gradient_of_symbol_is_one_by_default(self) -> None:
        x = SX.sym("x")

        result = gradient(x, x)

        evaluator = Function("g", [x], [result])
        self.assertEqual(evaluator(2.0), 1.0)

    def test_gradient_of_independent_symbol_is_zero(self) -> None:
        x = SX.sym("x")
        y = SX.sym("y")

        result = gradient(y, x)

        evaluator = Function("g", [x, y], [result])
        self.assertEqual(evaluator(2.0, 3.0), 0.0)

    def test_reverse_gradient_matches_expected_numeric_value(self) -> None:
        x = SX.sym("x")
        y = SX.sym("y")
        expr = (x * y) + x.sin()
        grad_x = gradient(expr, x)
        grad_y = gradient(expr, y)
        evaluator = Function("grad", [x, y], [grad_x, grad_y])

        result = evaluator(2.0, 5.0)

        self.assertAlmostEqual(result[0], 5.0 + math.cos(2.0))
        self.assertAlmostEqual(result[1], 2.0)

    def test_reverse_gradient_supports_vector_variables(self) -> None:
        x = SXVector.sym("x", 2)
        expr = x.dot(x)
        grad = gradient(expr, x)
        evaluator = Function("grad", [x], [grad])

        self.assertEqual(evaluator([3.0, 4.0]), (6.0, 8.0))

    def test_vector_output_vjp_returns_scalar_sensitivity(self) -> None:
        x = SX.sym("x")
        expr = SXVector((x.sin(), x * x))
        sensitivity = vjp(expr, x, [2.0, 3.0])
        evaluator = Function("vjp", [x], [sensitivity])

        result = evaluator(2.0)

        self.assertAlmostEqual(result, 2.0 * math.cos(2.0) + 3.0 * 4.0)

    def test_vector_output_vjp_returns_vector_sensitivity(self) -> None:
        x = SXVector.sym("x", 2)
        expr = SXVector((x.dot(x), x.sin()[0]))
        sensitivity = vjp(expr, x, [1.0, 2.0])
        evaluator = Function("vjp", [x], [sensitivity])

        result = evaluator([3.0, 4.0])

        self.assertEqual(result[1], 8.0)
        self.assertAlmostEqual(result[0], 6.0 + 2.0 * math.cos(3.0))

    def test_vector_output_vjp_requires_explicit_cotangent(self) -> None:
        x = SX.sym("x")
        expr = SXVector((x, x))

        with self.assertRaises(ValueError):
            vjp(expr, x)

    def test_function_vjp_builds_reverse_mode_function(self) -> None:
        x = SX.sym("x")
        y = SX.sym("y")
        f = Function("f", [x, y], [x * y + x.sin()])
        reverse = f.vjp(1.0, name="f_vjp")

        result = reverse(2.0, 5.0)

        self.assertEqual(reverse.name, "f_vjp")
        self.assertAlmostEqual(result[0], 5.0 + math.cos(2.0))
        self.assertAlmostEqual(result[1], 2.0)

    def test_function_vjp_supports_multiple_outputs(self) -> None:
        x = SX.sym("x")
        y = SX.sym("y")
        f = Function("f", [x, y], [x + y, x * y])
        reverse = f.vjp(2.0, 3.0)

        result = reverse(4.0, 5.0)

        self.assertEqual(result, (17.0, 14.0))

    def test_function_vjp_supports_vector_inputs(self) -> None:
        x = SXVector.sym("x", 2)
        f = Function("f", [x], [x.dot(x), x.sin()])
        reverse = f.vjp(1.0, [2.0, 0.0])

        result = reverse([3.0, 4.0])

        self.assertAlmostEqual(result[0], 6.0 + 2.0 * math.cos(3.0))
        self.assertEqual(result[1], 8.0)

    def test_function_vjp_validates_cotangent_argument_count(self) -> None:
        x = SX.sym("x")
        f = Function("f", [x], [x])

        with self.assertRaises(ValueError):
            f.vjp()


class JacobianTests(unittest.TestCase):
    def test_scalar_scalar_jacobian_matches_derivative(self) -> None:
        x = SX.sym("x")
        jac = jacobian(x * x, x)
        evaluator = Function("jac", [x], [jac])

        self.assertEqual(evaluator(3.0), 6.0)

    def test_scalar_vector_jacobian_returns_gradient_vector(self) -> None:
        x = SXVector.sym("x", 2)
        expr = x.dot(x)
        jac = jacobian(expr, x)
        evaluator = Function("jac", [x], [jac])

        self.assertEqual(evaluator([3.0, 4.0]), (6.0, 8.0))

    def test_vector_scalar_jacobian_returns_vector(self) -> None:
        x = SX.sym("x")
        expr = SXVector((x.sin(), x * x))
        jac = jacobian(expr, x)
        evaluator = Function("jac", [x], [jac])

        result = evaluator(2.0)

        self.assertAlmostEqual(result[0], math.cos(2.0))
        self.assertEqual(result[1], 4.0)

    def test_vector_vector_jacobian_returns_rows(self) -> None:
        x = SXVector.sym("x", 2)
        expr = SXVector((x[0] + x[1], x[0] * x[1]))
        jac = jacobian(expr, x)

        self.assertEqual(len(jac), 2)

        evaluator = Function("jac", [x], list(jac))
        result = evaluator([3.0, 4.0])

        self.assertEqual(result[0], (1.0, 1.0))
        self.assertEqual(result[1], (4.0, 3.0))

    def test_function_jacobian_for_scalar_input_block(self) -> None:
        x = SX.sym("x")
        y = SX.sym("y")
        f = Function("f", [x, y], [x + y, x * y])
        jac = f.jacobian(0)

        self.assertEqual(jac.name, "f_jacobian_i0")
        self.assertEqual(jac(3.0, 4.0), (1.0, 4.0))

    def test_function_jacobian_for_vector_input_block(self) -> None:
        x = SXVector.sym("x", 2)
        f = Function("f", [x], [x.dot(x), x.sin()])
        jac = f.jacobian(0)

        result = jac([3.0, 4.0])

        self.assertEqual(result[0], (6.0, 8.0))
        self.assertAlmostEqual(result[1][0], math.cos(3.0))
        self.assertEqual(result[1][1], 0.0)
        self.assertEqual(result[2][0], 0.0)
        self.assertAlmostEqual(result[2][1], math.cos(4.0))

    def test_function_jacobian_validates_index(self) -> None:
        x = SX.sym("x")
        f = Function("f", [x], [x])

        with self.assertRaises(IndexError):
            f.jacobian(1)


class HessianTests(unittest.TestCase):
    def test_scalar_scalar_hessian_matches_second_derivative(self) -> None:
        x = SX.sym("x")
        hes = hessian(x**3, x)
        evaluator = Function("hes", [x], [hes])

        self.assertEqual(evaluator(2.0), 12.0)

    def test_scalar_vector_hessian_returns_rows(self) -> None:
        x = SXVector.sym("x", 2)
        expr = (x[0] * x[0]) + (x[0] * x[1]) + (x[1] * x[1])
        hes = hessian(expr, x)

        self.assertEqual(len(hes), 2)

        evaluator = Function("hes", [x], list(hes))
        result = evaluator([3.0, 4.0])

        self.assertEqual(result[0], (2.0, 1.0))
        self.assertEqual(result[1], (1.0, 2.0))

    def test_hessian_of_separable_expression_is_diagonal(self) -> None:
        x = SXVector.sym("x", 2)
        expr = (x[0] * x[0]) + (x[1] * x[1])
        hes = hessian(expr, x)
        evaluator = Function("hes", [x], list(hes))

        result = evaluator([3.0, 4.0])

        self.assertEqual(result[0], (2.0, 0.0))
        self.assertEqual(result[1], (0.0, 2.0))

    def test_function_hessian_for_scalar_input_block(self) -> None:
        x = SX.sym("x")
        f = Function("f", [x], [x**3])
        hes = f.hessian(0)

        self.assertEqual(hes.name, "f_hessian_i0")
        self.assertEqual(hes(2.0), 12.0)

    def test_function_hessian_for_vector_input_block(self) -> None:
        x = SXVector.sym("x", 2)
        f = Function("f", [x], [(x[0] * x[0]) + (x[0] * x[1]) + (x[1] * x[1])])
        hes = f.hessian(0)

        result = hes([3.0, 4.0])

        self.assertEqual(result[0], (2.0, 1.0))
        self.assertEqual(result[1], (1.0, 2.0))

    def test_function_hessian_validates_index(self) -> None:
        x = SX.sym("x")
        f = Function("f", [x], [x])

        with self.assertRaises(IndexError):
            f.hessian(1)

    def test_function_hessian_requires_single_scalar_output(self) -> None:
        x = SX.sym("x")
        y = SXVector.sym("y", 2)

        with self.assertRaises(ValueError):
            Function("f", [x], [x, x]).hessian(0)

        with self.assertRaises(ValueError):
            Function("g", [y], [y]).hessian(0)


if __name__ == "__main__":
    unittest.main()
