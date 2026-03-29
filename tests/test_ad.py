import math
import unittest

from gradgen import Function, SX, SXVector, bilinear_form, derivative, gradient, hessian, jacobian, jvp, matvec, quadform, vjp


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

    def test_derivative_supports_extended_smooth_unary_math(self) -> None:
        x = SX.sym("x")
        expr = x.tan() + x.cosh() + x.tanh() + x.log1p() + x.expm1() + x.atan()
        df = derivative(expr, x)
        f = Function("f", [x], [df])

        value = 0.2
        result = f(value)
        expected = (
            1.0 / (math.cos(value) ** 2)
            + math.sinh(value)
            + (1.0 - math.tanh(value) ** 2)
            + 1.0 / (1.0 + value)
            + math.exp(value)
            + 1.0 / (1.0 + value * value)
        )
        self.assertAlmostEqual(result, expected)

    def test_derivative_supports_additional_smooth_elementary_math(self) -> None:
        x = SX.sym("x")
        y = SX.sym("y")
        expr = x.asinh() + x.acosh() + x.atanh() + x.cbrt() + x.erf() + x.erfc() + x.atan2(y) + x.hypot(y)
        dx = derivative(expr, x)
        evaluator = Function("df", [x, y], [dx])

        x_value = 1.5
        y_value = 2.0
        result = evaluator(x_value, y_value)
        expected = (
            1.0 / math.sqrt(x_value * x_value + 1.0)
            + 1.0 / (math.sqrt(x_value - 1.0) * math.sqrt(x_value + 1.0))
            + 1.0 / (1.0 - x_value * x_value)
            + 1.0 / (3.0 * (math.copysign(abs(x_value) ** (1.0 / 3.0), x_value) ** 2))
            + 1.1283791670955126 * math.exp(-(x_value * x_value))
            - 1.1283791670955126 * math.exp(-(x_value * x_value))
            + y_value / (x_value * x_value + y_value * y_value)
            + x_value / math.hypot(x_value, y_value)
        )
        self.assertAlmostEqual(result, expected)

    def test_ad_rejects_nonsmooth_elementary_math(self) -> None:
        x = SX.sym("x")

        for expr in (x.floor(), x.ceil(), x.round(), x.trunc(), x.fract(), x.signum()):
            with self.assertRaises(ValueError):
                _ = derivative(expr, x)

    def test_gradient_supports_norm_p_to_p_for_constant_p_greater_than_one(self) -> None:
        x = SXVector.sym("x", 2)
        f = Function("f", [x], [x.norm_p_to_p(3)])
        grad = f.gradient(0)

        result = grad([3.0, -4.0])

        self.assertAlmostEqual(result[0], 27.0)
        self.assertAlmostEqual(result[1], -48.0)

    def test_gradient_supports_norm_p_for_constant_p_greater_than_one(self) -> None:
        x = SXVector.sym("x", 2)
        f = Function("f", [x], [x.norm_p(3)])
        grad = f.gradient(0)

        result = grad([3.0, -4.0])
        denom = 91.0 ** (2.0 / 3.0)

        self.assertAlmostEqual(result[0], 9.0 / denom)
        self.assertAlmostEqual(result[1], -16.0 / denom)

    def test_ad_raises_for_norm_p_and_norm_p_to_p_when_p_is_one(self) -> None:
        x = SXVector.sym("x", 2)

        with self.assertRaises(ValueError):
            _ = gradient(x.norm_p(1), x)

        with self.assertRaises(ValueError):
            _ = gradient(x.norm_p_to_p(1), x)

    def test_gradient_supports_vector_sum_prod_and_mean(self) -> None:
        x = SXVector.sym("x", 3)
        f = Function("f", [x], [x.sum(), x.prod(), x.mean()])
        grad_sum = f.jacobian(0)

        result = grad_sum([3.0, -4.0, 1.0])

        self.assertEqual(result[0], (1.0, 1.0, 1.0))
        self.assertEqual(result[1], (-4.0, 3.0, -12.0))
        self.assertEqual(result[2], (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0))

    def test_ad_rejects_vector_max_and_min(self) -> None:
        x = SXVector.sym("x", 3)

        with self.assertRaises(ValueError):
            _ = gradient(x.max(), x)

        with self.assertRaises(ValueError):
            _ = gradient(x.min(), x)

    def test_quadratic_form_gradient_matches_two_p_x_for_symmetric_matrix(self) -> None:
        x = SXVector.sym("x", 2)
        matrix = [[2.0, 1.0], [1.0, 3.0]]
        f = Function("f", [x], [quadform(matrix, x)])
        grad = f.gradient(0)

        result = grad([1.0, 2.0])

        self.assertEqual(result, (8.0, 14.0))

    def test_quadratic_form_hvp_matches_constant_hessian_action(self) -> None:
        x = SXVector.sym("x", 2)
        matrix = [[2.0, 1.0], [1.0, 3.0]]
        f = Function("f", [x], [quadform(matrix, x)])
        hvp = f.hvp(0)

        result = hvp([1.0, 2.0], [3.0, 4.0])

        self.assertEqual(result, (20.0, 30.0))

    def test_bilinear_form_gradient_matches_expected_linear_maps(self) -> None:
        x = SXVector.sym("x", 2)
        y = SXVector.sym("y", 2)
        matrix = [[2.0, 1.0], [1.0, 3.0]]
        f = Function("f", [x, y], [bilinear_form(x, matrix, y)])

        grad_x = f.gradient(0)
        grad_y = f.gradient(1)

        self.assertEqual(grad_x([1.0, 2.0], [3.0, 4.0]), (10.0, 15.0))
        self.assertEqual(grad_y([1.0, 2.0], [3.0, 4.0]), (4.0, 7.0))

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

    def test_function_gradient_builds_scalar_output_gradient_function(self) -> None:
        x = SXVector.sym("x", 2)
        f = Function("f", [x], [x.dot(x)])
        grad = f.gradient(0)

        self.assertEqual(grad.name, "f_gradient_i0")
        self.assertEqual(grad([3.0, 4.0]), (6.0, 8.0))

    def test_function_gradient_requires_single_scalar_output(self) -> None:
        x = SXVector.sym("x", 2)

        with self.assertRaises(ValueError):
            Function("f", [x], [x]).gradient(0)

    def test_jacobian_blocks_returns_requested_blocks(self) -> None:
        x = SX.sym("x")
        y = SX.sym("y")
        f = Function("f", [x, y], [x + y, x * y], input_names=["x", "y"], output_names=["sum", "prod"])
        blocks = f.jacobian_blocks([1])

        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0].name, "f_jacobian_y")
        self.assertEqual(blocks[0](3.0, 4.0), (1.0, 3.0))

    def test_hessian_blocks_returns_requested_blocks(self) -> None:
        x = SXVector.sym("x", 2)
        y = SX.sym("y")
        f = Function(
            "f",
            [x, y],
            [(x[0] * x[0]) + (x[1] * x[1]) + (y * y)],
            input_names=["x", "y"],
            output_names=["cost"],
        )
        blocks = f.hessian_blocks([1])

        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0].name, "f_hessian_y")
        self.assertEqual(blocks[0]([3.0, 4.0], 2.0), 2.0)

    def test_function_hvp_builds_hessian_vector_product_function(self) -> None:
        x = SXVector.sym("x", 2)
        f = Function("f", [x], [x[0] * x[0] + x[0] * x[1] + x[1] * x[1]])
        hvp = f.hvp(0)

        self.assertEqual(hvp.name, "f_hvp_i0")
        self.assertEqual(hvp.input_names, ("i0", "v_i0"))
        self.assertEqual(hvp([3.0, 4.0], [1.0, 2.0]), (4.0, 5.0))

    def test_function_hvp_requires_single_scalar_output(self) -> None:
        x = SXVector.sym("x", 2)

        with self.assertRaises(ValueError):
            Function("f", [x], [x]).hvp(0)

    def test_block_helpers_validate_indices(self) -> None:
        x = SX.sym("x")
        f = Function("f", [x], [x * x])

        with self.assertRaises(IndexError):
            f.jacobian_blocks([1])

        with self.assertRaises(IndexError):
            f.hessian_blocks([1])


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

    def test_function_vjp_supports_runtime_seed_for_vector_output_and_vector_input_block(self) -> None:
        x = SXVector.sym("x", 2)
        f = Function("G", [x], [SXVector((x[0] + x[1], x[0] * x[1], x[1].sin()))])
        reverse = f.vjp(wrt_index=0)

        self.assertEqual(reverse.name, "G_vjp_i0")

        result = reverse([3.0, 4.0], [2.0, -1.0, 5.0])

        self.assertAlmostEqual(result[0], -2.0)
        self.assertAlmostEqual(result[1], 1.0 + 5.0 * math.cos(4.0))

    def test_function_jacobian_validates_index(self) -> None:
        x = SX.sym("x")
        f = Function("f", [x], [x])

        with self.assertRaises(IndexError):
            f.jacobian(1)

    def test_function_vjp_runtime_seed_validates_index(self) -> None:
        x = SX.sym("x")
        f = Function("f", [x], [x])

        with self.assertRaises(IndexError):
            f.vjp(wrt_index=1)

    def test_function_vjp_runtime_seed_rejects_explicit_cotangent_outputs(self) -> None:
        x = SX.sym("x")
        f = Function("f", [x], [x])

        with self.assertRaises(ValueError):
            f.vjp(1.0, wrt_index=0)

    def test_gradient_matches_jacobian_for_scalar_output(self) -> None:
        x = SXVector.sym("x", 2)
        f = Function("f", [x], [x.dot(x)])

        self.assertEqual(f.gradient(0)([3.0, 4.0]), f.jacobian(0)([3.0, 4.0]))


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

        self.assertEqual(result, (2.0, 1.0, 1.0, 2.0))

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

    def test_hessian_matches_jacobian_of_gradient(self) -> None:
        x = SXVector.sym("x", 2)
        f = Function("f", [x], [(x[0] * x[0]) + (x[0] * x[1]) + (x[1] * x[1])])

        hessian_function = f.hessian(0)
        jacobian_of_gradient = f.gradient(0).jacobian(0)

        jacobian_rows = jacobian_of_gradient([3.0, 4.0])
        flattened_jacobian = tuple(entry for row in jacobian_rows for entry in row)
        self.assertEqual(hessian_function([3.0, 4.0]), flattened_jacobian)

    def test_hessian_is_symmetric_for_scalar_real_function(self) -> None:
        x = SXVector.sym("x", 2)
        f = Function("f", [x], [(x[0] * x[0]) + (x[0] * x[1]) + (x[1] * x[1])])
        hes = f.hessian(0)

        flat = hes([3.0, 4.0])
        self.assertEqual(flat[1], flat[2])


if __name__ == "__main__":
    unittest.main()
