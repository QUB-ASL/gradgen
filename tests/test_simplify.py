import unittest

from gradgen import Function, SX, SXVector, derivative, hessian, simplify


class SimplifyTests(unittest.TestCase):
    def test_scalar_rules_simplify_basic_identities(self) -> None:
        x = SX.sym("x")

        self.assertIs(simplify(x + 0).node, x.node)
        self.assertIs(simplify(x * 1).node, x.node)
        self.assertEqual(simplify(x * 0).value, 0.0)
        self.assertEqual(simplify(x - x).value, 0.0)
        self.assertEqual(simplify(x / x).value, 1.0)
        self.assertIs(simplify(-(-x)).node, x.node)

    def test_constant_folding_simplifies_numeric_subexpressions(self) -> None:
        expr = (SX.const(2.0) + SX.const(3.0)) * SX.const(4.0)
        simplified = simplify(expr)

        self.assertEqual(simplified.op, "const")
        self.assertEqual(simplified.value, 20.0)

    def test_simplify_flattens_addition_and_collects_constants(self) -> None:
        x = SX.sym("x")
        simplified = simplify((x + 2.0) + 3.0, max_effort="medium")
        evaluator = Function("f", [x], [simplified])

        self.assertEqual(evaluator(4.0), 9.0)
        self.assertEqual(simplified.op, "add")
        self.assertEqual({arg.value for arg in simplified.args}, {None, 5.0})

    def test_simplify_collects_scaled_like_terms(self) -> None:
        x = SX.sym("x")
        simplified = simplify((2.0 * x) + (3.0 * x), max_effort="medium")
        evaluator = Function("f", [x], [simplified])

        self.assertEqual(evaluator(4.0), 20.0)
        self.assertEqual(simplified.op, "mul")
        self.assertEqual({arg.value for arg in simplified.args}, {None, 5.0})

    def test_simplify_combines_sum_of_negatives(self) -> None:
        x = SX.sym("x")
        y = SX.sym("y")
        simplified = simplify((-x) + (-y), max_effort="medium")
        evaluator = Function("f", [x, y], [simplified])

        self.assertEqual(evaluator(2.0, 3.0), -5.0)
        self.assertEqual(simplified.op, "neg")
        self.assertEqual(simplified.args[0].op, "add")
        self.assertEqual({arg.name for arg in simplified.args[0].args}, {"x", "y"})

    def test_simplify_cancels_double_negative_factors(self) -> None:
        x = SX.sym("x")
        y = SX.sym("y")
        simplified = simplify((-x) * (-y), max_effort="medium")
        evaluator = Function("f", [x, y], [simplified])

        self.assertEqual(evaluator(2.0, 3.0), 6.0)
        self.assertEqual(simplified.op, "mul")
        self.assertEqual({arg.name for arg in simplified.args}, {"x", "y"})

    def test_simplify_normalizes_single_negative_factors_in_products(self) -> None:
        x = SX.sym("x")
        y = SX.sym("y")

        left_negative = simplify(x * (-y), max_effort="medium")
        right_negative = simplify((-x) * y, max_effort="medium")
        evaluator = Function("f", [x, y], [left_negative, right_negative])

        self.assertEqual(evaluator(2.0, 3.0), (-6.0, -6.0))
        self.assertEqual(left_negative.op, "neg")
        self.assertEqual(right_negative.op, "neg")
        self.assertEqual(left_negative.args[0].op, "mul")
        self.assertEqual(right_negative.args[0].op, "mul")

    def test_simplify_combines_powers_with_same_base(self) -> None:
        x = SX.sym("x")
        simplified = simplify((x**2.0) * (x**3.0), max_effort="high")
        evaluator = Function("f", [x], [simplified])

        self.assertEqual(evaluator(2.0), 32.0)
        self.assertEqual(simplified.op, "pow")
        self.assertIs(simplified.args[0].node, x.node)
        self.assertEqual(simplified.args[1].value, 5.0)

    def test_simplify_rewrites_repeated_factor_as_square(self) -> None:
        x = SX.sym("x")
        simplified = simplify(x * x, max_effort="medium")

        self.assertEqual(simplified.op, "pow")
        self.assertIs(simplified.args[0].node, x.node)
        self.assertEqual(simplified.args[1].value, 2.0)

    def test_simplify_reduces_zero_times_x_plus_one_times_y_to_y(self) -> None:
        x = SX.sym("x")
        y = SX.sym("y")
        simplified = simplify((0.0 * x) + (1.0 * y), max_effort="medium")

        self.assertIs(simplified.node, y.node)

    def test_simplify_cancels_symbol_plus_negative_itself(self) -> None:
        x = SX.sym("x")
        simplified = simplify(x + (-x), max_effort="medium")

        self.assertEqual(simplified.op, "const")
        self.assertEqual(simplified.value, 0.0)

    def test_simplify_rewrites_subtract_negative_as_addition(self) -> None:
        x = SX.sym("x")
        y = SX.sym("y")
        simplified = simplify(x - (-y), max_effort="medium")
        evaluator = Function("f", [x, y], [simplified])

        self.assertEqual(evaluator(2.0, 3.0), 5.0)
        self.assertEqual(simplified.op, "add")
        self.assertEqual({arg.name for arg in simplified.args}, {"x", "y"})

    def test_simplify_normalizes_signed_division(self) -> None:
        x = SX.sym("x")
        y = SX.sym("y")
        same_sign = simplify((-x) / (-y), max_effort="medium")
        mixed_sign = simplify(x / (-y), max_effort="medium")
        evaluator = Function("f", [x, y], [same_sign, mixed_sign])

        self.assertEqual(evaluator(6.0, 3.0), (2.0, -2.0))
        self.assertEqual(same_sign.op, "div")
        self.assertEqual(mixed_sign.op, "neg")
        self.assertEqual(mixed_sign.args[0].op, "div")

    def test_simplify_applies_safe_trigonometric_identity(self) -> None:
        x = SX.sym("x")
        expr = (x.sin() ** 2.0) + (x.cos() ** 2.0)
        simplified = simplify(expr, max_effort="high")

        self.assertEqual(simplified.op, "const")
        self.assertEqual(simplified.value, 1.0)

    def test_simplify_applies_trigonometric_parity_rules(self) -> None:
        x = SX.sym("x")
        sin_expr = simplify((-x).sin(), max_effort="medium")
        cos_expr = simplify((-x).cos(), max_effort="medium")
        evaluator = Function("f", [x], [sin_expr, cos_expr])

        sin_value, cos_value = evaluator(0.5)

        self.assertAlmostEqual(sin_value, -Function("s", [x], [x.sin()])(0.5))
        self.assertAlmostEqual(cos_value, Function("c", [x], [x.cos()])(0.5))
        self.assertEqual(sin_expr.op, "neg")
        self.assertEqual(cos_expr.op, "cos")

    def test_simplify_evaluates_safe_unary_constant_identities(self) -> None:
        self.assertEqual(simplify(SX.const(0.0).sin()).value, 0.0)
        self.assertEqual(simplify(SX.const(0.0).exp()).value, 1.0)
        self.assertEqual(simplify(SX.const(1.0).log()).value, 0.0)
        self.assertEqual(simplify(SX.const(1.0).sqrt()).value, 1.0)
        self.assertEqual(simplify(SX.const(0.0).cos()).value, 1.0)

    def test_simplify_respects_effort_none(self) -> None:
        x = SX.sym("x")
        expr = x + SX.const(0.0)

        simplified = simplify(expr, max_effort="none")

        self.assertIs(simplified.node, expr.node)

    def test_simplify_reduces_derivative_expression(self) -> None:
        x = SX.sym("x")
        expr = derivative(x * x, x)
        simplified = simplify(expr, max_effort="medium")
        evaluator = Function("df", [x], [simplified])

        self.assertEqual(simplified.op, "mul")
        self.assertEqual(evaluator(3.0), 6.0)

    def test_simplify_reduces_repeated_derivative_terms(self) -> None:
        x = SX.sym("x")
        expr = derivative((x * x) + (x * x), x)
        simplified = simplify(expr, max_effort="high")
        evaluator = Function("df", [x], [simplified])

        self.assertEqual(evaluator(3.0), 12.0)
        self.assertEqual(simplified.op, "mul")
        self.assertEqual({arg.value for arg in simplified.args}, {None, 4.0})

    def test_simplify_function_preserves_values(self) -> None:
        x = SX.sym("x")
        f = Function("f", [x], [derivative(x * x, x)])
        simplified = f.simplify(max_effort="medium", name="f_s")

        self.assertEqual(simplified.name, "f_s")
        self.assertEqual(f(3.0), simplified(3.0))
        self.assertEqual(simplified.outputs[0].op, "mul")

    def test_simplify_can_reduce_codegen_workspace(self) -> None:
        x = SX.sym("x")
        f = Function("f", [x], [derivative((x * x) + (x * x), x)], input_names=["x"], output_names=["dx"])
        simplified = f.simplify(max_effort="high")

        self.assertLessEqual(
            simplified.generate_rust().workspace_size,
            f.generate_rust().workspace_size,
        )

    def test_simplify_jacobian_function(self) -> None:
        x = SXVector.sym("x", 2)
        jac = Function("f", [x], [x.dot(x)]).jacobian(0)
        simplified = jac.simplify(max_effort="medium")

        self.assertEqual(jac([3.0, 4.0]), simplified([3.0, 4.0]))
        self.assertEqual(simplified.outputs[0][0].op, "mul")
        self.assertEqual(simplified.outputs[0][1].op, "mul")

    def test_simplify_hessian_function(self) -> None:
        x = SXVector.sym("x", 2)
        f = Function("f", [x], [(x[0] * x[0]) + (x[0] * x[1]) + (x[1] * x[1])])
        hes = f.hessian(0)
        simplified = hes.simplify(max_effort="high")

        self.assertEqual(hes([3.0, 4.0]), simplified([3.0, 4.0]))
        self.assertEqual(simplified.outputs[0][0].value, 2.0)
        self.assertEqual(simplified.outputs[0][1].value, 1.0)
        self.assertEqual(simplified.outputs[0][2].value, 1.0)
        self.assertEqual(simplified.outputs[0][3].value, 2.0)

    def test_simplify_supports_hessian_rows_directly(self) -> None:
        x = SXVector.sym("x", 2)
        rows = hessian((x[0] * x[0]) + (x[1] * x[1]), x)
        simplified = simplify(rows, max_effort="high")

        self.assertEqual(simplified[0][0].value, 2.0)
        self.assertEqual(simplified[0][1].value, 0.0)
        self.assertEqual(simplified[1][0].value, 0.0)
        self.assertEqual(simplified[1][1].value, 2.0)

    def test_simplify_rejects_unknown_effort(self) -> None:
        x = SX.sym("x")

        with self.assertRaises(ValueError):
            simplify(x, max_effort="wild")


if __name__ == "__main__":
    unittest.main()
