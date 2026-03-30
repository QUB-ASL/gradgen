import unittest

from gradgen import (
    Function,
    SX,
    SXNode,
    SXVector,
    derivative,
    hessian,
    matvec,
    quadform,
    bilinear_form,
    register_elementary_function,
    clear_registered_elementary_functions,
    simplify,
)
from gradgen._custom_elementary.model import (
    custom_vector_hessian_entry,
    custom_vector_hvp_component,
    custom_vector_jacobian_component,
)


class SimplifyTests(unittest.TestCase):
    def tearDown(self) -> None:
        clear_registered_elementary_functions()

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

    def test_simplify_constant_folds_vector_reductions_and_norms(self) -> None:
        x = SXVector((SX.const(-3.0), SX.const(4.0)))

        self.assertEqual(simplify(x.sum()).value, 1.0)
        self.assertEqual(simplify(x.prod()).value, -12.0)
        self.assertEqual(simplify(x.max()).value, 4.0)
        self.assertEqual(simplify(x.min()).value, -3.0)
        self.assertEqual(simplify(x.mean()).value, 0.5)
        self.assertEqual(simplify(x.norm1()).value, 7.0)
        self.assertEqual(simplify(x.norm2sq()).value, 25.0)
        self.assertEqual(simplify(x.norm2()).value, 5.0)
        self.assertEqual(simplify(x.norm_inf()).value, 4.0)
        self.assertAlmostEqual(simplify(x.norm_p(3)).value, (27.0 + 64.0) ** (1.0 / 3.0))
        self.assertEqual(simplify(x.norm_p_to_p(3)).value, 91.0)

    def test_simplify_constant_folds_matrix_helpers(self) -> None:
        x = SXVector((SX.const(2.0), SX.const(3.0)))
        y = SXVector((SX.const(5.0), SX.const(7.0)))
        matrix = [[1.0, 2.0], [3.0, 4.0]]

        simplified_matvec = simplify(matvec(matrix, x))
        simplified_quadform = simplify(quadform(matrix, x))
        simplified_bilinear = simplify(bilinear_form(x, matrix, y))

        self.assertEqual(tuple(element.value for element in simplified_matvec), (8.0, 18.0))
        self.assertEqual(simplified_quadform.value, 70.0)
        self.assertEqual(simplified_bilinear.value, 167.0)

    def test_simplify_constant_folds_custom_function_derivatives(self) -> None:
        weighted_sqnorm = register_elementary_function(
            name="weighted_sqnorm_simplify",
            input_dimension=2,
            parameter_dimension=2,
            parameter_defaults=[1.0, 1.0],
            eval_python=lambda x, w: w[0] * x[0] * x[0] + w[1] * x[1] * x[1],
            jacobian=lambda x, w: [2 * w[0] * x[0], 2 * w[1] * x[1]],
            hessian=lambda x, w: [[2 * w[0], 0.0], [0.0, 2 * w[1]]],
            hvp=lambda x, v, w: [2 * w[0] * v[0], 2 * w[1] * v[1]],
        )

        x = SXVector((SX.const(1.0), SX.const(2.0)))
        params = weighted_sqnorm.resolve_parameters([2.0, 3.0])
        tangent = SXVector((SX.const(1.0), SX.const(1.0)))

        simplified_output = simplify(weighted_sqnorm(x, w=[2.0, 3.0]), max_effort="medium")
        simplified_grad = (
            simplify(custom_vector_jacobian_component(weighted_sqnorm.name, 0, x, params), max_effort="medium"),
            simplify(custom_vector_jacobian_component(weighted_sqnorm.name, 1, x, params), max_effort="medium"),
        )
        simplified_hessian = (
            simplify(custom_vector_hessian_entry(weighted_sqnorm.name, 0, 0, x, params), max_effort="medium"),
            simplify(custom_vector_hessian_entry(weighted_sqnorm.name, 0, 1, x, params), max_effort="medium"),
            simplify(custom_vector_hessian_entry(weighted_sqnorm.name, 1, 0, x, params), max_effort="medium"),
            simplify(custom_vector_hessian_entry(weighted_sqnorm.name, 1, 1, x, params), max_effort="medium"),
        )
        simplified_hvp = (
            simplify(custom_vector_hvp_component(weighted_sqnorm.name, 0, x, tangent, params), max_effort="medium"),
            simplify(custom_vector_hvp_component(weighted_sqnorm.name, 1, x, tangent, params), max_effort="medium"),
        )

        self.assertEqual(simplified_output.value, 14.0)
        self.assertEqual(tuple(item.value for item in simplified_grad), (4.0, 12.0))
        self.assertEqual(tuple(item.value for item in simplified_hessian), (4.0, 0.0, 0.0, 6.0))
        self.assertEqual(tuple(item.value for item in simplified_hvp), (4.0, 6.0))

    def test_simplify_constant_folds_extended_unary_and_binary_ops(self) -> None:
        self.assertAlmostEqual(simplify(SX.const(0.5).asinh()).value, 0.48121182505960347)
        self.assertAlmostEqual(simplify(SX.const(2.0).acosh()).value, 1.3169578969248166)
        self.assertAlmostEqual(simplify(SX.const(0.25).atanh()).value, 0.25541281188299536)
        self.assertEqual(simplify(SX.const(-8.0).cbrt()).value, -2.0)
        self.assertEqual(simplify(SX.const(-3.25).floor()).value, -4.0)
        self.assertEqual(simplify(SX.const(-3.25).ceil()).value, -3.0)
        self.assertEqual(simplify(SX.const(-3.25).trunc()).value, -3.0)
        self.assertEqual(simplify(SX.const(-3.25).fract()).value, -0.25)
        self.assertEqual(simplify(SX.const(-3.25).signum()).value, -1.0)
        self.assertEqual(simplify(SX.const(2.0).maximum(5.0)).value, 5.0)
        self.assertEqual(simplify(SX.const(2.0).minimum(5.0)).value, 2.0)
        self.assertEqual(simplify(SX.const(3.0).hypot(4.0)).value, 5.0)
        self.assertEqual(simplify(SX.const(1.0).atan2(1.0)).op, "const")

    def test_simplify_rejects_unknown_effort(self) -> None:
        x = SX.sym("x")

        with self.assertRaises(ValueError):
            simplify(x, max_effort="wild")


class SimplifyEdgeTests(unittest.TestCase):
    def tearDown(self) -> None:
        clear_registered_elementary_functions()

    def test_rejects_negative_effort(self) -> None:
        x = SX.sym("x")
        with self.assertRaises(ValueError):
            simplify(x, max_effort=-1)

    def test_empty_reductions_raise_or_return_defaults(self) -> None:
        # sum with no args -> 0.0
        empty_sum = SX(SXNode.make("sum", ()))
        self.assertEqual(simplify(empty_sum).value, 0.0)

        # prod with no args -> 1.0
        empty_prod = SX(SXNode.make("prod", ()))
        self.assertEqual(simplify(empty_prod).value, 1.0)

        # reduce_max/min/mean without args should raise ValueError
        empty_max = SX(SXNode.make("reduce_max", ()))
        empty_min = SX(SXNode.make("reduce_min", ()))
        empty_mean = SX(SXNode.make("mean", ()))

        with self.assertRaises(ValueError):
            simplify(empty_max)
        with self.assertRaises(ValueError):
            simplify(empty_min)
        with self.assertRaises(ValueError):
            simplify(empty_mean)

    def test_unknown_const_binary_op_raises(self) -> None:
        # create a made-up binary op with constant args -> should error
        node = SXNode.make("unknown_binop", (SX.const(1.0).node, SX.const(2.0).node))
        expr = SX(node)
        with self.assertRaises(ValueError):
            simplify(expr)

    def test_norm_p_short_circuits(self) -> None:
        # norm_p and norm_p_to_p with insufficient args return 0.0
        np_node = SXNode.make("norm_p", (SX.const(2.0).node,))
        npp_node = SXNode.make("norm_p_to_p", (SX.const(2.0).node,))
        self.assertEqual(simplify(SX(np_node)).value, 0.0)
        self.assertEqual(simplify(SX(npp_node)).value, 0.0)


if __name__ == "__main__":
    unittest.main()
