import unittest

from gradgen.sx import (
    SX,
    SXVector,
    acosh,
    acos,
    asinh,
    asin,
    atan2,
    atan,
    atanh,
    bilinear_form,
    cbrt,
    ceil,
    cos,
    cosh,
    erf,
    erfc,
    exp,
    expm1,
    floor,
    fract,
    hypot,
    log,
    log1p,
    matvec,
    maximum,
    minimum,
    quadform,
    round,
    signum,
    sin,
    sinh,
    sqrt,
    tan,
    tanh,
    trunc,
    vector,
)
import gradgen
from gradgen import Function


class SXTests(unittest.TestCase):
    def test_symbol_nodes_are_interned(self) -> None:
        x1 = SX.sym("x")
        x2 = SX.sym("x")

        self.assertIs(x1.node, x2.node)

    def test_symbol_metadata_participates_in_interning(self) -> None:
        x1 = SX.sym("x", metadata={"domain": "real"})
        x2 = SX.sym("x", metadata={"domain": "real"})
        xc = SX.sym("x", metadata={"domain": "complex"})

        self.assertIs(x1.node, x2.node)
        self.assertIsNot(x1.node, xc.node)

    def test_symbol_equality_and_hash_use_symbol_name(self) -> None:
        x1 = SX.sym("x", metadata={"domain": "real"})
        x2 = SX.sym("x", metadata={"domain": "complex"})
        y = SX.sym("y")

        self.assertEqual(x1, x2)
        self.assertEqual(hash(x1), hash(x2))
        self.assertNotEqual(x1, y)
        self.assertEqual({x1: "value"}[x2], "value")

    def test_vector_equality_uses_elementwise_symbol_names(self) -> None:
        p1 = SXVector.sym("p", 2)
        p2 = SXVector.sym("p", 2)
        q = SXVector.sym("q", 2)

        self.assertEqual(p1, p2)
        self.assertEqual(hash(p1), hash(p2))
        self.assertNotEqual(p1, q)
        self.assertEqual({p1: "value"}[p2], "value")

    def test_identical_binary_expressions_are_interned(self) -> None:
        x = SX.sym("x")
        y = SX.sym("y")

        expr1 = x + y
        expr2 = x + y

        self.assertIs(expr1.node, expr2.node)

    def test_commutative_nodes_are_canonicalized(self) -> None:
        x = SX.sym("x")
        y = SX.sym("y")

        self.assertIs((x + y).node, (y + x).node)
        self.assertIs((x * y).node, (y * x).node)

    def test_non_commutative_nodes_are_distinct(self) -> None:
        x = SX.sym("x")
        y = SX.sym("y")

        self.assertIsNot((x - y).node, (y - x).node)

    def test_constants_are_interned(self) -> None:
        c1 = SX.const(2.0)
        c2 = SX.const(2)

        self.assertIs(c1.node, c2.node)

    def test_unary_operations_create_expected_ops(self) -> None:
        x = SX.sym("x")

        self.assertEqual(sin(x).op, "sin")
        self.assertEqual(cos(x).op, "cos")
        self.assertEqual(tan(x).op, "tan")
        self.assertEqual(asin(x).op, "asin")
        self.assertEqual(acos(x).op, "acos")
        self.assertEqual(atan(x).op, "atan")
        self.assertEqual(asinh(x).op, "asinh")
        self.assertEqual(acosh(x).op, "acosh")
        self.assertEqual(atanh(x).op, "atanh")
        self.assertEqual(sinh(x).op, "sinh")
        self.assertEqual(cosh(x).op, "cosh")
        self.assertEqual(tanh(x).op, "tanh")
        self.assertEqual(exp(x).op, "exp")
        self.assertEqual(expm1(x).op, "expm1")
        self.assertEqual(log(x).op, "log")
        self.assertEqual(log1p(x).op, "log1p")
        self.assertEqual(sqrt(x).op, "sqrt")
        self.assertEqual(cbrt(x).op, "cbrt")
        self.assertEqual(erf(x).op, "erf")
        self.assertEqual(erfc(x).op, "erfc")
        self.assertEqual(floor(x).op, "floor")
        self.assertEqual(ceil(x).op, "ceil")
        self.assertEqual(round(x).op, "round")
        self.assertEqual(trunc(x).op, "trunc")
        self.assertEqual(fract(x).op, "fract")
        self.assertEqual(signum(x).op, "signum")
        self.assertEqual(atan2(x, 1).op, "atan2")
        self.assertEqual(hypot(x, 1).op, "hypot")
        self.assertEqual(x.abs().op, "abs")
        self.assertEqual(maximum(x, 1).op, "max")
        self.assertEqual(minimum(x, 1).op, "min")
        self.assertEqual((-x).op, "neg")

    def test_top_level_package_exports_unary_helpers(self) -> None:
        x = SX.sym("x")

        self.assertEqual(gradgen.sin(x).op, "sin")
        self.assertEqual(gradgen.cos(x).op, "cos")
        self.assertEqual(gradgen.tan(x).op, "tan")
        self.assertEqual(gradgen.asin(x).op, "asin")
        self.assertEqual(gradgen.acos(x).op, "acos")
        self.assertEqual(gradgen.atan(x).op, "atan")
        self.assertEqual(gradgen.asinh(x).op, "asinh")
        self.assertEqual(gradgen.acosh(x).op, "acosh")
        self.assertEqual(gradgen.atanh(x).op, "atanh")
        self.assertEqual(gradgen.sinh(x).op, "sinh")
        self.assertEqual(gradgen.cosh(x).op, "cosh")
        self.assertEqual(gradgen.tanh(x).op, "tanh")
        self.assertEqual(gradgen.exp(x).op, "exp")
        self.assertEqual(gradgen.expm1(x).op, "expm1")
        self.assertEqual(gradgen.log(x).op, "log")
        self.assertEqual(gradgen.log1p(x).op, "log1p")
        self.assertEqual(gradgen.sqrt(x).op, "sqrt")
        self.assertEqual(gradgen.cbrt(x).op, "cbrt")
        self.assertEqual(gradgen.erf(x).op, "erf")
        self.assertEqual(gradgen.erfc(x).op, "erfc")
        self.assertEqual(gradgen.floor(x).op, "floor")
        self.assertEqual(gradgen.ceil(x).op, "ceil")
        self.assertEqual(gradgen.round(x).op, "round")
        self.assertEqual(gradgen.trunc(x).op, "trunc")
        self.assertEqual(gradgen.fract(x).op, "fract")
        self.assertEqual(gradgen.signum(x).op, "signum")
        self.assertEqual(gradgen.atan2(x, 1).op, "atan2")
        self.assertEqual(gradgen.hypot(x, 1).op, "hypot")
        self.assertEqual(gradgen.maximum(x, 1).op, "max")
        self.assertEqual(gradgen.minimum(x, 1).op, "min")

    def test_numeric_operands_coerced_for_binary_ops(self) -> None:
        x = SX.sym("x")

        self.assertEqual({arg.value for arg in (x + 2).args}, {None, 2.0})
        self.assertEqual({arg.value for arg in (2 + x).args}, {None, 2.0})
        self.assertEqual({arg.value for arg in (x * 3).args}, {None, 3.0})
        self.assertEqual({arg.value for arg in (3 * x).args}, {None, 3.0})
        self.assertEqual((x / 4).args[1].value, 4.0)
        self.assertEqual((4 / x).args[0].value, 4.0)
        self.assertEqual((x**5).args[1].value, 5.0)
        self.assertEqual((5**x).args[0].value, 5.0)

    def test_reverse_non_commutative_binary_ops_preserve_order(self) -> None:
        x = SX.sym("x")

        expr_sub = 2 - x
        expr_div = 2 / x
        expr_pow = 2**x

        self.assertEqual(expr_sub.op, "sub")
        self.assertEqual(expr_sub.args[0].value, 2.0)
        self.assertEqual(expr_sub.args[1].name, "x")

        self.assertEqual(expr_div.op, "div")
        self.assertEqual(expr_div.args[0].value, 2.0)
        self.assertEqual(expr_div.args[1].name, "x")

        self.assertEqual(expr_pow.op, "pow")
        self.assertEqual(expr_pow.args[0].value, 2.0)
        self.assertEqual(expr_pow.args[1].name, "x")

    def test_invalid_operand_types_raise_type_error(self) -> None:
        x = SX.sym("x")

        with self.assertRaises(TypeError):
            _ = x + "bad"

        with self.assertRaises(TypeError):
            _ = sin("bad")

    def test_public_accessors_expose_node_metadata(self) -> None:
        x = SX.sym("x", metadata={"domain": "real"})
        c = SX.const(3)
        expr = x + c

        self.assertEqual(x.op, "symbol")
        self.assertEqual(x.name, "x")
        self.assertEqual(x.metadata, {"domain": "real"})
        self.assertIsNone(x.value)

        self.assertEqual(c.op, "const")
        self.assertIsNone(c.name)
        self.assertEqual(c.metadata, {})
        self.assertEqual(c.value, 3.0)

        self.assertEqual(expr.op, "add")
        self.assertEqual(expr.metadata, {})
        self.assertEqual(len(expr.args), 2)
        self.assertEqual({arg.name for arg in expr.args}, {"x", None})
        self.assertEqual({arg.value for arg in expr.args}, {None, 3.0})

    def test_repr_is_informative_for_common_node_shapes(self) -> None:
        x = SX.sym("x")
        xm = SX.sym("x", metadata={"domain": "real"})
        c = SX.const(2)

        self.assertEqual(repr(x), "SX.sym('x')")
        self.assertEqual(repr(xm), "SX.sym('x', metadata={'domain': 'real'})")
        self.assertEqual(repr(c), "SX.const(2.0)")
        self.assertEqual(repr(-x), "neg(SX.sym('x'))")
        self.assertEqual(repr(x + c), "add(SX.const(2.0), SX.sym('x'))")

    def test_symbol_metadata_is_validated(self) -> None:
        with self.assertRaises(TypeError):
            SX.sym("x", metadata="real")

        with self.assertRaises(TypeError):
            SX.sym("x", metadata={1: "real"})

        with self.assertRaises(TypeError):
            SX.sym("x", metadata={"tags": ["real"]})

    def test_nested_common_subexpressions_are_reused(self) -> None:
        x = SX.sym("x")
        y = SX.sym("y")

        subexpr = x + y
        expr = subexpr * subexpr

        self.assertIs(expr.args[0].node, expr.args[1].node)
        self.assertIs(expr.args[0].node, (x + y).node)


class SXVectorTests(unittest.TestCase):
    def test_symbolic_vector_uses_indexed_names(self) -> None:
        x = SXVector.sym("x", 3)

        self.assertEqual(len(x), 3)
        self.assertEqual([item.name for item in x], ["x_0", "x_1", "x_2"])

    def test_vector_slicing_returns_vector_view(self) -> None:
        x = SXVector.sym("x", 4)

        head = x[0:3]
        tail = x[2:]

        self.assertIsInstance(head, SXVector)
        self.assertIsInstance(tail, SXVector)
        self.assertEqual([item.name for item in head], ["x_0", "x_1", "x_2"])
        self.assertEqual([item.name for item in tail], ["x_2", "x_3"])

    def test_symbolic_vector_can_propagate_metadata(self) -> None:
        x = SXVector.sym("x", 2, metadata={"domain": "real"})

        self.assertEqual(
            [item.metadata for item in x], 
            [{"domain": "real"}, {"domain": "real"}])

    def test_empty_vectors_are_supported(self) -> None:
        x = SXVector.sym("x", 0)
        values = vector([])

        self.assertEqual(len(x), 0)
        self.assertEqual(len(values), 0)

    def test_empty_vector_dot_returns_zero_constant(self) -> None:
        x = SXVector.sym("x", 0)
        y = SXVector.sym("y", 0)

        dot = x.dot(y)

        self.assertEqual(dot.op, "const")
        self.assertEqual(dot.value, 0.0)

    def test_vector_constructor_coerces_scalar_like_values(self) -> None:
        values = vector([SX.sym("x"), 2, 3.5])

        self.assertEqual(len(values), 3)
        self.assertEqual(values[0].name, "x")
        self.assertEqual(values[1].value, 2.0)
        self.assertEqual(values[2].value, 3.5)

    def test_elementwise_vector_addition_and_subtraction(self) -> None:
        x = SXVector.sym("x", 2)
        y = SXVector.sym("y", 2)

        add_expr = x + y
        sub_expr = x - y
        reverse_sub_expr = y - x

        self.assertEqual(len(add_expr), 2)
        self.assertTrue(all(item.op == "add" for item in add_expr))
        self.assertEqual(sub_expr[0].op, "sub")
        self.assertEqual(sub_expr[0].args[0].name, "x_0")
        self.assertEqual(sub_expr[0].args[1].name, "y_0")
        self.assertEqual(reverse_sub_expr[0].args[0].name, "y_0")
        self.assertEqual(reverse_sub_expr[0].args[1].name, "x_0")

    def test_reverse_vector_addition_is_supported(self) -> None:
        x = SXVector.sym("x", 2)
        y = SXVector.sym("y", 2)

        expr = y.__radd__(x)

        self.assertEqual(len(expr), 2)
        self.assertTrue(all(item.op == "add" for item in expr))

    def test_scalar_vector_product_is_supported(self) -> None:
        x = SXVector.sym("x", 2)
        u = SX.sym("u")

        left = 2 * x
        right = x * 3
        symbolic_left = u * x
        symbolic_right = x * u

        self.assertTrue(all(item.op == "mul" for item in left))
        self.assertTrue(all(item.op == "mul" for item in right))
        self.assertTrue(all(item.op == "mul" for item in symbolic_left))
        self.assertTrue(all(item.op == "mul" for item in symbolic_right))
        self.assertEqual({arg.value for arg in left[0].args}, {None, 2.0})
        self.assertEqual({arg.value for arg in right[0].args}, {None, 3.0})
        self.assertEqual(
            {arg.name for arg in symbolic_left[0].args}, 
            {"u", "x_0"})
        self.assertEqual(
            {arg.name for arg in symbolic_right[1].args}, 
            {"u", "x_1"})

    def test_vector_vector_multiplication_is_not_supported(self) -> None:
        x = SXVector.sym("x", 2)
        y = SXVector.sym("y", 2)

        with self.assertRaises(TypeError):
            _ = x * y

        with self.assertRaises(TypeError):
            _ = x + 2

        with self.assertRaises(TypeError):
            _ = 2 + x

        with self.assertRaises(TypeError):
            _ = x / "bad"

        with self.assertRaises(TypeError):
            _ = SX.sym("s") / x

    def test_division_supports_scalar_and_vector_cases(self) -> None:
        x = SXVector.sym("x", 2)
        y = SXVector.sym("y", 2)

        scalar_div = x / 2
        reverse_div = 2 / x
        vector_div = x / y

        self.assertEqual(scalar_div[0].op, "div")
        self.assertEqual(scalar_div[0].args[1].value, 2.0)
        self.assertEqual(reverse_div[0].args[0].value, 2.0)
        self.assertEqual(vector_div[0].args[0].name, "x_0")
        self.assertEqual(vector_div[0].args[1].name, "y_0")

    def test_power_supports_scalar_and_vector_cases(self) -> None:
        x = SXVector.sym("x", 2)

        scalar_pow = x**2
        reverse_pow = 2**x

        self.assertEqual(scalar_pow[0].op, "pow")
        self.assertEqual(scalar_pow[0].args[1].value, 2.0)
        self.assertEqual(reverse_pow[0].op, "pow")
        self.assertEqual(reverse_pow[0].args[0].value, 2.0)

    def test_vector_power_rejects_vector_exponent(self) -> None:
        x = SXVector.sym("x", 2)
        y = SXVector.sym("y", 2)

        with self.assertRaises(TypeError):
            _ = x**y

        with self.assertRaises(TypeError):
            _ = y**x

    def test_vector_division_validates_lengths(self) -> None:
        x = SXVector.sym("x", 2)
        y = SXVector.sym("y", 3)

        with self.assertRaises(ValueError):
            _ = x / y

    def test_unary_vector_operations_apply_elementwise(self) -> None:
        x = SXVector.sym("x", 2)

        self.assertEqual(x.sin()[0].op, "sin")
        self.assertEqual(x.cos()[0].op, "cos")
        self.assertEqual(x.tan()[0].op, "tan")
        self.assertEqual(x.asin()[0].op, "asin")
        self.assertEqual(x.acos()[0].op, "acos")
        self.assertEqual(x.atan()[0].op, "atan")
        self.assertEqual(x.asinh()[0].op, "asinh")
        self.assertEqual(x.acosh()[0].op, "acosh")
        self.assertEqual(x.atanh()[0].op, "atanh")
        self.assertEqual(x.sinh()[0].op, "sinh")
        self.assertEqual(x.cosh()[0].op, "cosh")
        self.assertEqual(x.tanh()[0].op, "tanh")
        self.assertEqual(x.exp()[0].op, "exp")
        self.assertEqual(x.expm1()[0].op, "expm1")
        self.assertEqual(x.log()[0].op, "log")
        self.assertEqual(x.log1p()[0].op, "log1p")
        self.assertEqual(x.sqrt()[0].op, "sqrt")
        self.assertEqual(x.cbrt()[0].op, "cbrt")
        self.assertEqual(x.erf()[0].op, "erf")
        self.assertEqual(x.erfc()[0].op, "erfc")
        self.assertEqual(x.floor()[0].op, "floor")
        self.assertEqual(x.ceil()[0].op, "ceil")
        self.assertEqual(x.round()[0].op, "round")
        self.assertEqual(x.trunc()[0].op, "trunc")
        self.assertEqual(x.fract()[0].op, "fract")
        self.assertEqual(x.signum()[0].op, "signum")
        self.assertEqual(x.abs()[0].op, "abs")
        self.assertEqual((-x)[0].op, "neg")

    def test_dot_product_returns_scalar_expression(self) -> None:
        x = SXVector.sym("x", 2)
        y = SXVector.sym("y", 2)

        expr = x.dot(y)

        self.assertEqual(expr.op, "add")
        self.assertTrue(all(term.op == "mul" for term in expr.args))

    def test_dot_product_accepts_python_sequences(self) -> None:
        x = SXVector.sym("x", 2)
        expr = x.dot([1, 2])
        reference = x.dot(SXVector([1, 2]))

        self.assertEqual(expr.op, "add")
        self.assertIs(expr.node, reference.node)

    def test_vector_norms_build_expected_scalar_expressions(self) -> None:
        x = SXVector.sym("x", 2)

        norm1 = x.norm1()
        norm2 = x.norm2()
        norm2sq = x.norm2sq()
        norm_inf = x.norm_inf()
        norm_p = x.norm_p(3)
        norm_p_to_p = x.norm_p_to_p(3)

        self.assertEqual(norm1.op, "norm1")
        self.assertEqual(norm2.op, "norm2")
        self.assertEqual(norm2sq.op, "norm2sq")
        self.assertEqual(norm_inf.op, "norm_inf")
        self.assertEqual(norm_p.op, "norm_p")
        self.assertEqual(norm_p_to_p.op, "norm_p_to_p")

    def test_vector_reductions_build_expected_scalar_expressions(self) -> None:
        x = SXVector.sym("x", 3)

        total = x.sum()
        product = x.prod()
        maximum = x.max()
        minimum = x.min()
        mean = x.mean()

        self.assertEqual(total.op, "sum")
        self.assertEqual(product.op, "prod")
        self.assertEqual(maximum.op, "reduce_max")
        self.assertEqual(minimum.op, "reduce_min")
        self.assertEqual(mean.op, "mean")

    def test_constant_matrix_helpers_build_expected_expressions(self) -> None:
        x = SXVector.sym("x", 2)
        y = SXVector.sym("y", 2)
        matrix = [[2.0, 1.0], [1.0, 3.0]]

        matvec_expr = matvec(matrix, x)
        quadform_expr = quadform(matrix, x)
        bilinear_expr = bilinear_form(x, matrix, y)

        self.assertIsInstance(matvec_expr, SXVector)
        self.assertEqual(len(matvec_expr), 2)
        self.assertTrue(
            all(element.op == "matvec_component" for element in matvec_expr))
        self.assertEqual(quadform_expr.op, "quadform")
        self.assertEqual(bilinear_expr.op, "bilinear_form")

    def test_constant_matrix_helpers_validate_shapes(self) -> None:
        x = SXVector.sym("x", 2)
        y = SXVector.sym("y", 3)

        with self.assertRaises(ValueError):
            _ = matvec([[1.0, 2.0, 3.0]], x)

        with self.assertRaises(ValueError):
            _ = quadform([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], x)

        with self.assertRaises(ValueError):
            _ = quadform([[1.0, 2.0], [3.0, 4.0]], y)

        with self.assertRaises(ValueError):
            _ = bilinear_form(x, [[1.0, 2.0], [3.0, 4.0]], y)

    def test_constant_matrix_helpers_validate_matrix_entries(self) -> None:
        x = SXVector.sym("x", 2)

        with self.assertRaises(ValueError):
            _ = matvec([[1.0], [2.0, 3.0]], x)

        with self.assertRaises(TypeError):
            _ = matvec([["bad", 1.0], [2.0, 3.0]], x)

    def test_quadform_validates_symmetry_by_default(self) -> None:
        x = SXVector.sym("x", 2)

        with self.assertRaises(ValueError):
            _ = quadform([[1.0, 2.0], [3.0, 4.0]], x)

    def test_quadform_allows_non_symmetric_matrix_when_disabled(self) -> None:
        x = SXVector.sym("x", 2)
        expr = quadform([[1.0, 2.0], [3.0, 4.0]], x, is_symmetric=False)
        f = Function("quad_nonsymmetric", [x], [expr])

        # x^T P x = [2, 5] [[1,2],[3,4]] [2,5]^T = 154
        self.assertEqual(f([2.0, 5.0]), 154.0)

    def test_empty_vector_reductions_follow_documented_behavior(self) -> None:
        x = SXVector.sym("x", 0)

        self.assertEqual(x.sum().value, 0.0)
        self.assertEqual(x.prod().value, 1.0)

        with self.assertRaises(ValueError):
            _ = x.max()

        with self.assertRaises(ValueError):
            _ = x.min()

        with self.assertRaises(ValueError):
            _ = x.mean()

    def test_empty_vector_norms_return_zero_constants(self) -> None:
        x = SXVector.sym("x", 0)

        self.assertEqual(x.norm1().value, 0.0)
        self.assertEqual(x.norm2().value, 0.0)
        self.assertEqual(x.norm2sq().value, 0.0)
        self.assertEqual(x.norm_inf().value, 0.0)
        self.assertEqual(x.norm_p(3).value, 0.0)
        self.assertEqual(x.norm_p_to_p(3).value, 0.0)

    def test_vector_operations_validate_lengths(self) -> None:
        x = SXVector.sym("x", 2)
        y = SXVector.sym("y", 3)

        with self.assertRaises(ValueError):
            _ = x + y

        with self.assertRaises(ValueError):
            _ = x.dot(y)

    def test_vector_symbol_length_must_be_non_negative(self) -> None:
        with self.assertRaises(ValueError):
            SXVector.sym("x", -1)

    def test_vector_constructor_rejects_invalid_values(self) -> None:
        with self.assertRaises(TypeError):
            vector(["bad"])

    def test_scalar_vector_multiplication_reuses_canonical_nodes(self) -> None:
        x = SXVector.sym("x", 1)

        left = 2 * x
        right = x * 2

        self.assertIs(left.node, right.node)

    def test_singleton_vector_can_be_used_in_scalar_expressions(self) -> None:
        x = SXVector.sym("x", 3)
        u = SXVector.sym("u", 1)

        expr_left = u * x[0]
        expr_right = x[0] * u
        expr_add = x[0] + u

        self.assertIsInstance(expr_left, SX)
        self.assertIsInstance(expr_right, SX)
        self.assertIsInstance(expr_add, SX)
        self.assertEqual(expr_left.op, "mul")
        self.assertEqual(expr_right.op, "mul")
        self.assertEqual(expr_add.op, "add")
        self.assertEqual({arg.name for arg in expr_left.args}, {"u_0", "x_0"})

    def test_singleton_vector_supports_reverse_scalar_arithmetic(self) -> None:
        u = SXVector.sym("u", 1)

        expr_add = 2 + u
        expr_sub = 2 - u
        expr_mul = 2 * u

        self.assertIsInstance(expr_add, SX)
        self.assertIsInstance(expr_sub, SX)
        self.assertIsInstance(expr_mul, SX)
        self.assertEqual(expr_add.op, "add")
        self.assertEqual(expr_sub.op, "sub")
        self.assertEqual(expr_mul.op, "mul")
        self.assertEqual(expr_sub.args[0].value, 2.0)
        self.assertEqual(expr_sub.args[1].name, "u_0")

    def test_longer_vectors_rejected_in_scalar_expressions(self) -> None:
        x = SXVector.sym("x", 2)
        s = SX.sym("s")

        elementwise = s * x
        self.assertIsInstance(elementwise, SXVector)
        self.assertEqual(len(elementwise), 2)
        self.assertTrue(all(item.op == "mul" for item in elementwise))
        self.assertEqual(
            {arg.name for arg in elementwise[0].args}, 
            {"s", "x_0"})

        with self.assertRaises(TypeError):
            _ = x + s

    def test_vector_repr_is_informative(self) -> None:
        x = SXVector.sym("x", 2)

        self.assertEqual(repr(x), 
                         "SXVector(elements=(SX.sym('x_0'), SX.sym('x_1')))")

    def test_empty_matrix_helpers_support_zero_dimension_vectors(self) -> None:
        x = SXVector.sym("x", 0)
        y = SXVector.sym("y", 0)

        matvec_expr = matvec([], x)
        quadform_expr = quadform([], x)
        bilinear_expr = bilinear_form(x, [], y)

        self.assertEqual(len(matvec_expr), 0)
        self.assertEqual(quadform_expr.op, "quadform")
        self.assertEqual(bilinear_expr.op, "bilinear_form")

    def test_toplevel_scalar_helpers_reject_nonsingleton_vectors(self) -> None:
        x = SXVector.sym("x", 2)

        with self.assertRaises(TypeError):
            _ = maximum(x, 1)

        with self.assertRaises(TypeError):
            _ = minimum(x, 1)


if __name__ == "__main__":
    unittest.main()
