import unittest

from gradgen.sx import SX, cos, exp, log, sin, sqrt


class SXTests(unittest.TestCase):
    def test_symbol_nodes_are_interned(self) -> None:
        x1 = SX.sym("x")
        x2 = SX.sym("x")

        self.assertIs(x1.node, x2.node)

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
        self.assertEqual(exp(x).op, "exp")
        self.assertEqual(log(x).op, "log")
        self.assertEqual(sqrt(x).op, "sqrt")
        self.assertEqual((-x).op, "neg")


if __name__ == "__main__":
    unittest.main()
