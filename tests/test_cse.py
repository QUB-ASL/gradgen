import unittest

from gradgen import Function, SX, SXVector, cse, derivative


class CSETests(unittest.TestCase):
    def test_cse_extracts_repeated_scalar_subexpression(self) -> None:
        x = SX.sym("x")
        z = (x * x) + 1
        expr = z + (z * z)

        plan = cse([expr])

        self.assertEqual(len(plan.assignments), 2)
        self.assertEqual(plan.assignments[0].name, "w0")
        self.assertEqual(plan.assignments[0].expr.op, "mul")
        self.assertEqual(plan.assignments[1].expr.op, "add")
        self.assertEqual(plan.assignments[1].use_count, 3)

    def test_cse_skips_leaf_nodes(self) -> None:
        x = SX.sym("x")
        plan = cse([x])

        self.assertEqual(plan.assignments, ())
        self.assertEqual(len(plan.outputs), 1)

    def test_cse_respects_min_uses_threshold(self) -> None:
        x = SX.sym("x")
        z = (x * x) + 1
        expr = z + (z * z)

        plan = cse([expr], min_uses=4)

        self.assertEqual(plan.assignments, ())

    def test_cse_flattens_vector_outputs(self) -> None:
        x = SXVector.sym("x", 2)
        z = x.dot(x)
        plan = cse([x, SXVector((z, z))])

        self.assertEqual(len(plan.outputs), 4)
        self.assertEqual(plan.assignments[-1].expr.op, "add")

    def test_function_cse_builds_plan_for_outputs(self) -> None:
        x = SX.sym("x")
        z = (x * x) + 1
        f = Function("f", [x], [z + (z * z)])

        plan = f.cse(prefix="tmp")

        self.assertEqual(plan.assignments[0].name, "tmp0")
        self.assertEqual(plan.assignments[1].name, "tmp1")

    def test_cse_can_identify_repeated_derivative_subexpressions(self) -> None:
        x = SX.sym("x")
        expr = derivative((x * x + 1) * (x * x + 1), x)

        plan = cse([expr])

        self.assertTrue(
            any(assignment.expr.op == "add" \
                for assignment in plan.assignments))

    def test_function_cse_works_for_jacobian_functions(self) -> None:
        x = SXVector.sym("x", 2)
        jac = Function("f", [x], [x.dot(x)]).jacobian(0)

        plan = jac.cse()

        self.assertTrue(len(plan.assignments) >= 1)

    def test_cse_validates_min_uses(self) -> None:
        x = SX.sym("x")

        with self.assertRaises(ValueError):
            cse([x], min_uses=1)


if __name__ == "__main__":
    unittest.main()
