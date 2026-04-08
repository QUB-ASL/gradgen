import unittest

from gradgen import ComposedFunction, Function, SXVector


class ComposedFunctionTests(unittest.TestCase):
    def test_composed_packs_sym_params_evaluates(self) -> None:
        x = SXVector.sym("x", 2)
        state = SXVector.sym("state", 2)
        p = SXVector.sym("p", 2)

        g = Function(
            "G",
            [state, p],
            [SXVector((state[0] + p[0], state[1] * p[1]))],
            input_names=["state", "p"],
            output_names=["next_state"],
        )

        composed = (
            ComposedFunction("demo", x)
            .then(g, p=p)
            .repeat(g, params=[p, p])
            .finish()
        )

        compiled = composed.to_function()
        self.assertEqual(composed.input_names, ("x", "parameters"))
        self.assertEqual(composed.parameter_size, 6)
        self.assertEqual(compiled.input_names, ("x", "parameters"))
        self.assertEqual(compiled.output_names, ("y",))
        self.assertEqual(
            compiled([1.0, 2.0], [3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
            (16.0, 384.0),
        )
        self.assertEqual(
            composed.gradient().to_function()(
                [1.0, 2.0], [3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
            ),
            (1.0, 0.0, 0.0, 192.0),
        )

    def test_composed_function_accepts_keyword_arguments(self) -> None:
        x = SXVector.sym("x", 2)
        state = SXVector.sym("state", 2)
        p = SXVector.sym("p", 2)

        g = Function(
            "G",
            [state, p],
            [SXVector((state[0] + p[0], state[1] * p[1]))],
            input_names=["state", "p"],
            output_names=["next_state"],
        )

        composed = (
            ComposedFunction("demo", x)
            .then(g, p=p)
            .repeat(g, params=[p, p])
            .finish()
        )

        self.assertEqual(
            composed(
                x=[1.0, 2.0],
                parameters=[3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            ),
            (16.0, 384.0),
        )

    def test_repeat_requires_uniform_parameter_binding_kind(self) -> None:
        x = SXVector.sym("x", 2)
        state = SXVector.sym("state", 2)
        p = SXVector.sym("p", 2)

        g = Function(
            "G",
            [state, p],
            [SXVector((state[0] + p[0], state[1] + p[1]))],
            input_names=["state", "p"],
            output_names=["next_state"],
        )

        with self.assertRaises(ValueError):
            ComposedFunction("demo", x).repeat(g, params=[p, [1.0, 2.0]])

    def test_chain_accepts_function_parameter_pairs(self) -> None:
        x = SXVector.sym("x", 2)
        state = SXVector.sym("state", 2)
        p = SXVector.sym("p", 2)

        g1 = Function(
            "G1",
            [state, p],
            [SXVector((state[0] + p[0], state[1] + p[1]))],
            input_names=["state", "p"],
            output_names=["next_state"],
        )
        g2 = Function(
            "G2",
            [state, p],
            [SXVector((state[0] * p[0], state[1] * p[1]))],
            input_names=["state", "p"],
            output_names=["next_state"],
        )

        composed = (
            ComposedFunction("demo", x)
            .chain([(g1, [1.0, 2.0]), (g2, p)])
            .finish()
        )

        self.assertEqual(composed.parameter_size, 2)
        self.assertEqual(
            composed.to_function()([1.0, 3.0], [4.0, 5.0]),
            (8.0, 25.0),
        )

    def test_repeat_rejects_empty_parameter_list(self) -> None:
        x = SXVector.sym("x", 2)
        state = SXVector.sym("state", 2)
        p = SXVector.sym("p", 2)

        g = Function(
            "g",
            [state, p],
            [SXVector((state[0] + p[0], state[1] + p[1]))],
            input_names=["state", "p"],
            output_names=["next_state"],
        )

        with self.assertRaises(ValueError):
            ComposedFunction("demo", x).repeat(g, params=[])

    def test_finish_takes_no_arguments(self) -> None:
        x = SXVector.sym("x", 2)
        state = SXVector.sym("state", 2)
        p = SXVector.sym("p", 2)

        g = Function(
            "g",
            [state, p],
            [SXVector((state[0] + p[0], state[1] + p[1]))],
            input_names=["state", "p"],
            output_names=["next_state"],
        )

        with self.assertRaises(TypeError):
            ComposedFunction("demo", x).finish(g, p=[1.0, 2.0])

    def test_stage_edits_are_rejected_after_finish(self) -> None:
        x = SXVector.sym("x", 2)
        state = SXVector.sym("state", 2)
        p = SXVector.sym("p", 2)

        g = Function(
            "g",
            [state, p],
            [SXVector((state[0] + p[0], state[1] + p[1]))],
            input_names=["state", "p"],
            output_names=["next_state"],
        )

        composed = (
            ComposedFunction("demo", x)
            .then(g, p=[1.0, 2.0])
            .finish()
        )

        with self.assertRaises(ValueError):
            composed.then(g, p=[1.0, 2.0])
        with self.assertRaises(ValueError):
            composed.repeat(g, params=[[1.0, 2.0]])
        with self.assertRaises(ValueError):
            composed.finish()

    def test_stage_shape_validation_dim_mismatches(self) -> None:
        x = SXVector.sym("x", 2)
        state2 = SXVector.sym("state2", 2)
        p2 = SXVector.sym("p2", 2)

        bad_stage = Function(
            "bad_stage",
            [state2, p2],
            [SXVector((state2[0], state2[1], p2[0]))],
            input_names=["state", "p"],
            output_names=["next_state"],
        )

        with self.assertRaises(ValueError):
            ComposedFunction("demo", x).then(bad_stage, p=[1.0, 2.0])

        with self.assertRaises(ValueError):
            ComposedFunction("demo", x).finish()

    def test_state_input_must_be_symbolic(self) -> None:
        with self.assertRaises(ValueError):
            ComposedFunction("demo", SXVector((SXVector.sym("x", 2)[0], 1.0)))

        with self.assertRaises(ValueError):
            ComposedFunction("demo", 1.0 + SXVector.sym("x", 2)[0])


if __name__ == "__main__":
    unittest.main()
