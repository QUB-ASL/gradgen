import unittest

from gradgen import Function, SXVector, SquaredDistanceToSet


class SquaredDistanceToSetTests(unittest.TestCase):
    def test_squared_distance_supports_symbolic_composition_and_gradient(
        self,
    ) -> None:
        distance = (
            SquaredDistanceToSet(name="dist_to_axis")
            .with_sq_distance(lambda x: 0.5 * x[1] * x[1])
            .with_projection(lambda x: (x[0], 0.0))
        )

        x = SXVector.sym("x", 2)
        f = Function(
            "f",
            [x],
            [distance(2.0 * x)],
            input_names=["x"],
            output_names=["y"],
        )

        self.assertEqual(f([1.5, -3.0]), 18.0)
        self.assertEqual(f.gradient(0)([1.5, -3.0]), (0.0, -12.0))

    def test_squared_distance_generates_gradient_rust_from_projection(self) -> None:
        distance = (
            SquaredDistanceToSet(name="dist_to_axis_codegen")
            .with_sq_distance(lambda x: 0.5 * x[1] * x[1])
            .with_projection(lambda x: [x[0], 0.0])
            .with_rust_sq_distance(
                """
fn dist_to_axis_codegen(
    x: &[{{ scalar_type }}],
) -> {{ scalar_type }} {
    0.5_{{ scalar_type }} * x[1] * x[1]
}
"""
            )
            .with_rust_projection(
                """
fn dist_to_axis_codegen_projection(
    x: &[{{ scalar_type }}],
    out: &mut [{{ scalar_type }}],
) {
    out[0] = x[0];
    out[1] = 0.0_{{ scalar_type }};
}
"""
            )
        )

        x = SXVector.sym("x", 2)
        f = Function("f", [x], [distance(x)], input_names=["x"])

        generated = f.gradient(0).generate_rust(backend_mode="no_std")

        self.assertIn("dist_to_axis_codegen_projection", generated.source)
        self.assertIn("out[0] = x[0] - projection[0];", generated.source)
        self.assertIn("out[1] = x[1] - projection[1];", generated.source)

    def test_squared_distance_requires_projection_before_use(self) -> None:
        distance = SquaredDistanceToSet(name="missing_projection").with_sq_distance(
            lambda x: 0.5 * x[0] * x[0]
        )
        x = SXVector.sym("x", 1)

        with self.assertRaises(ValueError):
            _ = distance(x)
