import unittest
import math

from gradgen import Function, SX, SXVector, SquaredDistanceToSet


class SquaredDistanceToSetTests(unittest.TestCase):
    def test_common_set_factories_accept_custom_input_name(self) -> None:
        ball = SquaredDistanceToSet.euclidean_ball(
            name="named_ball",
            center=(0.0, 0.0),
            radius=1.0,
            input_name="z",
        )
        infinity_ball = SquaredDistanceToSet.infinity_ball(
            name="named_infinity_ball",
            center=(0.0, 0.0),
            radius=1.0,
            input_name="w",
        )
        rectangle = SquaredDistanceToSet.rectangle(
            name="named_rectangle",
            xmin=(-1.0, -math.inf),
            xmax=(math.inf, 2.0),
            input_name="r",
        )

        self.assertEqual(ball.to_function().input_names, ("z",))
        self.assertEqual(infinity_ball.to_function().input_names, ("w",))
        self.assertEqual(rectangle.to_function().input_names, ("r",))

    def test_second_order_cone_factory_supports_symbolic_use(self) -> None:
        distance = SquaredDistanceToSet.second_order_cone(
            name="named_soc",
            alpha=2.0,
            dimension=3,
            input_name="z",
        )

        x = SXVector.sym("z", 3)
        expr = distance(x)

        self.assertEqual(repr(expr), "named_soc(SXVector.sym('z', 3))")
        self.assertEqual(distance.to_function().input_names, ("z",))
        with self.assertRaisesRegex(
            ValueError,
            "does not support numeric evaluation in Python",
        ):
            distance([3.0, 4.0, 1.0])

    def test_rust_only_distance_supports_symbolic_use(self) -> None:
        distance = (
            SquaredDistanceToSet(name="dist_to_axis_rust_only")
            .with_rust_sq_distance(
                """
fn dist_to_axis_rust_only(
    x: &[{{ scalar_type }}],
    w: &[{{ scalar_type }}],
) -> {{ scalar_type }} {
    let _ = w;
    0.5_{{ scalar_type }} * x[1] * x[1]
}
"""
            )
            .with_rust_projection(
                """
fn dist_to_axis_rust_only_projection(
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
        expr = distance(x)

        self.assertEqual(
            repr(expr),
            "dist_to_axis_rust_only(SXVector.sym('x', 2))",
        )
        self.assertEqual(distance.to_function().input_names, ("x",))
        with self.assertRaisesRegex(
            ValueError,
            "does not support numeric evaluation in Python",
        ):
            distance([1.0, 2.0])

    def test_euclidean_ball_factory(self) -> None:
        distance = SquaredDistanceToSet.euclidean_ball(
            name="unit_ball",
            center=(0.0, 0.0),
            radius=1.0,
        )

        self.assertEqual(distance([3.0, 4.0]), 8.0)
        self.assertEqual(distance.jacobian()([3.0, 4.0]), (2.4, 3.2))
        self.assertEqual(distance([0.2, 0.3]), 0.0)

    def test_infinity_ball_factory(self) -> None:
        distance = SquaredDistanceToSet.infinity_ball(
            name="unit_infinity_ball",
            center=(0.0, 0.0),
            radius=1.0,
        )

        self.assertEqual(distance([2.0, 3.0]), 2.5)
        self.assertEqual(distance.jacobian()([2.0, 3.0]), (1.0, 2.0))

    def test_rectangle_factory_supports_infinite_bounds(self) -> None:
        distance = SquaredDistanceToSet.rectangle(
            name="half_infinite_box",
            xmin=(-1.0, -math.inf),
            xmax=(math.inf, 2.0),
        )

        self.assertEqual(distance([-3.0, 4.0]), 4.0)
        self.assertEqual(distance.jacobian()([-3.0, 4.0]), (-2.0, 2.0))

    def test_common_set_factories_validate_bounds(self) -> None:
        with self.assertRaises(ValueError):
            SquaredDistanceToSet.euclidean_ball(
                name="bad_ball",
                center=(0.0, 0.0),
                radius=0.0,
            )
        with self.assertRaises(ValueError):
            SquaredDistanceToSet.infinity_ball(
                name="bad_infinity_ball",
                center=(0.0, 0.0),
                radius=-1.0,
            )
        with self.assertRaises(ValueError):
            SquaredDistanceToSet.rectangle(
                name="bad_rectangle",
                xmin=(1.0, 0.0),
                xmax=(0.0, 1.0),
            )
        with self.assertRaises(ValueError):
            SquaredDistanceToSet.second_order_cone(
                name="bad_soc_alpha",
                alpha=0.0,
                dimension=3,
            )
        with self.assertRaises(ValueError):
            SquaredDistanceToSet.second_order_cone(
                name="bad_soc_dimension",
                alpha=1.0,
                dimension=1,
            )

    def test_squared_distance_behaves_like_a_function(self) -> None:
        x = SXVector.sym("x", 2)
        projection = Function(
            "proj_axis",
            [x],
            [SXVector((x[0], SX.const(0.0)))],
            input_names=["x"],
            output_names=["p"],
        )
        sq_distance = Function(
            "sqdist_axis",
            [x],
            [0.5 * x[1] * x[1]],
            input_names=["x"],
            output_names=["d"],
        )
        distance = (
            SquaredDistanceToSet(name="dist_to_axis_functional")
            .with_projection_function(projection)
            .with_sq_distance_function(sq_distance)
        )

        self.assertEqual(distance([1.0, 100.0]), 5000.0)
        self.assertEqual(distance.to_function()([1.0, 100.0]), 5000.0)
        self.assertEqual(distance.jacobian()([1.0, 100.0]), (0.0, 100.0))

    def test_symbolic_functions_without_rust_snippets(
        self,
    ) -> None:
        x = SXVector.sym("x", 2)
        projection = Function(
            "proj_axis",
            [x],
            [SXVector((x[0], SX.const(0.0)))],
            input_names=["x"],
            output_names=["p"],
        )
        sq_distance = Function(
            "sqdist_axis",
            [x],
            [0.5 * x[1] * x[1]],
            input_names=["x"],
            output_names=["d"],
        )
        distance = (
            SquaredDistanceToSet(name="dist_to_axis_symbolic")
            .with_projection_function(projection)
            .with_sq_distance_function(sq_distance)
        )

        f = Function(
            "f",
            [x],
            [distance(2.0 * x)],
            input_names=["x"],
            output_names=["y"],
        )

        self.assertEqual(f([1.5, -3.0]), 18.0)
        self.assertEqual(f.gradient(0)([1.5, -3.0]), (0.0, -12.0))

    def test_projection_function_alone(self) -> None:
        x = SXVector.sym("x", 2)
        projection = Function(
            "proj_axis_only",
            [x],
            [SXVector((x[0], SX.const(0.0)))],
            input_names=["x"],
            output_names=["p"],
        )
        distance = (
            SquaredDistanceToSet(name="dist_to_axis_from_projection")
            .with_projection_function(projection)
        )

        f = Function("f", [x], [distance(x)], input_names=["x"])

        self.assertEqual(f([1.5, -3.0]), 4.5)
        self.assertEqual(f.gradient(0)([1.5, -3.0]), (0.0, -3.0))

    def test_symbolic_composition_and_gradient(self) -> None:
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

    def test_generates_gradient_rust_from_projection(self) -> None:
        distance = (
            SquaredDistanceToSet(name="dist_to_axis_codegen")
            .with_sq_distance(lambda x: 0.5 * x[1] * x[1])
            .with_projection(lambda x: [x[0], 0.0])
            .with_rust_sq_distance(
                """
fn dist_to_axis_codegen(
    x: &[{{ scalar_type }}],
    w: &[{{ scalar_type }}],
) -> {{ scalar_type }} {
    let _ = w;
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

    def test_second_order_cone_rust_projection_defers_sqrt(self) -> None:
        distance = SquaredDistanceToSet.second_order_cone(
            name="soc_codegen",
            alpha=2.0,
            dimension=3,
        )
        x = SXVector.sym("x", 3)
        f = Function("f", [x], [distance(x)], input_names=["x"])

        generated = f.generate_rust(backend_mode="no_std")

        self.assertIn("let alpha_sq = alpha * alpha;", generated.source)
        self.assertIn(
            "if t <= zero && alpha_sq * sum_sq <= t_sq {",
            generated.source,
        )
        self.assertIn(
            "if t >= zero && sum_sq <= alpha_sq * t_sq {",
            generated.source,
        )
        self.assertIn("let norm_y = sum_sq.sqrt();", generated.source)
        self.assertIn(
            "return 0.5_f64 * (t_sq + sum_sq);",
            generated.source,
        )
        self.assertIn(
            "let dist_sq = y_scale * y_scale * sum_sq + dt * dt;",
            generated.source,
        )

    def test_requires_projection_before_use(self) -> None:
        distance = (
            SquaredDistanceToSet(name="missing_projection").with_sq_distance(
                lambda x: 0.5 * x[0] * x[0]
            )
        )
        x = SXVector.sym("x", 1)

        with self.assertRaises(ValueError):
            _ = distance(x)

    def test_validates_projection_function_shape(self) -> None:
        x = SXVector.sym("x", 2)
        projection = Function(
            "bad_projection",
            [x],
            [x[0]],
            input_names=["x"],
            output_names=["p"],
        )
        distance = SquaredDistanceToSet(name="bad_projection_shape")

        with self.assertRaises(TypeError):
            distance.with_projection_function(projection)
