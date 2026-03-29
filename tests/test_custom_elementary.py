import unittest

from gradgen import (
    Function,
    SX,
    SXVector,
    clear_registered_elementary_functions,
    register_elementary_function,
)


class CustomElementaryTests(unittest.TestCase):
    def tearDown(self) -> None:
        clear_registered_elementary_functions()

    def test_register_scalar_custom_function_supports_symbolic_and_numeric_use(self) -> None:
        square_shift = register_elementary_function(
            name="square_shift",
            input_dimension=1,
            parameter_dimension=1,
            parameter_defaults=[1.0],
            eval_python=lambda x, w: x * x + w[0],
            jacobian=lambda x, w: 2 * x,
            hessian=lambda x, w: SX.const(2.0),
            rust_primal="""
fn square_shift(x: {{ scalar_type }}, w: &[{{ scalar_type }}]) -> {{ scalar_type }} {
    x * x + w[0]
}
""",
            rust_jacobian="""
fn square_shift_jacobian(x: {{ scalar_type }}, w: &[{{ scalar_type }}]) -> {{ scalar_type }} {
    let _ = w;
    2.0_{{ scalar_type }} * x
}
""",
            rust_hvp="""
fn square_shift_hvp(
    x: {{ scalar_type }},
    v_x: {{ scalar_type }},
    w: &[{{ scalar_type }}],
) -> {{ scalar_type }} {
    let _ = x;
    let _ = w;
    2.0_{{ scalar_type }} * v_x
}
""",
            rust_hessian="""
fn square_shift_hessian(x: {{ scalar_type }}, w: &[{{ scalar_type }}]) -> {{ scalar_type }} {
    let _ = x;
    let _ = w;
    2.0_{{ scalar_type }}
}
""",
        )

        x = SX.sym("x")
        f = Function("f", [x], [square_shift(x, w=[3.0])], input_names=["x"], output_names=["y"])

        self.assertEqual(f(2.0), 7.0)
        self.assertEqual(f.gradient(0)(2.0), 4.0)
        self.assertEqual(f.hvp(0)(2.0, 5.0), 10.0)

    def test_register_vector_custom_function_supports_symbolic_and_numeric_use(self) -> None:
        weighted_sqnorm = register_elementary_function(
            name="weighted_sqnorm",
            input_dimension=2,
            parameter_dimension=2,
            parameter_defaults=[1.0, 1.0],
            eval_python=lambda x, w: w[0] * x[0] * x[0] + w[1] * x[1] * x[1],
            jacobian=lambda x, w: SXVector((2 * w[0] * x[0], 2 * w[1] * x[1])),
            hessian=lambda x, w: (
                SXVector((2 * w[0], SX.const(0.0))),
                SXVector((SX.const(0.0), 2 * w[1])),
            ),
            rust_primal="""
fn weighted_sqnorm(x: &[{{ scalar_type }}], w: &[{{ scalar_type }}]) -> {{ scalar_type }} {
    w[0] * x[0] * x[0] + w[1] * x[1] * x[1]
}
""",
            rust_jacobian="""
fn weighted_sqnorm_jacobian(
    x: &[{{ scalar_type }}],
    w: &[{{ scalar_type }}],
    out: &mut [{{ scalar_type }}],
) {
    out[0] = 2.0_{{ scalar_type }} * w[0] * x[0];
    out[1] = 2.0_{{ scalar_type }} * w[1] * x[1];
}
""",
            rust_hvp="""
fn weighted_sqnorm_hvp(
    x: &[{{ scalar_type }}],
    v_x: &[{{ scalar_type }}],
    w: &[{{ scalar_type }}],
    out: &mut [{{ scalar_type }}],
) {
    let _ = x;
    out[0] = 2.0_{{ scalar_type }} * w[0] * v_x[0];
    out[1] = 2.0_{{ scalar_type }} * w[1] * v_x[1];
}
""",
            rust_hessian="""
fn weighted_sqnorm_hessian(
    x: &[{{ scalar_type }}],
    w: &[{{ scalar_type }}],
    out: &mut [{{ scalar_type }}],
) {
    let _ = x;
    out[0] = 2.0_{{ scalar_type }} * w[0];
    out[1] = 0.0_{{ scalar_type }};
    out[2] = 0.0_{{ scalar_type }};
    out[3] = 2.0_{{ scalar_type }} * w[1];
}
""",
        )

        x = SXVector.sym("x", 2)
        f = Function("f", [x], [weighted_sqnorm(x, w=[2.0, 3.0])], input_names=["x"], output_names=["y"])

        self.assertEqual(f([1.0, 2.0]), 14.0)
        self.assertEqual(f.gradient(0)([1.0, 2.0]), (4.0, 12.0))
        self.assertEqual(f.hvp(0)([1.0, 2.0], [3.0, 4.0]), (12.0, 24.0))
        self.assertEqual(f.hessian(0)([1.0, 2.0]), (4.0, 0.0, 0.0, 6.0))

    def test_register_vector_custom_function_accepts_symbolic_parameter_inputs(self) -> None:
        weighted_sqnorm = register_elementary_function(
            name="weighted_sqnorm_symbolic_w",
            input_dimension=2,
            parameter_dimension=2,
            parameter_defaults=[1.0, 1.0],
            eval_python=lambda x, w: w[0] * x[0] * x[0] + w[1] * x[1] * x[1],
            jacobian=lambda x, w: [2 * w[0] * x[0], 2 * w[1] * x[1]],
            hessian=lambda x, w: [
                [2 * w[0], 0.0],
                [0.0, 2 * w[1]],
            ],
            hvp=lambda x, v, w: [2 * w[0] * v[0], 2 * w[1] * v[1]],
            rust_primal="""
fn weighted_sqnorm_symbolic_w(
    x: &[{{ scalar_type }}],
    w: &[{{ scalar_type }}],
) -> {{ scalar_type }} {
    w[0] * x[0] * x[0] + w[1] * x[1] * x[1]
}
""",
            rust_jacobian="""
fn weighted_sqnorm_symbolic_w_jacobian(
    x: &[{{ scalar_type }}],
    w: &[{{ scalar_type }}],
    out: &mut [{{ scalar_type }}],
) {
    out[0] = 2.0_{{ scalar_type }} * w[0] * x[0];
    out[1] = 2.0_{{ scalar_type }} * w[1] * x[1];
}
""",
            rust_hvp="""
fn weighted_sqnorm_symbolic_w_hvp(
    x: &[{{ scalar_type }}],
    v_x: &[{{ scalar_type }}],
    w: &[{{ scalar_type }}],
    out: &mut [{{ scalar_type }}],
) {
    let _ = x;
    out[0] = 2.0_{{ scalar_type }} * w[0] * v_x[0];
    out[1] = 2.0_{{ scalar_type }} * w[1] * v_x[1];
}
""",
            rust_hessian="""
fn weighted_sqnorm_symbolic_w_hessian(
    x: &[{{ scalar_type }}],
    w: &[{{ scalar_type }}],
    out: &mut [{{ scalar_type }}],
) {
    let _ = x;
    out[0] = 2.0_{{ scalar_type }} * w[0];
    out[1] = 0.0_{{ scalar_type }};
    out[2] = 0.0_{{ scalar_type }};
    out[3] = 2.0_{{ scalar_type }} * w[1];
}
""",
        )

        x = SXVector.sym("x", 2)
        w = SXVector.sym("w", 2)
        f = Function("f", [x, w], [weighted_sqnorm(x, w=w)], input_names=["x", "w"], output_names=["y"])

        self.assertEqual(f([1.0, 2.0], [2.0, 3.0]), 14.0)
        self.assertEqual(f.gradient(0)([1.0, 2.0], [2.0, 3.0]), (4.0, 12.0))
        self.assertIn("w: &[f64]", f.generate_rust(backend_mode="no_std").source)

    def test_register_vector_custom_function_accepts_plain_python_array_outputs(self) -> None:
        weighted_sqnorm = register_elementary_function(
            name="weighted_sqnorm_plain",
            input_dimension=2,
            parameter_dimension=2,
            parameter_defaults=[1.0, 1.0],
            eval_python=lambda x, w: w[0] * x[0] * x[0] + w[1] * x[1] * x[1],
            jacobian=lambda x, w: [2 * w[0] * x[0], 2 * w[1] * x[1]],
            hessian=lambda x, w: [
                [2 * w[0], 0],
                [0, 2 * w[1]],
            ],
            hvp=lambda x, v, w: [2 * w[0] * v[0], 2 * w[1] * v[1]],
        )

        x = SXVector.sym("x", 2)
        f = Function("f", [x], [weighted_sqnorm(x, w=[2.0, 3.0])], input_names=["x"], output_names=["y"])

        self.assertEqual(f([1.0, 2.0]), 14.0)
        self.assertEqual(f.gradient(0)([1.0, 2.0]), (4.0, 12.0))
        self.assertEqual(f.hessian(0)([1.0, 2.0]), (4.0, 0.0, 0.0, 6.0))
        self.assertEqual(f.hvp(0)([1.0, 2.0], [3.0, 4.0]), (12.0, 24.0))

    def test_register_custom_function_accepts_numpy_array_outputs_when_available(self) -> None:
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy is not installed")

        weighted_sqnorm = register_elementary_function(
            name="weighted_sqnorm_numpy",
            input_dimension=2,
            parameter_dimension=2,
            parameter_defaults=[1.0, 1.0],
            eval_python=lambda x, w: float(np.dot(np.asarray(w), np.asarray(x) ** 2)),
            jacobian=lambda x, w: np.array([2 * w[0] * x[0], 2 * w[1] * x[1]], dtype=object),
            hessian=lambda x, w: np.array([[2 * w[0], 0], [0, 2 * w[1]]], dtype=object),
            hvp=lambda x, v, w: np.array([2 * w[0] * v[0], 2 * w[1] * v[1]], dtype=object),
        )

        x = SXVector.sym("x", 2)
        f = Function("f", [x], [weighted_sqnorm(x, w=[2.0, 3.0])], input_names=["x"], output_names=["y"])

        self.assertEqual(f([1.0, 2.0]), 14.0)
        self.assertEqual(f.gradient(0)([1.0, 2.0]), (4.0, 12.0))
        self.assertEqual(f.hessian(0)([1.0, 2.0]), (4.0, 0.0, 0.0, 6.0))
        self.assertEqual(f.hvp(0)([1.0, 2.0], [3.0, 4.0]), (12.0, 24.0))

    def test_register_custom_function_allows_numpy_only_opaque_derivatives(self) -> None:
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy is not installed")

        weighted_sqnorm = register_elementary_function(
            name="weighted_sqnorm_exp2",
            input_dimension=2,
            parameter_dimension=2,
            eval_python=lambda x, w: np.exp2(w[0]) * x[0] * x[0] + w[1] * x[1] * x[1],
            jacobian=lambda x, w: [2 * np.exp2(w[0]) * x[0], 2 * w[1] * x[1]],
            hessian=lambda x, w: [
                [2 * np.exp2(w[0]), 0],
                [0, 2 * w[1]],
            ],
            hvp=lambda x, v, w: [
                2 * np.exp2(w[0]) * v[0],
                2 * w[1] * v[1],
            ],
        )

        x = SXVector.sym("x", 2)
        f = Function("f", [x], [weighted_sqnorm(x, w=[3.0, 3.0])], input_names=["x"], output_names=["y"])

        self.assertEqual(f([1.0, 2.0]), 20.0)
        self.assertEqual(f.gradient(0)([1.0, 2.0]), (16.0, 12.0))
        self.assertEqual(f.hessian(0)([1.0, 2.0]), (16.0, 0.0, 0.0, 6.0))
        self.assertEqual(f.hvp(0)([1.0, 2.0], [3.0, 4.0]), (48.0, 24.0))

    def test_registration_rejects_duplicate_names_and_bad_shapes(self) -> None:
        register_elementary_function(
            name="duplicate_me",
            input_dimension=1,
            jacobian=lambda x: SX.const(1.0),
            hessian=lambda x: SX.const(0.0),
        )

        with self.assertRaises(ValueError):
            register_elementary_function(
                name="duplicate_me",
                input_dimension=1,
                jacobian=lambda x: SX.const(1.0),
                hessian=lambda x: SX.const(0.0),
            )

        with self.assertRaises(TypeError):
            register_elementary_function(
                name="bad_vector_shape",
                input_dimension=2,
                jacobian=lambda x: SX.const(1.0),
                hessian=lambda x: (SXVector((SX.const(1.0), SX.const(0.0))),),
            )

    def test_rust_derivative_codegen_requires_matching_registered_helpers(self) -> None:
        missing_helpers = register_elementary_function(
            name="missing_helpers",
            input_dimension=1,
            parameter_dimension=1,
            parameter_defaults=[1.0],
            eval_python=lambda x, w: x * x + w[0],
            jacobian=lambda x, w: 2 * x,
            hessian=lambda x, w: SX.const(2.0),
            rust_primal="""
fn missing_helpers(x: {{ scalar_type }}, w: &[{{ scalar_type }}]) -> {{ scalar_type }} {
    x * x + w[0]
}
""",
        )

        x = SX.sym("x")
        f = Function("f", [x], [missing_helpers(x)], input_names=["x"], output_names=["y"])

        with self.assertRaises(ValueError):
            _ = f.gradient(0).generate_rust()

        with self.assertRaises(ValueError):
            _ = f.hvp(0).generate_rust()

    def test_vector_hessian_codegen_requires_matching_registered_helper(self) -> None:
        missing_vector_hessian = register_elementary_function(
            name="missing_vector_hessian",
            input_dimension=2,
            parameter_dimension=2,
            parameter_defaults=[1.0, 1.0],
            eval_python=lambda x, w: w[0] * x[0] * x[0] + w[1] * x[1] * x[1],
            jacobian=lambda x, w: SXVector((2 * w[0] * x[0], 2 * w[1] * x[1])),
            hessian=lambda x, w: (
                SXVector((2 * w[0], SX.const(0.0))),
                SXVector((SX.const(0.0), 2 * w[1])),
            ),
            rust_primal="""
fn missing_vector_hessian(
    x: &[{{ scalar_type }}],
    w: &[{{ scalar_type }}],
) -> {{ scalar_type }} {
    w[0] * x[0] * x[0] + w[1] * x[1] * x[1]
}
""",
        )

        x = SXVector.sym("x", 2)
        f = Function(
            "f",
            [x],
            [missing_vector_hessian(x, w=[2.0, 3.0])],
            input_names=["x"],
            output_names=["y"],
        )

        with self.assertRaises(ValueError):
            _ = f.hessian(0).generate_rust()

    def test_registration_rejects_malformed_numeric_hessian_and_hvp_shapes(self) -> None:
        with self.assertRaises(TypeError):
            register_elementary_function(
                name="bad_hessian_shape",
                input_dimension=2,
                parameter_dimension=2,
                parameter_defaults=[1.0, 1.0],
                eval_python=lambda x, w: w[0] * x[0] + w[1] * x[1],
                jacobian=lambda x, w: [w[0], w[1]],
                hessian=lambda x, w: [[0.0, 0.0]],
            )

        with self.assertRaises(TypeError):
            register_elementary_function(
                name="bad_hvp_shape",
                input_dimension=2,
                parameter_dimension=2,
                parameter_defaults=[1.0, 1.0],
                eval_python=lambda x, w: w[0] * x[0] * x[0] + w[1] * x[1] * x[1],
                jacobian=lambda x, w: [2 * w[0] * x[0], 2 * w[1] * x[1]],
                hessian=lambda x, w: [[2 * w[0], 0.0], [0.0, 2 * w[1]]],
                hvp=lambda x, v, w: [2 * w[0] * v[0]],
            )

    def test_custom_hvp_accepts_alternative_argument_order(self) -> None:
        weighted_sqnorm = register_elementary_function(
            name="weighted_sqnorm_alt_hvp_order",
            input_dimension=2,
            parameter_dimension=2,
            parameter_defaults=[1.0, 1.0],
            eval_python=lambda x, w: w[0] * x[0] * x[0] + w[1] * x[1] * x[1],
            jacobian=lambda x, w: [2 * w[0] * x[0], 2 * w[1] * x[1]],
            hessian=lambda x, w: [[2 * w[0], 0.0], [0.0, 2 * w[1]]],
            hvp=lambda x, w, v_x: [2 * w[0] * v_x[0], 2 * w[1] * v_x[1]],
        )

        x = SXVector.sym("x", 2)
        f = Function("f", [x], [weighted_sqnorm(x, w=[2.0, 3.0])], input_names=["x"], output_names=["y"])

        self.assertEqual(f.hvp(0)([1.0, 2.0], [3.0, 4.0]), (12.0, 24.0))

    def test_zero_parameter_scalar_callbacks_can_omit_parameter_argument(self) -> None:
        cubic = register_elementary_function(
            name="cubic_no_params",
            input_dimension=1,
            parameter_dimension=0,
            eval_python=lambda x: x * x * x,
            jacobian=lambda x: 3 * x * x,
            hessian=lambda x: 6 * x,
        )

        x = SX.sym("x")
        f = Function("f", [x], [cubic(x)], input_names=["x"], output_names=["y"])

        self.assertEqual(f(2.0), 8.0)
        self.assertEqual(f.gradient(0)(2.0), 12.0)
        self.assertEqual(f.hvp(0)(2.0, 5.0), 60.0)

    def test_rust_codegen_emits_registered_helpers(self) -> None:
        square_shift = register_elementary_function(
            name="square_shift_codegen",
            input_dimension=1,
            parameter_dimension=1,
            parameter_defaults=[1.0],
            eval_python=lambda x, w: x * x + w[0],
            jacobian=lambda x, w: 2 * x,
            hessian=lambda x, w: SX.const(2.0),
            rust_primal="""
fn square_shift_codegen(x: {{ scalar_type }}, w: &[{{ scalar_type }}]) -> {{ scalar_type }} {
    x * x + w[0]
}
""",
            rust_jacobian="""
fn square_shift_codegen_jacobian(
    x: {{ scalar_type }},
    w: &[{{ scalar_type }}],
) -> {{ scalar_type }} {
    let _ = w;
    2.0_{{ scalar_type }} * x
}
""",
            rust_hvp="""
fn square_shift_codegen_hvp(
    x: {{ scalar_type }},
    v_x: {{ scalar_type }},
    w: &[{{ scalar_type }}],
) -> {{ scalar_type }} {
    let _ = x;
    let _ = w;
    2.0_{{ scalar_type }} * v_x
}
""",
            rust_hessian="""
fn square_shift_codegen_hessian(
    x: {{ scalar_type }},
    w: &[{{ scalar_type }}],
) -> {{ scalar_type }} {
    let _ = x;
    let _ = w;
    2.0_{{ scalar_type }}
}
""",
        )

        x = SX.sym("x")
        f = Function("f", [x], [square_shift(x, w=[3.0])], input_names=["x"], output_names=["y"])

        primal = f.generate_rust()
        gradient = f.gradient(0).generate_rust()
        hvp = f.hvp(0).generate_rust()
        _ = f.hessian(0).generate_rust()

        self.assertIn("fn square_shift_codegen(x: f64, w: &[f64]) -> f64 {", primal.source)
        self.assertIn("square_shift_codegen(x[0], &[3.0_f64])", primal.source)
        self.assertIn("fn square_shift_codegen_jacobian(", gradient.source)
        self.assertIn("square_shift_codegen_jacobian(x[0], &[3.0_f64])", gradient.source)
        self.assertIn("fn square_shift_codegen_hvp(", hvp.source)
        self.assertIn("square_shift_codegen_hvp(x[0], v_x[0], &[3.0_f64])", hvp.source)

    def test_vector_custom_hessian_codegen_uses_flat_helper(self) -> None:
        weighted_sqnorm = register_elementary_function(
            name="weighted_sqnorm_codegen",
            input_dimension=2,
            parameter_dimension=2,
            parameter_defaults=[1.0, 1.0],
            eval_python=lambda x, w: w[0] * x[0] * x[0] + w[1] * x[1] * x[1],
            jacobian=lambda x, w: SXVector((2 * w[0] * x[0], 2 * w[1] * x[1])),
            hessian=lambda x, w: (
                SXVector((2 * w[0], SX.const(0.0))),
                SXVector((SX.const(0.0), 2 * w[1])),
            ),
            rust_primal="""
fn weighted_sqnorm_codegen(
    x: &[{{ scalar_type }}],
    w: &[{{ scalar_type }}],
) -> {{ scalar_type }} {
    w[0] * x[0] * x[0] + w[1] * x[1] * x[1]
}
""",
            rust_hessian="""
fn weighted_sqnorm_codegen_hessian(
    x: &[{{ scalar_type }}],
    w: &[{{ scalar_type }}],
    out: &mut [{{ scalar_type }}],
) {
    let _ = x;
    out[0] = 2.0_{{ scalar_type }} * w[0];
    out[1] = 0.0_{{ scalar_type }};
    out[2] = 0.0_{{ scalar_type }};
    out[3] = 2.0_{{ scalar_type }} * w[1];
}
""",
        )

        x = SXVector.sym("x", 2)
        hessian = Function("f", [x], [weighted_sqnorm(x, w=[2.0, 3.0])], input_names=["x"], output_names=["y"]).hessian(0).generate_rust()

        self.assertIn("fn weighted_sqnorm_codegen_hessian(", hessian.source)
        self.assertIn("weighted_sqnorm_codegen_hessian(x, &[2.0_f64, 3.0_f64], y);", hessian.source)

    def test_vector_custom_hessian_supports_larger_flat_row_major_shape(self) -> None:
        weighted_sqnorm3 = register_elementary_function(
            name="weighted_sqnorm3",
            input_dimension=3,
            parameter_dimension=3,
            parameter_defaults=[1.0, 1.0, 1.0],
            eval_python=lambda x, w: w[0] * x[0] * x[0] + w[1] * x[1] * x[1] + w[2] * x[2] * x[2],
            jacobian=lambda x, w: SXVector((2 * w[0] * x[0], 2 * w[1] * x[1], 2 * w[2] * x[2])),
            hessian=lambda x, w: (
                SXVector((2 * w[0], SX.const(0.0), SX.const(0.0))),
                SXVector((SX.const(0.0), 2 * w[1], SX.const(0.0))),
                SXVector((SX.const(0.0), SX.const(0.0), 2 * w[2])),
            ),
            rust_primal="""
fn weighted_sqnorm3(
    x: &[{{ scalar_type }}],
    w: &[{{ scalar_type }}],
) -> {{ scalar_type }} {
    w[0] * x[0] * x[0] + w[1] * x[1] * x[1] + w[2] * x[2] * x[2]
}
""",
            rust_hessian="""
fn weighted_sqnorm3_hessian(
    x: &[{{ scalar_type }}],
    w: &[{{ scalar_type }}],
    out: &mut [{{ scalar_type }}],
) {
    let _ = x;
    out[0] = 2.0_{{ scalar_type }} * w[0];
    out[1] = 0.0_{{ scalar_type }};
    out[2] = 0.0_{{ scalar_type }};
    out[3] = 0.0_{{ scalar_type }};
    out[4] = 2.0_{{ scalar_type }} * w[1];
    out[5] = 0.0_{{ scalar_type }};
    out[6] = 0.0_{{ scalar_type }};
    out[7] = 0.0_{{ scalar_type }};
    out[8] = 2.0_{{ scalar_type }} * w[2];
}
""",
        )

        x = SXVector.sym("x", 3)
        f = Function("f", [x], [weighted_sqnorm3(x, w=[2.0, 3.0, 4.0])], input_names=["x"], output_names=["y"])

        self.assertEqual(f.hessian(0)([1.0, 2.0, 3.0]), (4.0, 0.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 8.0))

        source = f.hessian(0).generate_rust().source
        self.assertIn("fn weighted_sqnorm3_hessian(", source)
        self.assertIn("weighted_sqnorm3_hessian(x, &[2.0_f64, 3.0_f64, 4.0_f64], y);", source)
