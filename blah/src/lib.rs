#![no_std]

/// Workspace length required by [`square_plus_one`].
pub const SQUARE_PLUS_ONE_WORK_SIZE: usize = 2;

/// Return the workspace length required by [`square_plus_one`].
pub fn square_plus_one_work_size() -> usize {
    SQUARE_PLUS_ONE_WORK_SIZE
}

/// Return the number of declared inputs of [`square_plus_one`].
pub fn square_plus_one_num_inputs() -> usize {
    1
}

/// Length of the `x` input slice for [`square_plus_one`].
pub const SQUARE_PLUS_ONE_INPUT_0_SIZE: usize = 1;

/// Return the length of the `x` input slice for [`square_plus_one`].
pub fn square_plus_one_input_0_size() -> usize {
    SQUARE_PLUS_ONE_INPUT_0_SIZE
}

/// Return the number of declared outputs of [`square_plus_one`].
pub fn square_plus_one_num_outputs() -> usize {
    1
}

/// Length of the `y` output slice for [`square_plus_one`].
pub const SQUARE_PLUS_ONE_OUTPUT_0_SIZE: usize = 1;

/// Return the length of the `y` output slice for [`square_plus_one`].
pub fn square_plus_one_output_0_size() -> usize {
    SQUARE_PLUS_ONE_OUTPUT_0_SIZE
}

/// Evaluate the generated symbolic function `square_plus_one`.
///
/// Inputs are passed as immutable slices, outputs are written into mutable slices,
/// and intermediate values are stored in `work`.
/// All numeric slices use the `f32` scalar type.
pub fn square_plus_one(x: &[f32], y: &mut [f32], work: &mut [f32]) {
    assert!(work.len() >= 2);
    assert_eq!(x.len(), 1);
    assert_eq!(y.len(), 1);
    work[0] = x[0] * x[0];
    work[1] = 1.0_f32 + work[0];
    y[0] = work[1];
}

/// Workspace length required by [`square_plus_one_gradient_x`].
pub const SQUARE_PLUS_ONE_GRADIENT_X_WORK_SIZE: usize = 5;

/// Return the workspace length required by [`square_plus_one_gradient_x`].
pub fn square_plus_one_gradient_x_work_size() -> usize {
    SQUARE_PLUS_ONE_GRADIENT_X_WORK_SIZE
}

/// Return the number of declared inputs of [`square_plus_one_gradient_x`].
pub fn square_plus_one_gradient_x_num_inputs() -> usize {
    1
}

/// Length of the `x` input slice for [`square_plus_one_gradient_x`].
pub const SQUARE_PLUS_ONE_GRADIENT_X_INPUT_0_SIZE: usize = 1;

/// Return the length of the `x` input slice for [`square_plus_one_gradient_x`].
pub fn square_plus_one_gradient_x_input_0_size() -> usize {
    SQUARE_PLUS_ONE_GRADIENT_X_INPUT_0_SIZE
}

/// Return the number of declared outputs of [`square_plus_one_gradient_x`].
pub fn square_plus_one_gradient_x_num_outputs() -> usize {
    1
}

/// Length of the `y` output slice for [`square_plus_one_gradient_x`].
pub const SQUARE_PLUS_ONE_GRADIENT_X_OUTPUT_0_SIZE: usize = 1;

/// Return the length of the `y` output slice for [`square_plus_one_gradient_x`].
pub fn square_plus_one_gradient_x_output_0_size() -> usize {
    SQUARE_PLUS_ONE_GRADIENT_X_OUTPUT_0_SIZE
}

/// Evaluate the generated symbolic function `square_plus_one_gradient_x`.
///
/// Inputs are passed as immutable slices, outputs are written into mutable slices,
/// and intermediate values are stored in `work`.
/// All numeric slices use the `f32` scalar type.
pub fn square_plus_one_gradient_x(x: &[f32], y: &mut [f32], work: &mut [f32]) {
    assert!(work.len() >= 5);
    assert_eq!(x.len(), 1);
    assert_eq!(y.len(), 1);
    work[0] = 0.0_f32 + 1.0_f32;
    work[1] = 0.0_f32 + work[0];
    work[2] = work[1] * x[0];
    work[3] = 0.0_f32 + work[2];
    work[4] = work[3] + work[2];
    y[0] = work[4];
}

/// Workspace length required by [`square_plus_one_hvp_x`].
pub const SQUARE_PLUS_ONE_HVP_X_WORK_SIZE: usize = 9;

/// Return the workspace length required by [`square_plus_one_hvp_x`].
pub fn square_plus_one_hvp_x_work_size() -> usize {
    SQUARE_PLUS_ONE_HVP_X_WORK_SIZE
}

/// Return the number of declared inputs of [`square_plus_one_hvp_x`].
pub fn square_plus_one_hvp_x_num_inputs() -> usize {
    2
}

/// Length of the `x` input slice for [`square_plus_one_hvp_x`].
pub const SQUARE_PLUS_ONE_HVP_X_INPUT_0_SIZE: usize = 1;

/// Return the length of the `x` input slice for [`square_plus_one_hvp_x`].
pub fn square_plus_one_hvp_x_input_0_size() -> usize {
    SQUARE_PLUS_ONE_HVP_X_INPUT_0_SIZE
}

/// Length of the `v_x` input slice for [`square_plus_one_hvp_x`].
pub const SQUARE_PLUS_ONE_HVP_X_INPUT_1_SIZE: usize = 1;

/// Return the length of the `v_x` input slice for [`square_plus_one_hvp_x`].
pub fn square_plus_one_hvp_x_input_1_size() -> usize {
    SQUARE_PLUS_ONE_HVP_X_INPUT_1_SIZE
}

/// Return the number of declared outputs of [`square_plus_one_hvp_x`].
pub fn square_plus_one_hvp_x_num_outputs() -> usize {
    1
}

/// Length of the `y` output slice for [`square_plus_one_hvp_x`].
pub const SQUARE_PLUS_ONE_HVP_X_OUTPUT_0_SIZE: usize = 1;

/// Return the length of the `y` output slice for [`square_plus_one_hvp_x`].
pub fn square_plus_one_hvp_x_output_0_size() -> usize {
    SQUARE_PLUS_ONE_HVP_X_OUTPUT_0_SIZE
}

/// Evaluate the generated symbolic function `square_plus_one_hvp_x`.
///
/// Inputs are passed as immutable slices, outputs are written into mutable slices,
/// and intermediate values are stored in `work`.
/// All numeric slices use the `f32` scalar type.
pub fn square_plus_one_hvp_x(x: &[f32], v_x: &[f32], y: &mut [f32], work: &mut [f32]) {
    assert!(work.len() >= 9);
    assert_eq!(x.len(), 1);
    assert_eq!(v_x.len(), 1);
    assert_eq!(y.len(), 1);
    work[0] = 0.0_f32 + 0.0_f32;
    work[1] = 0.0_f32 + work[0];
    work[2] = work[1] * x[0];
    work[3] = 0.0_f32 + 1.0_f32;
    work[4] = 0.0_f32 + work[3];
    work[5] = work[4] * v_x[0];
    work[6] = work[2] + work[5];
    work[7] = 0.0_f32 + work[6];
    work[8] = work[7] + work[6];
    y[0] = work[8];
}

/// Workspace length required by [`square_plus_one_primal_jacobian_x`].
pub const SQUARE_PLUS_ONE_PRIMAL_JACOBIAN_X_WORK_SIZE: usize = 3;

/// Return the workspace length required by [`square_plus_one_primal_jacobian_x`].
pub fn square_plus_one_primal_jacobian_x_work_size() -> usize {
    SQUARE_PLUS_ONE_PRIMAL_JACOBIAN_X_WORK_SIZE
}

/// Return the number of declared inputs of [`square_plus_one_primal_jacobian_x`].
pub fn square_plus_one_primal_jacobian_x_num_inputs() -> usize {
    1
}

/// Length of the `x` input slice for [`square_plus_one_primal_jacobian_x`].
pub const SQUARE_PLUS_ONE_PRIMAL_JACOBIAN_X_INPUT_0_SIZE: usize = 1;

/// Return the length of the `x` input slice for [`square_plus_one_primal_jacobian_x`].
pub fn square_plus_one_primal_jacobian_x_input_0_size() -> usize {
    SQUARE_PLUS_ONE_PRIMAL_JACOBIAN_X_INPUT_0_SIZE
}

/// Return the number of declared outputs of [`square_plus_one_primal_jacobian_x`].
pub fn square_plus_one_primal_jacobian_x_num_outputs() -> usize {
    2
}

/// Length of the `y` output slice for [`square_plus_one_primal_jacobian_x`].
pub const SQUARE_PLUS_ONE_PRIMAL_JACOBIAN_X_OUTPUT_0_SIZE: usize = 1;

/// Return the length of the `y` output slice for [`square_plus_one_primal_jacobian_x`].
pub fn square_plus_one_primal_jacobian_x_output_0_size() -> usize {
    SQUARE_PLUS_ONE_PRIMAL_JACOBIAN_X_OUTPUT_0_SIZE
}

/// Length of the `jacobian_y` output slice for [`square_plus_one_primal_jacobian_x`].
pub const SQUARE_PLUS_ONE_PRIMAL_JACOBIAN_X_OUTPUT_1_SIZE: usize = 1;

/// Return the length of the `jacobian_y` output slice for [`square_plus_one_primal_jacobian_x`].
pub fn square_plus_one_primal_jacobian_x_output_1_size() -> usize {
    SQUARE_PLUS_ONE_PRIMAL_JACOBIAN_X_OUTPUT_1_SIZE
}

/// Evaluate the generated symbolic function `square_plus_one_primal_jacobian_x`.
///
/// Inputs are passed as immutable slices, outputs are written into mutable slices,
/// and intermediate values are stored in `work`.
/// All numeric slices use the `f32` scalar type.
pub fn square_plus_one_primal_jacobian_x(x: &[f32], y: &mut [f32], jacobian_y: &mut [f32], work: &mut [f32]) {
    assert!(work.len() >= 3);
    assert_eq!(x.len(), 1);
    assert_eq!(y.len(), 1);
    assert_eq!(jacobian_y.len(), 1);
    work[0] = libm::powf(x[0], 2.0_f32);
    work[1] = 1.0_f32 + work[0];
    work[2] = 2.0_f32 * x[0];
    y[0] = work[1];
    jacobian_y[0] = work[2];
}
