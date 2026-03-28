/// Workspace length required by [`square_plus_one_jacobian_x`].
pub const SQUARE_PLUS_ONE_JACOBIAN_X_WORK_SIZE: usize = 1;

/// Return the workspace length required by [`square_plus_one_jacobian_x`].
pub fn square_plus_one_jacobian_x_work_size() -> usize {
    SQUARE_PLUS_ONE_JACOBIAN_X_WORK_SIZE
}

/// Return the number of declared inputs of [`square_plus_one_jacobian_x`].
pub fn square_plus_one_jacobian_x_num_inputs() -> usize {
    1
}

/// Length of the `x` input slice for [`square_plus_one_jacobian_x`].
pub const SQUARE_PLUS_ONE_JACOBIAN_X_INPUT_0_SIZE: usize = 1;

/// Return the length of the `x` input slice for [`square_plus_one_jacobian_x`].
pub fn square_plus_one_jacobian_x_input_0_size() -> usize {
    SQUARE_PLUS_ONE_JACOBIAN_X_INPUT_0_SIZE
}

/// Return the number of declared outputs of [`square_plus_one_jacobian_x`].
pub fn square_plus_one_jacobian_x_num_outputs() -> usize {
    1
}

/// Length of the `y` output slice for [`square_plus_one_jacobian_x`].
pub const SQUARE_PLUS_ONE_JACOBIAN_X_OUTPUT_0_SIZE: usize = 1;

/// Return the length of the `y` output slice for [`square_plus_one_jacobian_x`].
pub fn square_plus_one_jacobian_x_output_0_size() -> usize {
    SQUARE_PLUS_ONE_JACOBIAN_X_OUTPUT_0_SIZE
}

/// Evaluate the generated symbolic function `square_plus_one_jacobian_x`.
///
/// Inputs are passed as immutable slices, outputs are written into mutable slices,
/// and intermediate values are stored in `work`.
/// All numeric slices use the `f64` scalar type.
pub fn square_plus_one_jacobian_x(x: &[f64], y: &mut [f64], work: &mut [f64]) {
    assert!(work.len() >= 1);
    assert_eq!(x.len(), 1);
    assert_eq!(y.len(), 1);
    work[0] = 2.0_f64 * x[0];
    y[0] = work[0];
}
