#![no_std]

/// Workspace length required by [`eval_kernel`].
pub const EVAL_KERNEL_WORK_SIZE: usize = 3;

/// Return the workspace length required by [`eval_kernel`].
pub fn eval_kernel_work_size() -> usize {
    EVAL_KERNEL_WORK_SIZE
}

/// Return the number of declared inputs of [`eval_kernel`].
pub fn eval_kernel_num_inputs() -> usize {
    1
}

/// Length of the `x` input slice for [`eval_kernel`].
pub const EVAL_KERNEL_INPUT_0_SIZE: usize = 1;

/// Return the length of the `x` input slice for [`eval_kernel`].
pub fn eval_kernel_input_0_size() -> usize {
    EVAL_KERNEL_INPUT_0_SIZE
}

/// Return the number of declared outputs of [`eval_kernel`].
pub fn eval_kernel_num_outputs() -> usize {
    1
}

/// Length of the `y` output slice for [`eval_kernel`].
pub const EVAL_KERNEL_OUTPUT_0_SIZE: usize = 1;

/// Return the length of the `y` output slice for [`eval_kernel`].
pub fn eval_kernel_output_0_size() -> usize {
    EVAL_KERNEL_OUTPUT_0_SIZE
}

/// Evaluate the generated symbolic function `eval_kernel`.
///
/// Inputs are passed as immutable slices, outputs are written into mutable slices,
/// and intermediate values are stored in `work`.
/// All numeric slices use the `f32` scalar type.
pub fn eval_kernel(x: &[f32], y: &mut [f32], work: &mut [f32]) {
    assert!(work.len() >= 3);
    assert_eq!(x.len(), 1);
    assert_eq!(y.len(), 1);
    work[0] = libm::sinf(x[0]);
    work[1] = work[0] * x[0];
    work[2] = 1.0_f32 + work[1];
    y[0] = work[2];
}
