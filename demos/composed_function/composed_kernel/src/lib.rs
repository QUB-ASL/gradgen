#![no_std]
#![forbid(unsafe_code)]

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GradgenError {
    WorkspaceTooSmall(&'static str),
    InputTooSmall(&'static str),
    OutputTooSmall(&'static str),
}

/// Metadata describing a generated Rust function.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FunctionMetadata {
    /// Generated Rust function name.
    pub function_name: &'static str,
    /// Minimum required length of the mutable workspace slice.
    pub workspace_size: usize,
    /// Declared input names.
    pub input_names: &'static [&'static str],
    /// Declared input slice lengths.
    pub input_sizes: &'static [usize],
    /// Declared output names.
    pub output_names: &'static [&'static str],
    /// Declared output slice lengths.
    pub output_sizes: &'static [usize],
}

/// Return metadata describing [`composed_kernel_composed_demo_f`].
pub fn composed_kernel_composed_demo_f_meta() -> FunctionMetadata {
    FunctionMetadata {
        function_name: "composed_kernel_composed_demo_f",
        workspace_size: 6,
        input_names: &["x", "parameters"],
        input_sizes: &[2, 10],
        output_names: &["y"],
        output_sizes: &[2],
    }
}

/// Evaluate the generated symbolic function `composed_kernel_composed_demo_f`.
///
/// All numeric slices use the `f64` scalar type.
///
/// Arguments:
/// - `x`:
///   input slice for the declared argument `x`
///   Expected length: 2.
/// - `parameters`:
///   packed stage-parameter slice for the composed kernel; symbolic
///   parameter blocks are laid out in forward stage order
///   Expected length: 10.
/// - `y`:
///   primal output slice for the declared result `y`
///   Expected length: 2.
/// - `work`: mutable workspace slice used to store intermediate values
///   while evaluating this kernel. Expected length: at least 6.
pub fn composed_kernel_composed_demo_f(
    x: &[f64],
    parameters: &[f64],
    y: &mut [f64],
    work: &mut [f64],
) -> Result<(), GradgenError> {
    if work.len() < 6 {
        return Err(GradgenError::WorkspaceTooSmall("work expected at least 6"));
    };
    if x.len() != 2 {
        return Err(GradgenError::InputTooSmall("x expected length 2"));
    };
    if parameters.len() != 10 {
        return Err(GradgenError::InputTooSmall("parameters expected length 10"));
    };
    if y.len() != 2 {
        return Err(GradgenError::OutputTooSmall("y expected length 2"));
    };
    let (state_buffers, stage_work) = work.split_at_mut(4);
    let (current_state, next_state) = state_buffers.split_at_mut(2);
    current_state.copy_from_slice(x);
    for repeat_index in 0..5 {
        composed_kernel_composed_demo_repeat_0_g(
            current_state,
            &parameters[(repeat_index * 2)..((repeat_index + 1) * 2)],
            next_state,
            stage_work,
        );
        current_state.copy_from_slice(next_state);
    }
    y.copy_from_slice(current_state);
    Ok(())
}

fn composed_kernel_composed_demo_repeat_0_g(
    state: &[f64],
    p: &[f64],
    next_state: &mut [f64],
    work: &mut [f64],
) {
    work[0] = 0.9_f64 * state[0];
    work[0] += p[0];
    work[1] = 0.1_f64 * state[1];
    work[1] *= p[1];
    next_state[0] = work[0];
    next_state[1] = work[1];
}

/// Return metadata describing [`composed_kernel_composed_demo_grad_x`].
pub fn composed_kernel_composed_demo_grad_x_meta() -> FunctionMetadata {
    FunctionMetadata {
        function_name: "composed_kernel_composed_demo_grad_x",
        workspace_size: 1,
        input_names: &["x", "parameters"],
        input_sizes: &[2, 10],
        output_names: &["jacobian_y"],
        output_sizes: &[4],
    }
}

/// Evaluate the generated symbolic function `composed_kernel_composed_demo_grad_x`.
///
/// All numeric slices use the `f64` scalar type.
///
/// Arguments:
/// - `x`:
///   input slice for the declared argument `x`
///   Expected length: 2.
/// - `parameters`:
///   input slice for the declared argument `parameters`
///   Expected length: 10.
/// - `jacobian_y`:
///   output slice receiving the Jacobian block for declared result `y`
///   Expected length: 4.
/// - `work`: mutable workspace slice used to store intermediate values
///   while evaluating this kernel. Expected length: at least 1.
pub fn composed_kernel_composed_demo_grad_x(
    x: &[f64],
    parameters: &[f64],
    jacobian_y: &mut [f64],
    work: &mut [f64],
) -> Result<(), GradgenError> {
    if work.is_empty() {
        return Err(GradgenError::WorkspaceTooSmall("work expected at least 1"));
    };
    if x.len() != 2 {
        return Err(GradgenError::InputTooSmall("x expected length 2"));
    };
    if parameters.len() != 10 {
        return Err(GradgenError::InputTooSmall("parameters expected length 10"));
    };
    if jacobian_y.len() != 4 {
        return Err(GradgenError::OutputTooSmall("jacobian_y expected length 4"));
    };
    work[0] = 1.0000000000000004e-05_f64 * parameters[7];
    work[0] *= parameters[9];
    work[0] *= parameters[5];
    work[0] *= parameters[3];
    work[0] *= parameters[1];
    jacobian_y[0] = 0.5904900000000002_f64;
    jacobian_y[1] = 0.0_f64;
    jacobian_y[2] = 0.0_f64;
    jacobian_y[3] = work[0];
    Ok(())
}
