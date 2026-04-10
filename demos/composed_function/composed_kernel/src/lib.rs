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
    let composed_kernel_composed_demo_repeat_0_g_parameter_offsets: [usize; 5] = [0, 2, 4, 6, 8];
    for repeat_index in 0..5 {
        let parameter_offset =
            composed_kernel_composed_demo_repeat_0_g_parameter_offsets[repeat_index];
        composed_kernel_composed_demo_repeat_0_g(
            current_state,
            &parameters[parameter_offset..parameter_offset + 2],
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
        workspace_size: 20,
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
///   packed stage-parameter slice for the composed kernel; symbolic
///   parameter blocks are laid out in forward stage order
///   Expected length: 10.
/// - `jacobian_y`:
///   output slice receiving the Jacobian block for declared result `y`
///   Expected length: 4.
/// - `work`: mutable workspace slice used to store intermediate values
///   while evaluating this kernel. Expected length: at least 20.
pub fn composed_kernel_composed_demo_grad_x(
    x: &[f64],
    parameters: &[f64],
    jacobian_y: &mut [f64],
    work: &mut [f64],
) -> Result<(), GradgenError> {
    if work.len() < 20 {
        return Err(GradgenError::WorkspaceTooSmall("work expected at least 20"));
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
    let (state_history, rest) = work.split_at_mut(10);
    let (state_buffers, rest) = rest.split_at_mut(4);
    let (current_state, _next_state) = state_buffers.split_at_mut(2);
    let (lambda_buffers, stage_work) = rest.split_at_mut(4);
    let (lambda_a, lambda_b) = lambda_buffers.split_at_mut(2);
    current_state.copy_from_slice(x);
    let composed_kernel_composed_demo_repeat_0_g_parameter_offsets: [usize; 5] = [0, 2, 4, 6, 8];
    for repeat_index in 0..5 {
        let parameter_offset =
            composed_kernel_composed_demo_repeat_0_g_parameter_offsets[repeat_index];
        let stage_index = repeat_index;
        let stage_start = stage_index * 2;
        let stage_end = stage_start + 2;
        {
            let next_state = &mut state_history[stage_start..stage_end];
            composed_kernel_composed_demo_repeat_0_g(
                current_state,
                &parameters[parameter_offset..parameter_offset + 2],
                next_state,
                stage_work,
            );
            current_state.copy_from_slice(next_state);
        }
    }
    for output_index in 0..2 {
        lambda_a.fill(0.0_f64);
        lambda_b.fill(0.0_f64);
        lambda_a[output_index] = 1.0_f64;
        let mut current_lambda_is_a = true;
        let composed_kernel_composed_demo_repeat_0_g_parameter_offsets: [usize; 5] =
            [0, 2, 4, 6, 8];
        for repeat_index in (0..5).rev() {
            let parameter_offset =
                composed_kernel_composed_demo_repeat_0_g_parameter_offsets[repeat_index];
            let stage_index = repeat_index;
            if stage_index == 0 {
                if current_lambda_is_a {
                    composed_kernel_composed_demo_repeat_0_g_vjp(
                        x,
                        &parameters[parameter_offset..parameter_offset + 2],
                        &lambda_a[..],
                        lambda_b,
                        stage_work,
                    );
                } else {
                    composed_kernel_composed_demo_repeat_0_g_vjp(
                        x,
                        &parameters[parameter_offset..parameter_offset + 2],
                        &lambda_b[..],
                        lambda_a,
                        stage_work,
                    );
                }
            } else {
                let prev_start = (stage_index - 1) * 2;
                let prev_end = prev_start + 2;
                if current_lambda_is_a {
                    composed_kernel_composed_demo_repeat_0_g_vjp(
                        &state_history[prev_start..prev_end],
                        &parameters[parameter_offset..parameter_offset + 2],
                        &lambda_a[..],
                        lambda_b,
                        stage_work,
                    );
                } else {
                    composed_kernel_composed_demo_repeat_0_g_vjp(
                        &state_history[prev_start..prev_end],
                        &parameters[parameter_offset..parameter_offset + 2],
                        &lambda_b[..],
                        lambda_a,
                        stage_work,
                    );
                }
            }
            current_lambda_is_a = !current_lambda_is_a;
        }
        let gradient_row = if current_lambda_is_a {
            &lambda_a[..]
        } else {
            &lambda_b[..]
        };
        let row_start = output_index * 2;
        let row_end = row_start + 2;
        jacobian_y[row_start..row_end].copy_from_slice(gradient_row);
    }
    Ok(())
}

fn composed_kernel_composed_demo_repeat_0_g_vjp(
    _state: &[f64],
    p: &[f64],
    cotangent_next_state: &[f64],
    vjp_state: &mut [f64],
    work: &mut [f64],
) {
    work[0] = 0.9_f64 * cotangent_next_state[0];
    work[1] = 0.1_f64 * cotangent_next_state[1];
    work[1] *= p[1];
    vjp_state[0] = work[0];
    vjp_state[1] = work[1];
}
