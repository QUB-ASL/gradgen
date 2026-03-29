#![no_std]

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
        input_names: &[
            "x",
            "parameters",
        ],
        input_sizes: &[
            2,
            11,
        ],
        output_names: &[
            "y",
        ],
        output_sizes: &[
            1,
        ],
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
///   parameter blocks are laid out in forward stage order and the
///   terminal block is stored last
///   Expected length: 11.
/// - `y`:
///   primal output slice for the declared result `y`
///   Expected length: 1.
/// - `work`: mutable workspace slice used to store intermediate values
///   while evaluating this kernel. Expected length: at least 6.
pub fn composed_kernel_composed_demo_f(x: &[f64], parameters: &[f64], y: &mut [f64], work: &mut [f64]) {
    assert!(work.len() >= 6);
    assert_eq!(x.len(), 2);
    assert_eq!(parameters.len(), 11);
    assert_eq!(y.len(), 1);
    let (state_buffers, stage_work) = work.split_at_mut(4);
    let (current_state, next_state) = state_buffers.split_at_mut(2);
    current_state.copy_from_slice(x);
    for repeat_index in 0..5 {
        composed_kernel_composed_demo_f_repeat_0_G(current_state, &parameters[0 + (repeat_index * 2)..0 + ((repeat_index + 1) * 2)], next_state, stage_work);
        current_state.copy_from_slice(next_state);
    }
    composed_kernel_composed_demo_f_terminal_h(current_state, &parameters[10..11], y, stage_work);
}

fn composed_kernel_composed_demo_f_repeat_0_G(state: &[f64], p: &[f64], next_state: &mut [f64], work: &mut [f64]) {
    assert!(work.len() >= 2);
    assert_eq!(state.len(), 2);
    assert_eq!(p.len(), 2);
    assert_eq!(next_state.len(), 2);
    work[0] = 0.9_f64 * state[0];
    work[0] = work[0] + p[0];
    work[1] = 0.1_f64 * state[1];
    work[1] = work[1] * p[1];
    next_state[0] = work[0];
    next_state[1] = work[1];
}

fn composed_kernel_composed_demo_f_terminal_h(state: &[f64], pf: &[f64], y: &mut [f64], work: &mut [f64]) {
    assert!(work.len() >= 1);
    assert_eq!(state.len(), 2);
    assert_eq!(pf.len(), 1);
    assert_eq!(y.len(), 1);
    work[0] = 2.0_f64 * state[0];
    work[0] = work[0] - state[1];
    work[0] = work[0] + pf[0];
    y[0] = work[0];
}

/// Return metadata describing [`composed_kernel_composed_demo_grad_x`].
pub fn composed_kernel_composed_demo_grad_x_meta() -> FunctionMetadata {
    FunctionMetadata {
        function_name: "composed_kernel_composed_demo_grad_x",
        workspace_size: 18,
        input_names: &[
            "x",
            "parameters",
        ],
        input_sizes: &[
            2,
            11,
        ],
        output_names: &[
            "y",
        ],
        output_sizes: &[
            2,
        ],
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
///   parameter blocks are laid out in forward stage order and the
///   terminal block is stored last
///   Expected length: 11.
/// - `y`:
///   primal output slice for the declared result `y`
///   Expected length: 2.
/// - `work`: mutable workspace slice used to store intermediate values
///   while evaluating this kernel. Expected length: at least 18.
pub fn composed_kernel_composed_demo_grad_x(x: &[f64], parameters: &[f64], y: &mut [f64], work: &mut [f64]) {
    assert!(work.len() >= 18);
    assert_eq!(x.len(), 2);
    assert_eq!(parameters.len(), 11);
    assert_eq!(y.len(), 2);
    let (state_history, rest) = work.split_at_mut(10);
    let (current_state, rest) = rest.split_at_mut(2);
    let (lambda_buffers, stage_work) = rest.split_at_mut(4);
    let (lambda_a, lambda_b) = lambda_buffers.split_at_mut(2);
    current_state.copy_from_slice(x);
    for repeat_index in 0..5 {
        let stage_index = 0 + repeat_index;
        let stage_start = stage_index * 2;
        let stage_end = stage_start + 2;
        {
            let next_state = &mut state_history[stage_start..stage_end];
            composed_kernel_composed_demo_grad_x_repeat_0_G(current_state, &parameters[0 + (repeat_index * 2)..0 + ((repeat_index + 1) * 2)], next_state, stage_work);
            current_state.copy_from_slice(next_state);
        }
    }
    composed_kernel_composed_demo_grad_x_terminal_h_grad(current_state, &parameters[10..11], lambda_a, stage_work);
    let mut current_lambda_is_a = true;
    for repeat_index in (0..5).rev() {
        let stage_index = 0 + repeat_index;
        if stage_index == 0 {
            if current_lambda_is_a {
                composed_kernel_composed_demo_grad_x_repeat_0_G_vjp(x, &parameters[0 + (repeat_index * 2)..0 + ((repeat_index + 1) * 2)], &lambda_a[..], lambda_b, stage_work);
            } else {
                composed_kernel_composed_demo_grad_x_repeat_0_G_vjp(x, &parameters[0 + (repeat_index * 2)..0 + ((repeat_index + 1) * 2)], &lambda_b[..], lambda_a, stage_work);
            }
        } else {
            let prev_start = (stage_index - 1) * 2;
            let prev_end = prev_start + 2;
            if current_lambda_is_a {
                composed_kernel_composed_demo_grad_x_repeat_0_G_vjp(&state_history[prev_start..prev_end], &parameters[0 + (repeat_index * 2)..0 + ((repeat_index + 1) * 2)], &lambda_a[..], lambda_b, stage_work);
            } else {
                composed_kernel_composed_demo_grad_x_repeat_0_G_vjp(&state_history[prev_start..prev_end], &parameters[0 + (repeat_index * 2)..0 + ((repeat_index + 1) * 2)], &lambda_b[..], lambda_a, stage_work);
            }
        }
        current_lambda_is_a = !current_lambda_is_a;
    }
    let gradient = if current_lambda_is_a { &lambda_a[..] } else { &lambda_b[..] };
    y.copy_from_slice(gradient);
}

fn composed_kernel_composed_demo_grad_x_repeat_0_G(state: &[f64], p: &[f64], next_state: &mut [f64], work: &mut [f64]) {
    assert!(work.len() >= 2);
    assert_eq!(state.len(), 2);
    assert_eq!(p.len(), 2);
    assert_eq!(next_state.len(), 2);
    work[0] = 0.9_f64 * state[0];
    work[0] = work[0] + p[0];
    work[1] = 0.1_f64 * state[1];
    work[1] = work[1] * p[1];
    next_state[0] = work[0];
    next_state[1] = work[1];
}

fn composed_kernel_composed_demo_grad_x_repeat_0_G_vjp(state: &[f64], p: &[f64], cotangent_next_state: &[f64], vjp_state: &mut [f64], work: &mut [f64]) {
    assert!(work.len() >= 2);
    assert_eq!(state.len(), 2);
    assert_eq!(p.len(), 2);
    assert_eq!(cotangent_next_state.len(), 2);
    assert_eq!(vjp_state.len(), 2);
    work[0] = 0.0_f64 + cotangent_next_state[0];
    work[0] = 0.0_f64 + work[0];
    work[0] = 0.9_f64 * work[0];
    work[0] = 0.0_f64 + work[0];
    work[0] = 0.0_f64 + work[0];
    work[1] = 0.0_f64 + cotangent_next_state[1];
    work[1] = work[1] * p[1];
    work[1] = 0.0_f64 + work[1];
    work[1] = 0.1_f64 * work[1];
    work[1] = 0.0_f64 + work[1];
    work[1] = 0.0_f64 + work[1];
    vjp_state[0] = work[0];
    vjp_state[1] = work[1];
}

fn composed_kernel_composed_demo_grad_x_terminal_h_grad(state: &[f64], pf: &[f64], y: &mut [f64], work: &mut [f64]) {
    assert!(work.len() >= 2);
    assert_eq!(state.len(), 2);
    assert_eq!(pf.len(), 1);
    assert_eq!(y.len(), 2);
    work[0] = 0.0_f64 + 1.0_f64;
    work[0] = 0.0_f64 + work[0];
    work[1] = 0.0_f64 + work[0];
    work[1] = 2.0_f64 * work[1];
    work[1] = 0.0_f64 + work[1];
    work[0] = -work[0];
    work[0] = 0.0_f64 + work[0];
    y[0] = work[1];
    y[1] = work[0];
}
