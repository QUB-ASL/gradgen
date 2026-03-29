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

/// Return metadata describing [`composed_demo`].
pub fn composed_demo_meta() -> FunctionMetadata {
    FunctionMetadata {
        function_name: "composed_demo",
        workspace_size: 6,
        input_names: &[
            "x",
            "parameters",
        ],
        input_sizes: &[
            2,
            17,
        ],
        output_names: &[
            "y",
        ],
        output_sizes: &[
            1,
        ],
    }
}

/// Evaluate the generated symbolic function `composed_demo`.
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
///   Expected length: 17.
/// - `y`:
///   primal output slice for the declared result `y`
///   Expected length: 1.
/// - `work`: mutable workspace slice used to store intermediate values
///   while evaluating this kernel. Expected length: at least 6.
pub fn composed_demo(x: &[f64], parameters: &[f64], y: &mut [f64], work: &mut [f64]) {
    assert!(work.len() >= 6);
    assert_eq!(x.len(), 2);
    assert_eq!(parameters.len(), 17);
    assert_eq!(y.len(), 1);
    let (state_buffers, stage_work) = work.split_at_mut(4);
    let (current_state, next_state) = state_buffers.split_at_mut(2);
    current_state.copy_from_slice(x);
    for repeat_index in 0..8 {
        composed_demo_repeat_0_G(current_state, &parameters[0 + (repeat_index * 2)..0 + ((repeat_index + 1) * 2)], next_state, stage_work);
        current_state.copy_from_slice(next_state);
    }
    composed_demo_terminal_h(current_state, &parameters[16..17], y, stage_work);
}

fn composed_demo_repeat_0_G(state: &[f64], p: &[f64], next_state: &mut [f64], work: &mut [f64]) {
    assert!(work.len() >= 2);
    assert_eq!(state.len(), 2);
    assert_eq!(p.len(), 2);
    assert_eq!(next_state.len(), 2);
    work[0] = p[0] + state[0];
    work[1] = p[1] * state[1];
    next_state[0] = work[0];
    next_state[1] = work[1];
}

fn composed_demo_terminal_h(state: &[f64], pf: &[f64], y: &mut [f64], work: &mut [f64]) {
    assert!(work.len() >= 1);
    assert_eq!(state.len(), 2);
    assert_eq!(pf.len(), 1);
    assert_eq!(y.len(), 1);
    work[0] = state[0] + state[1];
    work[0] = work[0] + pf[0];
    y[0] = work[0];
}
