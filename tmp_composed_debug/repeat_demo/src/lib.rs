const REPEAT_DEMO_REPEAT_0_PARAMS: [[f64; 2]; 3] = [[1.0_f64, 2.0_f64], [3.0_f64, 4.0_f64], [5.0_f64, 6.0_f64]];
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

/// Return metadata describing [`repeat_demo`].
pub fn repeat_demo_meta() -> FunctionMetadata {
    FunctionMetadata {
        function_name: "repeat_demo",
        workspace_size: 6,
        input_names: &[
            "x",
        ],
        input_sizes: &[
            2,
        ],
        output_names: &[
            "y",
        ],
        output_sizes: &[
            1,
        ],
    }
}

/// Evaluate the generated symbolic function `repeat_demo`.
///
/// All numeric slices use the `f64` scalar type.
///
/// Arguments:
/// - `x`:
///   input slice for the declared argument `x`
///   Expected length: 2.
/// - `y`:
///   primal output slice for the declared result `y`
///   Expected length: 1.
/// - `work`: mutable workspace slice used to store intermediate values
///   while evaluating this kernel. Expected length: at least 6.
pub fn repeat_demo(x: &[f64], y: &mut [f64], work: &mut [f64]) {
    assert!(work.len() >= 6);
    assert_eq!(x.len(), 2);
    assert_eq!(y.len(), 1);
    let (state_buffers, stage_work) = work.split_at_mut(4);
    let (current_state, next_state) = state_buffers.split_at_mut(2);
    current_state.copy_from_slice(x);
    for repeat_index in 0..3 {
        repeat_demo_repeat_0_G(current_state, &REPEAT_DEMO_REPEAT_0_PARAMS[repeat_index], next_state, stage_work);
        current_state.copy_from_slice(next_state);
    }
    repeat_demo_terminal_h(current_state, &[7.0_f64], y, stage_work);
}

fn repeat_demo_repeat_0_G(state: &[f64], p: &[f64], next_state: &mut [f64], work: &mut [f64]) {
    assert!(work.len() >= 2);
    assert_eq!(state.len(), 2);
    assert_eq!(p.len(), 2);
    assert_eq!(next_state.len(), 2);
    work[0] = p[0] + state[0];
    work[1] = p[1] * state[1];
    next_state[0] = work[0];
    next_state[1] = work[1];
}

fn repeat_demo_terminal_h(state: &[f64], pf: &[f64], y: &mut [f64], work: &mut [f64]) {
    assert!(work.len() >= 1);
    assert_eq!(state.len(), 2);
    assert_eq!(pf.len(), 1);
    assert_eq!(y.len(), 1);
    work[0] = state[0] + state[1];
    work[0] = work[0] + pf[0];
    y[0] = work[0];
}

#[cfg(test)]
mod debug_test {
    use super::*;
    #[test]
    fn print_value() {
        let x = [1.0_f64, 2.0_f64];
        let mut y = [0.0_f64; 1];
        let mut work = [0.0_f64; 6];
        repeat_demo(&x, &mut y, &mut work);
        println!("y={:?} work={:?}", y, work);
        assert_eq!(y[0], 113.0_f64);
    }
}
