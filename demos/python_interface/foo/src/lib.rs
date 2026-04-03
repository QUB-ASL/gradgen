#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GradgenError {
    WorkspaceTooSmall(&'static str),
    InputTooSmall(&'static str),
    OutputTooSmall(&'static str),
}

fn norm2sq(values: &[f64]) -> f64 {
    values.iter().map(|value| *value * *value).sum()
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

/// Return metadata describing [`energy`].
pub fn energy_meta() -> FunctionMetadata {
    FunctionMetadata {
        function_name: "energy",
        workspace_size: 2,
        input_names: &["x", "w"],
        input_sizes: &[2, 1],
        output_names: &["cost", "state"],
        output_sizes: &[1, 1],
    }
}

/// Evaluate the generated symbolic function `energy`.
///
/// All numeric slices use the `f64` scalar type.
///
/// Arguments:
/// - `x`:
///   input slice for the declared argument `x`
///   Expected length: 2.
/// - `w`:
///   input slice for the declared argument `w`
///   Expected length: 1.
/// - `cost`:
///   primal output slice for the declared result `cost`
///   Expected length: 1.
/// - `state`:
///   primal output slice for the declared result `state`
///   Expected length: 1.
/// - `work`: mutable workspace slice used to store intermediate values
///   while evaluating this kernel. Expected length: at least 2.
pub fn energy(
    x: &[f64],
    w: &[f64],
    cost: &mut [f64],
    state: &mut [f64],
    work: &mut [f64],
) -> Result<(), GradgenError> {
    if work.len() < 2 {
        return Err(GradgenError::WorkspaceTooSmall("work expected at least 2"));
    };
    if x.len() != 2 {
        return Err(GradgenError::InputTooSmall("x expected length 2"));
    };
    if w.len() != 1 {
        return Err(GradgenError::InputTooSmall("w expected length 1"));
    };
    if cost.len() != 1 {
        return Err(GradgenError::OutputTooSmall("cost expected length 1"));
    };
    if state.len() != 1 {
        return Err(GradgenError::OutputTooSmall("state expected length 1"));
    };
    work[0] = norm2sq(x);
    work[0] += w[0];
    work[1] = x[0] + x[1];
    cost[0] = work[0];
    state[0] = work[1];
    Ok(())
}
