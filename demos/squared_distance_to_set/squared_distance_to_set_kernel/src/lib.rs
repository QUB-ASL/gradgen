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

/// Return metadata describing [`squared_distance_to_set_kernel_distance_energy_f`].
pub fn squared_distance_to_set_kernel_distance_energy_f_meta() -> FunctionMetadata {
    FunctionMetadata {
        function_name: "squared_distance_to_set_kernel_distance_energy_f",
        workspace_size: 1,
        input_names: &["x"],
        input_sizes: &[2],
        output_names: &["y"],
        output_sizes: &[1],
    }
}

/// Evaluate the generated symbolic function `squared_distance_to_set_kernel_distance_energy_f`.
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
///   while evaluating this kernel. Expected length: at least 1.
pub fn squared_distance_to_set_kernel_distance_energy_f(
    x: &[f64],
    y: &mut [f64],
    work: &mut [f64],
) -> Result<(), GradgenError> {
    if work.is_empty() {
        return Err(GradgenError::WorkspaceTooSmall("work expected at least 1"));
    };
    if x.len() != 2 {
        return Err(GradgenError::InputTooSmall("x expected length 2"));
    };
    if y.len() != 1 {
        return Err(GradgenError::OutputTooSmall("y expected length 1"));
    };
    work[0] = x[1] * x[1];
    work[0] *= 2.0_f64;
    y[0] = work[0];
    Ok(())
}

/// Return metadata describing [`squared_distance_to_set_kernel_distance_energy_grad_x_f`].
pub fn squared_distance_to_set_kernel_distance_energy_grad_x_f_meta() -> FunctionMetadata {
    FunctionMetadata {
        function_name: "squared_distance_to_set_kernel_distance_energy_grad_x_f",
        workspace_size: 1,
        input_names: &["x"],
        input_sizes: &[2],
        output_names: &["y"],
        output_sizes: &[2],
    }
}

/// Evaluate the generated symbolic function `squared_distance_to_set_kernel_distance_energy_grad_x_f`.
///
/// All numeric slices use the `f64` scalar type.
///
/// Arguments:
/// - `x`:
///   input slice for the declared argument `x`
///   Expected length: 2.
/// - `y`:
///   primal output slice for the declared result `y`
///   Expected length: 2.
/// - `work`: mutable workspace slice used to store intermediate values
///   while evaluating this kernel. Expected length: at least 1.
pub fn squared_distance_to_set_kernel_distance_energy_grad_x_f(
    x: &[f64],
    y: &mut [f64],
    work: &mut [f64],
) -> Result<(), GradgenError> {
    if work.is_empty() {
        return Err(GradgenError::WorkspaceTooSmall("work expected at least 1"));
    };
    if x.len() != 2 {
        return Err(GradgenError::InputTooSmall("x expected length 2"));
    };
    if y.len() != 2 {
        return Err(GradgenError::OutputTooSmall("y expected length 2"));
    };
    work[0] = 4.0_f64 * x[1];
    y[0] = 0.0_f64;
    y[1] = work[0];
    Ok(())
}
