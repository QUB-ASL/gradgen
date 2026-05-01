#![no_std]
#![forbid(unsafe_code)]

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GradgenError {
    WorkspaceTooSmall(&'static str),
    InputTooSmall(&'static str),
    OutputTooSmall(&'static str),
}

fn rust_only_sqdist(x: &[f64], w: &[f64]) -> f64 {
    let _ = w;
    0.5_f64 * x[1] * x[1]
}
fn rust_only_sqdist_projection(x: &[f64], out: &mut [f64]) {
    out[0] = x[0];
    out[1] = 0.0_f64;
}

fn rust_only_sqdist_jacobian(x: &[f64], w: &[f64], out: &mut [f64]) {
    let _ = w;
    let mut projection = [0.0_f64; 2];
    rust_only_sqdist_projection(x, &mut projection);
    out[0] = x[0] - projection[0];
    out[1] = x[1] - projection[1];
}
fn rust_only_sqdist_jacobian_component(index: usize, x: &[f64], w: &[f64]) -> f64 {
    let mut out = [0.0_f64; 2];
    rust_only_sqdist_jacobian(x, w, &mut out);
    out[index]
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

/// Return metadata describing [`squared_distance_to_set_rust_only_kernel_distance_energy_f`].
pub fn squared_distance_to_set_rust_only_kernel_distance_energy_f_meta() -> FunctionMetadata {
    FunctionMetadata {
        function_name: "squared_distance_to_set_rust_only_kernel_distance_energy_f",
        workspace_size: 2,
        input_names: &["x"],
        input_sizes: &[2],
        output_names: &["y"],
        output_sizes: &[1],
    }
}

/// Evaluate the generated symbolic function `squared_distance_to_set_rust_only_kernel_distance_energy_f`.
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
///   while evaluating this kernel. Expected length: at least 2.
pub fn squared_distance_to_set_rust_only_kernel_distance_energy_f(
    x: &[f64],
    y: &mut [f64],
    work: &mut [f64],
) -> Result<(), GradgenError> {
    if work.len() < 2 {
        return Err(GradgenError::WorkspaceTooSmall("work expected at least 2"));
    };
    if x.len() != 2 {
        return Err(GradgenError::InputTooSmall("x expected length 2"));
    };
    if y.len() != 1 {
        return Err(GradgenError::OutputTooSmall("y expected length 1"));
    };
    work[0] = 2.0_f64 * x[0];
    work[1] = 2.0_f64 * x[1];
    work[0] = rust_only_sqdist(work, &[]);
    y[0] = work[0];
    Ok(())
}

/// Return metadata describing [`squared_distance_to_set_rust_only_kernel_distance_energy_grad_x_f`].
pub fn squared_distance_to_set_rust_only_kernel_distance_energy_grad_x_f_meta() -> FunctionMetadata
{
    FunctionMetadata {
        function_name: "squared_distance_to_set_rust_only_kernel_distance_energy_grad_x_f",
        workspace_size: 3,
        input_names: &["x"],
        input_sizes: &[2],
        output_names: &["y"],
        output_sizes: &[2],
    }
}

/// Evaluate the generated symbolic function `squared_distance_to_set_rust_only_kernel_distance_energy_grad_x_f`.
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
///   while evaluating this kernel. Expected length: at least 3.
pub fn squared_distance_to_set_rust_only_kernel_distance_energy_grad_x_f(
    x: &[f64],
    y: &mut [f64],
    work: &mut [f64],
) -> Result<(), GradgenError> {
    if work.len() < 3 {
        return Err(GradgenError::WorkspaceTooSmall("work expected at least 3"));
    };
    if x.len() != 2 {
        return Err(GradgenError::InputTooSmall("x expected length 2"));
    };
    if y.len() != 2 {
        return Err(GradgenError::OutputTooSmall("y expected length 2"));
    };
    work[0] = 2.0_f64 * x[0];
    work[1] = 2.0_f64 * x[1];
    work[2] = rust_only_sqdist_jacobian_component(0, work, &[]);
    work[2] *= 2.0_f64;
    work[0] = rust_only_sqdist_jacobian_component(1, work, &[]);
    work[0] *= 2.0_f64;
    y[0] = work[2];
    y[1] = work[0];
    Ok(())
}
