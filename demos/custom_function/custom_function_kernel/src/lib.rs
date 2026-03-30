#![no_std]

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GradgenError {
    WorkspaceTooSmall(&'static str),
    InputTooSmall(&'static str),
    OutputTooSmall(&'static str),
}

fn custom_energy_demo(x: &[f64], w: &[f64]) -> f64 {
    libm::exp2(w[0]) * x[0] * x[0] + w[1] * x[1] * x[1] + libm::sin(x[0] * x[1])
}
fn custom_energy_demo_jacobian(x: &[f64], w: &[f64], out: &mut [f64]) {
    let xy = x[0] * x[1];
    out[0] = 2.0_f64 * libm::exp2(w[0]) * x[0] + x[1] * libm::cos(xy);
    out[1] = 2.0_f64 * w[1] * x[1] + x[0] * libm::cos(xy);
}
fn custom_energy_demo_hessian(x: &[f64], w: &[f64], out: &mut [f64]) {
    let xy = x[0] * x[1];
    let sin_xy = libm::sin(xy);
    let cross = libm::cos(xy) - x[0] * x[1] * sin_xy;
    out[0] = 2.0_f64 * libm::exp2(w[0]) - x[1] * x[1] * sin_xy;
    out[1] = cross;
    out[2] = cross;
    out[3] = 2.0_f64 * w[1] - x[0] * x[0] * sin_xy;
}
fn custom_energy_demo_hvp(x: &[f64], v_x: &[f64], w: &[f64], out: &mut [f64]) {
    let xy = x[0] * x[1];
    let sin_xy = libm::sin(xy);
    let cross = libm::cos(xy) - x[0] * x[1] * sin_xy;
    let h00 = 2.0_f64 * libm::exp2(w[0]) - x[1] * x[1] * sin_xy;
    let h11 = 2.0_f64 * w[1] - x[0] * x[0] * sin_xy;
    out[0] = h00 * v_x[0] + cross * v_x[1];
    out[1] = cross * v_x[0] + h11 * v_x[1];
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

/// Return metadata describing [`custom_function_kernel_custom_energy_f`].
pub fn custom_function_kernel_custom_energy_f_meta() -> FunctionMetadata {
    FunctionMetadata {
        function_name: "custom_function_kernel_custom_energy_f",
        workspace_size: 1,
        input_names: &["x", "w"],
        input_sizes: &[2, 2],
        output_names: &["y"],
        output_sizes: &[1],
    }
}

/// Evaluate the generated symbolic function `custom_function_kernel_custom_energy_f`.
///
/// All numeric slices use the `f64` scalar type.
///
/// Arguments:
/// - `x`:
///   input slice for the declared argument `x`
///   Expected length: 2.
/// - `w`:
///   input slice for the declared argument `w`
///   Expected length: 2.
/// - `y`:
///   primal output slice for the declared result `y`
///   Expected length: 1.
/// - `work`: mutable workspace slice used to store intermediate values
///   while evaluating this kernel. Expected length: at least 1.
pub fn custom_function_kernel_custom_energy_f(
    x: &[f64],
    w: &[f64],
    y: &mut [f64],
    work: &mut [f64],
) -> Result<(), GradgenError> {
    if work.is_empty() {
        return Err(GradgenError::WorkspaceTooSmall("work expected at least 1"));
    };
    if x.len() != 2 {
        return Err(GradgenError::InputTooSmall("x expected length 2"));
    };
    if w.len() != 2 {
        return Err(GradgenError::InputTooSmall("w expected length 2"));
    };
    if y.len() != 1 {
        return Err(GradgenError::OutputTooSmall("y expected length 1"));
    };
    work[0] = custom_energy_demo(x, w);
    y[0] = work[0];
    Ok(())
}

/// Return metadata describing [`custom_function_kernel_custom_energy_grad_x_f`].
pub fn custom_function_kernel_custom_energy_grad_x_f_meta() -> FunctionMetadata {
    FunctionMetadata {
        function_name: "custom_function_kernel_custom_energy_grad_x_f",
        workspace_size: 0,
        input_names: &["x", "w"],
        input_sizes: &[2, 2],
        output_names: &["y"],
        output_sizes: &[2],
    }
}

/// Evaluate the generated symbolic function `custom_function_kernel_custom_energy_grad_x_f`.
///
/// All numeric slices use the `f64` scalar type.
///
/// Arguments:
/// - `x`:
///   input slice for the declared argument `x`
///   Expected length: 2.
/// - `w`:
///   input slice for the declared argument `w`
///   Expected length: 2.
/// - `y`:
///   primal output slice for the declared result `y`
///   Expected length: 2.
/// - `work`: mutable workspace slice used to store intermediate values
///   while evaluating this kernel. Expected length: at least 0.
pub fn custom_function_kernel_custom_energy_grad_x_f(
    x: &[f64],
    w: &[f64],
    y: &mut [f64],
    _work: &mut [f64],
) -> Result<(), GradgenError> {
    if x.len() != 2 {
        return Err(GradgenError::InputTooSmall("x expected length 2"));
    };
    if w.len() != 2 {
        return Err(GradgenError::InputTooSmall("w expected length 2"));
    };
    if y.len() != 2 {
        return Err(GradgenError::OutputTooSmall("y expected length 2"));
    };
    custom_energy_demo_jacobian(x, w, y);
    Ok(())
}

/// Return metadata describing [`custom_function_kernel_custom_energy_hessian_x_f`].
pub fn custom_function_kernel_custom_energy_hessian_x_f_meta() -> FunctionMetadata {
    FunctionMetadata {
        function_name: "custom_function_kernel_custom_energy_hessian_x_f",
        workspace_size: 0,
        input_names: &["x", "w"],
        input_sizes: &[2, 2],
        output_names: &["y"],
        output_sizes: &[4],
    }
}

/// Evaluate the generated symbolic function `custom_function_kernel_custom_energy_hessian_x_f`.
///
/// All numeric slices use the `f64` scalar type.
///
/// Arguments:
/// - `x`:
///   input slice for the declared argument `x`
///   Expected length: 2.
/// - `w`:
///   input slice for the declared argument `w`
///   Expected length: 2.
/// - `y`:
///   primal output slice for the declared result `y`
///   Expected length: 4.
/// - `work`: mutable workspace slice used to store intermediate values
///   while evaluating this kernel. Expected length: at least 0.
pub fn custom_function_kernel_custom_energy_hessian_x_f(
    x: &[f64],
    w: &[f64],
    y: &mut [f64],
    _work: &mut [f64],
) -> Result<(), GradgenError> {
    if x.len() != 2 {
        return Err(GradgenError::InputTooSmall("x expected length 2"));
    };
    if w.len() != 2 {
        return Err(GradgenError::InputTooSmall("w expected length 2"));
    };
    if y.len() != 4 {
        return Err(GradgenError::OutputTooSmall("y expected length 4"));
    };
    custom_energy_demo_hessian(x, w, y);
    Ok(())
}

/// Return metadata describing [`custom_function_kernel_custom_energy_hvp_x_f`].
pub fn custom_function_kernel_custom_energy_hvp_x_f_meta() -> FunctionMetadata {
    FunctionMetadata {
        function_name: "custom_function_kernel_custom_energy_hvp_x_f",
        workspace_size: 0,
        input_names: &["x", "w", "v_x"],
        input_sizes: &[2, 2, 2],
        output_names: &["y"],
        output_sizes: &[2],
    }
}

/// Evaluate the generated symbolic function `custom_function_kernel_custom_energy_hvp_x_f`.
///
/// All numeric slices use the `f64` scalar type.
///
/// Arguments:
/// - `x`:
///   input slice for the declared argument `x`
///   Expected length: 2.
/// - `w`:
///   input slice for the declared argument `w`
///   Expected length: 2.
/// - `v_x`:
///   tangent or direction input associated with declared argument `x`;
///   use this slice when forming Hessian-vector-product or directional-
///   derivative terms
///   Expected length: 2.
/// - `y`:
///   primal output slice for the declared result `y`
///   Expected length: 2.
/// - `work`: mutable workspace slice used to store intermediate values
///   while evaluating this kernel. Expected length: at least 0.
pub fn custom_function_kernel_custom_energy_hvp_x_f(
    x: &[f64],
    w: &[f64],
    v_x: &[f64],
    y: &mut [f64],
    _work: &mut [f64],
) -> Result<(), GradgenError> {
    if x.len() != 2 {
        return Err(GradgenError::InputTooSmall("x expected length 2"));
    };
    if w.len() != 2 {
        return Err(GradgenError::InputTooSmall("w expected length 2"));
    };
    if v_x.len() != 2 {
        return Err(GradgenError::InputTooSmall("v_x expected length 2"));
    };
    if y.len() != 2 {
        return Err(GradgenError::OutputTooSmall("y expected length 2"));
    };
    custom_energy_demo_hvp(x, v_x, w, y);
    Ok(())
}
