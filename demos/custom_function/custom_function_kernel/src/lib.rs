fn custom_energy_demo(x: &[f64], w: &[f64]) -> f64 {
    w[0].exp2() * x[0] * x[0] + w[1] * x[1] * x[1] + (x[0] * x[1]).sin()
}
fn custom_energy_demo_jacobian(x: &[f64], w: &[f64], out: &mut [f64]) {
    let xy = x[0] * x[1];
    out[0] = 2.0_f64 * w[0].exp2() * x[0] + x[1] * xy.cos();
    out[1] = 2.0_f64 * w[1] * x[1] + x[0] * xy.cos();
}
fn custom_energy_demo_jacobian_component(index: usize, x: &[f64], w: &[f64]) -> f64 {
    let mut out = [0.0_f64; 2];
    custom_energy_demo_jacobian(x, w, &mut out);
    out[index]
}
fn custom_energy_demo_hessian(x: &[f64], w: &[f64], out: &mut [f64]) {
    let xy = x[0] * x[1];
    let sin_xy = xy.sin();
    let cross = xy.cos() - x[0] * x[1] * sin_xy;
    out[0] = 2.0_f64 * w[0].exp2() - x[1] * x[1] * sin_xy;
    out[1] = cross;
    out[2] = cross;
    out[3] = 2.0_f64 * w[1] - x[0] * x[0] * sin_xy;
}
fn custom_energy_demo_hessian_entry(row: usize, col: usize, x: &[f64], w: &[f64]) -> f64 {
    let mut out = [0.0_f64; 4];
    custom_energy_demo_hessian(x, w, &mut out);
    out[(row * 2) + col]
}
fn custom_energy_demo_hvp(x: &[f64], v_x: &[f64], w: &[f64], out: &mut [f64]) {
    let xy = x[0] * x[1];
    let sin_xy = xy.sin();
    let cross = xy.cos() - x[0] * x[1] * sin_xy;
    let h00 = 2.0_f64 * w[0].exp2() - x[1] * x[1] * sin_xy;
    let h11 = 2.0_f64 * w[1] - x[0] * x[0] * sin_xy;
    out[0] = h00 * v_x[0] + cross * v_x[1];
    out[1] = cross * v_x[0] + h11 * v_x[1];
}
fn custom_energy_demo_hvp_component(index: usize, x: &[f64], v_x: &[f64], w: &[f64]) -> f64 {
    let mut out = [0.0_f64; 2];
    custom_energy_demo_hvp(x, v_x, w, &mut out);
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

/// Return metadata describing [`custom_function_kernel_custom_energy_f`].
pub fn custom_function_kernel_custom_energy_f_meta() -> FunctionMetadata {
    FunctionMetadata {
        function_name: "custom_function_kernel_custom_energy_f",
        workspace_size: 1,
        input_names: &["x"],
        input_sizes: &[2],
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
/// - `y`:
///   primal output slice for the declared result `y`
///   Expected length: 1.
/// - `work`: mutable workspace slice used to store intermediate values
///   while evaluating this kernel. Expected length: at least 1.
pub fn custom_function_kernel_custom_energy_f(x: &[f64], y: &mut [f64], work: &mut [f64]) {
    assert!(work.len() >= 1);
    assert_eq!(x.len(), 2);
    assert_eq!(y.len(), 1);
    work[0] = custom_energy_demo(x, &[1.5_f64, 3.0_f64]);
    y[0] = work[0];
}

/// Return metadata describing [`custom_function_kernel_custom_energy_grad`].
pub fn custom_function_kernel_custom_energy_grad_meta() -> FunctionMetadata {
    FunctionMetadata {
        function_name: "custom_function_kernel_custom_energy_grad",
        workspace_size: 2,
        input_names: &["x"],
        input_sizes: &[2],
        output_names: &["y"],
        output_sizes: &[2],
    }
}

/// Evaluate the generated symbolic function `custom_function_kernel_custom_energy_grad`.
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
///   while evaluating this kernel. Expected length: at least 2.
pub fn custom_function_kernel_custom_energy_grad(x: &[f64], y: &mut [f64], work: &mut [f64]) {
    assert!(work.len() >= 2);
    assert_eq!(x.len(), 2);
    assert_eq!(y.len(), 2);
    work[0] = custom_energy_demo_jacobian_component(0, x, &[1.5_f64, 3.0_f64]);
    work[1] = custom_energy_demo_jacobian_component(1, x, &[1.5_f64, 3.0_f64]);
    custom_energy_demo_jacobian(x, &[1.5_f64, 3.0_f64], y);
}

/// Return metadata describing [`custom_function_kernel_custom_energy_hessian`].
pub fn custom_function_kernel_custom_energy_hessian_meta() -> FunctionMetadata {
    FunctionMetadata {
        function_name: "custom_function_kernel_custom_energy_hessian",
        workspace_size: 4,
        input_names: &["x"],
        input_sizes: &[2],
        output_names: &["y"],
        output_sizes: &[4],
    }
}

/// Evaluate the generated symbolic function `custom_function_kernel_custom_energy_hessian`.
///
/// All numeric slices use the `f64` scalar type.
///
/// Arguments:
/// - `x`:
///   input slice for the declared argument `x`
///   Expected length: 2.
/// - `y`:
///   primal output slice for the declared result `y`
///   Expected length: 4.
/// - `work`: mutable workspace slice used to store intermediate values
///   while evaluating this kernel. Expected length: at least 4.
pub fn custom_function_kernel_custom_energy_hessian(x: &[f64], y: &mut [f64], work: &mut [f64]) {
    assert!(work.len() >= 4);
    assert_eq!(x.len(), 2);
    assert_eq!(y.len(), 4);
    work[0] = custom_energy_demo_hessian_entry(0, 0, x, &[1.5_f64, 3.0_f64]);
    work[1] = custom_energy_demo_hessian_entry(0, 1, x, &[1.5_f64, 3.0_f64]);
    work[2] = custom_energy_demo_hessian_entry(1, 0, x, &[1.5_f64, 3.0_f64]);
    work[3] = custom_energy_demo_hessian_entry(1, 1, x, &[1.5_f64, 3.0_f64]);
    custom_energy_demo_hessian(x, &[1.5_f64, 3.0_f64], y);
}

/// Return metadata describing [`custom_function_kernel_custom_energy_hvp`].
pub fn custom_function_kernel_custom_energy_hvp_meta() -> FunctionMetadata {
    FunctionMetadata {
        function_name: "custom_function_kernel_custom_energy_hvp",
        workspace_size: 2,
        input_names: &["x", "v_x"],
        input_sizes: &[2, 2],
        output_names: &["y"],
        output_sizes: &[2],
    }
}

/// Evaluate the generated symbolic function `custom_function_kernel_custom_energy_hvp`.
///
/// All numeric slices use the `f64` scalar type.
///
/// Arguments:
/// - `x`:
///   input slice for the declared argument `x`
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
///   while evaluating this kernel. Expected length: at least 2.
pub fn custom_function_kernel_custom_energy_hvp(
    x: &[f64],
    v_x: &[f64],
    y: &mut [f64],
    work: &mut [f64],
) {
    assert!(work.len() >= 2);
    assert_eq!(x.len(), 2);
    assert_eq!(v_x.len(), 2);
    assert_eq!(y.len(), 2);
    work[0] = custom_energy_demo_hvp_component(0, x, v_x, &[1.5_f64, 3.0_f64]);
    work[1] = custom_energy_demo_hvp_component(1, x, v_x, &[1.5_f64, 3.0_f64]);
    custom_energy_demo_hvp(x, v_x, &[1.5_f64, 3.0_f64], y);
}
