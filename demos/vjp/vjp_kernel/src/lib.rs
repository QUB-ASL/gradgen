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

/// Return metadata describing [`vjp_kernel_g_f`].
pub fn vjp_kernel_g_f_meta() -> FunctionMetadata {
    FunctionMetadata {
        function_name: "vjp_kernel_g_f",
        workspace_size: 3,
        input_names: &["x"],
        input_sizes: &[2],
        output_names: &["y"],
        output_sizes: &[3],
    }
}

/// Evaluate the generated symbolic function `vjp_kernel_g_f`.
///
/// All numeric slices use the `f64` scalar type.
///
/// Arguments:
/// - `x`:
///   input slice for the declared argument `x`
///   Expected length: 2.
/// - `y`:
///   primal output slice for the declared result `y`
///   Expected length: 3.
/// - `work`: mutable workspace slice used to store intermediate values
///   while evaluating this kernel. Expected length: at least 3.
pub fn vjp_kernel_g_f(x: &[f64], y: &mut [f64], work: &mut [f64]) {
    assert!(work.len() >= 3);
    assert_eq!(x.len(), 2);
    assert_eq!(y.len(), 3);
    work[0] = x[0] + x[1];
    work[1] = x[0] * x[1];
    work[2] = libm::sin(x[1]);
    y[0] = work[0];
    y[1] = work[1];
    y[2] = work[2];
}

/// Return metadata describing [`vjp_kernel_g_jf`].
pub fn vjp_kernel_g_jf_meta() -> FunctionMetadata {
    FunctionMetadata {
        function_name: "vjp_kernel_g_jf",
        workspace_size: 1,
        input_names: &["x"],
        input_sizes: &[2],
        output_names: &["jacobian_y"],
        output_sizes: &[6],
    }
}

/// Evaluate the generated symbolic function `vjp_kernel_g_jf`.
///
/// All numeric slices use the `f64` scalar type.
///
/// Arguments:
/// - `x`:
///   input slice for the declared argument `x`
///   Expected length: 2.
/// - `jacobian_y`:
///   output slice receiving the Jacobian block for declared result `y`
///   Expected length: 6.
/// - `work`: mutable workspace slice used to store intermediate values
///   while evaluating this kernel. Expected length: at least 1.
pub fn vjp_kernel_g_jf(x: &[f64], jacobian_y: &mut [f64], work: &mut [f64]) {
    assert!(!work.is_empty());
    assert_eq!(x.len(), 2);
    assert_eq!(jacobian_y.len(), 6);
    work[0] = libm::cos(x[1]);
    jacobian_y[0] = 1.0_f64;
    jacobian_y[1] = 1.0_f64;
    jacobian_y[2] = x[1];
    jacobian_y[3] = x[0];
    jacobian_y[4] = 0.0_f64;
    jacobian_y[5] = work[0];
}

/// Return metadata describing [`vjp_kernel_g_vjp`].
pub fn vjp_kernel_g_vjp_meta() -> FunctionMetadata {
    FunctionMetadata {
        function_name: "vjp_kernel_g_vjp",
        workspace_size: 3,
        input_names: &["x", "cotangent_y"],
        input_sizes: &[2, 3],
        output_names: &["vjp_x"],
        output_sizes: &[2],
    }
}

/// Evaluate the generated symbolic function `vjp_kernel_g_vjp`.
///
/// All numeric slices use the `f64` scalar type.
///
/// Arguments:
/// - `x`:
///   input slice for the declared argument `x`
///   Expected length: 2.
/// - `cotangent_y`:
///   cotangent seed associated with declared result `y`; use this slice
///   when forming Jacobian-transpose-vector or reverse-mode sensitivity
///   terms
///   Expected length: 3.
/// - `vjp_x`:
///   output slice receiving the vector-Jacobian product for declared
///   input `x`
///   Expected length: 2.
/// - `work`: mutable workspace slice used to store intermediate values
///   while evaluating this kernel. Expected length: at least 3.
pub fn vjp_kernel_g_vjp(x: &[f64], cotangent_y: &[f64], vjp_x: &mut [f64], work: &mut [f64]) {
    assert!(work.len() >= 3);
    assert_eq!(x.len(), 2);
    assert_eq!(cotangent_y.len(), 3);
    assert_eq!(vjp_x.len(), 2);
    work[0] = cotangent_y[1] * x[1];
    work[0] += cotangent_y[0];
    work[1] = libm::cos(x[1]);
    work[1] *= cotangent_y[2];
    work[1] += cotangent_y[0];
    work[2] = cotangent_y[1] * x[0];
    work[1] += work[2];
    vjp_x[0] = work[0];
    vjp_x[1] = work[1];
}
