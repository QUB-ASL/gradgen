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

/// Return metadata describing [`codegen_kernel_f`].
pub fn codegen_kernel_f_meta() -> FunctionMetadata {
    FunctionMetadata {
        function_name: "codegen_kernel_f",
        workspace_size: 2,
        input_names: &["x", "u"],
        input_sizes: &[3, 1],
        output_names: &["y"],
        output_sizes: &[1],
    }
}

/// Evaluate the generated symbolic function `codegen_kernel_f`.
///
/// All numeric slices use the `f64` scalar type.
///
/// Arguments:
/// - `x`:
///   input slice for the declared argument `x`
///   Expected length: 3.
/// - `u`:
///   input slice for the declared argument `u`
///   Expected length: 1.
/// - `y`:
///   primal output slice for the declared result `y`
///   Expected length: 1.
/// - `work`: mutable workspace slice used to store intermediate values
///   while evaluating this kernel. Expected length: at least 2.
pub fn codegen_kernel_f(x: &[f64], u: &[f64], y: &mut [f64], work: &mut [f64]) {
    assert!(work.len() >= 2);
    assert_eq!(x.len(), 3);
    assert_eq!(u.len(), 1);
    assert_eq!(y.len(), 1);
    work[0] = x[0].sin();
    work[0] = work[0] * u[0];
    work[1] = norm2sq(x);
    work[0] = work[0] + work[1];
    work[1] = x[1] * x[2];
    work[0] = work[0] + work[1];
    y[0] = work[0];
}

/// Return metadata describing [`codegen_kernel_jf_x`].
pub fn codegen_kernel_jf_x_meta() -> FunctionMetadata {
    FunctionMetadata {
        function_name: "codegen_kernel_jf_x",
        workspace_size: 3,
        input_names: &["x", "u"],
        input_sizes: &[3, 1],
        output_names: &["y"],
        output_sizes: &[3],
    }
}

/// Evaluate the generated symbolic function `codegen_kernel_jf_x`.
///
/// All numeric slices use the `f64` scalar type.
///
/// Arguments:
/// - `x`:
///   input slice for the declared argument `x`
///   Expected length: 3.
/// - `u`:
///   input slice for the declared argument `u`
///   Expected length: 1.
/// - `y`:
///   primal output slice for the declared result `y`
///   Expected length: 3.
/// - `work`: mutable workspace slice used to store intermediate values
///   while evaluating this kernel. Expected length: at least 3.
pub fn codegen_kernel_jf_x(x: &[f64], u: &[f64], y: &mut [f64], work: &mut [f64]) {
    assert!(work.len() >= 3);
    assert_eq!(x.len(), 3);
    assert_eq!(u.len(), 1);
    assert_eq!(y.len(), 3);
    work[0] = 2.0_f64 * x[0];
    work[1] = x[0].cos();
    work[1] = work[1] * u[0];
    work[0] = work[0] + work[1];
    work[1] = 2.0_f64 * x[1];
    work[1] = work[1] + x[2];
    work[2] = 2.0_f64 * x[2];
    work[2] = work[2] + x[1];
    y[0] = work[0];
    y[1] = work[1];
    y[2] = work[2];
}

/// Return metadata describing [`codegen_kernel_jf_u`].
pub fn codegen_kernel_jf_u_meta() -> FunctionMetadata {
    FunctionMetadata {
        function_name: "codegen_kernel_jf_u",
        workspace_size: 1,
        input_names: &["x", "u"],
        input_sizes: &[3, 1],
        output_names: &["y"],
        output_sizes: &[1],
    }
}

/// Evaluate the generated symbolic function `codegen_kernel_jf_u`.
///
/// All numeric slices use the `f64` scalar type.
///
/// Arguments:
/// - `x`:
///   input slice for the declared argument `x`
///   Expected length: 3.
/// - `u`:
///   input slice for the declared argument `u`
///   Expected length: 1.
/// - `y`:
///   primal output slice for the declared result `y`
///   Expected length: 1.
/// - `work`: mutable workspace slice used to store intermediate values
///   while evaluating this kernel. Expected length: at least 1.
pub fn codegen_kernel_jf_u(x: &[f64], u: &[f64], y: &mut [f64], work: &mut [f64]) {
    assert!(work.len() >= 1);
    assert_eq!(x.len(), 3);
    assert_eq!(u.len(), 1);
    assert_eq!(y.len(), 1);
    work[0] = x[0].sin();
    y[0] = work[0];
}
