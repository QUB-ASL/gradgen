#![no_std]

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

/// Return metadata describing [`multi_function_kernel_energy_f`].
pub fn multi_function_kernel_energy_f_meta() -> FunctionMetadata {
    FunctionMetadata {
        function_name: "multi_function_kernel_energy_f",
        workspace_size: 2,
        input_names: &[
            "x",
            "u",
        ],
        input_sizes: &[
            2,
            1,
        ],
        output_names: &[
            "energy",
        ],
        output_sizes: &[
            1,
        ],
    }
}

/// Evaluate the generated symbolic function `multi_function_kernel_energy_f`.
///
/// All numeric slices use the `f64` scalar type.
///
/// Arguments:
/// - `x`:
///   input slice for the declared argument `x`
///   Expected length: 2.
/// - `u`:
///   input slice for the declared argument `u`
///   Expected length: 1.
/// - `energy`:
///   primal output slice for the declared result `energy`
///   Expected length: 1.
/// - `work`: mutable workspace slice used to store intermediate values
///   while evaluating this kernel. Expected length: at least 2.
pub fn multi_function_kernel_energy_f(x: &[f64], u: &[f64], energy: &mut [f64], work: &mut [f64]) {
    assert!(work.len() >= 2);
    assert_eq!(x.len(), 2);
    assert_eq!(u.len(), 1);
    assert_eq!(energy.len(), 1);
    work[0] = u[0] * x[0];
    work[1] = norm2sq(x);
    work[0] = work[0] + work[1];
    energy[0] = work[0];
}

/// Return metadata describing [`multi_function_kernel_energy_jf_x`].
pub fn multi_function_kernel_energy_jf_x_meta() -> FunctionMetadata {
    FunctionMetadata {
        function_name: "multi_function_kernel_energy_jf_x",
        workspace_size: 2,
        input_names: &[
            "x",
            "u",
        ],
        input_sizes: &[
            2,
            1,
        ],
        output_names: &[
            "jacobian_energy",
        ],
        output_sizes: &[
            2,
        ],
    }
}

/// Evaluate the generated symbolic function `multi_function_kernel_energy_jf_x`.
///
/// All numeric slices use the `f64` scalar type.
///
/// Arguments:
/// - `x`:
///   input slice for the declared argument `x`
///   Expected length: 2.
/// - `u`:
///   input slice for the declared argument `u`
///   Expected length: 1.
/// - `jacobian_energy`:
///   output slice receiving the Jacobian block for declared result
///   `energy`
///   Expected length: 2.
/// - `work`: mutable workspace slice used to store intermediate values
///   while evaluating this kernel. Expected length: at least 2.
pub fn multi_function_kernel_energy_jf_x(x: &[f64], u: &[f64], jacobian_energy: &mut [f64], work: &mut [f64]) {
    assert!(work.len() >= 2);
    assert_eq!(x.len(), 2);
    assert_eq!(u.len(), 1);
    assert_eq!(jacobian_energy.len(), 2);
    work[0] = 2.0_f64 * x[0];
    work[0] = work[0] + u[0];
    work[1] = 2.0_f64 * x[1];
    jacobian_energy[0] = work[0];
    jacobian_energy[1] = work[1];
}

/// Return metadata describing [`multi_function_kernel_energy_jf_u`].
pub fn multi_function_kernel_energy_jf_u_meta() -> FunctionMetadata {
    FunctionMetadata {
        function_name: "multi_function_kernel_energy_jf_u",
        workspace_size: 0,
        input_names: &[
            "x",
            "u",
        ],
        input_sizes: &[
            2,
            1,
        ],
        output_names: &[
            "jacobian_energy",
        ],
        output_sizes: &[
            1,
        ],
    }
}

/// Evaluate the generated symbolic function `multi_function_kernel_energy_jf_u`.
///
/// All numeric slices use the `f64` scalar type.
///
/// Arguments:
/// - `x`:
///   input slice for the declared argument `x`
///   Expected length: 2.
/// - `u`:
///   input slice for the declared argument `u`
///   Expected length: 1.
/// - `jacobian_energy`:
///   output slice receiving the Jacobian block for declared result
///   `energy`
///   Expected length: 1.
/// - `work`: mutable workspace slice used to store intermediate values
///   while evaluating this kernel. Expected length: at least 0.
pub fn multi_function_kernel_energy_jf_u(x: &[f64], u: &[f64], jacobian_energy: &mut [f64], _work: &mut [f64]) {
    assert_eq!(x.len(), 2);
    assert_eq!(u.len(), 1);
    assert_eq!(jacobian_energy.len(), 1);
    jacobian_energy[0] = x[0];
}

/// Return metadata describing [`multi_function_kernel_coupling_f`].
pub fn multi_function_kernel_coupling_f_meta() -> FunctionMetadata {
    FunctionMetadata {
        function_name: "multi_function_kernel_coupling_f",
        workspace_size: 2,
        input_names: &[
            "x",
            "u",
        ],
        input_sizes: &[
            2,
            1,
        ],
        output_names: &[
            "coupling",
        ],
        output_sizes: &[
            1,
        ],
    }
}

/// Evaluate the generated symbolic function `multi_function_kernel_coupling_f`.
///
/// All numeric slices use the `f64` scalar type.
///
/// Arguments:
/// - `x`:
///   input slice for the declared argument `x`
///   Expected length: 2.
/// - `u`:
///   input slice for the declared argument `u`
///   Expected length: 1.
/// - `coupling`:
///   primal output slice for the declared result `coupling`
///   Expected length: 1.
/// - `work`: mutable workspace slice used to store intermediate values
///   while evaluating this kernel. Expected length: at least 2.
pub fn multi_function_kernel_coupling_f(x: &[f64], u: &[f64], coupling: &mut [f64], work: &mut [f64]) {
    assert!(work.len() >= 2);
    assert_eq!(x.len(), 2);
    assert_eq!(u.len(), 1);
    assert_eq!(coupling.len(), 1);
    work[0] = libm::cos(u[0]);
    work[1] = x[0] * x[1];
    work[0] = work[0] + work[1];
    coupling[0] = work[0];
}

/// Return metadata describing [`multi_function_kernel_coupling_grad_x`].
pub fn multi_function_kernel_coupling_grad_x_meta() -> FunctionMetadata {
    FunctionMetadata {
        function_name: "multi_function_kernel_coupling_grad_x",
        workspace_size: 0,
        input_names: &[
            "x",
            "u",
        ],
        input_sizes: &[
            2,
            1,
        ],
        output_names: &[
            "coupling",
        ],
        output_sizes: &[
            2,
        ],
    }
}

/// Evaluate the generated symbolic function `multi_function_kernel_coupling_grad_x`.
///
/// All numeric slices use the `f64` scalar type.
///
/// Arguments:
/// - `x`:
///   input slice for the declared argument `x`
///   Expected length: 2.
/// - `u`:
///   input slice for the declared argument `u`
///   Expected length: 1.
/// - `coupling`:
///   primal output slice for the declared result `coupling`
///   Expected length: 2.
/// - `work`: mutable workspace slice used to store intermediate values
///   while evaluating this kernel. Expected length: at least 0.
pub fn multi_function_kernel_coupling_grad_x(x: &[f64], u: &[f64], coupling: &mut [f64], _work: &mut [f64]) {
    assert_eq!(x.len(), 2);
    assert_eq!(u.len(), 1);
    assert_eq!(coupling.len(), 2);
    coupling[0] = x[1];
    coupling[1] = x[0];
}

/// Return metadata describing [`multi_function_kernel_coupling_grad_u`].
pub fn multi_function_kernel_coupling_grad_u_meta() -> FunctionMetadata {
    FunctionMetadata {
        function_name: "multi_function_kernel_coupling_grad_u",
        workspace_size: 1,
        input_names: &[
            "x",
            "u",
        ],
        input_sizes: &[
            2,
            1,
        ],
        output_names: &[
            "coupling",
        ],
        output_sizes: &[
            1,
        ],
    }
}

/// Evaluate the generated symbolic function `multi_function_kernel_coupling_grad_u`.
///
/// All numeric slices use the `f64` scalar type.
///
/// Arguments:
/// - `x`:
///   input slice for the declared argument `x`
///   Expected length: 2.
/// - `u`:
///   input slice for the declared argument `u`
///   Expected length: 1.
/// - `coupling`:
///   primal output slice for the declared result `coupling`
///   Expected length: 1.
/// - `work`: mutable workspace slice used to store intermediate values
///   while evaluating this kernel. Expected length: at least 1.
pub fn multi_function_kernel_coupling_grad_u(x: &[f64], u: &[f64], coupling: &mut [f64], work: &mut [f64]) {
    assert!(work.len() >= 1);
    assert_eq!(x.len(), 2);
    assert_eq!(u.len(), 1);
    assert_eq!(coupling.len(), 1);
    work[0] = libm::sin(u[0]);
    work[0] = -work[0];
    coupling[0] = work[0];
}

/// Return metadata describing [`multi_function_kernel_coupling_hvp_x`].
pub fn multi_function_kernel_coupling_hvp_x_meta() -> FunctionMetadata {
    FunctionMetadata {
        function_name: "multi_function_kernel_coupling_hvp_x",
        workspace_size: 0,
        input_names: &[
            "x",
            "u",
            "v_x",
        ],
        input_sizes: &[
            2,
            1,
            2,
        ],
        output_names: &[
            "coupling",
        ],
        output_sizes: &[
            2,
        ],
    }
}

/// Evaluate the generated symbolic function `multi_function_kernel_coupling_hvp_x`.
///
/// All numeric slices use the `f64` scalar type.
///
/// Arguments:
/// - `x`:
///   input slice for the declared argument `x`
///   Expected length: 2.
/// - `u`:
///   input slice for the declared argument `u`
///   Expected length: 1.
/// - `v_x`:
///   tangent or direction input associated with declared argument `x`;
///   use this slice when forming Hessian-vector-product or directional-
///   derivative terms
///   Expected length: 2.
/// - `coupling`:
///   primal output slice for the declared result `coupling`
///   Expected length: 2.
/// - `work`: mutable workspace slice used to store intermediate values
///   while evaluating this kernel. Expected length: at least 0.
pub fn multi_function_kernel_coupling_hvp_x(x: &[f64], u: &[f64], v_x: &[f64], coupling: &mut [f64], _work: &mut [f64]) {
    assert_eq!(x.len(), 2);
    assert_eq!(u.len(), 1);
    assert_eq!(v_x.len(), 2);
    assert_eq!(coupling.len(), 2);
    coupling[0] = v_x[1];
    coupling[1] = v_x[0];
}

/// Return metadata describing [`multi_function_kernel_coupling_hvp_u`].
pub fn multi_function_kernel_coupling_hvp_u_meta() -> FunctionMetadata {
    FunctionMetadata {
        function_name: "multi_function_kernel_coupling_hvp_u",
        workspace_size: 1,
        input_names: &[
            "x",
            "u",
            "v_u",
        ],
        input_sizes: &[
            2,
            1,
            1,
        ],
        output_names: &[
            "coupling",
        ],
        output_sizes: &[
            1,
        ],
    }
}

/// Evaluate the generated symbolic function `multi_function_kernel_coupling_hvp_u`.
///
/// All numeric slices use the `f64` scalar type.
///
/// Arguments:
/// - `x`:
///   input slice for the declared argument `x`
///   Expected length: 2.
/// - `u`:
///   input slice for the declared argument `u`
///   Expected length: 1.
/// - `v_u`:
///   tangent or direction input associated with declared argument `u`;
///   use this slice when forming Hessian-vector-product or directional-
///   derivative terms
///   Expected length: 1.
/// - `coupling`:
///   primal output slice for the declared result `coupling`
///   Expected length: 1.
/// - `work`: mutable workspace slice used to store intermediate values
///   while evaluating this kernel. Expected length: at least 1.
pub fn multi_function_kernel_coupling_hvp_u(x: &[f64], u: &[f64], v_u: &[f64], coupling: &mut [f64], work: &mut [f64]) {
    assert!(work.len() >= 1);
    assert_eq!(x.len(), 2);
    assert_eq!(u.len(), 1);
    assert_eq!(v_u.len(), 1);
    assert_eq!(coupling.len(), 1);
    work[0] = libm::cos(u[0]);
    work[0] = work[0] * v_u[0];
    work[0] = -work[0];
    coupling[0] = work[0];
}

/// Return metadata describing [`multi_function_kernel_coupling_f_jf_x`].
pub fn multi_function_kernel_coupling_f_jf_x_meta() -> FunctionMetadata {
    FunctionMetadata {
        function_name: "multi_function_kernel_coupling_f_jf_x",
        workspace_size: 2,
        input_names: &[
            "x",
            "u",
        ],
        input_sizes: &[
            2,
            1,
        ],
        output_names: &[
            "coupling",
            "jacobian_coupling",
        ],
        output_sizes: &[
            1,
            2,
        ],
    }
}

/// Evaluate the generated symbolic function `multi_function_kernel_coupling_f_jf_x`.
///
/// All numeric slices use the `f64` scalar type.
///
/// Arguments:
/// - `x`:
///   input slice for the declared argument `x`
///   Expected length: 2.
/// - `u`:
///   input slice for the declared argument `u`
///   Expected length: 1.
/// - `coupling`:
///   primal output slice for the declared result `coupling`
///   Expected length: 1.
/// - `jacobian_coupling`:
///   output slice receiving the Jacobian block for declared result
///   `coupling`
///   Expected length: 2.
/// - `work`: mutable workspace slice used to store intermediate values
///   while evaluating this kernel. Expected length: at least 2.
pub fn multi_function_kernel_coupling_f_jf_x(x: &[f64], u: &[f64], coupling: &mut [f64], jacobian_coupling: &mut [f64], work: &mut [f64]) {
    assert!(work.len() >= 2);
    assert_eq!(x.len(), 2);
    assert_eq!(u.len(), 1);
    assert_eq!(coupling.len(), 1);
    assert_eq!(jacobian_coupling.len(), 2);
    work[0] = libm::cos(u[0]);
    work[1] = x[0] * x[1];
    work[0] = work[0] + work[1];
    coupling[0] = work[0];
    jacobian_coupling[0] = x[1];
    jacobian_coupling[1] = x[0];
}
