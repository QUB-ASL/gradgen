#![forbid(unsafe_code)]
#![forbid(missing_docs)]
//!
//! Generated Rust kernels emitted by gradgen.

/// Errors returned by generated Rust kernels when their input slices,
/// output slices, or workspace slice are too small.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GradgenError {
    /// The mutable workspace slice was smaller than required.
    WorkspaceTooSmall(&'static str),
    /// An input slice was smaller than required.
    InputTooSmall(&'static str),
    /// An output slice was smaller than required.
    OutputTooSmall(&'static str),
}

fn soc_penalty(x: &[f64], w: &[f64]) -> f64 {
    let _ = w;
    let alpha = 2.0_f64;
    let one = 1.0_f64;
    let zero = 0.0_f64;
    let alpha_sq = alpha * alpha;
    let alpha_sq_plus_one = alpha_sq + one;
    let last = x.len() - 1;
    let t = x[last];
    let t_sq = t * t;
    let mut sum_sq = zero;
    for value in &x[..last] {
        sum_sq += *value * *value;
    }

    if t <= zero && alpha_sq * sum_sq <= t_sq {
        return 0.5_f64 * (t_sq + sum_sq);
    }

    if t >= zero && sum_sq <= alpha_sq * t_sq {
        return zero;
    }

    let norm_y = (sum_sq).sqrt();
    let beta = (alpha * norm_y + t) / alpha_sq_plus_one;
    let y_scale = one - (alpha * beta / norm_y);
    let dt = t - beta;
    let dist_sq = y_scale * y_scale * sum_sq + dt * dt;
    0.5_f64 * dist_sq
}
fn soc_penalty_projection(x: &[f64], out: &mut [f64]) {
    let alpha = 2.0_f64;
    let one = 1.0_f64;
    let zero = 0.0_f64;
    let alpha_sq = alpha * alpha;
    let alpha_sq_plus_one = alpha_sq + one;
    let last = x.len() - 1;
    let t = x[last];
    let t_sq = t * t;
    let mut sum_sq = zero;
    for value in &x[..last] {
        sum_sq += *value * *value;
    }

    if t <= zero && alpha_sq * sum_sq <= t_sq {
        out.fill(zero);
        return;
    }

    if t >= zero && sum_sq <= alpha_sq * t_sq {
        out.copy_from_slice(x);
        return;
    }

    let norm_y = (sum_sq).sqrt();
    let beta = (alpha * norm_y + t) / alpha_sq_plus_one;
    let scale = alpha * beta / norm_y;
    for index in 0..last {
        out[index] = scale * x[index];
    }
    out[last] = beta;
}

fn soc_penalty_jacobian(x: &[f64], w: &[f64], out: &mut [f64]) {
    let _ = w;
    let mut projection = [0.0_f64; 3];
    soc_penalty_projection(x, &mut projection);
    for index in 0..3 {
        out[index] = x[index] - projection[index];
    }
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

/// Return metadata describing [`soc_kernel_energy`].
pub fn soc_kernel_energy_meta() -> FunctionMetadata {
    FunctionMetadata {
        function_name: "soc_kernel_energy",
        workspace_size: 0,
        input_names: &["x"],
        input_sizes: &[3],
        output_names: &["energy"],
        output_sizes: &[1],
    }
}

/// Evaluate the generated symbolic function `soc_kernel_energy`.
///
/// All numeric slices use the `f64` scalar type.
///
/// Arguments:
/// - `x`:
///   input slice for the declared argument `x`
///   Expected length: 3.
/// - `energy`:
///   primal output slice for the declared result `energy`
///   Expected length: 1.
/// - `work`: mutable workspace slice used to store intermediate values
///   while evaluating this kernel. Expected length: at least 0.
pub fn soc_kernel_energy(
    x: &[f64],
    energy: &mut [f64],
    _work: &mut [f64],
) -> Result<(), GradgenError> {
    if x.len() != 3 {
        return Err(GradgenError::InputTooSmall("x expected length 3"));
    };
    if energy.len() != 1 {
        return Err(GradgenError::OutputTooSmall("energy expected length 1"));
    };
    energy[0] = soc_penalty(x, &[]);
    Ok(())
}

/// Return metadata describing [`soc_kernel_energy_grad_x`].
pub fn soc_kernel_energy_grad_x_meta() -> FunctionMetadata {
    FunctionMetadata {
        function_name: "soc_kernel_energy_grad_x",
        workspace_size: 0,
        input_names: &["x"],
        input_sizes: &[3],
        output_names: &["energy"],
        output_sizes: &[3],
    }
}

/// Evaluate the generated symbolic function `soc_kernel_energy_grad_x`.
///
/// All numeric slices use the `f64` scalar type.
///
/// Arguments:
/// - `x`:
///   input slice for the declared argument `x`
///   Expected length: 3.
/// - `energy`:
///   primal output slice for the declared result `energy`
///   Expected length: 3.
/// - `work`: mutable workspace slice used to store intermediate values
///   while evaluating this kernel. Expected length: at least 0.
pub fn soc_kernel_energy_grad_x(
    x: &[f64],
    energy: &mut [f64],
    _work: &mut [f64],
) -> Result<(), GradgenError> {
    if x.len() != 3 {
        return Err(GradgenError::InputTooSmall("x expected length 3"));
    };
    if energy.len() != 3 {
        return Err(GradgenError::OutputTooSmall("energy expected length 3"));
    };
    soc_penalty_jacobian(x, &[], energy);
    Ok(())
}
