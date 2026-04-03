#![no_std]

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

/// Return metadata describing [`reduce_map_kernel_mapped_seq_f`].
pub fn reduce_map_kernel_mapped_seq_f_meta() -> FunctionMetadata {
    FunctionMetadata {
        function_name: "reduce_map_kernel_mapped_seq_f",
        workspace_size: 2,
        input_names: &["x_seq"],
        input_sizes: &[3],
        output_names: &["m"],
        output_sizes: &[3],
    }
}

/// Evaluate the generated symbolic function `reduce_map_kernel_mapped_seq_f`.
///
/// All numeric slices use the `f64` scalar type.
///
/// Arguments:
/// - `x_seq`:
///   input slice for the declared argument `x_seq`
///   Expected length: 3.
/// - `m`:
///   primal output slice for the declared result `m`
///   Expected length: 3.
/// - `work`: mutable workspace slice used to store intermediate values
///   while evaluating this kernel. Expected length: at least 2.
pub fn reduce_map_kernel_mapped_seq_f(
    x_seq: &[f64],
    m: &mut [f64],
    work: &mut [f64],
) -> Result<(), GradgenError> {
    if work.len() < 2 {
        return Err(GradgenError::WorkspaceTooSmall("work expected at least 2"));
    };
    if x_seq.len() != 3 {
        return Err(GradgenError::InputTooSmall("x_seq expected length 3"));
    };
    if m.len() != 3 {
        return Err(GradgenError::OutputTooSmall("m expected length 3"));
    };
    let helper_work = &mut work[..2];
    for stage_index in 0..3 {
        let x_seq_stage = &x_seq[stage_index..stage_index + 1];
        let m_stage = &mut m[stage_index..stage_index + 1];
        reduce_map_kernel_mapped_seq_f_helper(x_seq_stage, m_stage, helper_work);
    }
    Ok(())
}

fn reduce_map_kernel_mapped_seq_f_helper(x: &[f64], m: &mut [f64], work: &mut [f64]) {
    work[0] = x[0] * x[0];
    work[1] = libm::sin(x[0]);
    work[0] += work[1];
    m[0] = work[0];
}

/// Return metadata describing [`reduce_map_kernel_mapped_seq_jf_x_seq`].
pub fn reduce_map_kernel_mapped_seq_jf_x_seq_meta() -> FunctionMetadata {
    FunctionMetadata {
        function_name: "reduce_map_kernel_mapped_seq_jf_x_seq",
        workspace_size: 3,
        input_names: &["x_seq"],
        input_sizes: &[3],
        output_names: &["jacobian_m"],
        output_sizes: &[9],
    }
}

/// Evaluate the generated symbolic function `reduce_map_kernel_mapped_seq_jf_x_seq`.
///
/// All numeric slices use the `f64` scalar type.
///
/// Arguments:
/// - `x_seq`:
///   input slice for the declared argument `x_seq`
///   Expected length: 3.
/// - `jacobian_m`:
///   output slice receiving the Jacobian block for declared result `m`
///   Expected length: 9.
/// - `work`: mutable workspace slice used to store intermediate values
///   while evaluating this kernel. Expected length: at least 3.
pub fn reduce_map_kernel_mapped_seq_jf_x_seq(
    x_seq: &[f64],
    jacobian_m: &mut [f64],
    work: &mut [f64],
) -> Result<(), GradgenError> {
    if work.len() < 3 {
        return Err(GradgenError::WorkspaceTooSmall("work expected at least 3"));
    };
    if x_seq.len() != 3 {
        return Err(GradgenError::InputTooSmall("x_seq expected length 3"));
    };
    if jacobian_m.len() != 9 {
        return Err(GradgenError::OutputTooSmall("jacobian_m expected length 9"));
    };
    jacobian_m.fill(0.0_f64);
    let (temp_jacobian_m, helper_work) = work.split_at_mut(1);
    for stage_index in 0..3 {
        let x_seq_stage = &x_seq[stage_index..stage_index + 1];
        reduce_map_kernel_mapped_seq_jf_x_seq_helper(x_seq_stage, temp_jacobian_m, helper_work);
        for local_row in 0..1 {
            let dest_row = stage_index + local_row;
            let dest_start = (dest_row * 3) + stage_index;
            let src_start = local_row;
            jacobian_m[dest_start..(dest_start + 1)]
                .copy_from_slice(&temp_jacobian_m[src_start..(src_start + 1)]);
        }
    }
    Ok(())
}

fn reduce_map_kernel_mapped_seq_jf_x_seq_helper(
    x: &[f64],
    jacobian_m: &mut [f64],
    work: &mut [f64],
) {
    work[0] = libm::cos(x[0]);
    work[1] = 2.0_f64 * x[0];
    work[0] += work[1];
    jacobian_m[0] = work[0];
}

/// Return metadata describing [`reduce_map_kernel_reduced_scalar_f`].
pub fn reduce_map_kernel_reduced_scalar_f_meta() -> FunctionMetadata {
    FunctionMetadata {
        function_name: "reduce_map_kernel_reduced_scalar_f",
        workspace_size: 4,
        input_names: &["acc0", "m_seq"],
        input_sizes: &[1, 3],
        output_names: &["acc_final"],
        output_sizes: &[1],
    }
}

/// Evaluate the generated symbolic function `reduce_map_kernel_reduced_scalar_f`.
///
/// All numeric slices use the `f64` scalar type.
///
/// Arguments:
/// - `acc0`:
///   input slice for the declared argument `acc0`
///   Expected length: 1.
/// - `m_seq`:
///   input slice for the declared argument `m_seq`
///   Expected length: 3.
/// - `acc_final`:
///   primal output slice for the declared result `acc_final`
///   Expected length: 1.
/// - `work`: mutable workspace slice used to store intermediate values
///   while evaluating this kernel. Expected length: at least 4.
pub fn reduce_map_kernel_reduced_scalar_f(
    acc0: &[f64],
    m_seq: &[f64],
    acc_final: &mut [f64],
    work: &mut [f64],
) -> Result<(), GradgenError> {
    if work.len() < 4 {
        return Err(GradgenError::WorkspaceTooSmall("work expected at least 4"));
    };
    if acc0.len() != 1 {
        return Err(GradgenError::InputTooSmall("acc0 expected length 1"));
    };
    if m_seq.len() != 3 {
        return Err(GradgenError::InputTooSmall("m_seq expected length 3"));
    };
    if acc_final.len() != 1 {
        return Err(GradgenError::OutputTooSmall("acc_final expected length 1"));
    };
    let (acc_work, helper_work) = work.split_at_mut(2);
    let (acc_curr_buf, acc_next_buf) = acc_work.split_at_mut(1);
    acc_curr_buf.copy_from_slice(acc0);
    for stage_index in 0..3 {
        let x_stage = &m_seq[stage_index..stage_index + 1];
        acc_next_buf.fill(0.0_f64);
        reduce_map_kernel_reduced_scalar_f_helper(acc_curr_buf, x_stage, acc_next_buf, helper_work);
        acc_curr_buf.copy_from_slice(acc_next_buf);
    }
    acc_final.copy_from_slice(acc_curr_buf);
    Ok(())
}

fn reduce_map_kernel_reduced_scalar_f_helper(
    acc: &[f64],
    m: &[f64],
    acc_next: &mut [f64],
    work: &mut [f64],
) {
    work[0] = acc[0] * m[0];
    work[1] = libm::sin(acc[0]);
    work[0] += work[1];
    work[1] = m[0] * m[0];
    work[0] += work[1];
    acc_next[0] = work[0];
}
