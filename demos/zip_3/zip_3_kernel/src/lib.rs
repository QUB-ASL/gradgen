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

/// Return metadata describing [`zip_3_kernel_zip3_f`].
pub fn zip_3_kernel_zip3_f_meta() -> FunctionMetadata {
    FunctionMetadata {
        function_name: "zip_3_kernel_zip3_f",
        workspace_size: 2,
        input_names: &["a_seq", "b_seq", "c_seq"],
        input_sizes: &[10, 5, 5],
        output_names: &["y"],
        output_sizes: &[5],
    }
}

/// Evaluate the generated symbolic function `zip_3_kernel_zip3_f`.
///
/// All numeric slices use the `f64` scalar type.
///
/// Arguments:
/// - `a_seq`:
///   input slice for the declared argument `a_seq`
///   Expected length: 10.
/// - `b_seq`:
///   input slice for the declared argument `b_seq`
///   Expected length: 5.
/// - `c_seq`:
///   input slice for the declared argument `c_seq`
///   Expected length: 5.
/// - `y`:
///   primal output slice for the declared result `y`
///   Expected length: 5.
/// - `work`: mutable workspace slice used to store intermediate values
///   while evaluating this kernel. Expected length: at least 2.
pub fn zip_3_kernel_zip3_f(
    a_seq: &[f64],
    b_seq: &[f64],
    c_seq: &[f64],
    y: &mut [f64],
    work: &mut [f64],
) -> Result<(), GradgenError> {
    if work.len() < 2 {
        return Err(GradgenError::WorkspaceTooSmall("work expected at least 2"));
    };
    if a_seq.len() != 10 {
        return Err(GradgenError::InputTooSmall("a_seq expected length 10"));
    };
    if b_seq.len() != 5 {
        return Err(GradgenError::InputTooSmall("b_seq expected length 5"));
    };
    if c_seq.len() != 5 {
        return Err(GradgenError::InputTooSmall("c_seq expected length 5"));
    };
    if y.len() != 5 {
        return Err(GradgenError::OutputTooSmall("y expected length 5"));
    };
    let helper_work = &mut work[..2];
    for stage_index in 0..5 {
        let a_seq_stage = &a_seq[stage_index * 2..((stage_index + 1) * 2)];
        let b_seq_stage = &b_seq[stage_index..stage_index + 1];
        let c_seq_stage = &c_seq[stage_index..stage_index + 1];
        let y_stage = &mut y[stage_index..stage_index + 1];
        zip_3_kernel_zip3_f_helper(a_seq_stage, b_seq_stage, c_seq_stage, y_stage, helper_work);
    }
    Ok(())
}

fn zip_3_kernel_zip3_f_helper(a: &[f64], b: &[f64], c: &[f64], y: &mut [f64], work: &mut [f64]) {
    work[0] = a[0] * b[0];
    work[0] += a[1];
    work[1] = libm::sin(c[0]);
    work[0] += work[1];
    y[0] = work[0];
}

/// Return metadata describing [`zip_3_kernel_zip3_jf_a_seq`].
pub fn zip_3_kernel_zip3_jf_a_seq_meta() -> FunctionMetadata {
    FunctionMetadata {
        function_name: "zip_3_kernel_zip3_jf_a_seq",
        workspace_size: 2,
        input_names: &["a_seq", "b_seq", "c_seq"],
        input_sizes: &[10, 5, 5],
        output_names: &["jacobian_y"],
        output_sizes: &[50],
    }
}

/// Evaluate the generated symbolic function `zip_3_kernel_zip3_jf_a_seq`.
///
/// All numeric slices use the `f64` scalar type.
///
/// Arguments:
/// - `a_seq`:
///   input slice for the declared argument `a_seq`
///   Expected length: 10.
/// - `b_seq`:
///   input slice for the declared argument `b_seq`
///   Expected length: 5.
/// - `c_seq`:
///   input slice for the declared argument `c_seq`
///   Expected length: 5.
/// - `jacobian_y`:
///   output slice receiving the Jacobian block for declared result `y`
///   Expected length: 50.
/// - `work`: mutable workspace slice used to store intermediate values
///   while evaluating this kernel. Expected length: at least 2.
pub fn zip_3_kernel_zip3_jf_a_seq(
    a_seq: &[f64],
    b_seq: &[f64],
    c_seq: &[f64],
    jacobian_y: &mut [f64],
    work: &mut [f64],
) -> Result<(), GradgenError> {
    if work.len() < 2 {
        return Err(GradgenError::WorkspaceTooSmall("work expected at least 2"));
    };
    if a_seq.len() != 10 {
        return Err(GradgenError::InputTooSmall("a_seq expected length 10"));
    };
    if b_seq.len() != 5 {
        return Err(GradgenError::InputTooSmall("b_seq expected length 5"));
    };
    if c_seq.len() != 5 {
        return Err(GradgenError::InputTooSmall("c_seq expected length 5"));
    };
    if jacobian_y.len() != 50 {
        return Err(GradgenError::OutputTooSmall(
            "jacobian_y expected length 50",
        ));
    };
    jacobian_y.fill(0.0_f64);
    let (temp_jacobian_y, helper_work) = work.split_at_mut(2);
    for stage_index in 0..5 {
        let a_seq_stage = &a_seq[stage_index * 2..((stage_index + 1) * 2)];
        let b_seq_stage = &b_seq[stage_index..stage_index + 1];
        let c_seq_stage = &c_seq[stage_index..stage_index + 1];
        zip_3_kernel_zip3_jf_a_seq_helper(
            a_seq_stage,
            b_seq_stage,
            c_seq_stage,
            temp_jacobian_y,
            helper_work,
        );
        for local_row in 0..1 {
            let dest_row = stage_index + local_row;
            let dest_start = (dest_row * 10) + stage_index * 2;
            let src_start = local_row * 2;
            jacobian_y[dest_start..(dest_start + 2)]
                .copy_from_slice(&temp_jacobian_y[src_start..(src_start + 2)]);
        }
    }
    Ok(())
}

fn zip_3_kernel_zip3_jf_a_seq_helper(
    _a: &[f64],
    b: &[f64],
    _c: &[f64],
    jacobian_y: &mut [f64],
    _work: &mut [f64],
) {
    jacobian_y[0] = b[0];
    jacobian_y[1] = 1.0_f64;
}

/// Return metadata describing [`zip_3_kernel_zip3_jf_b_seq`].
pub fn zip_3_kernel_zip3_jf_b_seq_meta() -> FunctionMetadata {
    FunctionMetadata {
        function_name: "zip_3_kernel_zip3_jf_b_seq",
        workspace_size: 1,
        input_names: &["a_seq", "b_seq", "c_seq"],
        input_sizes: &[10, 5, 5],
        output_names: &["jacobian_y"],
        output_sizes: &[25],
    }
}

/// Evaluate the generated symbolic function `zip_3_kernel_zip3_jf_b_seq`.
///
/// All numeric slices use the `f64` scalar type.
///
/// Arguments:
/// - `a_seq`:
///   input slice for the declared argument `a_seq`
///   Expected length: 10.
/// - `b_seq`:
///   input slice for the declared argument `b_seq`
///   Expected length: 5.
/// - `c_seq`:
///   input slice for the declared argument `c_seq`
///   Expected length: 5.
/// - `jacobian_y`:
///   output slice receiving the Jacobian block for declared result `y`
///   Expected length: 25.
/// - `work`: mutable workspace slice used to store intermediate values
///   while evaluating this kernel. Expected length: at least 1.
pub fn zip_3_kernel_zip3_jf_b_seq(
    a_seq: &[f64],
    b_seq: &[f64],
    c_seq: &[f64],
    jacobian_y: &mut [f64],
    work: &mut [f64],
) -> Result<(), GradgenError> {
    if work.is_empty() {
        return Err(GradgenError::WorkspaceTooSmall("work expected at least 1"));
    };
    if a_seq.len() != 10 {
        return Err(GradgenError::InputTooSmall("a_seq expected length 10"));
    };
    if b_seq.len() != 5 {
        return Err(GradgenError::InputTooSmall("b_seq expected length 5"));
    };
    if c_seq.len() != 5 {
        return Err(GradgenError::InputTooSmall("c_seq expected length 5"));
    };
    if jacobian_y.len() != 25 {
        return Err(GradgenError::OutputTooSmall(
            "jacobian_y expected length 25",
        ));
    };
    jacobian_y.fill(0.0_f64);
    let (temp_jacobian_y, helper_work) = work.split_at_mut(1);
    for stage_index in 0..5 {
        let a_seq_stage = &a_seq[stage_index * 2..((stage_index + 1) * 2)];
        let b_seq_stage = &b_seq[stage_index..stage_index + 1];
        let c_seq_stage = &c_seq[stage_index..stage_index + 1];
        zip_3_kernel_zip3_jf_b_seq_helper(
            a_seq_stage,
            b_seq_stage,
            c_seq_stage,
            temp_jacobian_y,
            helper_work,
        );
        for local_row in 0..1 {
            let dest_row = stage_index + local_row;
            let dest_start = (dest_row * 5) + stage_index;
            let src_start = local_row;
            jacobian_y[dest_start..(dest_start + 1)]
                .copy_from_slice(&temp_jacobian_y[src_start..(src_start + 1)]);
        }
    }
    Ok(())
}

fn zip_3_kernel_zip3_jf_b_seq_helper(
    a: &[f64],
    _b: &[f64],
    _c: &[f64],
    jacobian_y: &mut [f64],
    _work: &mut [f64],
) {
    jacobian_y[0] = a[0];
}

/// Return metadata describing [`zip_3_kernel_zip3_jf_c_seq`].
pub fn zip_3_kernel_zip3_jf_c_seq_meta() -> FunctionMetadata {
    FunctionMetadata {
        function_name: "zip_3_kernel_zip3_jf_c_seq",
        workspace_size: 2,
        input_names: &["a_seq", "b_seq", "c_seq"],
        input_sizes: &[10, 5, 5],
        output_names: &["jacobian_y"],
        output_sizes: &[25],
    }
}

/// Evaluate the generated symbolic function `zip_3_kernel_zip3_jf_c_seq`.
///
/// All numeric slices use the `f64` scalar type.
///
/// Arguments:
/// - `a_seq`:
///   input slice for the declared argument `a_seq`
///   Expected length: 10.
/// - `b_seq`:
///   input slice for the declared argument `b_seq`
///   Expected length: 5.
/// - `c_seq`:
///   input slice for the declared argument `c_seq`
///   Expected length: 5.
/// - `jacobian_y`:
///   output slice receiving the Jacobian block for declared result `y`
///   Expected length: 25.
/// - `work`: mutable workspace slice used to store intermediate values
///   while evaluating this kernel. Expected length: at least 2.
pub fn zip_3_kernel_zip3_jf_c_seq(
    a_seq: &[f64],
    b_seq: &[f64],
    c_seq: &[f64],
    jacobian_y: &mut [f64],
    work: &mut [f64],
) -> Result<(), GradgenError> {
    if work.len() < 2 {
        return Err(GradgenError::WorkspaceTooSmall("work expected at least 2"));
    };
    if a_seq.len() != 10 {
        return Err(GradgenError::InputTooSmall("a_seq expected length 10"));
    };
    if b_seq.len() != 5 {
        return Err(GradgenError::InputTooSmall("b_seq expected length 5"));
    };
    if c_seq.len() != 5 {
        return Err(GradgenError::InputTooSmall("c_seq expected length 5"));
    };
    if jacobian_y.len() != 25 {
        return Err(GradgenError::OutputTooSmall(
            "jacobian_y expected length 25",
        ));
    };
    jacobian_y.fill(0.0_f64);
    let (temp_jacobian_y, helper_work) = work.split_at_mut(1);
    for stage_index in 0..5 {
        let a_seq_stage = &a_seq[stage_index * 2..((stage_index + 1) * 2)];
        let b_seq_stage = &b_seq[stage_index..stage_index + 1];
        let c_seq_stage = &c_seq[stage_index..stage_index + 1];
        zip_3_kernel_zip3_jf_c_seq_helper(
            a_seq_stage,
            b_seq_stage,
            c_seq_stage,
            temp_jacobian_y,
            helper_work,
        );
        for local_row in 0..1 {
            let dest_row = stage_index + local_row;
            let dest_start = (dest_row * 5) + stage_index;
            let src_start = local_row;
            jacobian_y[dest_start..(dest_start + 1)]
                .copy_from_slice(&temp_jacobian_y[src_start..(src_start + 1)]);
        }
    }
    Ok(())
}

fn zip_3_kernel_zip3_jf_c_seq_helper(
    _a: &[f64],
    _b: &[f64],
    c: &[f64],
    jacobian_y: &mut [f64],
    work: &mut [f64],
) {
    work[0] = libm::cos(c[0]);
    jacobian_y[0] = work[0];
}
