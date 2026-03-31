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

/// Return metadata describing [`map_zip_kernel_unary_map_f`].
pub fn map_zip_kernel_unary_map_f_meta() -> FunctionMetadata {
    FunctionMetadata {
        function_name: "map_zip_kernel_unary_map_f",
        workspace_size: 2,
        input_names: &["x_seq"],
        input_sizes: &[6],
        output_names: &["y"],
        output_sizes: &[6],
    }
}

/// Evaluate the generated symbolic function `map_zip_kernel_unary_map_f`.
///
/// All numeric slices use the `f64` scalar type.
///
/// Arguments:
/// - `x_seq`:
///   input slice for the declared argument `x_seq`
///   Expected length: 6.
/// - `y`:
///   primal output slice for the declared result `y`
///   Expected length: 6.
/// - `work`: mutable workspace slice used to store intermediate values
///   while evaluating this kernel. Expected length: at least 2.
pub fn map_zip_kernel_unary_map_f(
    x_seq: &[f64],
    y: &mut [f64],
    work: &mut [f64],
) -> Result<(), GradgenError> {
    if work.len() < 2 {
        return Err(GradgenError::WorkspaceTooSmall("work expected at least 2"));
    };
    if x_seq.len() != 6 {
        return Err(GradgenError::InputTooSmall("x_seq expected length 6"));
    };
    if y.len() != 6 {
        return Err(GradgenError::OutputTooSmall("y expected length 6"));
    };
    let helper_work = &mut work[..2];
    for stage_index in 0..3 {
        let x_seq_stage = &x_seq[stage_index * 2..((stage_index + 1) * 2)];
        let y_stage = &mut y[stage_index * 2..((stage_index + 1) * 2)];
        map_zip_kernel_unary_map_f_helper(x_seq_stage, y_stage, helper_work);
    }
    Ok(())
}

fn map_zip_kernel_unary_map_f_helper(x: &[f64], y: &mut [f64], work: &mut [f64]) {
    work[0] = x[0] * x[0];
    work[1] = libm::sin(x[1]);
    work[0] += work[1];
    work[1] = 0.5_f64 * x[1];
    work[1] = -work[1];
    work[1] += x[0];
    y[0] = work[0];
    y[1] = work[1];
}

/// Return metadata describing [`map_zip_kernel_unary_map_jf_x_seq`].
pub fn map_zip_kernel_unary_map_jf_x_seq_meta() -> FunctionMetadata {
    FunctionMetadata {
        function_name: "map_zip_kernel_unary_map_jf_x_seq",
        workspace_size: 6,
        input_names: &["x_seq"],
        input_sizes: &[6],
        output_names: &["jacobian_y"],
        output_sizes: &[36],
    }
}

/// Evaluate the generated symbolic function `map_zip_kernel_unary_map_jf_x_seq`.
///
/// All numeric slices use the `f64` scalar type.
///
/// Arguments:
/// - `x_seq`:
///   input slice for the declared argument `x_seq`
///   Expected length: 6.
/// - `jacobian_y`:
///   output slice receiving the Jacobian block for declared result `y`
///   Expected length: 36.
/// - `work`: mutable workspace slice used to store intermediate values
///   while evaluating this kernel. Expected length: at least 6.
pub fn map_zip_kernel_unary_map_jf_x_seq(
    x_seq: &[f64],
    jacobian_y: &mut [f64],
    work: &mut [f64],
) -> Result<(), GradgenError> {
    if work.len() < 6 {
        return Err(GradgenError::WorkspaceTooSmall("work expected at least 6"));
    };
    if x_seq.len() != 6 {
        return Err(GradgenError::InputTooSmall("x_seq expected length 6"));
    };
    if jacobian_y.len() != 36 {
        return Err(GradgenError::OutputTooSmall(
            "jacobian_y expected length 36",
        ));
    };
    jacobian_y.fill(0.0_f64);
    let (temp_jacobian_y, helper_work) = work.split_at_mut(4);
    for stage_index in 0..3 {
        let x_seq_stage = &x_seq[stage_index * 2..((stage_index + 1) * 2)];
        map_zip_kernel_unary_map_jf_x_seq_helper(x_seq_stage, temp_jacobian_y, helper_work);
        for local_row in 0..2 {
            let dest_row = stage_index * 2 + local_row;
            let dest_start = (dest_row * 6) + stage_index * 2;
            let src_start = local_row * 2;
            jacobian_y[dest_start..(dest_start + 2)]
                .copy_from_slice(&temp_jacobian_y[src_start..(src_start + 2)]);
        }
    }
    Ok(())
}

fn map_zip_kernel_unary_map_jf_x_seq_helper(x: &[f64], jacobian_y: &mut [f64], work: &mut [f64]) {
    work[0] = 2.0_f64 * x[0];
    work[1] = libm::cos(x[1]);
    jacobian_y[0] = work[0];
    jacobian_y[1] = work[1];
    jacobian_y[2] = 1.0_f64;
    jacobian_y[3] = -0.5_f64;
}

/// Return metadata describing [`map_zip_kernel_binary_zip_f`].
pub fn map_zip_kernel_binary_zip_f_meta() -> FunctionMetadata {
    FunctionMetadata {
        function_name: "map_zip_kernel_binary_zip_f",
        workspace_size: 3,
        input_names: &["a_seq", "b_seq"],
        input_sizes: &[6, 6],
        output_names: &["z"],
        output_sizes: &[6],
    }
}

/// Evaluate the generated symbolic function `map_zip_kernel_binary_zip_f`.
///
/// All numeric slices use the `f64` scalar type.
///
/// Arguments:
/// - `a_seq`:
///   input slice for the declared argument `a_seq`
///   Expected length: 6.
/// - `b_seq`:
///   input slice for the declared argument `b_seq`
///   Expected length: 6.
/// - `z`:
///   primal output slice for the declared result `z`
///   Expected length: 6.
/// - `work`: mutable workspace slice used to store intermediate values
///   while evaluating this kernel. Expected length: at least 3.
pub fn map_zip_kernel_binary_zip_f(
    a_seq: &[f64],
    b_seq: &[f64],
    z: &mut [f64],
    work: &mut [f64],
) -> Result<(), GradgenError> {
    if work.len() < 3 {
        return Err(GradgenError::WorkspaceTooSmall("work expected at least 3"));
    };
    if a_seq.len() != 6 {
        return Err(GradgenError::InputTooSmall("a_seq expected length 6"));
    };
    if b_seq.len() != 6 {
        return Err(GradgenError::InputTooSmall("b_seq expected length 6"));
    };
    if z.len() != 6 {
        return Err(GradgenError::OutputTooSmall("z expected length 6"));
    };
    let helper_work = &mut work[..3];
    for stage_index in 0..3 {
        let a_seq_stage = &a_seq[stage_index * 2..((stage_index + 1) * 2)];
        let b_seq_stage = &b_seq[stage_index * 2..((stage_index + 1) * 2)];
        let z_stage = &mut z[stage_index * 2..((stage_index + 1) * 2)];
        map_zip_kernel_binary_zip_f_helper(a_seq_stage, b_seq_stage, z_stage, helper_work);
    }
    Ok(())
}

fn map_zip_kernel_binary_zip_f_helper(a: &[f64], b: &[f64], z: &mut [f64], work: &mut [f64]) {
    work[0] = 2.0_f64 * b[0];
    work[0] += a[0];
    work[1] = libm::cos(a[0]);
    work[2] = a[1] * b[1];
    work[1] += work[2];
    z[0] = work[0];
    z[1] = work[1];
}

/// Return metadata describing [`map_zip_kernel_binary_zip_jf_a_seq`].
pub fn map_zip_kernel_binary_zip_jf_a_seq_meta() -> FunctionMetadata {
    FunctionMetadata {
        function_name: "map_zip_kernel_binary_zip_jf_a_seq",
        workspace_size: 5,
        input_names: &["a_seq", "b_seq"],
        input_sizes: &[6, 6],
        output_names: &["jacobian_z"],
        output_sizes: &[36],
    }
}

/// Evaluate the generated symbolic function `map_zip_kernel_binary_zip_jf_a_seq`.
///
/// All numeric slices use the `f64` scalar type.
///
/// Arguments:
/// - `a_seq`:
///   input slice for the declared argument `a_seq`
///   Expected length: 6.
/// - `b_seq`:
///   input slice for the declared argument `b_seq`
///   Expected length: 6.
/// - `jacobian_z`:
///   output slice receiving the Jacobian block for declared result `z`
///   Expected length: 36.
/// - `work`: mutable workspace slice used to store intermediate values
///   while evaluating this kernel. Expected length: at least 5.
pub fn map_zip_kernel_binary_zip_jf_a_seq(
    a_seq: &[f64],
    b_seq: &[f64],
    jacobian_z: &mut [f64],
    work: &mut [f64],
) -> Result<(), GradgenError> {
    if work.len() < 5 {
        return Err(GradgenError::WorkspaceTooSmall("work expected at least 5"));
    };
    if a_seq.len() != 6 {
        return Err(GradgenError::InputTooSmall("a_seq expected length 6"));
    };
    if b_seq.len() != 6 {
        return Err(GradgenError::InputTooSmall("b_seq expected length 6"));
    };
    if jacobian_z.len() != 36 {
        return Err(GradgenError::OutputTooSmall(
            "jacobian_z expected length 36",
        ));
    };
    jacobian_z.fill(0.0_f64);
    let (temp_jacobian_z, helper_work) = work.split_at_mut(4);
    for stage_index in 0..3 {
        let a_seq_stage = &a_seq[stage_index * 2..((stage_index + 1) * 2)];
        let b_seq_stage = &b_seq[stage_index * 2..((stage_index + 1) * 2)];
        map_zip_kernel_binary_zip_jf_a_seq_helper(
            a_seq_stage,
            b_seq_stage,
            temp_jacobian_z,
            helper_work,
        );
        for local_row in 0..2 {
            let dest_row = stage_index * 2 + local_row;
            let dest_start = (dest_row * 6) + stage_index * 2;
            let src_start = local_row * 2;
            jacobian_z[dest_start..(dest_start + 2)]
                .copy_from_slice(&temp_jacobian_z[src_start..(src_start + 2)]);
        }
    }
    Ok(())
}

fn map_zip_kernel_binary_zip_jf_a_seq_helper(
    a: &[f64],
    b: &[f64],
    jacobian_z: &mut [f64],
    work: &mut [f64],
) {
    work[0] = libm::sin(a[0]);
    work[0] = -work[0];
    jacobian_z[0] = 1.0_f64;
    jacobian_z[1] = 0.0_f64;
    jacobian_z[2] = work[0];
    jacobian_z[3] = b[1];
}

/// Return metadata describing [`map_zip_kernel_binary_zip_jf_b_seq`].
pub fn map_zip_kernel_binary_zip_jf_b_seq_meta() -> FunctionMetadata {
    FunctionMetadata {
        function_name: "map_zip_kernel_binary_zip_jf_b_seq",
        workspace_size: 4,
        input_names: &["a_seq", "b_seq"],
        input_sizes: &[6, 6],
        output_names: &["jacobian_z"],
        output_sizes: &[36],
    }
}

/// Evaluate the generated symbolic function `map_zip_kernel_binary_zip_jf_b_seq`.
///
/// All numeric slices use the `f64` scalar type.
///
/// Arguments:
/// - `a_seq`:
///   input slice for the declared argument `a_seq`
///   Expected length: 6.
/// - `b_seq`:
///   input slice for the declared argument `b_seq`
///   Expected length: 6.
/// - `jacobian_z`:
///   output slice receiving the Jacobian block for declared result `z`
///   Expected length: 36.
/// - `work`: mutable workspace slice used to store intermediate values
///   while evaluating this kernel. Expected length: at least 4.
pub fn map_zip_kernel_binary_zip_jf_b_seq(
    a_seq: &[f64],
    b_seq: &[f64],
    jacobian_z: &mut [f64],
    work: &mut [f64],
) -> Result<(), GradgenError> {
    if work.len() < 4 {
        return Err(GradgenError::WorkspaceTooSmall("work expected at least 4"));
    };
    if a_seq.len() != 6 {
        return Err(GradgenError::InputTooSmall("a_seq expected length 6"));
    };
    if b_seq.len() != 6 {
        return Err(GradgenError::InputTooSmall("b_seq expected length 6"));
    };
    if jacobian_z.len() != 36 {
        return Err(GradgenError::OutputTooSmall(
            "jacobian_z expected length 36",
        ));
    };
    jacobian_z.fill(0.0_f64);
    let (temp_jacobian_z, helper_work) = work.split_at_mut(4);
    for stage_index in 0..3 {
        let a_seq_stage = &a_seq[stage_index * 2..((stage_index + 1) * 2)];
        let b_seq_stage = &b_seq[stage_index * 2..((stage_index + 1) * 2)];
        map_zip_kernel_binary_zip_jf_b_seq_helper(
            a_seq_stage,
            b_seq_stage,
            temp_jacobian_z,
            helper_work,
        );
        for local_row in 0..2 {
            let dest_row = stage_index * 2 + local_row;
            let dest_start = (dest_row * 6) + stage_index * 2;
            let src_start = local_row * 2;
            jacobian_z[dest_start..(dest_start + 2)]
                .copy_from_slice(&temp_jacobian_z[src_start..(src_start + 2)]);
        }
    }
    Ok(())
}

fn map_zip_kernel_binary_zip_jf_b_seq_helper(
    a: &[f64],
    _b: &[f64],
    jacobian_z: &mut [f64],
    _work: &mut [f64],
) {
    jacobian_z[0] = 2.0_f64;
    jacobian_z[1] = 0.0_f64;
    jacobian_z[2] = 0.0_f64;
    jacobian_z[3] = a[1];
}

/// Return metadata describing [`map_zip_kernel_composed_map_zip_f`].
pub fn map_zip_kernel_composed_map_zip_f_meta() -> FunctionMetadata {
    FunctionMetadata {
        function_name: "map_zip_kernel_composed_map_zip_f",
        workspace_size: 8,
        input_names: &["x_seq", "b_seq"],
        input_sizes: &[6, 6],
        output_names: &["z_seq"],
        output_sizes: &[6],
    }
}

/// Evaluate the generated symbolic function `map_zip_kernel_composed_map_zip_f`.
///
/// All numeric slices use the `f64` scalar type.
///
/// Arguments:
/// - `x_seq`:
///   input slice for the declared argument `x_seq`
///   Expected length: 6.
/// - `b_seq`:
///   input slice for the declared argument `b_seq`
///   Expected length: 6.
/// - `z_seq`:
///   primal output slice for the declared result `z_seq`
///   Expected length: 6.
/// - `work`: mutable workspace slice used to store intermediate values
///   while evaluating this kernel. Expected length: at least 8.
pub fn map_zip_kernel_composed_map_zip_f(
    x_seq: &[f64],
    b_seq: &[f64],
    z_seq: &mut [f64],
    work: &mut [f64],
) -> Result<(), GradgenError> {
    if work.len() < 8 {
        return Err(GradgenError::WorkspaceTooSmall("work expected at least 8"));
    };
    if x_seq.len() != 6 {
        return Err(GradgenError::InputTooSmall("x_seq expected length 6"));
    };
    if b_seq.len() != 6 {
        return Err(GradgenError::InputTooSmall("b_seq expected length 6"));
    };
    if z_seq.len() != 6 {
        return Err(GradgenError::OutputTooSmall("z_seq expected length 6"));
    };
    work[0] = x_seq[0] * x_seq[0];
    work[1] = libm::sin(x_seq[1]);
    work[0] += work[1];
    work[1] = b_seq[0] * b_seq[0];
    work[2] = libm::sin(b_seq[1]);
    work[1] += work[2];
    work[1] *= 2.0_f64;
    work[1] += work[0];
    work[0] = libm::cos(work[0]);
    work[2] = 0.5_f64 * b_seq[1];
    work[2] = -work[2];
    work[2] += b_seq[0];
    work[3] = 0.5_f64 * x_seq[1];
    work[3] = -work[3];
    work[3] += x_seq[0];
    work[2] *= work[3];
    work[0] += work[2];
    work[2] = x_seq[2] * x_seq[2];
    work[3] = libm::sin(x_seq[3]);
    work[2] += work[3];
    work[3] = b_seq[2] * b_seq[2];
    work[4] = libm::sin(b_seq[3]);
    work[3] += work[4];
    work[3] *= 2.0_f64;
    work[3] += work[2];
    work[2] = libm::cos(work[2]);
    work[4] = 0.5_f64 * b_seq[3];
    work[4] = -work[4];
    work[4] += b_seq[2];
    work[5] = 0.5_f64 * x_seq[3];
    work[5] = -work[5];
    work[5] += x_seq[2];
    work[4] *= work[5];
    work[2] += work[4];
    work[4] = x_seq[4] * x_seq[4];
    work[5] = libm::sin(x_seq[5]);
    work[4] += work[5];
    work[5] = b_seq[4] * b_seq[4];
    work[6] = libm::sin(b_seq[5]);
    work[5] += work[6];
    work[5] *= 2.0_f64;
    work[5] += work[4];
    work[4] = libm::cos(work[4]);
    work[6] = 0.5_f64 * b_seq[5];
    work[6] = -work[6];
    work[6] += b_seq[4];
    work[7] = 0.5_f64 * x_seq[5];
    work[7] = -work[7];
    work[7] += x_seq[4];
    work[6] *= work[7];
    work[4] += work[6];
    z_seq[0] = work[1];
    z_seq[1] = work[0];
    z_seq[2] = work[3];
    z_seq[3] = work[2];
    z_seq[4] = work[5];
    z_seq[5] = work[4];
    Ok(())
}

/// Return metadata describing [`map_zip_kernel_composed_map_zip_jf_x_seq`].
pub fn map_zip_kernel_composed_map_zip_jf_x_seq_meta() -> FunctionMetadata {
    FunctionMetadata {
        function_name: "map_zip_kernel_composed_map_zip_jf_x_seq",
        workspace_size: 13,
        input_names: &["x_seq", "b_seq"],
        input_sizes: &[6, 6],
        output_names: &["jacobian_z_seq"],
        output_sizes: &[36],
    }
}

/// Evaluate the generated symbolic function `map_zip_kernel_composed_map_zip_jf_x_seq`.
///
/// All numeric slices use the `f64` scalar type.
///
/// Arguments:
/// - `x_seq`:
///   input slice for the declared argument `x_seq`
///   Expected length: 6.
/// - `b_seq`:
///   input slice for the declared argument `b_seq`
///   Expected length: 6.
/// - `jacobian_z_seq`:
///   output slice receiving the Jacobian block for declared result
///   `z_seq`
///   Expected length: 36.
/// - `work`: mutable workspace slice used to store intermediate values
///   while evaluating this kernel. Expected length: at least 13.
pub fn map_zip_kernel_composed_map_zip_jf_x_seq(
    x_seq: &[f64],
    b_seq: &[f64],
    jacobian_z_seq: &mut [f64],
    work: &mut [f64],
) -> Result<(), GradgenError> {
    if work.len() < 13 {
        return Err(GradgenError::WorkspaceTooSmall("work expected at least 13"));
    };
    if x_seq.len() != 6 {
        return Err(GradgenError::InputTooSmall("x_seq expected length 6"));
    };
    if b_seq.len() != 6 {
        return Err(GradgenError::InputTooSmall("b_seq expected length 6"));
    };
    if jacobian_z_seq.len() != 36 {
        return Err(GradgenError::OutputTooSmall(
            "jacobian_z_seq expected length 36",
        ));
    };
    work[0] = 2.0_f64 * x_seq[0];
    work[1] = libm::cos(x_seq[1]);
    work[2] = 0.5_f64 * b_seq[1];
    work[2] = -work[2];
    work[2] += b_seq[0];
    work[3] = x_seq[0] * x_seq[0];
    work[4] = libm::sin(x_seq[1]);
    work[3] += work[4];
    work[3] = libm::sin(work[3]);
    work[4] = work[3] * x_seq[0];
    work[4] *= -2.0_f64;
    work[4] += work[2];
    work[2] *= -0.5_f64;
    work[3] *= work[1];
    work[3] = -work[3];
    work[2] += work[3];
    work[3] = 2.0_f64 * x_seq[2];
    work[5] = libm::cos(x_seq[3]);
    work[6] = 0.5_f64 * b_seq[3];
    work[6] = -work[6];
    work[6] += b_seq[2];
    work[7] = x_seq[2] * x_seq[2];
    work[8] = libm::sin(x_seq[3]);
    work[7] += work[8];
    work[7] = libm::sin(work[7]);
    work[8] = work[7] * x_seq[2];
    work[8] *= -2.0_f64;
    work[8] += work[6];
    work[6] *= -0.5_f64;
    work[7] *= work[5];
    work[7] = -work[7];
    work[6] += work[7];
    work[7] = 2.0_f64 * x_seq[4];
    work[9] = libm::cos(x_seq[5]);
    work[10] = 0.5_f64 * b_seq[5];
    work[10] = -work[10];
    work[10] += b_seq[4];
    work[11] = x_seq[4] * x_seq[4];
    work[12] = libm::sin(x_seq[5]);
    work[11] += work[12];
    work[11] = libm::sin(work[11]);
    work[12] = work[11] * x_seq[4];
    work[12] *= -2.0_f64;
    work[12] += work[10];
    work[10] *= -0.5_f64;
    work[11] *= work[9];
    work[11] = -work[11];
    work[10] += work[11];
    jacobian_z_seq[0] = work[0];
    jacobian_z_seq[1] = work[1];
    jacobian_z_seq[2] = 0.0_f64;
    jacobian_z_seq[3] = 0.0_f64;
    jacobian_z_seq[4] = 0.0_f64;
    jacobian_z_seq[5] = 0.0_f64;
    jacobian_z_seq[6] = work[4];
    jacobian_z_seq[7] = work[2];
    jacobian_z_seq[8] = 0.0_f64;
    jacobian_z_seq[9] = 0.0_f64;
    jacobian_z_seq[10] = 0.0_f64;
    jacobian_z_seq[11] = 0.0_f64;
    jacobian_z_seq[12] = 0.0_f64;
    jacobian_z_seq[13] = 0.0_f64;
    jacobian_z_seq[14] = work[3];
    jacobian_z_seq[15] = work[5];
    jacobian_z_seq[16] = 0.0_f64;
    jacobian_z_seq[17] = 0.0_f64;
    jacobian_z_seq[18] = 0.0_f64;
    jacobian_z_seq[19] = 0.0_f64;
    jacobian_z_seq[20] = work[8];
    jacobian_z_seq[21] = work[6];
    jacobian_z_seq[22] = 0.0_f64;
    jacobian_z_seq[23] = 0.0_f64;
    jacobian_z_seq[24] = 0.0_f64;
    jacobian_z_seq[25] = 0.0_f64;
    jacobian_z_seq[26] = 0.0_f64;
    jacobian_z_seq[27] = 0.0_f64;
    jacobian_z_seq[28] = work[7];
    jacobian_z_seq[29] = work[9];
    jacobian_z_seq[30] = 0.0_f64;
    jacobian_z_seq[31] = 0.0_f64;
    jacobian_z_seq[32] = 0.0_f64;
    jacobian_z_seq[33] = 0.0_f64;
    jacobian_z_seq[34] = work[12];
    jacobian_z_seq[35] = work[10];
    Ok(())
}

/// Return metadata describing [`map_zip_kernel_composed_map_zip_jf_b_seq`].
pub fn map_zip_kernel_composed_map_zip_jf_b_seq_meta() -> FunctionMetadata {
    FunctionMetadata {
        function_name: "map_zip_kernel_composed_map_zip_jf_b_seq",
        workspace_size: 12,
        input_names: &["x_seq", "b_seq"],
        input_sizes: &[6, 6],
        output_names: &["jacobian_z_seq"],
        output_sizes: &[36],
    }
}

/// Evaluate the generated symbolic function `map_zip_kernel_composed_map_zip_jf_b_seq`.
///
/// All numeric slices use the `f64` scalar type.
///
/// Arguments:
/// - `x_seq`:
///   input slice for the declared argument `x_seq`
///   Expected length: 6.
/// - `b_seq`:
///   input slice for the declared argument `b_seq`
///   Expected length: 6.
/// - `jacobian_z_seq`:
///   output slice receiving the Jacobian block for declared result
///   `z_seq`
///   Expected length: 36.
/// - `work`: mutable workspace slice used to store intermediate values
///   while evaluating this kernel. Expected length: at least 12.
pub fn map_zip_kernel_composed_map_zip_jf_b_seq(
    x_seq: &[f64],
    b_seq: &[f64],
    jacobian_z_seq: &mut [f64],
    work: &mut [f64],
) -> Result<(), GradgenError> {
    if work.len() < 12 {
        return Err(GradgenError::WorkspaceTooSmall("work expected at least 12"));
    };
    if x_seq.len() != 6 {
        return Err(GradgenError::InputTooSmall("x_seq expected length 6"));
    };
    if b_seq.len() != 6 {
        return Err(GradgenError::InputTooSmall("b_seq expected length 6"));
    };
    if jacobian_z_seq.len() != 36 {
        return Err(GradgenError::OutputTooSmall(
            "jacobian_z_seq expected length 36",
        ));
    };
    work[0] = 4.0_f64 * b_seq[0];
    work[1] = libm::cos(b_seq[1]);
    work[1] *= 2.0_f64;
    work[2] = 0.5_f64 * x_seq[1];
    work[2] = -work[2];
    work[2] += x_seq[0];
    work[3] = -0.5_f64 * work[2];
    work[4] = 4.0_f64 * b_seq[2];
    work[5] = libm::cos(b_seq[3]);
    work[5] *= 2.0_f64;
    work[6] = 0.5_f64 * x_seq[3];
    work[6] = -work[6];
    work[6] += x_seq[2];
    work[7] = -0.5_f64 * work[6];
    work[8] = 4.0_f64 * b_seq[4];
    work[9] = libm::cos(b_seq[5]);
    work[9] *= 2.0_f64;
    work[10] = 0.5_f64 * x_seq[5];
    work[10] = -work[10];
    work[10] += x_seq[4];
    work[11] = -0.5_f64 * work[10];
    jacobian_z_seq[0] = work[0];
    jacobian_z_seq[1] = work[1];
    jacobian_z_seq[2] = 0.0_f64;
    jacobian_z_seq[3] = 0.0_f64;
    jacobian_z_seq[4] = 0.0_f64;
    jacobian_z_seq[5] = 0.0_f64;
    jacobian_z_seq[6] = work[2];
    jacobian_z_seq[7] = work[3];
    jacobian_z_seq[8] = 0.0_f64;
    jacobian_z_seq[9] = 0.0_f64;
    jacobian_z_seq[10] = 0.0_f64;
    jacobian_z_seq[11] = 0.0_f64;
    jacobian_z_seq[12] = 0.0_f64;
    jacobian_z_seq[13] = 0.0_f64;
    jacobian_z_seq[14] = work[4];
    jacobian_z_seq[15] = work[5];
    jacobian_z_seq[16] = 0.0_f64;
    jacobian_z_seq[17] = 0.0_f64;
    jacobian_z_seq[18] = 0.0_f64;
    jacobian_z_seq[19] = 0.0_f64;
    jacobian_z_seq[20] = work[6];
    jacobian_z_seq[21] = work[7];
    jacobian_z_seq[22] = 0.0_f64;
    jacobian_z_seq[23] = 0.0_f64;
    jacobian_z_seq[24] = 0.0_f64;
    jacobian_z_seq[25] = 0.0_f64;
    jacobian_z_seq[26] = 0.0_f64;
    jacobian_z_seq[27] = 0.0_f64;
    jacobian_z_seq[28] = work[8];
    jacobian_z_seq[29] = work[9];
    jacobian_z_seq[30] = 0.0_f64;
    jacobian_z_seq[31] = 0.0_f64;
    jacobian_z_seq[32] = 0.0_f64;
    jacobian_z_seq[33] = 0.0_f64;
    jacobian_z_seq[34] = work[10];
    jacobian_z_seq[35] = work[11];
    Ok(())
}
