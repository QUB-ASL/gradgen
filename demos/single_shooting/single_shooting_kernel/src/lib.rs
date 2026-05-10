#![no_std]
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

/// Return metadata describing [`single_shooting_kernel_mpc_cost_f_states`].
pub fn single_shooting_kernel_mpc_cost_f_states_meta() -> FunctionMetadata {
    FunctionMetadata {
        function_name: "single_shooting_kernel_mpc_cost_f_states",
        workspace_size: 5,
        input_names: &["x0", "u_seq", "p"],
        input_sizes: &[2, 5, 2],
        output_names: &["cost", "x_traj"],
        output_sizes: &[1, 12],
    }
}

/// Evaluate the generated symbolic function `single_shooting_kernel_mpc_cost_f_states`.
///
/// All numeric slices use the `f64` scalar type.
///
/// Arguments:
/// - `x0`:
///   initial state vector for the single-shooting rollout
///   Expected length: 2.
/// - `u_seq`:
///   packed control-sequence slice laid out stage-major over the horizon
///   Expected length: 5.
/// - `p`:
///   shared parameter slice used at every stage and terminal evaluation
///   Expected length: 2.
/// - `cost`:
///   scalar rollout cost
///   Expected length: 1.
/// - `x_traj`:
///   packed rollout state trajectory
///   Expected length: 12.
/// - `work`: mutable workspace slice used to store intermediate values
///   while evaluating this kernel. Expected length: at least 5.
pub fn single_shooting_kernel_mpc_cost_f_states(
    x0: &[f64],
    u_seq: &[f64],
    p: &[f64],
    cost: &mut [f64],
    x_traj: &mut [f64],
    work: &mut [f64],
) -> Result<(), GradgenError> {
    if work.len() < 5 {
        return Err(GradgenError::WorkspaceTooSmall("work expected at least 5"));
    };
    if x0.len() != 2 {
        return Err(GradgenError::InputTooSmall("x0 expected length 2"));
    };
    if u_seq.len() != 5 {
        return Err(GradgenError::InputTooSmall("u_seq expected length 5"));
    };
    if p.len() != 2 {
        return Err(GradgenError::InputTooSmall("p expected length 2"));
    };
    if cost.len() != 1 {
        return Err(GradgenError::OutputTooSmall("cost expected length 1"));
    };
    if x_traj.len() != 12 {
        return Err(GradgenError::OutputTooSmall("x_traj expected length 12"));
    };
    let rest = work;
    let (state_buffers, rest) = rest.split_at_mut(4);
    let (current_state_buf, next_state_buf) = state_buffers.split_at_mut(2);
    let mut current_state = current_state_buf;
    let mut next_state = next_state_buf;
    let (scalar_buffer, stage_work) = rest.split_at_mut(1);
    current_state.copy_from_slice(x0);
    x_traj[0..2].copy_from_slice(x0);
    let mut total_cost = 0.0_f64;
    for stage_index in 0..5 {
        let u_t = &u_seq[stage_index..(stage_index + 1)];
        single_shooting_kernel_mpc_cost_stage_cost(
            current_state,
            u_t,
            p,
            scalar_buffer,
            stage_work,
        );
        total_cost += scalar_buffer[0];
        single_shooting_kernel_mpc_cost_dynamics(current_state, u_t, p, next_state, stage_work);
        x_traj[((stage_index + 1) * 2)..((stage_index + 2) * 2)].copy_from_slice(next_state);
        core::mem::swap(&mut current_state, &mut next_state);
    }
    single_shooting_kernel_mpc_cost_terminal_cost(current_state, p, scalar_buffer, stage_work);
    total_cost += scalar_buffer[0];
    cost[0] = total_cost;
    Ok(())
}

/// Evaluate the generated symbolic function `single_shooting_kernel_mpc_cost_dynamics`.
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
/// - `p`:
///   input slice for the declared argument `p`
///   Expected length: 2.
/// - `x_next`:
///   primal output slice for the declared result `x_next`
///   Expected length: 2.
/// - `work`: mutable workspace slice used to store intermediate values
///   while evaluating this kernel. Expected length: at least 0.
fn single_shooting_kernel_mpc_cost_dynamics(
    x: &[f64],
    u: &[f64],
    p: &[f64],
    x_next: &mut [f64],
    _work: &mut [f64],
) {
    x_next[0] = p[0] * x[1];
    x_next[0] += x[0];
    x_next[0] += u[0];
    x_next[1] = p[1] * u[0];
    x_next[1] += x[1];
    x_next[1] -= 0.5_f64 * x[0];
}

/// Evaluate the generated symbolic function `single_shooting_kernel_mpc_cost_terminal_cost`.
///
/// All numeric slices use the `f64` scalar type.
///
/// Arguments:
/// - `x`:
///   input slice for the declared argument `x`
///   Expected length: 2.
/// - `p`:
///   input slice for the declared argument `p`
///   Expected length: 2.
/// - `vf`:
///   primal output slice for the declared result `vf`
///   Expected length: 1.
/// - `work`: mutable workspace slice used to store intermediate values
///   while evaluating this kernel. Expected length: at least 0.
fn single_shooting_kernel_mpc_cost_terminal_cost(
    x: &[f64],
    p: &[f64],
    vf: &mut [f64],
    _work: &mut [f64],
) {
    vf[0] = 0.5_f64 * (x[1] * x[1]);
    vf[0] += 3.0_f64 * (x[0] * x[0]);
    vf[0] += p[1] * x[0];
}

/// Evaluate the generated symbolic function `single_shooting_kernel_mpc_cost_stage_cost`.
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
/// - `p`:
///   input slice for the declared argument `p`
///   Expected length: 2.
/// - `ell`:
///   primal output slice for the declared result `ell`
///   Expected length: 1.
/// - `work`: mutable workspace slice used to store intermediate values
///   while evaluating this kernel. Expected length: at least 0.
fn single_shooting_kernel_mpc_cost_stage_cost(
    x: &[f64],
    u: &[f64],
    p: &[f64],
    ell: &mut [f64],
    _work: &mut [f64],
) {
    ell[0] = 2.0_f64 * (x[1] * x[1]);
    ell[0] += x[0] * x[0];
    ell[0] += 0.3_f64 * (u[0] * u[0]);
    ell[0] += p[0] * u[0];
}

/// Return metadata describing [`single_shooting_kernel_mpc_cost_grad_states_u_seq`].
pub fn single_shooting_kernel_mpc_cost_grad_states_u_seq_meta() -> FunctionMetadata {
    FunctionMetadata {
        function_name: "single_shooting_kernel_mpc_cost_grad_states_u_seq",
        workspace_size: 11,
        input_names: &["x0", "u_seq", "p"],
        input_sizes: &[2, 5, 2],
        output_names: &["gradient_u_seq", "x_traj"],
        output_sizes: &[5, 12],
    }
}

/// Evaluate the generated symbolic function `single_shooting_kernel_mpc_cost_grad_states_u_seq`.
///
/// All numeric slices use the `f64` scalar type.
///
/// Arguments:
/// - `x0`:
///   initial state vector for the single-shooting rollout
///   Expected length: 2.
/// - `u_seq`:
///   packed control-sequence slice laid out stage-major over the horizon
///   Expected length: 5.
/// - `p`:
///   shared parameter slice used at every stage and terminal evaluation
///   Expected length: 2.
/// - `gradient_u_seq`:
///   packed gradient with respect to the control sequence
///   Expected length: 5.
/// - `x_traj`:
///   packed rollout state trajectory
///   Expected length: 12.
/// - `work`: mutable workspace slice used to store intermediate values
///   while evaluating this kernel. Expected length: at least 11.
pub fn single_shooting_kernel_mpc_cost_grad_states_u_seq(
    x0: &[f64],
    u_seq: &[f64],
    p: &[f64],
    gradient_u_seq: &mut [f64],
    x_traj: &mut [f64],
    work: &mut [f64],
) -> Result<(), GradgenError> {
    if work.len() < 11 {
        return Err(GradgenError::WorkspaceTooSmall("work expected at least 11"));
    };
    if x0.len() != 2 {
        return Err(GradgenError::InputTooSmall("x0 expected length 2"));
    };
    if u_seq.len() != 5 {
        return Err(GradgenError::InputTooSmall("u_seq expected length 5"));
    };
    if p.len() != 2 {
        return Err(GradgenError::InputTooSmall("p expected length 2"));
    };
    if gradient_u_seq.len() != 5 {
        return Err(GradgenError::OutputTooSmall(
            "gradient_u_seq expected length 5",
        ));
    };
    if x_traj.len() != 12 {
        return Err(GradgenError::OutputTooSmall("x_traj expected length 12"));
    };
    let rest = work;
    let (state_buffers, rest) = rest.split_at_mut(4);
    let (current_state_buf, next_state_buf) = state_buffers.split_at_mut(2);
    let mut current_state = current_state_buf;
    let mut next_state = next_state_buf;
    let (lambda_buffers, rest) = rest.split_at_mut(4);
    let (lambda_current, lambda_next) = lambda_buffers.split_at_mut(2);
    let (temp_state, rest) = rest.split_at_mut(2);
    let (temp_control, rest) = rest.split_at_mut(1);
    let stage_work = rest;
    current_state.copy_from_slice(x0);
    x_traj[0..2].copy_from_slice(x0);
    let state_history = &mut x_traj[2..];
    for stage_index in 0..5 {
        let u_t = &u_seq[stage_index..(stage_index + 1)];
        single_shooting_kernel_mpc_cost_dynamics(current_state, u_t, p, next_state, stage_work);
        state_history[(stage_index * 2)..((stage_index + 1) * 2)].copy_from_slice(next_state);
        core::mem::swap(&mut current_state, &mut next_state);
    }
    single_shooting_kernel_mpc_cost_terminal_cost_grad_x(
        current_state,
        p,
        lambda_current,
        stage_work,
    );
    for stage_index in (1..5).rev() {
        let x_t = &state_history[((stage_index - 1) * 2)..(stage_index * 2)];
        let u_t = &u_seq[stage_index..(stage_index + 1)];
        let grad_u_t = &mut gradient_u_seq[stage_index..(stage_index + 1)];
        single_shooting_kernel_mpc_cost_stage_cost_grad(
            x_t,
            u_t,
            p,
            lambda_next,
            grad_u_t,
            stage_work,
        );
        single_shooting_kernel_mpc_cost_dynamics_vjp(
            x_t,
            u_t,
            p,
            &lambda_current[..],
            temp_state,
            temp_control,
            stage_work,
        );
        lambda_next[0] += temp_state[0];
        lambda_next[1] += temp_state[1];
        grad_u_t[0] += temp_control[0];
        lambda_current.copy_from_slice(lambda_next);
    }
    let u_t = &u_seq[0..1];
    let grad_u_t = &mut gradient_u_seq[0..1];
    single_shooting_kernel_mpc_cost_stage_cost_grad(x0, u_t, p, temp_state, grad_u_t, stage_work);
    single_shooting_kernel_mpc_cost_dynamics_vjp(
        x0,
        u_t,
        p,
        &lambda_current[..],
        temp_state,
        temp_control,
        stage_work,
    );
    lambda_next[0] += temp_state[0];
    lambda_next[1] += temp_state[1];
    grad_u_t[0] += temp_control[0];
    Ok(())
}

/// Evaluate the generated symbolic function `single_shooting_kernel_mpc_cost_dynamics_vjp`.
///
/// All numeric slices use the `f64` scalar type.
///
/// Arguments:
/// - `_x`:
///   input slice for the declared argument `x`
///   Expected length: 2.
/// - `_u`:
///   input slice for the declared argument `u`
///   Expected length: 1.
/// - `p`:
///   input slice for the declared argument `p`
///   Expected length: 2.
/// - `cotangent_x_next`:
///   cotangent seed associated with declared result `x_next`; use this
///   slice when forming Jacobian-transpose-vector or reverse-mode
///   sensitivity terms
///   Expected length: 2.
/// - `vjp_x`:
///   output slice receiving the vector-Jacobian product for declared
///   input `x`
///   Expected length: 2.
/// - `vjp_u`:
///   output slice receiving the vector-Jacobian product for declared
///   input `u`
///   Expected length: 1.
/// - `work`: mutable workspace slice used to store intermediate values
///   while evaluating this kernel. Expected length: at least 0.
fn single_shooting_kernel_mpc_cost_dynamics_vjp(
    _x: &[f64],
    _u: &[f64],
    p: &[f64],
    cotangent_x_next: &[f64],
    vjp_x: &mut [f64],
    vjp_u: &mut [f64],
    _work: &mut [f64],
) {
    vjp_x[0] = -0.5_f64 * cotangent_x_next[1];
    vjp_x[0] += cotangent_x_next[0];
    vjp_x[1] = cotangent_x_next[0] * p[0];
    vjp_x[1] += cotangent_x_next[1];
    vjp_u[0] = cotangent_x_next[1] * p[1];
    vjp_u[0] += cotangent_x_next[0];
}

/// Evaluate the generated symbolic function `single_shooting_kernel_mpc_cost_stage_cost_grad`.
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
/// - `p`:
///   input slice for the declared argument `p`
///   Expected length: 2.
/// - `grad_x`:
///   primal output slice for the declared result `grad_x`
///   Expected length: 2.
/// - `grad_u`:
///   primal output slice for the declared result `grad_u`
///   Expected length: 1.
/// - `work`: mutable workspace slice used to store intermediate values
///   while evaluating this kernel. Expected length: at least 0.
fn single_shooting_kernel_mpc_cost_stage_cost_grad(
    x: &[f64],
    u: &[f64],
    p: &[f64],
    grad_x: &mut [f64],
    grad_u: &mut [f64],
    _work: &mut [f64],
) {
    grad_x[0] = 2.0_f64 * x[0];
    grad_x[1] = 4.0_f64 * x[1];
    grad_u[0] = 0.6_f64 * u[0];
    grad_u[0] += p[0];
}

/// Evaluate the generated symbolic function `single_shooting_kernel_mpc_cost_terminal_cost_grad_x`.
///
/// All numeric slices use the `f64` scalar type.
///
/// Arguments:
/// - `x`:
///   input slice for the declared argument `x`
///   Expected length: 2.
/// - `p`:
///   input slice for the declared argument `p`
///   Expected length: 2.
/// - `vf`:
///   primal output slice for the declared result `vf`
///   Expected length: 2.
/// - `work`: mutable workspace slice used to store intermediate values
///   while evaluating this kernel. Expected length: at least 0.
fn single_shooting_kernel_mpc_cost_terminal_cost_grad_x(
    x: &[f64],
    p: &[f64],
    vf: &mut [f64],
    _work: &mut [f64],
) {
    vf[0] = 6.0_f64 * x[0];
    vf[0] += p[1];
    vf[1] = x[1];
}

/// Return metadata describing [`single_shooting_kernel_mpc_cost_hvp_states_u_seq`].
pub fn single_shooting_kernel_mpc_cost_hvp_states_u_seq_meta() -> FunctionMetadata {
    FunctionMetadata {
        function_name: "single_shooting_kernel_mpc_cost_hvp_states_u_seq",
        workspace_size: 29,
        input_names: &["x0", "u_seq", "p", "v_u_seq"],
        input_sizes: &[2, 5, 2, 5],
        output_names: &["hvp_u_seq", "x_traj"],
        output_sizes: &[5, 12],
    }
}

/// Evaluate the generated symbolic function `single_shooting_kernel_mpc_cost_hvp_states_u_seq`.
///
/// All numeric slices use the `f64` scalar type.
///
/// Arguments:
/// - `x0`:
///   initial state vector for the single-shooting rollout
///   Expected length: 2.
/// - `u_seq`:
///   packed control-sequence slice laid out stage-major over the horizon
///   Expected length: 5.
/// - `p`:
///   shared parameter slice used at every stage and terminal evaluation
///   Expected length: 2.
/// - `v_u_seq`:
///   packed control-direction vector for the single-shooting HVP
///   Expected length: 5.
/// - `hvp_u_seq`:
///   packed Hessian-vector product with respect to the control sequence
///   Expected length: 5.
/// - `x_traj`:
///   packed rollout state trajectory
///   Expected length: 12.
/// - `work`: mutable workspace slice used to store intermediate values
///   while evaluating this kernel. Expected length: at least 29.
pub fn single_shooting_kernel_mpc_cost_hvp_states_u_seq(
    x0: &[f64],
    u_seq: &[f64],
    p: &[f64],
    v_u_seq: &[f64],
    hvp_u_seq: &mut [f64],
    x_traj: &mut [f64],
    work: &mut [f64],
) -> Result<(), GradgenError> {
    if work.len() < 29 {
        return Err(GradgenError::WorkspaceTooSmall("work expected at least 29"));
    };
    if x0.len() != 2 {
        return Err(GradgenError::InputTooSmall("x0 expected length 2"));
    };
    if u_seq.len() != 5 {
        return Err(GradgenError::InputTooSmall("u_seq expected length 5"));
    };
    if p.len() != 2 {
        return Err(GradgenError::InputTooSmall("p expected length 2"));
    };
    if v_u_seq.len() != 5 {
        return Err(GradgenError::InputTooSmall("v_u_seq expected length 5"));
    };
    if hvp_u_seq.len() != 5 {
        return Err(GradgenError::OutputTooSmall("hvp_u_seq expected length 5"));
    };
    if x_traj.len() != 12 {
        return Err(GradgenError::OutputTooSmall("x_traj expected length 12"));
    };
    let rest = work;
    let (tangent_history, rest) = rest.split_at_mut(10);
    let (state_buffers, rest) = rest.split_at_mut(4);
    let (current_state_buf, next_state_buf) = state_buffers.split_at_mut(2);
    let mut current_state = current_state_buf;
    let mut next_state = next_state_buf;
    let (tangent_buffers, rest) = rest.split_at_mut(4);
    let (current_tangent, next_tangent) = tangent_buffers.split_at_mut(2);
    let (lambda_buffers, rest) = rest.split_at_mut(4);
    let (lambda_current, lambda_next) = lambda_buffers.split_at_mut(2);
    let (mu_buffers, rest) = rest.split_at_mut(4);
    let (mu_current, mu_next) = mu_buffers.split_at_mut(2);
    let (temp_state, rest) = rest.split_at_mut(2);
    let (temp_control, rest) = rest.split_at_mut(1);
    let stage_work = rest;
    current_state.copy_from_slice(x0);
    current_tangent.fill(0.0_f64);
    x_traj[0..2].copy_from_slice(x0);
    let state_history = &mut x_traj[2..];
    for stage_index in 0..5 {
        let u_t = &u_seq[stage_index..(stage_index + 1)];
        let v_u_t = &v_u_seq[stage_index..(stage_index + 1)];
        single_shooting_kernel_mpc_cost_dynamics(current_state, u_t, p, next_state, stage_work);
        single_shooting_kernel_mpc_cost_dynamics_jvp(
            current_state,
            u_t,
            p,
            current_tangent,
            v_u_t,
            next_tangent,
            stage_work,
        );
        state_history[(stage_index * 2)..((stage_index + 1) * 2)].copy_from_slice(next_state);
        tangent_history[(stage_index * 2)..((stage_index + 1) * 2)].copy_from_slice(next_tangent);
        core::mem::swap(&mut current_state, &mut next_state);
        current_tangent.copy_from_slice(next_tangent);
    }
    single_shooting_kernel_mpc_cost_terminal_cost_grad_x(
        current_state,
        p,
        lambda_current,
        stage_work,
    );
    single_shooting_kernel_mpc_cost_terminal_cost_grad_x_jvp(
        current_state,
        p,
        current_tangent,
        mu_current,
        stage_work,
    );
    for stage_index in (1..5).rev() {
        let x_t = &state_history[((stage_index - 1) * 2)..(stage_index * 2)];
        let u_t = &u_seq[stage_index..(stage_index + 1)];
        let tangent_x_t = &tangent_history[((stage_index - 1) * 2)..(stage_index * 2)];
        let v_u_t = &v_u_seq[stage_index..(stage_index + 1)];
        let hvp_u_t = &mut hvp_u_seq[stage_index..(stage_index + 1)];
        single_shooting_kernel_mpc_cost_stage_cost_grad_u_jvp(
            x_t,
            u_t,
            p,
            tangent_x_t,
            v_u_t,
            hvp_u_t,
            stage_work,
        );
        single_shooting_kernel_mpc_cost_dynamics_vjp_u_jvp(
            x_t,
            u_t,
            p,
            &lambda_current[..],
            tangent_x_t,
            v_u_t,
            &mu_current[..],
            temp_control,
            stage_work,
        );
        hvp_u_t[0] += temp_control[0];
        single_shooting_kernel_mpc_cost_stage_cost_grad_x(x_t, u_t, p, lambda_next, stage_work);
        single_shooting_kernel_mpc_cost_dynamics_vjp_x(
            x_t,
            u_t,
            p,
            &lambda_current[..],
            temp_state,
            stage_work,
        );
        lambda_next[0] += temp_state[0];
        lambda_next[1] += temp_state[1];
        single_shooting_kernel_mpc_cost_stage_cost_grad_x_jvp(
            x_t,
            u_t,
            p,
            tangent_x_t,
            v_u_t,
            mu_next,
            stage_work,
        );
        single_shooting_kernel_mpc_cost_dynamics_vjp_x_jvp(
            x_t,
            u_t,
            p,
            &lambda_current[..],
            tangent_x_t,
            v_u_t,
            &mu_current[..],
            temp_state,
            stage_work,
        );
        mu_next[0] += temp_state[0];
        mu_next[1] += temp_state[1];
        lambda_current.copy_from_slice(lambda_next);
        mu_current.copy_from_slice(mu_next);
    }
    let u_t = &u_seq[0..1];
    next_tangent.fill(0.0_f64);
    let v_u_t = &v_u_seq[0..1];
    let hvp_u_t = &mut hvp_u_seq[0..1];
    single_shooting_kernel_mpc_cost_stage_cost_grad_u_jvp(
        x0,
        u_t,
        p,
        next_tangent,
        v_u_t,
        hvp_u_t,
        stage_work,
    );
    single_shooting_kernel_mpc_cost_dynamics_vjp_u_jvp(
        x0,
        u_t,
        p,
        &lambda_current[..],
        next_tangent,
        v_u_t,
        &mu_current[..],
        temp_control,
        stage_work,
    );
    hvp_u_t[0] += temp_control[0];
    Ok(())
}

/// Evaluate the generated symbolic function `single_shooting_kernel_mpc_cost_dynamics_jvp`.
///
/// All numeric slices use the `f64` scalar type.
///
/// Arguments:
/// - `_x`:
///   input slice for the declared argument `x`
///   Expected length: 2.
/// - `_u`:
///   input slice for the declared argument `u`
///   Expected length: 1.
/// - `p`:
///   input slice for the declared argument `p`
///   Expected length: 2.
/// - `tangent_x`:
///   input slice for the declared argument `tangent_x`
///   Expected length: 2.
/// - `tangent_u`:
///   input slice for the declared argument `tangent_u`
///   Expected length: 1.
/// - `x_next`:
///   primal output slice for the declared result `x_next`
///   Expected length: 2.
/// - `work`: mutable workspace slice used to store intermediate values
///   while evaluating this kernel. Expected length: at least 0.
fn single_shooting_kernel_mpc_cost_dynamics_jvp(
    _x: &[f64],
    _u: &[f64],
    p: &[f64],
    tangent_x: &[f64],
    tangent_u: &[f64],
    x_next: &mut [f64],
    _work: &mut [f64],
) {
    x_next[0] = p[0] * tangent_x[1];
    x_next[0] += tangent_x[0];
    x_next[0] += tangent_u[0];
    x_next[1] = -0.5_f64 * tangent_x[0];
    x_next[1] += tangent_x[1];
    x_next[1] += p[1] * tangent_u[0];
}

/// Evaluate the generated symbolic function `single_shooting_kernel_mpc_cost_dynamics_vjp_x`.
///
/// All numeric slices use the `f64` scalar type.
///
/// Arguments:
/// - `_x`:
///   input slice for the declared argument `x`
///   Expected length: 2.
/// - `_u`:
///   input slice for the declared argument `u`
///   Expected length: 1.
/// - `p`:
///   input slice for the declared argument `p`
///   Expected length: 2.
/// - `cotangent_x_next`:
///   cotangent seed associated with declared result `x_next`; use this
///   slice when forming Jacobian-transpose-vector or reverse-mode
///   sensitivity terms
///   Expected length: 2.
/// - `vjp_x`:
///   output slice receiving the vector-Jacobian product for declared
///   input `x`
///   Expected length: 2.
/// - `work`: mutable workspace slice used to store intermediate values
///   while evaluating this kernel. Expected length: at least 0.
fn single_shooting_kernel_mpc_cost_dynamics_vjp_x(
    _x: &[f64],
    _u: &[f64],
    p: &[f64],
    cotangent_x_next: &[f64],
    vjp_x: &mut [f64],
    _work: &mut [f64],
) {
    vjp_x[0] = -0.5_f64 * cotangent_x_next[1];
    vjp_x[0] += cotangent_x_next[0];
    vjp_x[1] = cotangent_x_next[0] * p[0];
    vjp_x[1] += cotangent_x_next[1];
}

/// Evaluate the generated symbolic function `single_shooting_kernel_mpc_cost_stage_cost_grad_x`.
///
/// All numeric slices use the `f64` scalar type.
///
/// Arguments:
/// - `x`:
///   input slice for the declared argument `x`
///   Expected length: 2.
/// - `_u`:
///   input slice for the declared argument `u`
///   Expected length: 1.
/// - `_p`:
///   input slice for the declared argument `p`
///   Expected length: 2.
/// - `ell`:
///   primal output slice for the declared result `ell`
///   Expected length: 2.
/// - `work`: mutable workspace slice used to store intermediate values
///   while evaluating this kernel. Expected length: at least 0.
fn single_shooting_kernel_mpc_cost_stage_cost_grad_x(
    x: &[f64],
    _u: &[f64],
    _p: &[f64],
    ell: &mut [f64],
    _work: &mut [f64],
) {
    ell[0] = 2.0_f64 * x[0];
    ell[1] = 4.0_f64 * x[1];
}

/// Evaluate the generated symbolic function `single_shooting_kernel_mpc_cost_dynamics_vjp_x_jvp`.
///
/// All numeric slices use the `f64` scalar type.
///
/// Arguments:
/// - `_x`:
///   input slice for the declared argument `x`
///   Expected length: 2.
/// - `_u`:
///   input slice for the declared argument `u`
///   Expected length: 1.
/// - `p`:
///   input slice for the declared argument `p`
///   Expected length: 2.
/// - `_cotangent_x_next`:
///   cotangent seed associated with declared result `x_next`; use this
///   slice when forming Jacobian-transpose-vector or reverse-mode
///   sensitivity terms
///   Expected length: 2.
/// - `_tangent_x`:
///   input slice for the declared argument `tangent_x`
///   Expected length: 2.
/// - `_tangent_u`:
///   input slice for the declared argument `tangent_u`
///   Expected length: 1.
/// - `tangent_cotangent_x_next`:
///   input slice for the declared argument `tangent_cotangent_x_next`
///   Expected length: 2.
/// - `vjp_x`:
///   output slice receiving the vector-Jacobian product for declared
///   input `x`
///   Expected length: 2.
/// - `work`: mutable workspace slice used to store intermediate values
///   while evaluating this kernel. Expected length: at least 0.
#[allow(clippy::too_many_arguments)]
fn single_shooting_kernel_mpc_cost_dynamics_vjp_x_jvp(
    _x: &[f64],
    _u: &[f64],
    p: &[f64],
    _cotangent_x_next: &[f64],
    _tangent_x: &[f64],
    _tangent_u: &[f64],
    tangent_cotangent_x_next: &[f64],
    vjp_x: &mut [f64],
    _work: &mut [f64],
) {
    vjp_x[0] = -0.5_f64 * tangent_cotangent_x_next[1];
    vjp_x[0] += tangent_cotangent_x_next[0];
    vjp_x[1] = p[0] * tangent_cotangent_x_next[0];
    vjp_x[1] += tangent_cotangent_x_next[1];
}

/// Evaluate the generated symbolic function `single_shooting_kernel_mpc_cost_dynamics_vjp_u_jvp`.
///
/// All numeric slices use the `f64` scalar type.
///
/// Arguments:
/// - `_x`:
///   input slice for the declared argument `x`
///   Expected length: 2.
/// - `_u`:
///   input slice for the declared argument `u`
///   Expected length: 1.
/// - `p`:
///   input slice for the declared argument `p`
///   Expected length: 2.
/// - `_cotangent_x_next`:
///   cotangent seed associated with declared result `x_next`; use this
///   slice when forming Jacobian-transpose-vector or reverse-mode
///   sensitivity terms
///   Expected length: 2.
/// - `_tangent_x`:
///   input slice for the declared argument `tangent_x`
///   Expected length: 2.
/// - `_tangent_u`:
///   input slice for the declared argument `tangent_u`
///   Expected length: 1.
/// - `tangent_cotangent_x_next`:
///   input slice for the declared argument `tangent_cotangent_x_next`
///   Expected length: 2.
/// - `vjp_u`:
///   output slice receiving the vector-Jacobian product for declared
///   input `u`
///   Expected length: 1.
/// - `work`: mutable workspace slice used to store intermediate values
///   while evaluating this kernel. Expected length: at least 0.
#[allow(clippy::too_many_arguments)]
fn single_shooting_kernel_mpc_cost_dynamics_vjp_u_jvp(
    _x: &[f64],
    _u: &[f64],
    p: &[f64],
    _cotangent_x_next: &[f64],
    _tangent_x: &[f64],
    _tangent_u: &[f64],
    tangent_cotangent_x_next: &[f64],
    vjp_u: &mut [f64],
    _work: &mut [f64],
) {
    vjp_u[0] = p[1] * tangent_cotangent_x_next[1];
    vjp_u[0] += tangent_cotangent_x_next[0];
}

/// Evaluate the generated symbolic function `single_shooting_kernel_mpc_cost_stage_cost_grad_x_jvp`.
///
/// All numeric slices use the `f64` scalar type.
///
/// Arguments:
/// - `_x`:
///   input slice for the declared argument `x`
///   Expected length: 2.
/// - `_u`:
///   input slice for the declared argument `u`
///   Expected length: 1.
/// - `_p`:
///   input slice for the declared argument `p`
///   Expected length: 2.
/// - `tangent_x`:
///   input slice for the declared argument `tangent_x`
///   Expected length: 2.
/// - `_tangent_u`:
///   input slice for the declared argument `tangent_u`
///   Expected length: 1.
/// - `ell`:
///   primal output slice for the declared result `ell`
///   Expected length: 2.
/// - `work`: mutable workspace slice used to store intermediate values
///   while evaluating this kernel. Expected length: at least 0.
fn single_shooting_kernel_mpc_cost_stage_cost_grad_x_jvp(
    _x: &[f64],
    _u: &[f64],
    _p: &[f64],
    tangent_x: &[f64],
    _tangent_u: &[f64],
    ell: &mut [f64],
    _work: &mut [f64],
) {
    ell[0] = 2.0_f64 * tangent_x[0];
    ell[1] = 4.0_f64 * tangent_x[1];
}

/// Evaluate the generated symbolic function `single_shooting_kernel_mpc_cost_stage_cost_grad_u_jvp`.
///
/// All numeric slices use the `f64` scalar type.
///
/// Arguments:
/// - `_x`:
///   input slice for the declared argument `x`
///   Expected length: 2.
/// - `_u`:
///   input slice for the declared argument `u`
///   Expected length: 1.
/// - `_p`:
///   input slice for the declared argument `p`
///   Expected length: 2.
/// - `_tangent_x`:
///   input slice for the declared argument `tangent_x`
///   Expected length: 2.
/// - `tangent_u`:
///   input slice for the declared argument `tangent_u`
///   Expected length: 1.
/// - `ell`:
///   primal output slice for the declared result `ell`
///   Expected length: 1.
/// - `work`: mutable workspace slice used to store intermediate values
///   while evaluating this kernel. Expected length: at least 0.
fn single_shooting_kernel_mpc_cost_stage_cost_grad_u_jvp(
    _x: &[f64],
    _u: &[f64],
    _p: &[f64],
    _tangent_x: &[f64],
    tangent_u: &[f64],
    ell: &mut [f64],
    _work: &mut [f64],
) {
    ell[0] = 0.6_f64 * tangent_u[0];
}

/// Evaluate the generated symbolic function `single_shooting_kernel_mpc_cost_terminal_cost_grad_x_jvp`.
///
/// All numeric slices use the `f64` scalar type.
///
/// Arguments:
/// - `_x`:
///   input slice for the declared argument `x`
///   Expected length: 2.
/// - `_p`:
///   input slice for the declared argument `p`
///   Expected length: 2.
/// - `tangent_x`:
///   input slice for the declared argument `tangent_x`
///   Expected length: 2.
/// - `vf`:
///   primal output slice for the declared result `vf`
///   Expected length: 2.
/// - `work`: mutable workspace slice used to store intermediate values
///   while evaluating this kernel. Expected length: at least 0.
fn single_shooting_kernel_mpc_cost_terminal_cost_grad_x_jvp(
    _x: &[f64],
    _p: &[f64],
    tangent_x: &[f64],
    vf: &mut [f64],
    _work: &mut [f64],
) {
    vf[0] = 6.0_f64 * tangent_x[0];
    vf[1] = tangent_x[1];
}

/// Return metadata describing [`single_shooting_kernel_mpc_cost_f_grad_states_u_seq`].
pub fn single_shooting_kernel_mpc_cost_f_grad_states_u_seq_meta() -> FunctionMetadata {
    FunctionMetadata {
        function_name: "single_shooting_kernel_mpc_cost_f_grad_states_u_seq",
        workspace_size: 22,
        input_names: &["x0", "u_seq", "p"],
        input_sizes: &[2, 5, 2],
        output_names: &["cost", "gradient_u_seq", "x_traj"],
        output_sizes: &[1, 5, 12],
    }
}

/// Evaluate the generated symbolic function `single_shooting_kernel_mpc_cost_f_grad_states_u_seq`.
///
/// All numeric slices use the `f64` scalar type.
///
/// Arguments:
/// - `x0`:
///   initial state vector for the single-shooting rollout
///   Expected length: 2.
/// - `u_seq`:
///   packed control-sequence slice laid out stage-major over the horizon
///   Expected length: 5.
/// - `p`:
///   shared parameter slice used at every stage and terminal evaluation
///   Expected length: 2.
/// - `cost`:
///   scalar rollout cost
///   Expected length: 1.
/// - `gradient_u_seq`:
///   packed gradient with respect to the control sequence
///   Expected length: 5.
/// - `x_traj`:
///   packed rollout state trajectory
///   Expected length: 12.
/// - `work`: mutable workspace slice used to store intermediate values
///   while evaluating this kernel. Expected length: at least 22.
pub fn single_shooting_kernel_mpc_cost_f_grad_states_u_seq(
    x0: &[f64],
    u_seq: &[f64],
    p: &[f64],
    cost: &mut [f64],
    gradient_u_seq: &mut [f64],
    x_traj: &mut [f64],
    work: &mut [f64],
) -> Result<(), GradgenError> {
    if work.len() < 22 {
        return Err(GradgenError::WorkspaceTooSmall("work expected at least 22"));
    };
    if x0.len() != 2 {
        return Err(GradgenError::InputTooSmall("x0 expected length 2"));
    };
    if u_seq.len() != 5 {
        return Err(GradgenError::InputTooSmall("u_seq expected length 5"));
    };
    if p.len() != 2 {
        return Err(GradgenError::InputTooSmall("p expected length 2"));
    };
    if cost.len() != 1 {
        return Err(GradgenError::OutputTooSmall("cost expected length 1"));
    };
    if gradient_u_seq.len() != 5 {
        return Err(GradgenError::OutputTooSmall(
            "gradient_u_seq expected length 5",
        ));
    };
    if x_traj.len() != 12 {
        return Err(GradgenError::OutputTooSmall("x_traj expected length 12"));
    };
    let rest = work;
    let (stage_gradient_history, rest) = rest.split_at_mut(10);
    let (state_buffers, rest) = rest.split_at_mut(4);
    let (current_state_buf, next_state_buf) = state_buffers.split_at_mut(2);
    let mut current_state = current_state_buf;
    let mut next_state = next_state_buf;
    let (lambda_buffers, rest) = rest.split_at_mut(4);
    let (lambda_current, lambda_next) = lambda_buffers.split_at_mut(2);
    let (temp_state, rest) = rest.split_at_mut(2);
    let (temp_control, rest) = rest.split_at_mut(1);
    let (scalar_buffer, stage_work) = rest.split_at_mut(1);
    current_state.copy_from_slice(x0);
    x_traj[0..2].copy_from_slice(x0);
    let state_history = &mut x_traj[2..];
    let mut total_cost = 0.0_f64;
    for stage_index in 0..5 {
        let u_t = &u_seq[stage_index..(stage_index + 1)];
        let stage_grad_x_t =
            &mut stage_gradient_history[(stage_index * 2)..((stage_index + 1) * 2)];
        let grad_u_t = &mut gradient_u_seq[stage_index..(stage_index + 1)];
        single_shooting_kernel_mpc_cost_stage_cost_joint(
            current_state,
            u_t,
            p,
            scalar_buffer,
            stage_grad_x_t,
            grad_u_t,
            stage_work,
        );
        total_cost += scalar_buffer[0];
        single_shooting_kernel_mpc_cost_dynamics(current_state, u_t, p, next_state, stage_work);
        state_history[(stage_index * 2)..((stage_index + 1) * 2)].copy_from_slice(next_state);
        core::mem::swap(&mut current_state, &mut next_state);
    }
    single_shooting_kernel_mpc_cost_terminal_cost(current_state, p, scalar_buffer, stage_work);
    total_cost += scalar_buffer[0];
    cost[0] = total_cost;
    single_shooting_kernel_mpc_cost_terminal_cost_grad_x(
        current_state,
        p,
        lambda_current,
        stage_work,
    );
    for stage_index in (1..5).rev() {
        let x_t = &state_history[((stage_index - 1) * 2)..(stage_index * 2)];
        let u_t = &u_seq[stage_index..(stage_index + 1)];
        let grad_x_t = &stage_gradient_history[(stage_index * 2)..((stage_index + 1) * 2)];
        let grad_u_t = &mut gradient_u_seq[stage_index..(stage_index + 1)];
        lambda_next.copy_from_slice(grad_x_t);
        single_shooting_kernel_mpc_cost_dynamics_vjp(
            x_t,
            u_t,
            p,
            &lambda_current[..],
            temp_state,
            temp_control,
            stage_work,
        );
        lambda_next[0] += temp_state[0];
        lambda_next[1] += temp_state[1];
        grad_u_t[0] += temp_control[0];
        lambda_current.copy_from_slice(lambda_next);
    }
    let u_t = &u_seq[0..1];
    let grad_x_t = &stage_gradient_history[0..2];
    let grad_u_t = &mut gradient_u_seq[0..1];
    lambda_next.copy_from_slice(grad_x_t);
    single_shooting_kernel_mpc_cost_dynamics_vjp(
        x0,
        u_t,
        p,
        &lambda_current[..],
        temp_state,
        temp_control,
        stage_work,
    );
    lambda_next[0] += temp_state[0];
    lambda_next[1] += temp_state[1];
    grad_u_t[0] += temp_control[0];
    Ok(())
}

/// Evaluate the generated symbolic function `single_shooting_kernel_mpc_cost_stage_cost_joint`.
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
/// - `p`:
///   input slice for the declared argument `p`
///   Expected length: 2.
/// - `ell`:
///   primal output slice for the declared result `ell`
///   Expected length: 1.
/// - `grad_x`:
///   primal output slice for the declared result `grad_x`
///   Expected length: 2.
/// - `grad_u`:
///   primal output slice for the declared result `grad_u`
///   Expected length: 1.
/// - `work`: mutable workspace slice used to store intermediate values
///   while evaluating this kernel. Expected length: at least 0.
fn single_shooting_kernel_mpc_cost_stage_cost_joint(
    x: &[f64],
    u: &[f64],
    p: &[f64],
    ell: &mut [f64],
    grad_x: &mut [f64],
    grad_u: &mut [f64],
    _work: &mut [f64],
) {
    ell[0] = 2.0_f64 * (x[1] * x[1]);
    ell[0] += x[0] * x[0];
    ell[0] += 0.3_f64 * (u[0] * u[0]);
    ell[0] += p[0] * u[0];
    grad_x[0] = 2.0_f64 * x[0];
    grad_x[1] = 4.0_f64 * x[1];
    grad_u[0] = 0.6_f64 * u[0];
    grad_u[0] += p[0];
}
