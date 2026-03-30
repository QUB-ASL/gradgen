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

/// Return metadata describing [`single_shooting_kernel_mpc_cost_f_states`].
pub fn single_shooting_kernel_mpc_cost_f_states_meta() -> FunctionMetadata {
    FunctionMetadata {
        function_name: "single_shooting_kernel_mpc_cost_f_states",
        workspace_size: 8,
        input_names: &["x0", "u_seq", "p"],
        input_sizes: &[2, 20, 2],
        output_names: &["cost", "x_traj"],
        output_sizes: &[1, 42],
    }
}

/// Evaluate the generated symbolic function `single_shooting_kernel_mpc_cost_f_states`.
///
/// All numeric slices use the `f64` scalar type.
///
/// Arguments:
/// - `x0`:
///   initial state slice
///   Expected length: 2.
/// - `u_seq`:
///   packed control-sequence slice laid out stage-major over the horizon
///   Expected length: 20.
/// - `p`:
///   shared parameter slice used at every stage and terminal evaluation
///   Expected length: 2.
/// - `cost`:
///   total cost output
///   Expected length: 1.
/// - `x_traj`:
///   packed rollout state trajectory including x0 through xN
///   Expected length: 42.
/// - `work`: mutable workspace slice used to store intermediate values
///   while evaluating this kernel. Expected length: at least 8.
pub fn single_shooting_kernel_mpc_cost_f_states(
    x0: &[f64],
    u_seq: &[f64],
    p: &[f64],
    cost: &mut [f64],
    x_traj: &mut [f64],
    work: &mut [f64],
) -> Result<(), GradgenError> {
    if work.len() < 8 {
        return Err(GradgenError::WorkspaceTooSmall("work expected at least 8"));
    };
    if x0.len() != 2 {
        return Err(GradgenError::InputTooSmall("x0 expected length 2"));
    };
    if u_seq.len() != 20 {
        return Err(GradgenError::InputTooSmall("u_seq expected length 20"));
    };
    if p.len() != 2 {
        return Err(GradgenError::InputTooSmall("p expected length 2"));
    };
    if cost.len() != 1 {
        return Err(GradgenError::OutputTooSmall("cost expected length 1"));
    };
    if x_traj.len() != 42 {
        return Err(GradgenError::OutputTooSmall("x_traj expected length 42"));
    };
    let rest = work;
    let (state_buffers, rest) = rest.split_at_mut(4);
    let (current_state, next_state) = state_buffers.split_at_mut(2);
    let (scalar_buffer, stage_work) = rest.split_at_mut(1);
    current_state.copy_from_slice(x0);
    x_traj[0..2].copy_from_slice(x0);
    let mut total_cost = 0.0_f64;
    for stage_index in 0..20 {
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
        current_state.copy_from_slice(next_state);
        x_traj[((stage_index + 1) * 2)..((stage_index + 2) * 2)].copy_from_slice(next_state);
    }
    single_shooting_kernel_mpc_cost_terminal_cost(current_state, p, scalar_buffer, stage_work);
    total_cost += scalar_buffer[0];
    cost[0] = total_cost;
    Ok(())
}

fn single_shooting_kernel_mpc_cost_dynamics(
    x: &[f64],
    u: &[f64],
    p: &[f64],
    x_next: &mut [f64],
    work: &mut [f64],
) {
    assert!(
        work.len() >= 3,
        "work is length {} but should be at least 3",
        work.len()
    );
    assert_eq!(x.len(), 2, "x is length {} but should be 2", x.len());
    assert_eq!(u.len(), 1, "u is length {} but should be 1", u.len());
    assert_eq!(p.len(), 2, "p is length {} but should be 2", p.len());
    assert_eq!(
        x_next.len(),
        2,
        "x_next is length {} but should be 2",
        x_next.len()
    );
    work[0] = p[0] * x[1];
    work[0] += x[0];
    work[0] += u[0];
    work[1] = p[1] * u[0];
    work[1] += x[1];
    work[2] = 0.5_f64 * x[0];
    work[1] -= work[2];
    x_next[0] = work[0];
    x_next[1] = work[1];
}

fn single_shooting_kernel_mpc_cost_stage_cost(
    x: &[f64],
    u: &[f64],
    p: &[f64],
    ell: &mut [f64],
    work: &mut [f64],
) {
    assert!(
        work.len() >= 2,
        "work is length {} but should be at least 2",
        work.len()
    );
    assert_eq!(x.len(), 2, "x is length {} but should be 2", x.len());
    assert_eq!(u.len(), 1, "u is length {} but should be 1", u.len());
    assert_eq!(p.len(), 2, "p is length {} but should be 2", p.len());
    assert_eq!(ell.len(), 1, "ell is length {} but should be 1", ell.len());
    work[0] = 2.0_f64 * x[1];
    work[0] *= x[1];
    work[1] = x[0] * x[0];
    work[0] += work[1];
    work[1] = 0.3_f64 * u[0];
    work[1] *= u[0];
    work[0] += work[1];
    work[1] = p[0] * u[0];
    work[0] += work[1];
    ell[0] = work[0];
}

fn single_shooting_kernel_mpc_cost_terminal_cost(
    x: &[f64],
    p: &[f64],
    vf: &mut [f64],
    work: &mut [f64],
) {
    assert!(
        work.len() >= 2,
        "work is length {} but should be at least 2",
        work.len()
    );
    assert_eq!(x.len(), 2, "x is length {} but should be 2", x.len());
    assert_eq!(p.len(), 2, "p is length {} but should be 2", p.len());
    assert_eq!(vf.len(), 1, "vf is length {} but should be 1", vf.len());
    work[0] = 0.5_f64 * x[1];
    work[0] *= x[1];
    work[1] = 3.0_f64 * x[0];
    work[1] *= x[0];
    work[0] += work[1];
    work[1] = p[1] * x[0];
    work[0] += work[1];
    vf[0] = work[0];
}

/// Return metadata describing [`single_shooting_kernel_mpc_cost_grad_states_u_seq`].
pub fn single_shooting_kernel_mpc_cost_grad_states_u_seq_meta() -> FunctionMetadata {
    FunctionMetadata {
        function_name: "single_shooting_kernel_mpc_cost_grad_states_u_seq",
        workspace_size: 54,
        input_names: &["x0", "u_seq", "p"],
        input_sizes: &[2, 20, 2],
        output_names: &["gradient_u_seq", "x_traj"],
        output_sizes: &[20, 42],
    }
}

/// Evaluate the generated symbolic function `single_shooting_kernel_mpc_cost_grad_states_u_seq`.
///
/// All numeric slices use the `f64` scalar type.
///
/// Arguments:
/// - `x0`:
///   initial state slice
///   Expected length: 2.
/// - `u_seq`:
///   packed control-sequence slice laid out stage-major over the horizon
///   Expected length: 20.
/// - `p`:
///   shared parameter slice used at every stage and terminal evaluation
///   Expected length: 2.
/// - `gradient_u_seq`:
///   gradient with respect to the packed control sequence
///   Expected length: 20.
/// - `x_traj`:
///   packed rollout state trajectory including x0 through xN
///   Expected length: 42.
/// - `work`: mutable workspace slice used to store intermediate values
///   while evaluating this kernel. Expected length: at least 54.
pub fn single_shooting_kernel_mpc_cost_grad_states_u_seq(
    x0: &[f64],
    u_seq: &[f64],
    p: &[f64],
    gradient_u_seq: &mut [f64],
    x_traj: &mut [f64],
    work: &mut [f64],
) -> Result<(), GradgenError> {
    if work.len() < 54 {
        return Err(GradgenError::WorkspaceTooSmall("work expected at least 54"));
    };
    if x0.len() != 2 {
        return Err(GradgenError::InputTooSmall("x0 expected length 2"));
    };
    if u_seq.len() != 20 {
        return Err(GradgenError::InputTooSmall("u_seq expected length 20"));
    };
    if p.len() != 2 {
        return Err(GradgenError::InputTooSmall("p expected length 2"));
    };
    if gradient_u_seq.len() != 20 {
        return Err(GradgenError::OutputTooSmall(
            "gradient_u_seq expected length 20",
        ));
    };
    if x_traj.len() != 42 {
        return Err(GradgenError::OutputTooSmall("x_traj expected length 42"));
    };
    let (state_history, rest) = work.split_at_mut(40);
    let (state_buffers, rest) = rest.split_at_mut(4);
    let (current_state, next_state) = state_buffers.split_at_mut(2);
    let (lambda_buffers, rest) = rest.split_at_mut(4);
    let (lambda_current, lambda_next) = lambda_buffers.split_at_mut(2);
    let (temp_state, rest) = rest.split_at_mut(2);
    let (temp_control, rest) = rest.split_at_mut(1);
    let stage_work = rest;
    current_state.copy_from_slice(x0);
    x_traj[0..2].copy_from_slice(x0);
    for stage_index in 0..20 {
        let u_t = &u_seq[stage_index..(stage_index + 1)];
        single_shooting_kernel_mpc_cost_dynamics(current_state, u_t, p, next_state, stage_work);
        state_history[(stage_index * 2)..((stage_index + 1) * 2)].copy_from_slice(next_state);
        current_state.copy_from_slice(next_state);
        x_traj[((stage_index + 1) * 2)..((stage_index + 2) * 2)].copy_from_slice(next_state);
    }
    single_shooting_kernel_mpc_cost_terminal_cost_grad_x(
        current_state,
        p,
        lambda_current,
        stage_work,
    );
    for stage_index in (1..20).rev() {
        let x_t = &state_history[((stage_index - 1) * 2)..(stage_index * 2)];
        let u_t = &u_seq[stage_index..(stage_index + 1)];
        let grad_u_t = &mut gradient_u_seq[stage_index..(stage_index + 1)];
        single_shooting_kernel_mpc_cost_stage_cost_grad_u(x_t, u_t, p, grad_u_t, stage_work);
        single_shooting_kernel_mpc_cost_dynamics_vjp_u(
            x_t,
            u_t,
            p,
            &lambda_current[..],
            temp_control,
            stage_work,
        );
        grad_u_t[0] += temp_control[0];
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
        lambda_current.copy_from_slice(lambda_next);
    }
    let u_t = &u_seq[0..1];
    let grad_u_t = &mut gradient_u_seq[0..1];
    single_shooting_kernel_mpc_cost_stage_cost_grad_u(x0, u_t, p, grad_u_t, stage_work);
    single_shooting_kernel_mpc_cost_dynamics_vjp_u(
        x0,
        u_t,
        p,
        &lambda_current[..],
        temp_control,
        stage_work,
    );
    grad_u_t[0] += temp_control[0];
    Ok(())
}

fn single_shooting_kernel_mpc_cost_dynamics_vjp_x(
    x: &[f64],
    u: &[f64],
    p: &[f64],
    cotangent_x_next: &[f64],
    vjp_x: &mut [f64],
    work: &mut [f64],
) {
    assert!(
        work.len() >= 3,
        "work is length {} but should be at least 3",
        work.len()
    );
    assert_eq!(x.len(), 2, "x is length {} but should be 2", x.len());
    assert_eq!(u.len(), 1, "u is length {} but should be 1", u.len());
    assert_eq!(p.len(), 2, "p is length {} but should be 2", p.len());
    assert_eq!(
        cotangent_x_next.len(),
        2,
        "cotangent_x_next is length {} but should be 2",
        cotangent_x_next.len()
    );
    assert_eq!(
        vjp_x.len(),
        2,
        "vjp_x is length {} but should be 2",
        vjp_x.len()
    );
    work[0] = 0.0_f64 + cotangent_x_next[0];
    work[0] += 0.0_f64;
    work[1] = 0.0_f64 + cotangent_x_next[1];
    work[2] = -work[1];
    work[2] += 0.0_f64;
    work[2] *= 0.5_f64;
    work[2] += 0.0_f64;
    work[2] += work[0];
    work[2] += 0.0_f64;
    work[1] += 0.0_f64;
    work[1] += 0.0_f64;
    work[0] += 0.0_f64;
    work[0] *= p[0];
    work[0] += work[1];
    work[0] += 0.0_f64;
    vjp_x[0] = work[2];
    vjp_x[1] = work[0];
}

fn single_shooting_kernel_mpc_cost_dynamics_vjp_u(
    x: &[f64],
    u: &[f64],
    p: &[f64],
    cotangent_x_next: &[f64],
    vjp_u: &mut [f64],
    work: &mut [f64],
) {
    assert!(
        work.len() >= 2,
        "work is length {} but should be at least 2",
        work.len()
    );
    assert_eq!(x.len(), 2, "x is length {} but should be 2", x.len());
    assert_eq!(u.len(), 1, "u is length {} but should be 1", u.len());
    assert_eq!(p.len(), 2, "p is length {} but should be 2", p.len());
    assert_eq!(
        cotangent_x_next.len(),
        2,
        "cotangent_x_next is length {} but should be 2",
        cotangent_x_next.len()
    );
    assert_eq!(
        vjp_u.len(),
        1,
        "vjp_u is length {} but should be 1",
        vjp_u.len()
    );
    work[0] = 0.0_f64 + cotangent_x_next[1];
    work[0] += 0.0_f64;
    work[0] += 0.0_f64;
    work[0] *= p[1];
    work[0] += 0.0_f64;
    work[1] = 0.0_f64 + cotangent_x_next[0];
    work[0] += work[1];
    work[0] += 0.0_f64;
    vjp_u[0] = work[0];
}

fn single_shooting_kernel_mpc_cost_stage_cost_grad_x(
    x: &[f64],
    u: &[f64],
    p: &[f64],
    ell: &mut [f64],
    work: &mut [f64],
) {
    assert!(
        work.len() >= 3,
        "work is length {} but should be at least 3",
        work.len()
    );
    assert_eq!(x.len(), 2, "x is length {} but should be 2", x.len());
    assert_eq!(u.len(), 1, "u is length {} but should be 1", u.len());
    assert_eq!(p.len(), 2, "p is length {} but should be 2", p.len());
    assert_eq!(ell.len(), 2, "ell is length {} but should be 2", ell.len());
    work[0] = 0.0_f64 + 1.0_f64;
    work[0] += 0.0_f64;
    work[0] += 0.0_f64;
    work[0] += 0.0_f64;
    work[1] = work[0] * x[0];
    work[2] = 0.0_f64 + work[1];
    work[1] += work[2];
    work[2] = 2.0_f64 * x[1];
    work[2] *= work[0];
    work[2] += 0.0_f64;
    work[0] *= x[1];
    work[0] += 0.0_f64;
    work[0] *= 2.0_f64;
    work[0] += work[2];
    ell[0] = work[1];
    ell[1] = work[0];
}

fn single_shooting_kernel_mpc_cost_stage_cost_grad_u(
    x: &[f64],
    u: &[f64],
    p: &[f64],
    ell: &mut [f64],
    work: &mut [f64],
) {
    assert!(
        work.len() >= 3,
        "work is length {} but should be at least 3",
        work.len()
    );
    assert_eq!(x.len(), 2, "x is length {} but should be 2", x.len());
    assert_eq!(u.len(), 1, "u is length {} but should be 1", u.len());
    assert_eq!(p.len(), 2, "p is length {} but should be 2", p.len());
    assert_eq!(ell.len(), 1, "ell is length {} but should be 1", ell.len());
    work[0] = 0.0_f64 + 1.0_f64;
    work[0] += 0.0_f64;
    work[1] = work[0] * p[0];
    work[1] += 0.0_f64;
    work[0] += 0.0_f64;
    work[2] = 0.3_f64 * u[0];
    work[2] *= work[0];
    work[1] += work[2];
    work[0] *= u[0];
    work[0] += 0.0_f64;
    work[0] *= 0.3_f64;
    work[0] += work[1];
    ell[0] = work[0];
}

fn single_shooting_kernel_mpc_cost_terminal_cost_grad_x(
    x: &[f64],
    p: &[f64],
    vf: &mut [f64],
    work: &mut [f64],
) {
    assert!(
        work.len() >= 3,
        "work is length {} but should be at least 3",
        work.len()
    );
    assert_eq!(x.len(), 2, "x is length {} but should be 2", x.len());
    assert_eq!(p.len(), 2, "p is length {} but should be 2", p.len());
    assert_eq!(vf.len(), 2, "vf is length {} but should be 2", vf.len());
    work[0] = 0.0_f64 + 1.0_f64;
    work[0] += 0.0_f64;
    work[1] = work[0] * p[1];
    work[1] += 0.0_f64;
    work[0] += 0.0_f64;
    work[2] = 3.0_f64 * x[0];
    work[2] *= work[0];
    work[1] += work[2];
    work[2] = work[0] * x[0];
    work[2] += 0.0_f64;
    work[2] *= 3.0_f64;
    work[1] += work[2];
    work[2] = 0.5_f64 * x[1];
    work[2] *= work[0];
    work[2] += 0.0_f64;
    work[0] *= x[1];
    work[0] += 0.0_f64;
    work[0] *= 0.5_f64;
    work[0] += work[2];
    vf[0] = work[1];
    vf[1] = work[0];
}

/// Return metadata describing [`single_shooting_kernel_mpc_cost_hvp_states_u_seq`].
pub fn single_shooting_kernel_mpc_cost_hvp_states_u_seq_meta() -> FunctionMetadata {
    FunctionMetadata {
        function_name: "single_shooting_kernel_mpc_cost_hvp_states_u_seq",
        workspace_size: 108,
        input_names: &["x0", "u_seq", "p", "v_u_seq"],
        input_sizes: &[2, 20, 2, 20],
        output_names: &["hvp_u_seq", "x_traj"],
        output_sizes: &[20, 42],
    }
}

/// Evaluate the generated symbolic function `single_shooting_kernel_mpc_cost_hvp_states_u_seq`.
///
/// All numeric slices use the `f64` scalar type.
///
/// Arguments:
/// - `x0`:
///   initial state slice
///   Expected length: 2.
/// - `u_seq`:
///   packed control-sequence slice laid out stage-major over the horizon
///   Expected length: 20.
/// - `p`:
///   shared parameter slice used at every stage and terminal evaluation
///   Expected length: 2.
/// - `v_u_seq`:
///   packed control-sequence direction slice laid out stage-major over
///   the horizon
///   Expected length: 20.
/// - `hvp_u_seq`:
///   Hessian-vector product with respect to the packed control sequence
///   Expected length: 20.
/// - `x_traj`:
///   packed rollout state trajectory including x0 through xN
///   Expected length: 42.
/// - `work`: mutable workspace slice used to store intermediate values
///   while evaluating this kernel. Expected length: at least 108.
pub fn single_shooting_kernel_mpc_cost_hvp_states_u_seq(
    x0: &[f64],
    u_seq: &[f64],
    p: &[f64],
    v_u_seq: &[f64],
    hvp_u_seq: &mut [f64],
    x_traj: &mut [f64],
    work: &mut [f64],
) -> Result<(), GradgenError> {
    if work.len() < 108 {
        return Err(GradgenError::WorkspaceTooSmall(
            "work expected at least 108",
        ));
    };
    if x0.len() != 2 {
        return Err(GradgenError::InputTooSmall("x0 expected length 2"));
    };
    if u_seq.len() != 20 {
        return Err(GradgenError::InputTooSmall("u_seq expected length 20"));
    };
    if p.len() != 2 {
        return Err(GradgenError::InputTooSmall("p expected length 2"));
    };
    if v_u_seq.len() != 20 {
        return Err(GradgenError::InputTooSmall("v_u_seq expected length 20"));
    };
    if hvp_u_seq.len() != 20 {
        return Err(GradgenError::OutputTooSmall("hvp_u_seq expected length 20"));
    };
    if x_traj.len() != 42 {
        return Err(GradgenError::OutputTooSmall("x_traj expected length 42"));
    };
    let (state_history, rest) = work.split_at_mut(40);
    let (tangent_history, rest) = rest.split_at_mut(40);
    let (state_buffers, rest) = rest.split_at_mut(4);
    let (current_state, next_state) = state_buffers.split_at_mut(2);
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
    for stage_index in 0..20 {
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
        current_state.copy_from_slice(next_state);
        current_tangent.copy_from_slice(next_tangent);
        x_traj[((stage_index + 1) * 2)..((stage_index + 2) * 2)].copy_from_slice(next_state);
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
    for stage_index in (1..20).rev() {
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

fn single_shooting_kernel_mpc_cost_dynamics_jvp(
    x: &[f64],
    u: &[f64],
    p: &[f64],
    tangent_x: &[f64],
    tangent_u: &[f64],
    x_next: &mut [f64],
    work: &mut [f64],
) {
    assert!(
        work.len() >= 5,
        "work is length {} but should be at least 5",
        work.len()
    );
    assert_eq!(x.len(), 2, "x is length {} but should be 2", x.len());
    assert_eq!(u.len(), 1, "u is length {} but should be 1", u.len());
    assert_eq!(p.len(), 2, "p is length {} but should be 2", p.len());
    assert_eq!(
        tangent_x.len(),
        2,
        "tangent_x is length {} but should be 2",
        tangent_x.len()
    );
    assert_eq!(
        tangent_u.len(),
        1,
        "tangent_u is length {} but should be 1",
        tangent_u.len()
    );
    assert_eq!(
        x_next.len(),
        2,
        "x_next is length {} but should be 2",
        x_next.len()
    );
    work[0] = 0.0_f64 * x[1];
    work[1] = p[0] * tangent_x[1];
    work[1] += work[0];
    work[1] += tangent_x[0];
    work[1] += 0.0_f64;
    work[1] += 0.0_f64;
    work[2] = 0.0_f64 * p[0];
    work[0] += work[2];
    work[0] += 0.0_f64;
    work[0] += tangent_u[0];
    work[0] += work[1];
    work[1] = 0.0_f64 * p[1];
    work[2] = 0.0_f64 * u[0];
    work[1] += work[2];
    work[1] += tangent_x[1];
    work[3] = 0.0_f64 * x[0];
    work[4] = 0.5_f64 * tangent_x[0];
    work[4] += work[3];
    work[1] -= work[4];
    work[1] += 0.0_f64;
    work[4] = p[1] * tangent_u[0];
    work[2] += work[4];
    work[2] += 0.0_f64;
    work[4] = 0.0_f64 * 0.5_f64;
    work[3] += work[4];
    work[2] -= work[3];
    work[1] += work[2];
    x_next[0] = work[0];
    x_next[1] = work[1];
}

#[allow(clippy::too_many_arguments)]
fn single_shooting_kernel_mpc_cost_dynamics_vjp_x_jvp(
    x: &[f64],
    u: &[f64],
    p: &[f64],
    cotangent_x_next: &[f64],
    tangent_x: &[f64],
    tangent_u: &[f64],
    tangent_cotangent_x_next: &[f64],
    vjp_x: &mut [f64],
    work: &mut [f64],
) {
    assert!(
        work.len() >= 6,
        "work is length {} but should be at least 6",
        work.len()
    );
    assert_eq!(x.len(), 2, "x is length {} but should be 2", x.len());
    assert_eq!(u.len(), 1, "u is length {} but should be 1", u.len());
    assert_eq!(p.len(), 2, "p is length {} but should be 2", p.len());
    assert_eq!(
        cotangent_x_next.len(),
        2,
        "cotangent_x_next is length {} but should be 2",
        cotangent_x_next.len()
    );
    assert_eq!(
        tangent_x.len(),
        2,
        "tangent_x is length {} but should be 2",
        tangent_x.len()
    );
    assert_eq!(
        tangent_u.len(),
        1,
        "tangent_u is length {} but should be 1",
        tangent_u.len()
    );
    assert_eq!(
        tangent_cotangent_x_next.len(),
        2,
        "tangent_cotangent_x_next is length {} but should be 2",
        tangent_cotangent_x_next.len()
    );
    assert_eq!(
        vjp_x.len(),
        2,
        "vjp_x is length {} but should be 2",
        vjp_x.len()
    );
    work[0] = 0.0_f64 + tangent_cotangent_x_next[0];
    work[0] += 0.0_f64;
    work[1] = 0.0_f64 + cotangent_x_next[1];
    work[1] = -work[1];
    work[1] += 0.0_f64;
    work[1] *= 0.0_f64;
    work[2] = 0.0_f64 + tangent_cotangent_x_next[1];
    work[3] = -work[2];
    work[3] += 0.0_f64;
    work[3] *= 0.5_f64;
    work[3] += work[1];
    work[3] += 0.0_f64;
    work[3] += work[0];
    work[3] += 0.0_f64;
    work[4] = 0.0_f64 + 0.0_f64;
    work[5] = 0.0_f64 + work[4];
    work[4] = -work[4];
    work[4] += 0.0_f64;
    work[4] *= 0.5_f64;
    work[1] += work[4];
    work[1] += 0.0_f64;
    work[1] += work[5];
    work[1] += 0.0_f64;
    work[4] = 0.0_f64 + work[1];
    work[1] += work[4];
    work[1] += work[3];
    work[2] += 0.0_f64;
    work[2] += 0.0_f64;
    work[3] = 0.0_f64 + cotangent_x_next[0];
    work[3] += 0.0_f64;
    work[3] += 0.0_f64;
    work[3] *= 0.0_f64;
    work[0] += 0.0_f64;
    work[0] *= p[0];
    work[0] += work[3];
    work[0] += work[2];
    work[0] += 0.0_f64;
    work[2] = 0.0_f64 + work[5];
    work[4] = work[2] * p[0];
    work[3] += work[4];
    work[2] += work[3];
    work[2] += 0.0_f64;
    work[3] = 0.0_f64 + work[2];
    work[2] += work[3];
    work[0] += work[2];
    vjp_x[0] = work[1];
    vjp_x[1] = work[0];
}

#[allow(clippy::too_many_arguments)]
fn single_shooting_kernel_mpc_cost_dynamics_vjp_u_jvp(
    x: &[f64],
    u: &[f64],
    p: &[f64],
    cotangent_x_next: &[f64],
    tangent_x: &[f64],
    tangent_u: &[f64],
    tangent_cotangent_x_next: &[f64],
    vjp_u: &mut [f64],
    work: &mut [f64],
) {
    assert!(
        work.len() >= 4,
        "work is length {} but should be at least 4",
        work.len()
    );
    assert_eq!(x.len(), 2, "x is length {} but should be 2", x.len());
    assert_eq!(u.len(), 1, "u is length {} but should be 1", u.len());
    assert_eq!(p.len(), 2, "p is length {} but should be 2", p.len());
    assert_eq!(
        cotangent_x_next.len(),
        2,
        "cotangent_x_next is length {} but should be 2",
        cotangent_x_next.len()
    );
    assert_eq!(
        tangent_x.len(),
        2,
        "tangent_x is length {} but should be 2",
        tangent_x.len()
    );
    assert_eq!(
        tangent_u.len(),
        1,
        "tangent_u is length {} but should be 1",
        tangent_u.len()
    );
    assert_eq!(
        tangent_cotangent_x_next.len(),
        2,
        "tangent_cotangent_x_next is length {} but should be 2",
        tangent_cotangent_x_next.len()
    );
    assert_eq!(
        vjp_u.len(),
        1,
        "vjp_u is length {} but should be 1",
        vjp_u.len()
    );
    work[0] = 0.0_f64 + cotangent_x_next[1];
    work[0] += 0.0_f64;
    work[0] += 0.0_f64;
    work[0] *= 0.0_f64;
    work[1] = 0.0_f64 + tangent_cotangent_x_next[1];
    work[1] += 0.0_f64;
    work[1] += 0.0_f64;
    work[1] *= p[1];
    work[1] += work[0];
    work[1] += 0.0_f64;
    work[2] = 0.0_f64 + tangent_cotangent_x_next[0];
    work[1] += work[2];
    work[1] += 0.0_f64;
    work[2] = 0.0_f64 + 0.0_f64;
    work[3] = 0.0_f64 + work[2];
    work[3] += 0.0_f64;
    work[3] *= p[1];
    work[0] += work[3];
    work[0] += 0.0_f64;
    work[0] += work[2];
    work[0] += 0.0_f64;
    work[2] = 0.0_f64 + work[0];
    work[0] += work[2];
    work[0] += work[1];
    vjp_u[0] = work[0];
}

fn single_shooting_kernel_mpc_cost_stage_cost_grad_x_jvp(
    x: &[f64],
    u: &[f64],
    p: &[f64],
    tangent_x: &[f64],
    tangent_u: &[f64],
    ell: &mut [f64],
    work: &mut [f64],
) {
    assert!(
        work.len() >= 9,
        "work is length {} but should be at least 9",
        work.len()
    );
    assert_eq!(x.len(), 2, "x is length {} but should be 2", x.len());
    assert_eq!(u.len(), 1, "u is length {} but should be 1", u.len());
    assert_eq!(p.len(), 2, "p is length {} but should be 2", p.len());
    assert_eq!(
        tangent_x.len(),
        2,
        "tangent_x is length {} but should be 2",
        tangent_x.len()
    );
    assert_eq!(
        tangent_u.len(),
        1,
        "tangent_u is length {} but should be 1",
        tangent_u.len()
    );
    assert_eq!(ell.len(), 2, "ell is length {} but should be 2", ell.len());
    work[0] = 0.0_f64 + 0.0_f64;
    work[0] += 0.0_f64;
    work[0] += 0.0_f64;
    work[0] += 0.0_f64;
    work[1] = work[0] * x[0];
    work[2] = 0.0_f64 + 1.0_f64;
    work[2] += 0.0_f64;
    work[2] += 0.0_f64;
    work[2] += 0.0_f64;
    work[3] = work[2] * tangent_x[0];
    work[3] += work[1];
    work[4] = 0.0_f64 + work[3];
    work[3] += work[4];
    work[3] += 0.0_f64;
    work[4] = 0.0_f64 * work[2];
    work[1] += work[4];
    work[5] = 0.0_f64 + work[1];
    work[1] += work[5];
    work[1] += work[3];
    work[3] = 2.0_f64 * x[1];
    work[3] *= work[0];
    work[5] = 0.0_f64 * x[1];
    work[6] = 2.0_f64 * tangent_x[1];
    work[6] += work[5];
    work[6] *= work[2];
    work[6] += work[3];
    work[6] += 0.0_f64;
    work[7] = work[2] * x[1];
    work[7] += 0.0_f64;
    work[7] *= 0.0_f64;
    work[0] *= x[1];
    work[8] = work[2] * tangent_x[1];
    work[8] += work[0];
    work[8] += 0.0_f64;
    work[8] *= 2.0_f64;
    work[8] += work[7];
    work[6] += work[8];
    work[6] += 0.0_f64;
    work[8] = 0.0_f64 * 2.0_f64;
    work[5] += work[8];
    work[2] *= work[5];
    work[2] += work[3];
    work[2] += 0.0_f64;
    work[0] += work[4];
    work[0] += 0.0_f64;
    work[0] *= 2.0_f64;
    work[0] += work[7];
    work[0] += work[2];
    work[0] += work[6];
    ell[0] = work[1];
    ell[1] = work[0];
}

fn single_shooting_kernel_mpc_cost_stage_cost_grad_u_jvp(
    x: &[f64],
    u: &[f64],
    p: &[f64],
    tangent_x: &[f64],
    tangent_u: &[f64],
    ell: &mut [f64],
    work: &mut [f64],
) {
    assert!(
        work.len() >= 8,
        "work is length {} but should be at least 8",
        work.len()
    );
    assert_eq!(x.len(), 2, "x is length {} but should be 2", x.len());
    assert_eq!(u.len(), 1, "u is length {} but should be 1", u.len());
    assert_eq!(p.len(), 2, "p is length {} but should be 2", p.len());
    assert_eq!(
        tangent_x.len(),
        2,
        "tangent_x is length {} but should be 2",
        tangent_x.len()
    );
    assert_eq!(
        tangent_u.len(),
        1,
        "tangent_u is length {} but should be 1",
        tangent_u.len()
    );
    assert_eq!(ell.len(), 1, "ell is length {} but should be 1", ell.len());
    work[0] = 0.0_f64 + 1.0_f64;
    work[0] += 0.0_f64;
    work[1] = 0.0_f64 * work[0];
    work[2] = 0.0_f64 + 0.0_f64;
    work[2] += 0.0_f64;
    work[3] = work[2] * p[0];
    work[1] += work[3];
    work[1] += 0.0_f64;
    work[2] += 0.0_f64;
    work[3] = 0.3_f64 * u[0];
    work[3] *= work[2];
    work[0] += 0.0_f64;
    work[4] = 0.0_f64 * 0.3_f64;
    work[5] = 0.0_f64 * u[0];
    work[4] += work[5];
    work[4] *= work[0];
    work[4] += work[3];
    work[4] += work[1];
    work[6] = work[0] * u[0];
    work[6] += 0.0_f64;
    work[6] *= 0.0_f64;
    work[7] = 0.0_f64 * work[0];
    work[2] *= u[0];
    work[7] += work[2];
    work[7] += 0.0_f64;
    work[7] *= 0.3_f64;
    work[7] += work[6];
    work[4] += work[7];
    work[4] += 0.0_f64;
    work[7] = 0.3_f64 * tangent_u[0];
    work[5] += work[7];
    work[5] *= work[0];
    work[3] += work[5];
    work[1] += work[3];
    work[0] *= tangent_u[0];
    work[0] += work[2];
    work[0] += 0.0_f64;
    work[0] *= 0.3_f64;
    work[0] += work[6];
    work[0] += work[1];
    work[0] += work[4];
    ell[0] = work[0];
}

fn single_shooting_kernel_mpc_cost_terminal_cost_grad_x_jvp(
    x: &[f64],
    p: &[f64],
    tangent_x: &[f64],
    vf: &mut [f64],
    work: &mut [f64],
) {
    assert!(
        work.len() >= 6,
        "work is length {} but should be at least 6",
        work.len()
    );
    assert_eq!(x.len(), 2, "x is length {} but should be 2", x.len());
    assert_eq!(p.len(), 2, "p is length {} but should be 2", p.len());
    assert_eq!(
        tangent_x.len(),
        2,
        "tangent_x is length {} but should be 2",
        tangent_x.len()
    );
    assert_eq!(vf.len(), 2, "vf is length {} but should be 2", vf.len());
    work[0] = 0.0_f64 + 1.0_f64;
    work[0] += 0.0_f64;
    work[1] = 0.0_f64 * work[0];
    work[2] = 0.0_f64 + 0.0_f64;
    work[2] += 0.0_f64;
    work[3] = work[2] * p[1];
    work[1] += work[3];
    work[1] += 0.0_f64;
    work[2] += 0.0_f64;
    work[3] = 3.0_f64 * x[0];
    work[3] *= work[2];
    work[0] += 0.0_f64;
    work[4] = 0.0_f64 * x[0];
    work[5] = 3.0_f64 * tangent_x[0];
    work[4] += work[5];
    work[4] *= work[0];
    work[3] += work[4];
    work[1] += work[3];
    work[3] = work[0] * x[0];
    work[3] += 0.0_f64;
    work[3] *= 0.0_f64;
    work[4] = work[2] * x[0];
    work[5] = work[0] * tangent_x[0];
    work[4] += work[5];
    work[4] += 0.0_f64;
    work[4] *= 3.0_f64;
    work[3] += work[4];
    work[1] += work[3];
    work[1] += 0.0_f64;
    work[3] = 0.5_f64 * x[1];
    work[3] *= work[2];
    work[4] = 0.0_f64 * x[1];
    work[5] = 0.5_f64 * tangent_x[1];
    work[4] += work[5];
    work[4] *= work[0];
    work[3] += work[4];
    work[3] += 0.0_f64;
    work[4] = work[0] * x[1];
    work[4] += 0.0_f64;
    work[4] *= 0.0_f64;
    work[2] *= x[1];
    work[0] *= tangent_x[1];
    work[0] += work[2];
    work[0] += 0.0_f64;
    work[0] *= 0.5_f64;
    work[0] += work[4];
    work[0] += work[3];
    work[0] += 0.0_f64;
    vf[0] = work[1];
    vf[1] = work[0];
}

/// Return metadata describing [`single_shooting_kernel_mpc_cost_f_grad_states_u_seq`].
pub fn single_shooting_kernel_mpc_cost_f_grad_states_u_seq_meta() -> FunctionMetadata {
    FunctionMetadata {
        function_name: "single_shooting_kernel_mpc_cost_f_grad_states_u_seq",
        workspace_size: 55,
        input_names: &["x0", "u_seq", "p"],
        input_sizes: &[2, 20, 2],
        output_names: &["cost", "gradient_u_seq", "x_traj"],
        output_sizes: &[1, 20, 42],
    }
}

/// Evaluate the generated symbolic function `single_shooting_kernel_mpc_cost_f_grad_states_u_seq`.
///
/// All numeric slices use the `f64` scalar type.
///
/// Arguments:
/// - `x0`:
///   initial state slice
///   Expected length: 2.
/// - `u_seq`:
///   packed control-sequence slice laid out stage-major over the horizon
///   Expected length: 20.
/// - `p`:
///   shared parameter slice used at every stage and terminal evaluation
///   Expected length: 2.
/// - `cost`:
///   total cost output
///   Expected length: 1.
/// - `gradient_u_seq`:
///   gradient with respect to the packed control sequence
///   Expected length: 20.
/// - `x_traj`:
///   packed rollout state trajectory including x0 through xN
///   Expected length: 42.
/// - `work`: mutable workspace slice used to store intermediate values
///   while evaluating this kernel. Expected length: at least 55.
pub fn single_shooting_kernel_mpc_cost_f_grad_states_u_seq(
    x0: &[f64],
    u_seq: &[f64],
    p: &[f64],
    cost: &mut [f64],
    gradient_u_seq: &mut [f64],
    x_traj: &mut [f64],
    work: &mut [f64],
) -> Result<(), GradgenError> {
    if work.len() < 55 {
        return Err(GradgenError::WorkspaceTooSmall("work expected at least 55"));
    };
    if x0.len() != 2 {
        return Err(GradgenError::InputTooSmall("x0 expected length 2"));
    };
    if u_seq.len() != 20 {
        return Err(GradgenError::InputTooSmall("u_seq expected length 20"));
    };
    if p.len() != 2 {
        return Err(GradgenError::InputTooSmall("p expected length 2"));
    };
    if cost.len() != 1 {
        return Err(GradgenError::OutputTooSmall("cost expected length 1"));
    };
    if gradient_u_seq.len() != 20 {
        return Err(GradgenError::OutputTooSmall(
            "gradient_u_seq expected length 20",
        ));
    };
    if x_traj.len() != 42 {
        return Err(GradgenError::OutputTooSmall("x_traj expected length 42"));
    };
    let (state_history, rest) = work.split_at_mut(40);
    let (state_buffers, rest) = rest.split_at_mut(4);
    let (current_state, next_state) = state_buffers.split_at_mut(2);
    let (lambda_buffers, rest) = rest.split_at_mut(4);
    let (lambda_current, lambda_next) = lambda_buffers.split_at_mut(2);
    let (temp_state, rest) = rest.split_at_mut(2);
    let (temp_control, rest) = rest.split_at_mut(1);
    let (scalar_buffer, stage_work) = rest.split_at_mut(1);
    current_state.copy_from_slice(x0);
    x_traj[0..2].copy_from_slice(x0);
    let mut total_cost = 0.0_f64;
    for stage_index in 0..20 {
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
        state_history[(stage_index * 2)..((stage_index + 1) * 2)].copy_from_slice(next_state);
        current_state.copy_from_slice(next_state);
        x_traj[((stage_index + 1) * 2)..((stage_index + 2) * 2)].copy_from_slice(next_state);
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
    for stage_index in (1..20).rev() {
        let x_t = &state_history[((stage_index - 1) * 2)..(stage_index * 2)];
        let u_t = &u_seq[stage_index..(stage_index + 1)];
        let grad_u_t = &mut gradient_u_seq[stage_index..(stage_index + 1)];
        single_shooting_kernel_mpc_cost_stage_cost_grad_u(x_t, u_t, p, grad_u_t, stage_work);
        single_shooting_kernel_mpc_cost_dynamics_vjp_u(
            x_t,
            u_t,
            p,
            &lambda_current[..],
            temp_control,
            stage_work,
        );
        grad_u_t[0] += temp_control[0];
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
        lambda_current.copy_from_slice(lambda_next);
    }
    let u_t = &u_seq[0..1];
    let grad_u_t = &mut gradient_u_seq[0..1];
    single_shooting_kernel_mpc_cost_stage_cost_grad_u(x0, u_t, p, grad_u_t, stage_work);
    single_shooting_kernel_mpc_cost_dynamics_vjp_u(
        x0,
        u_t,
        p,
        &lambda_current[..],
        temp_control,
        stage_work,
    );
    grad_u_t[0] += temp_control[0];
    Ok(())
}
