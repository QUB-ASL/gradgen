// This file is automatically generated by GradGen

use casadi_{{name}}::*;

/// Number of states
pub fn num_states() -> usize {
    return NX;
}

/// Number of inputs
pub fn num_inputs() -> usize {
    return NU;
}

#[derive(Debug)]
pub struct BackwardGradientWorkspace {
    pub(crate) w: Vec<f64>,
    pub(crate) w_new: Vec<f64>,
    pub(crate) x_seq: Vec<f64>,
    pub(crate) temp_nx: Vec<f64>,
    pub(crate) temp_nu: Vec<f64>,


}

/// Workspace structure
impl BackwardGradientWorkspace {

    /// Create new instance of workspaces
    ///
    /// # Arguments
    ///
    /// * `n` - prediction horizon
    ///
    pub fn new(n_pred: usize) -> BackwardGradientWorkspace {
        BackwardGradientWorkspace {
            w: vec![0.0; NX],
            w_new: vec![0.0; NX],
            x_seq: vec![0.0; NX * (n_pred + 1)],
            temp_nx: vec![0.0; NX],
            temp_nu: vec![0.0; NU],


        }
    }
}

fn a_plus_eq_b(a: &mut [f64], b: &[f64]) {
    a.iter_mut().zip(b.iter()).for_each(|(ai, bi)| *ai += *bi);
}

/// Gradient of the total cost function with backward method
///
/// # Arguments
///
/// * `x0` - initial state
/// * `u_seq` - sequence of inputs
/// * `grad` - gradient of total cost function (result)
/// * `ws` - workspace of type `BackwardGradientWorkspace`
/// * `n` - prediction horizon
///
///
pub fn total_cost_gradient_bw(
    x0: &[f64],
    u_seq: &[f64],
    grad: &mut [f64],
    ws: &mut BackwardGradientWorkspace,
    n: usize,
    vn: &mut f64
) {
    *vn = 0.0;
    ws.x_seq[..NX].copy_from_slice(x0);
    let mut temp_vf=0.0;
    let mut temp_vn=0.0;
    /* Simulation and add ell*/
    for i in 0..=n - 1 {
        let xi = &ws.x_seq[i * NX..(i + 1) * NX];
        let ui = &u_seq[i * NU..(i + 1) * NU];
        f(xi, ui, &mut ws.w);
        ell(xi, ui,&mut temp_vn);
        ws.x_seq[(i + 1) * NX..(i + 2) * NX].copy_from_slice(&ws.w);
        *vn = *vn + temp_vn;
    }

    /* add Vf */
    let x_npred = &ws.x_seq[n * NX..(n + 1) * NX];
    vf(x_npred, &mut temp_vf);
    *vn = *vn +  temp_vf;

    /* initial w */
    vfx(
        x_npred,
        &mut ws.w,
    );

    /* backward method */
    for j in 1..=n-1 {
        let x_npred_j = &ws.x_seq[(n - j) * NX..(n - j + 1) * NX];
        let u_npred_j = &u_seq[(n - j) * NU..(n - j + 1) * NU];
        let grad_npred_j = &mut grad[(n - j) * NU..(n - j + 1) * NU];

        fu(x_npred_j, u_npred_j, &ws.w, grad_npred_j);
        ellu(x_npred_j, u_npred_j, &mut ws.temp_nu);
        a_plus_eq_b(grad_npred_j, &ws.temp_nu);

        fx(x_npred_j, u_npred_j, &ws.w, &mut ws.w_new);
        ellx(x_npred_j, u_npred_j, &mut ws.temp_nx);
        a_plus_eq_b(&mut ws.w_new, &ws.temp_nx);
        ws.w.copy_from_slice(&ws.w_new);
    }

    /* first coordinate (t=0) */
    let x_npred_j = &ws.x_seq[0..NX];
    let u_npred_j = &u_seq[0..NU];
    let grad_npred_j = &mut grad[..NU];

    fu(x_npred_j, u_npred_j, &ws.w, grad_npred_j);
    ellu(x_npred_j, u_npred_j, &mut ws.temp_nu);
    a_plus_eq_b(grad_npred_j, &ws.temp_nu);

}
/// The total cost function
///
/// # Arguments
///
/// * `x0` - initial state
/// * `u_seq` - sequence of inputs
/// * `ws` - workspace of type `BackwardGradientWorkspace`
/// * `n` - prediction horizon
///
///
pub fn total_cost(
    x0: &[f64],
    u_seq: &[f64],
    ws: &mut BackwardGradientWorkspace,
    n: usize
) -> f64 {
    let mut temp_vf=0.0;
    let mut temp_vn=0.0;
    let mut temp_nx_new= vec![0.0; NX];
    let mut vn = 0.0;
    ws.temp_nx[..NX].copy_from_slice(x0);
    let mut xi= vec![0.0; NX];
    xi.copy_from_slice(&x0);
    for i in 0..=n - 1 {
        f(&ws.temp_nx[..NX], &u_seq[i * NU..(i + 1) * NU], &mut temp_nx_new[..NX]);
        ell(&ws.temp_nx[..NX], &u_seq[i * NU..(i + 1) * NU],&mut temp_vn);
        ws.temp_nx.copy_from_slice(&temp_nx_new);
        vn = vn + temp_vn;
    }
    vf(&ws.temp_nx[..NX], &mut temp_vf);
    vn = vn +  temp_vf;
    vn
}