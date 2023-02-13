// nothing
use casadi_inverted_pendulum_test::*;

pub fn num_states() -> usize {
    return NX;
}

pub fn num_inputs() -> usize {
    return NU;
}

#[derive(Debug)]
pub struct BackwardGradientWorkspace {
    pub(crate) w: Vec<f64>,
    pub(crate) w_new: Vec<f64>,
    pub(crate) x_seq: Vec<f64>,
    pub(crate) temp_nu: Vec<f64>,
    pub(crate) temp_nx: Vec<f64>,
}

impl BackwardGradientWorkspace {
    pub fn new(n_pred: usize) -> BackwardGradientWorkspace {
        BackwardGradientWorkspace {
            w: vec![0.0; NX],
            w_new : vec![0.0; NX],
            x_seq: vec![0.0; NX * (n_pred + 1)],
            temp_nu: vec! [0.0; NU],
            temp_nx: vec! [0.0; NX],
        }
    }
}


/// a = a + b
fn add(a: &mut [f64], b: &[f64]) {
    a.iter_mut().zip(b.iter()).for_each(|(ai, bi)| *ai += *bi);
}



pub fn total_cost_gradient_bw(
    x0: &[f64],
    u_seq: &[f64],
    grad: &mut [f64],
    workspace: &mut BackwardGradientWorkspace,
    n: usize
) {

    /*
    * Simulate the system starting from x0 and using the
    * sequence of inputs, u_seq = (u(0), u(1), .., U(N-1))
    * and store the states, x_seq = (x(0), x(1),..., x(N))
    * in workspace. x_seq = (x(0), x(1), . . . , x (N))
    */
    workspace.x_seq[0..NX].copy_from_slice(x0);
    /* Simulation */
    for t in 0..= n - 1 {
        let xt = &workspace.x_seq[t * NX.. (t + 1) * NX];
        let ut = &u_seq[t * NU.. (t + 1) * NU];
        // W= f(xt, ut)
        f(xt, ut, &mut workspace.w);
        // x seq[next]<- W (CODY)
        workspace.x_seq[(t + 1) * NX.. (t + 2) * NX].copy_from_slice (&workspace.w);
    }

    /* Store Vfx in w */
    let xn = &workspace.x_seq[n * NX..(n + 1) * NX];
    vfx(xn, &mut workspace.w);

    /* backward for-loop */
    for j in 1..=n{

        let xnj = &workspace.x_seq[(n-j)* NX.. (n-j+ 1) * NX];
        let unj = &u_seq[(n-j) * NU.. (n-j+1)* NU];
        let gradnj = &mut grad[(n-j) * NU.. (n-j+1)* NU];

        // gradnj := f^u(w) at t=N-j
        jfu(xnj, unj, &mut workspace.w, gradnj);
        // temp_nu := ellu at t=N-j
        ellu(xnj, unj,&mut workspace.temp_nu );
        // gradnj += temp_nu
        add(gradnj, &workspace.temp_nu);


        // gradnj += &mut workspace.temp_nx
        jfx(xnj, unj, &mut workspace.w, &mut workspace.w_new);
        // temp_nx <-- ell^x_N_minus_j
        ellx(xnj, unj,&mut workspace.temp_nx );
        // grad_V_N_minus_j += temp_nx
        add(&mut workspace.w_new, &workspace.temp_nx);
        workspace.w.copy_from_slice (&workspace.w_new);
    }




}

#[cfg(test)]
mod tests {

    use casadi_inverted_pendulum_test::*;

    #[test]
    fn tst_nothing() {
        let n: usize= 6 ;
        let mut ws = super::BackwardGradientWorkspace::new(n);
        let x0 = vec![1.0; NX];
        let u_seq = vec![1.0; NU * n];
        let mut grad = vec![0.0; NU * n];
        super::total_cost_gradient_bw(&x0, &u_seq, &mut grad, &mut ws, n);
        println!("The gradient of total cost function is {:?}", grad);
        println!("The state variable x is {:?}", ws.x_seq);
    }
}

