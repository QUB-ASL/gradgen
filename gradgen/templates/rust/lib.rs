// nothing
use casadi_{{name}}::*;

#[derive(Debug)]
pub struct BackwardGradientWorkspace {
    pub(crate) w: Vec<f64>,
    pub(crate) w_new: Vec<f64>,
    pub(crate) x_seq: Vec<f64>,
    pub(crate) temp_nu: Vec<f64>,
    pub(crate) temp_nx: Vec<f64>,
}

impl BackwardGradientWorkspace {
    pub fn new() -> BackwardGradientWorkspace {
        BackwardGradientWorkspace {
            w: vec![0.0; NX],
            w_new : vec![0.0; NX],
            x_seq: vec![0.0; NX * (NPRED + 1)],
            temp_nu: vec! [0.0; NU],
            temp_nx: vec! [0.0; NX],
        }
    }
}


/// a = a + b
fn add(a: &mut [f64], b: &[f64]) {
    a.iter_mut().zip(b.iter()).for_each(|(ai, bi)| *ai += bi);
}



pub fn total_cost_gradient_bw(
    x0: &[f64],
    u_seq: &[f64],
    grad: &mut [f64],
    workspace: &mut BackwardGradientWorkspace,
) {

    /*
    * Simulate the system starting from x0 and using the
    * sequence of inputs, u_seq = (u(0), u(1), .., U(N-1))
    * and store the states, x_seq = (x(0), x(1),..., x(N))
    * in workspace. x_seq = (x(0), x(1), . . . , x (N))
    */
    workspace.x_seq[0..NX].copy_from_slice(x0);
    /* Simulation */
    for t in 0..=NPRED - 1 {
        let xt = &workspace.x_seq[t * NX.. (t + 1) * NX];
        let ut = &u_seq[t * NU.. (t + 1) * NU];
        // W= f(xt, ut)
        f(xt, ut, &mut workspace.w);
        // x seq[next]<- W (CODY)
        workspace.x_seq[(t + 1) * NX.. (t + 2) * NX].copy_from_slice (&workspace.w);
    }

    /* Store Vfx in w */
    let xn = &workspace.x_seq[NPRED * NX..(NPRED + 1) * NX];
    vfx(xn, &mut workspace.w);

    /* backward for-loop */
    for j in 1..=NPRED {

        let xnj = &workspace.x_seq[(NPRED-j)* NX.. (NPRED-j+ 1) * NX];
        let unj = &u_seq[(NPRED-j) * NU.. (NPRED-j+1)* NU];
        let gradnj = &mut grad[(NPRED-j) * NU.. (NPRED-j+1)* NU];

        // gradnj := f^u(w) at t=N-j
        jfu(xnj, unj, &mut workspace.w, gradnj);
        // temp_nu := ellu at t=N-j
        ellu(xnj, unj,&mut workspace.temp_nu );
        // gradnj += temp_nu
        add(gradnj, &workspace.temp_nu);


        // &mut workspace.w_new := f^x(w) at t=N-j
        jfx(xnj, unj, &mut workspace.w, &mut workspace.w_new);
        // temp_nx := ellx at t=N-j
        ellx(xnj, unj,&mut workspace.temp_nx );
        // &mut workspace.w_new += temp_nx
        add(&mut workspace.w_new, &workspace.temp_nx);
        // update w
        workspace.w.copy_from_slice (&workspace.w_new);
    }




}

#[cfg(test)]
mod tests {

    #[test]
    fn tst_nothing() {
        let mut ws = super::BackwardGradientWorkspace::new();
        let x0 = vec![1.0; casadi_{{name}}::NX];
        let u_seq = vec![0.0; casadi_{{name}}::NU * casadi_{{name}}::NPRED];
        let mut grad = vec![0.0; casadi_{{name}}::NU * casadi_{{name}}::NPRED];
        super::total_cost_gradient_bw(&x0, &u_seq, &mut grad, &mut ws);
        println!("{:?}", ws);
    }
}
