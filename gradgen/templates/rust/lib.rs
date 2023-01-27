// nothing

#[derive(Debug)]
pub struct BackwardGradientWorkspace {
    pub(crate) w: Vec<f64>,
    pub(crate) x_seq: Vec<f64>,
}

impl BackwardGradientWorkspace {
    pub fn new() -> BackwardGradientWorkspace {
        BackwardGradientWorkspace {
            w: vec![0.0; casadi_{{name}}::NX],
            x_seq: vec![0.0; casadi_{{name}}::NX * (casadi_{{name}}::NPRED + 1)],
        }
    }
}

pub fn total_cost_gradient_bw(
    x0: &[f64],
    u_seq: &[f64],
    grad: &mut [f64],
    ws: &mut BackwardGradientWorkspace,
) {
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
