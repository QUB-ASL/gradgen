fn main() {
    let n_pred = 5;
    let mut ws = pantelis::BackwardGradientWorkspace::new(n_pred);
    let nu = pantelis::num_inputs();
    let x0 = vec![0.5, 0.1, -0.2];
    let u_seq = vec![0.1; nu * n_pred];
    let mut grad = vec![0.0; nu * n_pred];
    pantelis::total_cost_gradient_bw(&x0, &u_seq, &mut grad, &mut ws, n_pred);
    println!("{:?}", grad);
}