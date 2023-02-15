fn main() {

    let n_pred = 5;
    let max_num_iterations= 2;
    let tol:f64 = 0.000000001;

    let mut ws = inverted_pendulum_test::BackwardGradientWorkspace::new(n_pred);
    let nu =inverted_pendulum_test::num_inputs();
    let nx = inverted_pendulum_test::num_states();
    let x0 = vec![0.0; nx];
    let gamma =vec![0.1; nu * n_pred];
    let mut u_seq = vec! [1.0; nu * n_pred];
    let mut grad = vec! [0.0; nu * n_pred];
    let mut u_seq_new = vec! [1.0; nu * n_pred];


    fn sub(vec_a: Vec<f64>, vec_b: Vec<f64>)-> Vec<f64>  {
    vec_a.into_iter().zip(vec_b).map(|(a, b)| a - b).collect() }

    fn mul(vec_a: Vec<f64>, vec_b: Vec<f64>)-> Vec<f64>  {
    vec_a.into_iter().zip(vec_b).map(|(a, b)| a * b).collect() }



    // let mut store= vec! [0.0;  n_runs];
    for i in (1..=max_num_iterations).step_by(1) {
        println!("{:?}", u_seq);
        inverted_pendulum_test::total_cost_gradient_bw(&x0, &u_seq, &mut grad, &mut ws, n_pred);
        println!("{:?},{:?}",grad,i);
        let df = grad.clone();
        let dff = mul(gamma.clone(), df);
        u_seq_new = sub(u_seq.clone(), dff.clone());
        let error =sub( u_seq_new.clone() ,u_seq.clone());
        let error_max= error.iter().max_by(|a, b| a.abs().total_cmp(&b.abs()));
        let error_max_abs =error_max.expect("REASON").abs();
        if error_max_abs < tol {
            break;
        }
        // println!("{:?}", error_max_abs);
        u_seq =u_seq_new;
        println!("{:?}", u_seq);

    }
}


