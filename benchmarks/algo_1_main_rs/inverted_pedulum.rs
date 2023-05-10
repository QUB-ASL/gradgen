use std::time::Instant;


fn main() {
    let n_pred_max = 5;
    let n_runs= 2;
    let mut ws = inverted_pendulum_test::BackwardGradientWorkspace::new(n_pred_max);
    let nu = inverted_pendulum_test::num_inputs();
    let nx = inverted_pendulum_test::num_states();
    let x0 = vec![0.0; nx];

    for n_pred in (5..=n_pred_max).step_by(5){
        let u_seq = vec! [1.0; nu * n_pred];
        let mut grad = vec! [0.0; nu * n_pred];
        let mut store= vec! [0.0;  n_runs];
        for _j in 1..n_runs {
            let now = Instant::now();
            inverted_pendulum_test::total_cost_gradient_bw(&x0, &u_seq, &mut grad, &mut ws, n_pred);
            let elapsed = now.elapsed();
            let duration = elapsed.as_secs() as f64 + elapsed.subsec_nanos() as f64 * 1e-9;
            store[_j-1]=1e6 *duration;
        }
        //print horizon size N and
        //the average runtime (in microsecond) for calculating the total gradient once
        let data_mean = inverted_pendulum_test::mean(&store);
        let data_std_deviation = inverted_pendulum_test::std_deviation(&store);
        println!("{},{:?},{:?}",
                 n_pred,
                 data_mean.unwrap_or(-1.0),
                 data_std_deviation.unwrap_or(-1.0));
    }
}