use std::time::Instant;


fn main() {
    let n_pred = 30;
    let n_runs = 1000000;
    let mut ws = bm_algo_1_inv_pend_gradgen::BackwardGradientWorkspace::new(n_pred);
    let nu = bm_algo_1_inv_pend_gradgen::num_inputs();
    let nx = bm_algo_1_inv_pend_gradgen::num_states();
    let x0 = vec![0.0; nx];
    let mut vn = 0.0;

    let u_seq = vec! [1.0; nu * n_pred];
    let mut grad = vec! [0.0; nu * n_pred];
    let mut store= vec! [0.0;  n_runs];
    for _j in 1..n_runs {
        let now = Instant::now();
        bm_algo_1_inv_pend_gradgen::total_cost_gradient_bw(&x0, &u_seq, &mut grad, &mut ws,
                                                           n_pred, &mut vn);
        let elapsed = now.elapsed();
        let duration = elapsed.as_secs() as f64 + elapsed.subsec_nanos() as f64 * 1e-9;
        store[_j-1]=1e6 *duration;
    }
    //print horizon size N and
    //the average runtime (in microsecond) for calculating the total gradient once
    let data_mean = bm_algo_1_inv_pend_gradgen::mean(&store);
    let data_std_deviation = bm_algo_1_inv_pend_gradgen::std_deviation(&store);
    println!("{},{:?},{:?}",
             n_pred,
             data_mean.unwrap_or(-1.0),
             data_std_deviation.unwrap_or(-1.0));
}