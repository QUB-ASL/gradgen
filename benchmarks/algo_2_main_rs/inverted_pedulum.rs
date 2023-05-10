use std::time::Instant;


fn main() {
    let n_runs= 10;
    let mut ws = bm_algo_2_inv_pend_gradgen::BackwardGradientWorkspace::new();
    let nu = bm_algo_2_inv_pend_gradgen::num_inputs();
    let nx = bm_algo_2_inv_pend_gradgen::num_states();
    let nnl = bm_algo_2_inv_pend_gradgen::num_nonleaf_nodes();
    let x0 = vec![0.0; nx];
    let u_nodes = vec! [1.0; nu * nnl];
    let mut grad = vec! [0.0; nu * nnl];
    let mut store= vec! [0.0;  n_runs];
    for _j in 1..n_runs {
        let now = Instant::now();
        bm_algo_2_inv_pend_gradgen::total_cost_gradient_bw(&x0, &u_nodes, &mut grad, &mut ws);
        let elapsed = now.elapsed();
        let duration = elapsed.as_secs() as f64 + elapsed.subsec_nanos() as f64 * 1e-9;
        store[_j-1]=1e6 *duration;
    }
    //print the average runtime (in microsecond) for calculating the total gradient once
    let data_mean = bm_algo_2_inv_pend_gradgen::mean(&store);
    let data_std_deviation = bm_algo_2_inv_pend_gradgen::std_deviation(&store);
    println!("mean:{:?}, std_dev:{:?}",
             data_mean.unwrap_or(-1.0),
             data_std_deviation.unwrap_or(-1.0));
}