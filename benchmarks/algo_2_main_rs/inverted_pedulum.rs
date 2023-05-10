use std::time::Instant;


fn main() {
    let n_runs: i64 = 100 as i64;
    let total: f64 = n_runs as f64;
    let mut ws = bm_algo_2_inv_pend_gradgen::BackwardGradientWorkspace::new();
    let nu = bm_algo_2_inv_pend_gradgen::num_inputs();
    let nx = bm_algo_2_inv_pend_gradgen::num_states();
    let nnl = bm_algo_2_inv_pend_gradgen::num_nonleaf_nodes();
    let x0 = vec![0.0; nx];
    let u_nodes = vec! [1.0; nu * nnl];
    let mut grad = vec! [0.0; nu * nnl];
    let now = Instant::now();
    for _ in 1..n_runs {
        bm_algo_2_inv_pend_gradgen::total_cost_gradient_bw(&x0, &u_nodes, &mut grad, &mut ws);
    }
    let elapsed = now.elapsed();
    let duration = elapsed.as_secs() as f64 + elapsed.subsec_nanos() as f64 * 1e-9;
    //print the average runtime (in microsecond) for calculating the total gradient once
    let data_mean = duration * 1e6 / total;
    println!("mean:{:?}", data_mean);
}