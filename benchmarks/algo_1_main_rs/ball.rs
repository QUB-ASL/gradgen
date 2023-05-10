use std::time::Instant;


fn main() {
    let n_pred_max = 500;
    let n_runs= 5000;
    let mut ws = ball_test::BackwardGradientWorkspace::new(n_pred_max);
    let nu = ball_test:: num_inputs();
    let nx = ball_test::num_states();
    let x0 = vec![1.0; nx];


    fn mean(data: &[f64]) -> Option<f64> {
        let sum = data.iter().sum::<f64>() as f64;
        let count = data.len();
        match count {
            positive if positive > 0 => Some(sum/count as f64),
            _ => None,
        }
    }
    fn std_deviation(data: &[f64]) -> Option<f64> {
        match (mean(data), data.len()) {
            (Some(data_mean), count) if count > 0 => {
                let variance = data.iter().map(|value| {
                    let diff = data_mean - (*value as f64);

                    diff * diff
                }).sum::<f64>()/count as f64;

                Some(variance.sqrt())
            },
            _ => None
        }
    }



    for n_pred in (10..=n_pred_max).step_by(5){
        let u_seq = vec! [0.01; nu * n_pred];
        let mut grad = vec! [0.0; nu * n_pred];
        let mut store= vec! [0.0;  n_runs];
        for _j in 1..n_runs {
            let now = Instant::now();
            ball_test::total_cost_gradient_bw(&x0, &u_seq, &mut grad, &mut ws, n_pred);
            let elapsed = now.elapsed();
            let duration = elapsed.as_secs() as f64 + elapsed.subsec_nanos() as f64 * 1e-9;
            store[_j-1]=1e6 *duration;
        }

        let data_mean = mean(&store);
        let data_std_deviation = std_deviation(&store);
        println!("{},{:?},{:?}", n_pred, data_mean.unwrap_or(-1.0), data_std_deviation.unwrap_or(-1.0));


    }
}