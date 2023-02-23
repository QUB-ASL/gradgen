fn main() {

    let n_pred = 5;
    let max_num_iterations= 100;
    let mut tol:f64 = 1e-4;
    let mut re_tol:f64 = 1e-4;

    let mut ws = ball_test::BackwardGradientWorkspace::new(n_pred);
    let nu =ball_test::num_inputs();
    let nx = ball_test::num_states();

    let x0 = vec![1.0;nx];
    let mut u_seq = vec! [0.1; nu * n_pred];
    let mut u_seq_new = vec! [1.0; nu * n_pred];
    let mut vn =0.0;
    let mut grad = vec! [0.0; nu * n_pred];


    fn sub(vec_a: Vec<f64>, vec_b: Vec<f64>)-> Vec<f64>  {
        vec_a.into_iter().zip(vec_b).map(|(a, b)| a - b).collect()
    }

    fn mul(vec_a: Vec<f64>, vec_b: Vec<f64>)-> Vec<f64>  {
        vec_a.into_iter().zip(vec_b).map(|(a, b)| a * b).collect()
    }

    fn armijo_condition(x: &[f64], u: &[f64], alpha: f64, c: f64, grad:&[f64], mut vn: f64) -> f64 {
    let n_pred = 5;
    let mut ws = ball_test::BackwardGradientWorkspace::new(n_pred);
    let mut alpha = alpha;
    let mut u_new = vec![0.0; u.len()];
    let mut vn_new = 0.0;
    let mut grad_new = vec![0.0; u_new.len()];

    loop {
        for i in 0..u.len() {
            u_new[i] = u[i] - alpha * grad[i];
        }
        ball_test::total_cost_gradient_bw(&x, &u_new, &mut grad_new, &mut ws, n_pred, &mut vn_new);

        let rhs = vn - c * alpha * dot(&grad, &grad);
        let lhs = vn_new;
        println!("lhs{:?},rhs{:?}", lhs, rhs);

        if lhs > rhs {
            break;
        }
        else {
            alpha *= 0.99;
            if alpha < 1e-8 {
                panic!("Armijo condition failed: alpha too small");
            }
        }
    }

    alpha
}

    fn dot(a: &[f64], b: &[f64]) -> f64 {
        let mut sum = 0.0;
        for i in 0..a.len() {
            sum += a[i] * b[i];
        }
        sum
    }



    let alpha = 1.0;
    let c = 0.5; // Constant value for Armijo condition
    ball_test::total_cost_gradient_bw(&x0, &u_seq, &mut grad, &mut ws, n_pred, &mut vn );
    let a=armijo_condition(&x0,&u_seq, alpha, c, &grad, vn);
    println!("alpha{:?}", a);

    // let gamma =vec![0.2; nu * n_pred];
    let gamma =vec![a; nu * n_pred];


    //error
    for i in (1..=max_num_iterations).step_by(1) {
        // println!("i{:?}", i);
        // println!("u {:?}", u_seq);
        ball_test::total_cost_gradient_bw(&x0, &u_seq, &mut grad, &mut ws, n_pred, &mut vn );
        // println!("grad {:?}",grad);
        let df = grad.clone();
        // dff = gamma * df
        let dff = mul(gamma.clone(), df);
        // u_seq_new= u_seq - gamma * df
        u_seq_new = sub(u_seq.clone(), dff.clone());

        let error =sub( u_seq_new.clone() ,u_seq.clone());
        let error_max= error.iter().max_by(|a, b| a.abs().total_cmp(&b.abs()));
        let error_max_abs =error_max.expect("REASON").abs();
        println!("{:?},{:?}", i,error_max_abs);
        if error_max_abs < tol {
            break;
        }
        if i==1{
            re_tol=tol*error_max_abs;
            tol=tol+re_tol*tol;
        }
        // println!("{:?}", error_max_abs);
        u_seq =u_seq_new.clone();

        // println!("----------------");

    }
    // println!("u_new {:?}", u_seq_new);



}