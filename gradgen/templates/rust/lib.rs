// This file is automatically generated by GradGen

use casadi_{{name}}::*;
use std::time::Instant;
use rand::prelude::*;


/// Number of states
pub fn num_states() -> usize {
    return NX;
}

/// Number of inputs
pub fn num_inputs() -> usize {
    return NU;
}


#[derive(Debug)]
pub struct BackwardGradientWorkspace {
    pub(crate) w: Vec<f64>,
    pub(crate) w_new: Vec<f64>,
    pub(crate) x_seq: Vec<f64>,
    pub(crate) temp_nu: Vec<f64>,
    pub(crate) temp_nx: Vec<f64>,
}
/// Workspace structure
impl BackwardGradientWorkspace {

    /// Create new instance of workspaces
    ///
    /// # Arguments
    ///
    /// * `n` - prediction horizon
    ///
    pub fn new(n_pred: usize) -> BackwardGradientWorkspace {
        BackwardGradientWorkspace {
            w: vec![0.0; NX],
            w_new: vec![0.0; NX],
            x_seq: vec![0.0; NX * (n_pred + 1)],
            temp_nx: vec![0.0; NX],
            temp_nu: vec![0.0; NU],
        }
    }
}

fn a_plus_eq_b(a: &mut [f64], b: &[f64]) {
    a.iter_mut().zip(b.iter()).for_each(|(ai, bi)| *ai += *bi);
}

/// Gradient of the total cost function with backward method
///
/// # Arguments
///
/// * `x0` - initial state
/// * `u_seq` - sequence of inputs
/// * `grad` - gradient of total cost function (result)
/// * `ws` - workspace of type `BackwardGradientWorkspace`
/// * `n` - prediction horizon
///
///
pub fn total_cost_gradient_bw(
    x0: &[f64],
    u_seq: &[f64],
    grad: &mut [f64],
    ws: &mut BackwardGradientWorkspace,
    n: usize,
    vn: &mut f64
) {
    *vn = 0.0;
    ws.x_seq[..NX].copy_from_slice(x0);
    let mut temp_vf=0.0;
    let mut temp_vn=0.0;
    /* Simulation */
    for i in 0..=n - 1 {
        let xi = &ws.x_seq[i * NX..(i + 1) * NX];
        let ui = &u_seq[i * NU..(i + 1) * NU];
        f(xi, ui, &mut ws.w);
        ell(xi, ui,&mut temp_vn);
        ws.x_seq[(i + 1) * NX..(i + 2) * NX].copy_from_slice(&ws.w);
        *vn = *vn + temp_vn;
    }

    /* add Vf */
    let x_npred = &ws.x_seq[n * NX..(n + 1) * NX];
    vf(x_npred, &mut temp_vf);
    *vn = *vn +  temp_vf;

    /* initial w */
    vfx(
        x_npred,
        &mut ws.w,
    );

    /* backward method */
    for j in 1..=n-1 {
        let x_npred_j = &ws.x_seq[(n - j) * NX..(n - j + 1) * NX];
        let u_npred_j = &u_seq[(n - j) * NU..(n - j + 1) * NU];
        let grad_npred_j = &mut grad[(n - j) * NU..(n - j + 1) * NU];

        fu(x_npred_j, u_npred_j, &ws.w, grad_npred_j);
        ellu(x_npred_j, u_npred_j, &mut ws.temp_nu);
        a_plus_eq_b(grad_npred_j, &ws.temp_nu);

        fx(x_npred_j, u_npred_j, &ws.w, &mut ws.w_new);
        ellx(x_npred_j, u_npred_j, &mut ws.temp_nx);
        a_plus_eq_b(&mut ws.w_new, &ws.temp_nx);
        ws.w.copy_from_slice(&ws.w_new);
    }


    /* first coordinate (t=0) */
    let x_npred_j = &ws.x_seq[0..NX];
    let u_npred_j = &u_seq[0..NU];
    let grad_npred_j = &mut grad[..NU];

    fu(x_npred_j, u_npred_j, &ws.w, grad_npred_j);
    ellu(x_npred_j, u_npred_j, &mut ws.temp_nu);
    a_plus_eq_b(grad_npred_j, &ws.temp_nu);

}
/// The total cost function
///
/// # Arguments
///
/// * `x0` - initial state
/// * `u_seq` - sequence of inputs
/// * `ws` - workspace of type `BackwardGradientWorkspace`
/// * `n` - prediction horizon
///
///
pub fn total_cost(
    x0: &[f64],
    u_seq: &[f64],
    ws: &mut BackwardGradientWorkspace,
    n: usize
) -> f64 {
    let mut temp_vf=0.0;
    let mut temp_vn=0.0;
    let mut temp_nx_new= vec![0.0; NX];
    let mut vn = 0.0;
    ws.temp_nx[..NX].copy_from_slice(x0);
    let mut xi= vec![0.0; NX];
    xi.copy_from_slice(&x0);
    for i in 0..=n - 1 {
        f(&ws.temp_nx[..NX], &u_seq[i * NU..(i + 1) * NU], &mut temp_nx_new[..NX]);
        ell(&ws.temp_nx[..NX], &u_seq[i * NU..(i + 1) * NU],&mut temp_vn);
        ws.temp_nx.copy_from_slice(&temp_nx_new);
        vn = vn + temp_vn;
    }
    vf(&ws.temp_nx[..NX], &mut temp_vf);
    vn = vn +  temp_vf;
    vn
}



pub fn gradient_descent_arm_time(){
    let n_pred = 50;
    let n_pred_max = 50;
    let n_runs= 50;
    let max_num_iterations= 1000;



    for n_pred in (5..=n_pred_max).step_by(1){
            let mut store= vec! [0.0;  n_runs];
            let mut box_v = vec! [0.0; n_runs];

            let mut k =1;
            let mut j=1;
            let mut sign =true;

            for _j in 1..n_runs {
            j=_j-k;
            sign =true;

            let mut tol:f64 = 1e-5;
            let mut re_tol:f64 = 1e-4;
            let mut vn =0.0;
            let mut f_new =0.0;

            let nu = num_inputs();
            let nx = num_states();
            let mut rng = rand::thread_rng();

            // let x1 = rng.gen_range(0.1..=0.5);
            // let x2 = rng.gen_range(0.1..=1.0);
            let x1 = rng.gen_range(-0.524..=0.524);
            let x2 = rng.gen_range(-0.1..=0.1);

            let x0 = vec![x1, x2];
            // let x0 = vec![0.1, 0.1];


            let mut ws =  BackwardGradientWorkspace::new(n_pred);
            let mut u_seq = vec! [0.1; nu * n_pred];
            let mut u_seq_new = vec! [1.0; nu * n_pred];
            let mut grad = vec! [0.0; nu * n_pred];
            let mut grad_prev = vec![0.0; nu * n_pred];


            // ** value for Armijo condition
            let c1 = 0.5;
            let update=0.5;
            let mut alpha = 1.0;

            // ** value for curvature condition
            // let c2 = 0.9;


            let now = Instant::now();
             // ** Armijo condition
            'outer:for v in (1..=max_num_iterations).step_by(1)  {
                // println!(" u_seq {:?}", u_seq);
                total_cost_gradient_bw(&x0, &u_seq, &mut grad, &mut ws, n_pred, &mut vn);
                let df = grad.clone();
                let f = vn.clone();
                let nrm=dot(&df, &df);
                // println!("df {:?}",df);
                // println!("f {:?}",f);
                // println!("nrm{:?}",nrm);


                // ** initial alpha
                 if v == 1 {
                     let epsilon=vec![1e-9; nu * n_pred];
                     let epsilon_n=1e-9;
                     let r=vec![1.0; nu * n_pred];

                     let grad_f_x0 = grad.clone();
                     for i in 0..nu * n_pred {
                            u_seq_new[i] = u_seq[i] +  epsilon_n * r[i];
                            // ** projection step **
                            // if u_seq_new[i] < p[i] {
                            //     u_seq_new[i] = p[i];
                            // }
                         }
                     total_cost_gradient_bw(&x0, &u_seq_new, &mut grad, &mut ws, n_pred, &mut vn);
                     let grad_f_x1 = grad.clone();
                     let numerator = euclidean_norm(&sub(grad_f_x1, grad_f_x0));
                     let denominator = euclidean_norm(&mul(r, epsilon));
                     let L=numerator / denominator;
                     alpha=0.99/L;
                     // println!("alpha_in{:?}",alpha);
                 }

                if v >= 2 {
                let norm_grad_prev = dot(&grad_prev, &grad_prev);
                let norm_grad = dot(&grad, &grad);
                alpha = alpha * norm_grad_prev / norm_grad;
                // println!("alpha_in{:?}",alpha);
                }

                for i in 0..nu * n_pred {
                        u_seq_new[i] = u_seq[i] - alpha * df[i];
                    }
                f_new = total_cost(&x0, &u_seq_new, &mut ws, n_pred);
                // println!("f_new{:?}", f_new);
                // println!("f{:?}", f);
                let mut vv=v as f64;
                while f_new > f - c1 * alpha *nrm{
                    alpha *= update;
                    // println!("alpha{:?}",alpha);
                    if alpha < 1e-8 {
                        // panic!("Armijo condition failed: alpha too small");
                        vv  =vv +0.1;
                        sign=false;
                        break 'outer;
                    }
                    for i in 0..nu * n_pred {
                        u_seq_new[i] = u_seq[i] - alpha * df[i];
                    }
                    f_new = total_cost(&x0, &u_seq_new, &mut ws, n_pred);
                }
                // println!("u_seq_new{:?}",u_seq_new);
                // println!("u_seq{:?}",u_seq);
                let error =sub( u_seq_new.clone() ,u_seq.clone());
                let error_max= error.iter().max_by(|a, b| a.abs().total_cmp(&b.abs()));
                let error_max_abs =error_max.expect("REASON").abs();
                // println!("v{:?},err{:?},alpha{:?}", v,error_max_abs,alpha);
                // println!("{:?}", error_max_abs);
                if v==1{
                re_tol=tol*error_max_abs;
                // println!("re_tol{:?}", re_tol);
                tol = tol + re_tol;
                // println!("tol = {:?}, err_max = {:?}", tol, error_max_abs);
                }
                if error_max_abs < tol {
                    box_v[j]=vv;
                    break;
                }
                if v  ==max_num_iterations{
                    // panic!("Reach maximum iterations");
                    // println!("Reach maximum iterations");
                    sign=false;
                    break;
                }
                u_seq =u_seq_new.clone();
                grad_prev.copy_from_slice(&grad);
                // println!("----------------");
            }
            if sign==true{
                let elapsed = now.elapsed();
                let duration = 1e3 *(elapsed.as_secs() as f64 + elapsed.subsec_nanos() as f64 * 1e-9);
                // println!("j {:?}", j);
                store[j]=1e3 *duration;
            }
            else {
                let elapsed = now.elapsed();
                // let duration =0.0;
                k=k+1;
            }
        }
        // if iteration has 0.1 means in this run, alpha is too small
        println!("box_v {:?}", box_v[..n_runs-1].to_vec());
        // println!("store {:?}", store);
        let mut store_new = store[..j+1].to_vec();
        store_new.truncate(j+1);
        let data_mean = mean(&store_new);
        let data_std_deviation = std_deviation(&store_new);
        println!("{},{:?},{:?}", n_pred, data_mean.unwrap_or(-1.0), data_std_deviation.unwrap_or(-1.0));


    }

}




// ** fundamental calculation **
pub fn dot(a: &[f64], b: &[f64]) -> f64 {
        let mut sum = 0.0;
        for i in 0..a.len() {
            sum += a[i] * b[i];
        }
        sum
    }
pub fn sub(vec_a: Vec<f64>, vec_b: Vec<f64>)-> Vec<f64>  {
        vec_a.into_iter().zip(vec_b).map(|(a, b)| a - b).collect()
    }
pub fn mul(vec_a: Vec<f64>, vec_b: Vec<f64>)-> Vec<f64>  {
        vec_a.into_iter().zip(vec_b).map(|(a, b)| a * b).collect()
    }
pub fn mean(data: &[f64]) -> Option<f64> {
        let sum = data.iter().sum::<f64>() as f64;
        let count = data.len();
        match count {
            positive if positive > 0 => Some(sum/count as f64),
            _ => None,
        }
    }
pub fn std_deviation(data: &[f64]) -> Option<f64> {
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
pub fn max(vec_a: &Vec<f64>, vec_b: &Vec<f64>) -> Vec<f64> {
        let mut vec_c=vec! [0.0;vec_a.len()];
        for i in 0..vec_a.len() {
            for i in 0..vec_b.len() {
                if vec_a[i] > vec_b[i] {
                    vec_c[i] = vec_a[i];
                }
                if vec_a[i] < vec_b[i] {
                    vec_c[i] = vec_b[i];
                }
            }
        }
        vec_c.to_vec()
    }
pub fn euclidean_norm(v: &[f64]) -> f64 {
    v.iter().map(|&x| x * x).sum::<f64>().sqrt()
    }
