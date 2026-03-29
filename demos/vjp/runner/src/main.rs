use vjp_kernel::{
    vjp_kernel_g_f, vjp_kernel_g_f_meta, vjp_kernel_g_jf, vjp_kernel_g_jf_meta, vjp_kernel_g_vjp,
    vjp_kernel_g_vjp_meta,
};

fn main() {
    let x = [3.0_f64, 4.0_f64];
    let cotangent_y = [2.0_f64, -1.0_f64, 5.0_f64];

    let mut y = [0.0_f64; 3];
    let mut y_work = vec![0.0_f64; vjp_kernel_g_f_meta().workspace_size];
    vjp_kernel_g_f(&x, &mut y, &mut y_work);
    println!("G(x) = {:?}", y);

    let mut jacobian_y = [0.0_f64; 6];
    let mut jacobian_work = vec![0.0_f64; vjp_kernel_g_jf_meta().workspace_size];
    vjp_kernel_g_jf(&x, &mut jacobian_y, &mut jacobian_work);
    println!("J_G(x) flat row-major = {:?}", jacobian_y);

    let mut vjp_x = [0.0_f64; 2];
    let mut vjp_work = vec![0.0_f64; vjp_kernel_g_vjp_meta().workspace_size];
    vjp_kernel_g_vjp(&x, &cotangent_y, &mut vjp_x, &mut vjp_work);
    println!("J_G(x)^T v = {:?}", vjp_x);
}
