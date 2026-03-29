use multi_function_kernel::{
    multi_function_kernel_coupling_f, multi_function_kernel_coupling_f_jf_x,
    multi_function_kernel_coupling_f_jf_x_meta, multi_function_kernel_coupling_f_meta,
    multi_function_kernel_coupling_grad_x, multi_function_kernel_coupling_grad_x_meta,
    multi_function_kernel_coupling_hvp_x, multi_function_kernel_coupling_hvp_x_meta,
    multi_function_kernel_energy_f, multi_function_kernel_energy_f_meta,
    multi_function_kernel_energy_jf_x, multi_function_kernel_energy_jf_x_meta,
};

fn main() {
    let x = [1.5_f64, -0.25_f64];
    let u = [0.75_f64];
    let v_x = [0.1_f64, -0.3_f64];

    let mut energy = [0.0_f64; 1];
    let mut energy_work = vec![0.0_f64; multi_function_kernel_energy_f_meta().workspace_size];
    multi_function_kernel_energy_f(&x, &u, &mut energy, &mut energy_work);
    println!("energy(x, u) = {:?}", energy);

    let mut energy_jf_x = [0.0_f64; 2];
    let mut energy_jf_x_work =
        vec![0.0_f64; multi_function_kernel_energy_jf_x_meta().workspace_size];
    multi_function_kernel_energy_jf_x(&x, &u, &mut energy_jf_x, &mut energy_jf_x_work);
    println!("J_energy wrt x = {:?}", energy_jf_x);

    let mut coupling = [0.0_f64; 1];
    let mut coupling_work = vec![0.0_f64; multi_function_kernel_coupling_f_meta().workspace_size];
    multi_function_kernel_coupling_f(&x, &u, &mut coupling, &mut coupling_work);
    println!("coupling(x, u) = {:?}", coupling);

    let mut coupling_grad_x = [0.0_f64; 2];
    let mut coupling_grad_x_work =
        vec![0.0_f64; multi_function_kernel_coupling_grad_x_meta().workspace_size];
    multi_function_kernel_coupling_grad_x(&x, &u, &mut coupling_grad_x, &mut coupling_grad_x_work);
    println!("grad coupling wrt x = {:?}", coupling_grad_x);

    let mut coupling_hvp_x = [0.0_f64; 2];
    let mut coupling_hvp_x_work =
        vec![0.0_f64; multi_function_kernel_coupling_hvp_x_meta().workspace_size];
    multi_function_kernel_coupling_hvp_x(
        &x,
        &u,
        &v_x,
        &mut coupling_hvp_x,
        &mut coupling_hvp_x_work,
    );
    println!("hvp coupling wrt x = {:?}", coupling_hvp_x);

    let mut joint_coupling = [0.0_f64; 1];
    let mut joint_jacobian = [0.0_f64; 2];
    let mut joint_work = vec![0.0_f64; multi_function_kernel_coupling_f_jf_x_meta().workspace_size];
    multi_function_kernel_coupling_f_jf_x(
        &x,
        &u,
        &mut joint_coupling,
        &mut joint_jacobian,
        &mut joint_work,
    );
    println!(
        "joint coupling and J_coupling wrt x = {:?}, {:?}",
        joint_coupling, joint_jacobian
    );
}
