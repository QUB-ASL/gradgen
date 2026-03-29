use codegen_kernel::{
    codegen_kernel_coupling_f, codegen_kernel_coupling_jf_u, codegen_kernel_coupling_jf_u_meta,
    codegen_kernel_coupling_jf_x, codegen_kernel_coupling_jf_x_meta, codegen_kernel_energy_f,
    codegen_kernel_energy_jf_u, codegen_kernel_energy_jf_u_meta, codegen_kernel_energy_jf_x,
    codegen_kernel_energy_jf_x_meta,
};

fn main() {
    let x = [1.0_f64, 2.0_f64, -0.5_f64];
    let u = [3.0_f64];

    let mut energy = [0.0_f64; 1];
    let mut energy_work = vec![0.0_f64; 4];
    codegen_kernel_energy_f(&x, &u, &mut energy, &mut energy_work);
    println!("energy(x, u) = {:?}", energy);

    let mut energy_jf_x = [0.0_f64; 3];
    let mut energy_jf_x_work = vec![0.0_f64; codegen_kernel_energy_jf_x_meta().workspace_size];
    codegen_kernel_energy_jf_x(&x, &u, &mut energy_jf_x, &mut energy_jf_x_work);
    println!("J_energy wrt x = {:?}", energy_jf_x);

    let mut energy_jf_u = [0.0_f64; 1];
    let mut energy_jf_u_work = vec![0.0_f64; codegen_kernel_energy_jf_u_meta().workspace_size];
    codegen_kernel_energy_jf_u(&x, &u, &mut energy_jf_u, &mut energy_jf_u_work);
    println!("J_energy wrt u = {:?}", energy_jf_u);

    let mut coupling = [0.0_f64; 1];
    let mut coupling_work = vec![0.0_f64; 2];
    codegen_kernel_coupling_f(&x, &u, &mut coupling, &mut coupling_work);
    println!("coupling(x, u) = {:?}", coupling);

    let mut coupling_jf_x = [0.0_f64; 3];
    let mut coupling_jf_x_work = vec![0.0_f64; codegen_kernel_coupling_jf_x_meta().workspace_size];
    codegen_kernel_coupling_jf_x(&x, &u, &mut coupling_jf_x, &mut coupling_jf_x_work);
    println!("J_coupling wrt x = {:?}", coupling_jf_x);

    let mut coupling_jf_u = [0.0_f64; 1];
    let mut coupling_jf_u_work = vec![0.0_f64; codegen_kernel_coupling_jf_u_meta().workspace_size];
    codegen_kernel_coupling_jf_u(&x, &u, &mut coupling_jf_u, &mut coupling_jf_u_work);
    println!("J_coupling wrt u = {:?}", coupling_jf_u);
}
