use multi_function_kernel::{
    multi_function_kernel_coupling_f, multi_function_kernel_coupling_f_jf_x,
    multi_function_kernel_coupling_f_jf_x_meta, multi_function_kernel_coupling_f_meta,
    multi_function_kernel_coupling_grad_u_meta, multi_function_kernel_coupling_grad_x,
    multi_function_kernel_coupling_grad_x_meta, multi_function_kernel_coupling_hvp_u_meta,
    multi_function_kernel_coupling_hvp_x, multi_function_kernel_coupling_hvp_x_meta,
    multi_function_kernel_energy_f, multi_function_kernel_energy_f_meta,
    multi_function_kernel_energy_jf_u_meta, multi_function_kernel_energy_jf_x,
    multi_function_kernel_energy_jf_x_meta, FunctionMetadata,
};

fn print_metadata(label: &str, metadata: FunctionMetadata) {
    println!("{label}: {metadata:#?}");
}

fn format_slice(values: &[f64]) -> String {
    let items: Vec<String> = values.iter().map(|value| format!("{value:.4}")).collect();
    format!("[{}]", items.join(", "))
}

fn main() {
    let energy_f_metadata = multi_function_kernel_energy_f_meta();
    let energy_jf_x_metadata = multi_function_kernel_energy_jf_x_meta();
    let energy_jf_u_metadata = multi_function_kernel_energy_jf_u_meta();
    let coupling_f_metadata = multi_function_kernel_coupling_f_meta();
    let coupling_grad_x_metadata = multi_function_kernel_coupling_grad_x_meta();
    let coupling_grad_u_metadata = multi_function_kernel_coupling_grad_u_meta();
    let coupling_hvp_x_metadata = multi_function_kernel_coupling_hvp_x_meta();
    let coupling_hvp_u_metadata = multi_function_kernel_coupling_hvp_u_meta();
    let coupling_joint_metadata = multi_function_kernel_coupling_f_jf_x_meta();

    print_metadata("energy_f metadata", energy_f_metadata);
    print_metadata("energy_jf_x metadata", energy_jf_x_metadata);
    print_metadata("energy_jf_u metadata", energy_jf_u_metadata);
    print_metadata("coupling_f metadata", coupling_f_metadata);
    print_metadata("coupling_grad_x metadata", coupling_grad_x_metadata);
    print_metadata("coupling_grad_u metadata", coupling_grad_u_metadata);
    print_metadata("coupling_hvp_x metadata", coupling_hvp_x_metadata);
    print_metadata("coupling_hvp_u metadata", coupling_hvp_u_metadata);
    print_metadata("coupling_f_jf_x metadata", coupling_joint_metadata);

    let x = [1.5_f64, -0.25_f64];
    let u = [0.75_f64];
    let v_x = [0.1_f64, -0.3_f64];

    let mut energy = vec![0.0_f64; energy_f_metadata.output_sizes[0]];
    let mut energy_work = vec![0.0_f64; energy_f_metadata.workspace_size];
    multi_function_kernel_energy_f(&x, &u, &mut energy, &mut energy_work).unwrap();
    println!("energy(x, u) = {}", format_slice(&energy));

    let mut energy_jf_x = vec![0.0_f64; energy_jf_x_metadata.output_sizes[0]];
    let mut energy_jf_x_work = vec![0.0_f64; energy_jf_x_metadata.workspace_size];
    multi_function_kernel_energy_jf_x(&x, &u, &mut energy_jf_x, &mut energy_jf_x_work).unwrap();
    println!("J_energy wrt x = {}", format_slice(&energy_jf_x));

    let mut coupling = vec![0.0_f64; coupling_f_metadata.output_sizes[0]];
    let mut coupling_work = vec![0.0_f64; coupling_f_metadata.workspace_size];
    multi_function_kernel_coupling_f(&x, &u, &mut coupling, &mut coupling_work).unwrap();
    println!("coupling(x, u) = {}", format_slice(&coupling));

    let mut coupling_grad_x = vec![0.0_f64; coupling_grad_x_metadata.output_sizes[0]];
    let mut coupling_grad_x_work = vec![0.0_f64; coupling_grad_x_metadata.workspace_size];
    multi_function_kernel_coupling_grad_x(&x, &u, &mut coupling_grad_x, &mut coupling_grad_x_work)
        .unwrap();
    println!("grad coupling wrt x = {}", format_slice(&coupling_grad_x));

    let mut coupling_hvp_x = vec![0.0_f64; coupling_hvp_x_metadata.output_sizes[0]];
    let mut coupling_hvp_x_work = vec![0.0_f64; coupling_hvp_x_metadata.workspace_size];
    multi_function_kernel_coupling_hvp_x(
        &x,
        &u,
        &v_x,
        &mut coupling_hvp_x,
        &mut coupling_hvp_x_work,
    )
    .unwrap();
    println!("hvp coupling wrt x = {}", format_slice(&coupling_hvp_x));

    let mut joint_coupling = vec![0.0_f64; coupling_joint_metadata.output_sizes[0]];
    let mut joint_jacobian = vec![0.0_f64; coupling_joint_metadata.output_sizes[1]];
    let mut joint_work = vec![0.0_f64; coupling_joint_metadata.workspace_size];
    multi_function_kernel_coupling_f_jf_x(
        &x,
        &u,
        &mut joint_coupling,
        &mut joint_jacobian,
        &mut joint_work,
    )
    .unwrap();
    println!(
        "joint coupling and J_coupling wrt x = {}, {}",
        format_slice(&joint_coupling),
        format_slice(&joint_jacobian)
    );
}
