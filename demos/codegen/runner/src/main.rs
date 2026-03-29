use codegen_kernel::{
    codegen_kernel_coupling_f, codegen_kernel_coupling_f_meta, codegen_kernel_coupling_jf_u,
    codegen_kernel_coupling_jf_u_meta, codegen_kernel_coupling_jf_x,
    codegen_kernel_coupling_jf_x_meta, codegen_kernel_energy_f, codegen_kernel_energy_f_meta,
    codegen_kernel_energy_jf_u, codegen_kernel_energy_jf_u_meta, codegen_kernel_energy_jf_x,
    codegen_kernel_energy_jf_x_meta, FunctionMetadata,
};

fn print_metadata(label: &str, metadata: FunctionMetadata) {
    println!("{label}: {metadata:#?}");
}

fn format_slice(values: &[f64]) -> String {
    let items: Vec<String> = values.iter().map(|value| format!("{value:.4}")).collect();
    format!("[{}]", items.join(", "))
}

fn main() {
    let energy_f_metadata = codegen_kernel_energy_f_meta();
    let energy_jf_x_metadata = codegen_kernel_energy_jf_x_meta();
    let energy_jf_u_metadata = codegen_kernel_energy_jf_u_meta();
    let coupling_f_metadata = codegen_kernel_coupling_f_meta();
    let coupling_jf_x_metadata = codegen_kernel_coupling_jf_x_meta();
    let coupling_jf_u_metadata = codegen_kernel_coupling_jf_u_meta();

    print_metadata("energy_f metadata", energy_f_metadata);
    print_metadata("energy_jf_x metadata", energy_jf_x_metadata);
    print_metadata("energy_jf_u metadata", energy_jf_u_metadata);
    print_metadata("coupling_f metadata", coupling_f_metadata);
    print_metadata("coupling_jf_x metadata", coupling_jf_x_metadata);
    print_metadata("coupling_jf_u metadata", coupling_jf_u_metadata);

    let x = [1.0_f64, 2.0_f64, -0.5_f64];
    let u = [3.0_f64];

    let mut energy = vec![0.0_f64; energy_f_metadata.output_sizes[0]];
    let mut energy_work = vec![0.0_f64; energy_f_metadata.workspace_size];
    codegen_kernel_energy_f(&x, &u, &mut energy, &mut energy_work);
    println!("energy(x, u) = {}", format_slice(&energy));

    let mut energy_jf_x = vec![0.0_f64; energy_jf_x_metadata.output_sizes[0]];
    let mut energy_jf_x_work = vec![0.0_f64; energy_jf_x_metadata.workspace_size];
    codegen_kernel_energy_jf_x(&x, &u, &mut energy_jf_x, &mut energy_jf_x_work);
    println!("J_energy wrt x = {}", format_slice(&energy_jf_x));

    let mut energy_jf_u = vec![0.0_f64; energy_jf_u_metadata.output_sizes[0]];
    let mut energy_jf_u_work = vec![0.0_f64; energy_jf_u_metadata.workspace_size];
    codegen_kernel_energy_jf_u(&x, &u, &mut energy_jf_u, &mut energy_jf_u_work);
    println!("J_energy wrt u = {}", format_slice(&energy_jf_u));

    let mut coupling = vec![0.0_f64; coupling_f_metadata.output_sizes[0]];
    let mut coupling_work = vec![0.0_f64; coupling_f_metadata.workspace_size];
    codegen_kernel_coupling_f(&x, &u, &mut coupling, &mut coupling_work);
    println!("coupling(x, u) = {}", format_slice(&coupling));

    let mut coupling_jf_x = vec![0.0_f64; coupling_jf_x_metadata.output_sizes[0]];
    let mut coupling_jf_x_work = vec![0.0_f64; coupling_jf_x_metadata.workspace_size];
    codegen_kernel_coupling_jf_x(&x, &u, &mut coupling_jf_x, &mut coupling_jf_x_work);
    println!("J_coupling wrt x = {}", format_slice(&coupling_jf_x));

    let mut coupling_jf_u = vec![0.0_f64; coupling_jf_u_metadata.output_sizes[0]];
    let mut coupling_jf_u_work = vec![0.0_f64; coupling_jf_u_metadata.workspace_size];
    codegen_kernel_coupling_jf_u(&x, &u, &mut coupling_jf_u, &mut coupling_jf_u_work);
    println!("J_coupling wrt u = {}", format_slice(&coupling_jf_u));
}
