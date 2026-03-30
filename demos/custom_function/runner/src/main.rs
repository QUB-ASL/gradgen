use custom_function_kernel::{
    custom_function_kernel_custom_energy_f, custom_function_kernel_custom_energy_f_meta,
    custom_function_kernel_custom_energy_grad_x_f,
    custom_function_kernel_custom_energy_grad_x_f_meta,
    custom_function_kernel_custom_energy_hessian_x_f,
    custom_function_kernel_custom_energy_hessian_x_f_meta,
    custom_function_kernel_custom_energy_hvp_x_f,
    custom_function_kernel_custom_energy_hvp_x_f_meta, FunctionMetadata,
};

fn print_metadata(label: &str, metadata: FunctionMetadata) {
    println!("{label}: {metadata:#?}");
}

fn format_slice(values: &[f64]) -> String {
    let items: Vec<String> = values.iter().map(|value| format!("{value:.4}")).collect();
    format!("[{}]", items.join(", "))
}

fn main() {
    let primal_metadata = custom_function_kernel_custom_energy_f_meta();
    let gradient_metadata = custom_function_kernel_custom_energy_grad_x_f_meta();
    let hessian_metadata = custom_function_kernel_custom_energy_hessian_x_f_meta();
    let hvp_metadata = custom_function_kernel_custom_energy_hvp_x_f_meta();

    print_metadata("custom_energy_f metadata", primal_metadata);
    print_metadata("custom_energy_grad_x metadata", gradient_metadata);
    print_metadata("custom_energy_hessian_x metadata", hessian_metadata);
    print_metadata("custom_energy_hvp_x metadata", hvp_metadata);

    let x = [1.2_f64, -0.7_f64];
    let w = [1.5_f64, 3.0_f64];
    let v_x = [0.5_f64, -1.0_f64];

    let mut y = vec![0.0_f64; primal_metadata.output_sizes[0]];
    let mut y_work = vec![0.0_f64; primal_metadata.workspace_size];
    custom_function_kernel_custom_energy_f(&x, &w, &mut y, &mut y_work).unwrap();
    println!("custom_energy(x, w) = {}", format_slice(&y));

    let mut grad = vec![0.0_f64; gradient_metadata.output_sizes[0]];
    let mut grad_work = vec![0.0_f64; gradient_metadata.workspace_size];
    custom_function_kernel_custom_energy_grad_x_f(&x, &w, &mut grad, &mut grad_work).unwrap();
    println!("grad_x custom_energy(x, w) = {}", format_slice(&grad));

    let mut hessian = vec![0.0_f64; hessian_metadata.output_sizes[0]];
    let mut hessian_work = vec![0.0_f64; hessian_metadata.workspace_size];
    custom_function_kernel_custom_energy_hessian_x_f(&x, &w, &mut hessian, &mut hessian_work)
        .unwrap();
    println!("hessian_x custom_energy(x, w) = {}", format_slice(&hessian));

    let mut hvp = vec![0.0_f64; hvp_metadata.output_sizes[0]];
    let mut hvp_work = vec![0.0_f64; hvp_metadata.workspace_size];
    custom_function_kernel_custom_energy_hvp_x_f(&x, &w, &v_x, &mut hvp, &mut hvp_work).unwrap();
    println!("hvp_x custom_energy(x, w, v_x) = {}", format_slice(&hvp));
}
