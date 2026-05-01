use squared_distance_to_set_rust_only_kernel::*;

fn print_metadata(label: &str, metadata: FunctionMetadata) {
    println!("{label}: {metadata:#?}");
}

fn format_slice(values: &[f64]) -> String {
    let items: Vec<String> = values
        .iter()
        .map(|value| format!("{value:.4}"))
        .collect();
    format!("[{}]", items.join(", "))
}

fn main() {
    let primal_metadata =
        squared_distance_to_set_rust_only_kernel_distance_energy_f_meta();
    let gradient_metadata =
        squared_distance_to_set_rust_only_kernel_distance_energy_grad_x_f_meta(
        );

    print_metadata("distance_energy metadata", primal_metadata);
    print_metadata("distance_energy_grad_x metadata", gradient_metadata);

    let x = [1.5_f64, -3.0_f64];

    let mut y = vec![0.0_f64; primal_metadata.output_sizes[0]];
    let mut y_work = vec![0.0_f64; primal_metadata.workspace_size];
    squared_distance_to_set_rust_only_kernel_distance_energy_f(
        &x,
        &mut y,
        &mut y_work,
    )
    .unwrap();
    println!("distance_energy(x) = {}", format_slice(&y));

    let mut grad = vec![0.0_f64; gradient_metadata.output_sizes[0]];
    let mut grad_work = vec![0.0_f64; gradient_metadata.workspace_size];
    squared_distance_to_set_rust_only_kernel_distance_energy_grad_x_f(
        &x,
        &mut grad,
        &mut grad_work,
    )
    .unwrap();
    println!("grad_x distance_energy(x) = {}", format_slice(&grad));
}
