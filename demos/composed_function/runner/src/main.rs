use composed_kernel::{
    composed_kernel_composed_demo_f, composed_kernel_composed_demo_f_meta,
    composed_kernel_composed_demo_grad_x, composed_kernel_composed_demo_grad_x_meta,
    FunctionMetadata,
};

fn print_metadata(label: &str, metadata: FunctionMetadata) {
    println!("{label}: {metadata:#?}");
}

fn format_slice(values: &[f64]) -> String {
    let items: Vec<String> = values.iter().map(|value| format!("{value:.4}")).collect();
    format!("[{}]", items.join(", "))
}

fn build_parameters(parameter_len: usize) -> Vec<f64> {
    assert!(
        parameter_len >= 1 && (parameter_len - 1).is_multiple_of(2),
        "expected packed composed parameters to contain N 2D stage blocks plus one terminal scalar",
    );

    let repeats = (parameter_len - 1) / 2;
    let mut values = Vec::with_capacity(parameter_len);
    for repeat_index in 0..repeats {
        values.push((2 * repeat_index + 3) as f64);
        values.push((2 * repeat_index + 4) as f64);
    }
    values.push((2 * repeats + 3) as f64);
    values
}

fn main() {
    let primal_metadata = composed_kernel_composed_demo_f_meta();
    let gradient_metadata = composed_kernel_composed_demo_grad_x_meta();
    print_metadata("composed_demo_f metadata", primal_metadata);
    print_metadata("composed_demo_grad_x metadata", gradient_metadata);

    let x = [1.0_f64, 2.0_f64];
    let parameters = build_parameters(primal_metadata.input_sizes[1]);

    let mut y = [0.0_f64; 1];
    let mut work = vec![0.0_f64; primal_metadata.workspace_size];
    composed_kernel_composed_demo_f(&x, &parameters, &mut y, &mut work);
    println!("f(x, parameters) = {}", format_slice(&y));

    let mut grad = [0.0_f64; 2];
    let mut grad_work = vec![0.0_f64; gradient_metadata.workspace_size];
    composed_kernel_composed_demo_grad_x(&x, &parameters, &mut grad, &mut grad_work);
    println!("grad f(x, parameters) = {}", format_slice(&grad));
}
