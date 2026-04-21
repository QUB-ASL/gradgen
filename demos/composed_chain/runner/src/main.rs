use composed_chain_kernel::*;

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

fn build_parameters(parameter_len: usize) -> Vec<f64> {
    assert_eq!(
        parameter_len,
        2,
        "expected the aliased chain demo to use one 2D parameter block",
    );

    vec![0.6, -1.5]
}

fn main() {
    let primal_metadata = composed_chain_kernel_chain_demo_f_meta();
    let gradient_metadata = composed_chain_kernel_chain_demo_grad_x_meta();
    print_metadata("chain_demo_f metadata", primal_metadata);
    print_metadata("chain_demo_grad_x metadata", gradient_metadata);

    let x = [0.5_f64, -0.4_f64];
    let parameters = build_parameters(primal_metadata.input_sizes[1]);

    let mut y = [0.0_f64; 2];
    let mut work = vec![0.0_f64; primal_metadata.workspace_size];
    composed_chain_kernel_chain_demo_f(&x, &parameters, &mut y, &mut work)
        .unwrap();
    println!("f(x, parameters) = {}", format_slice(&y));

    let mut grad = [0.0_f64; 4];
    let mut grad_work = vec![0.0_f64; gradient_metadata.workspace_size];
    composed_chain_kernel_chain_demo_grad_x(
        &x,
        &parameters,
        &mut grad,
        &mut grad_work,
    )
    .unwrap();
    println!("jacobian f(x, parameters) = {}", format_slice(&grad));
}
