use vjp_kernel::{
    vjp_kernel_g_f, vjp_kernel_g_f_meta, vjp_kernel_g_jf, vjp_kernel_g_jf_meta, vjp_kernel_g_vjp,
    vjp_kernel_g_vjp_meta, FunctionMetadata,
};

fn print_metadata(label: &str, metadata: FunctionMetadata) {
    println!("{label}: {metadata:#?}");
}

fn format_slice(values: &[f64]) -> String {
    let items: Vec<String> = values.iter().map(|value| format!("{value:.4}")).collect();
    format!("[{}]", items.join(", "))
}

fn main() {
    let primal_metadata = vjp_kernel_g_f_meta();
    let jacobian_metadata = vjp_kernel_g_jf_meta();
    let vjp_metadata = vjp_kernel_g_vjp_meta();
    print_metadata("g_f metadata", primal_metadata);
    print_metadata("g_jf metadata", jacobian_metadata);
    print_metadata("g_vjp metadata", vjp_metadata);

    let x = [3.0_f64, 4.0_f64];
    let cotangent_y = [2.0_f64, -1.0_f64, 5.0_f64];

    let mut y = vec![0.0_f64; primal_metadata.output_sizes[0]];
    let mut y_work = vec![0.0_f64; primal_metadata.workspace_size];
    vjp_kernel_g_f(&x, &mut y, &mut y_work).unwrap();
    println!("G(x) = {}", format_slice(&y));

    let mut jacobian_y = vec![0.0_f64; jacobian_metadata.output_sizes[0]];
    let mut jacobian_work = vec![0.0_f64; jacobian_metadata.workspace_size];
    vjp_kernel_g_jf(&x, &mut jacobian_y, &mut jacobian_work).unwrap();
    println!("J_G(x) flat row-major = {}", format_slice(&jacobian_y));

    let mut vjp_x = vec![0.0_f64; vjp_metadata.output_sizes[0]];
    let mut vjp_work = vec![0.0_f64; vjp_metadata.workspace_size];
    vjp_kernel_g_vjp(&x, &cotangent_y, &mut vjp_x, &mut vjp_work).unwrap();
    println!("J_G(x)^T v = {}", format_slice(&vjp_x));
}
