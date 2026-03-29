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
    print_metadata("g_f metadata", vjp_kernel_g_f_meta());
    print_metadata("g_jf metadata", vjp_kernel_g_jf_meta());
    print_metadata("g_vjp metadata", vjp_kernel_g_vjp_meta());

    let x = [3.0_f64, 4.0_f64];
    let cotangent_y = [2.0_f64, -1.0_f64, 5.0_f64];

    let mut y = [0.0_f64; 3];
    let mut y_work = vec![0.0_f64; vjp_kernel_g_f_meta().workspace_size];
    vjp_kernel_g_f(&x, &mut y, &mut y_work);
    println!("G(x) = {}", format_slice(&y));

    let mut jacobian_y = [0.0_f64; 6];
    let mut jacobian_work = vec![0.0_f64; vjp_kernel_g_jf_meta().workspace_size];
    vjp_kernel_g_jf(&x, &mut jacobian_y, &mut jacobian_work);
    println!("J_G(x) flat row-major = {}", format_slice(&jacobian_y));

    let mut vjp_x = [0.0_f64; 2];
    let mut vjp_work = vec![0.0_f64; vjp_kernel_g_vjp_meta().workspace_size];
    vjp_kernel_g_vjp(&x, &cotangent_y, &mut vjp_x, &mut vjp_work);
    println!("J_G(x)^T v = {}", format_slice(&vjp_x));
}
