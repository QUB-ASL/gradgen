use custom_function_kernel::{
    custom_function_kernel_custom_energy_f, custom_function_kernel_custom_energy_f_meta,
    custom_function_kernel_custom_energy_grad, custom_function_kernel_custom_energy_grad_meta,
    custom_function_kernel_custom_energy_hessian,
    custom_function_kernel_custom_energy_hessian_meta, custom_function_kernel_custom_energy_hvp,
    custom_function_kernel_custom_energy_hvp_meta, FunctionMetadata,
};

fn print_metadata(label: &str, metadata: FunctionMetadata) {
    println!("{label}: {metadata:#?}");
}

fn format_slice(values: &[f64]) -> String {
    let items: Vec<String> = values.iter().map(|value| format!("{value:.4}")).collect();
    format!("[{}]", items.join(", "))
}

fn main() {
    print_metadata(
        "custom_energy_f metadata",
        custom_function_kernel_custom_energy_f_meta(),
    );
    print_metadata(
        "custom_energy_grad metadata",
        custom_function_kernel_custom_energy_grad_meta(),
    );
    print_metadata(
        "custom_energy_hessian metadata",
        custom_function_kernel_custom_energy_hessian_meta(),
    );
    print_metadata(
        "custom_energy_hvp metadata",
        custom_function_kernel_custom_energy_hvp_meta(),
    );

    let x = [1.2_f64, -0.7_f64];
    let v_x = [0.5_f64, -1.0_f64];

    let mut y = [0.0_f64; 1];
    let mut y_work = vec![0.0_f64; custom_function_kernel_custom_energy_f_meta().workspace_size];
    custom_function_kernel_custom_energy_f(&x, &mut y, &mut y_work);
    println!("custom_energy(x) = {}", format_slice(&y));

    let mut grad = [0.0_f64; 2];
    let mut grad_work =
        vec![0.0_f64; custom_function_kernel_custom_energy_grad_meta().workspace_size];
    custom_function_kernel_custom_energy_grad(&x, &mut grad, &mut grad_work);
    println!("grad custom_energy(x) = {}", format_slice(&grad));

    let mut hessian = [0.0_f64; 4];
    let mut hessian_work =
        vec![0.0_f64; custom_function_kernel_custom_energy_hessian_meta().workspace_size];
    custom_function_kernel_custom_energy_hessian(&x, &mut hessian, &mut hessian_work);
    println!("hessian custom_energy(x) = {}", format_slice(&hessian));

    let mut hvp = [0.0_f64; 2];
    let mut hvp_work =
        vec![0.0_f64; custom_function_kernel_custom_energy_hvp_meta().workspace_size];
    custom_function_kernel_custom_energy_hvp(&x, &v_x, &mut hvp, &mut hvp_work);
    println!("hvp custom_energy(x, v_x) = {}", format_slice(&hvp));
}
