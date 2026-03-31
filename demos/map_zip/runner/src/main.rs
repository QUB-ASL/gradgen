use map_zip_kernel::*;

fn print_metadata(label: &str, metadata: FunctionMetadata) {
    println!("{label}: {metadata:#?}");
}

fn format_slice(values: &[f64]) -> String {
    let items: Vec<String> = values.iter().map(|value| format!("{value:.4}")).collect();
    format!("[{}]", items.join(", "))
}

fn main() {
    let unary_map_f_metadata = map_zip_kernel_unary_map_f_meta();
    let unary_map_jf_metadata = map_zip_kernel_unary_map_jf_x_seq_meta();
    let binary_zip_f_metadata = map_zip_kernel_binary_zip_f_meta();
    let binary_zip_jf_a_seq_metadata = map_zip_kernel_binary_zip_jf_a_seq_meta();
    let binary_zip_jf_b_seq_metadata = map_zip_kernel_binary_zip_jf_b_seq_meta();

    print_metadata("unary_map_f metadata", unary_map_f_metadata);
    print_metadata("unary_map_jf metadata", unary_map_jf_metadata);
    print_metadata("binary_zip_f metadata", binary_zip_f_metadata);
    print_metadata("binary_zip_jf_a_seq metadata", binary_zip_jf_a_seq_metadata);
    print_metadata("binary_zip_jf_b_seq metadata", binary_zip_jf_b_seq_metadata);

    let x_seq = [1.0_f64, 0.5_f64, -0.2_f64, 2.0_f64, 0.8_f64, -1.5_f64];
    let a_seq = [1.0_f64, 0.5_f64, -0.3_f64, 2.0_f64, 0.7_f64, -1.2_f64];
    let b_seq = [0.2_f64, 1.1_f64, 0.4_f64, -0.5_f64, -1.3_f64, 0.9_f64];

    let mut unary_map_y = vec![0.0_f64; unary_map_f_metadata.output_sizes[0]];
    let mut unary_map_work = vec![0.0_f64; unary_map_f_metadata.workspace_size];
    map_zip_kernel_unary_map_f(&x_seq, &mut unary_map_y, &mut unary_map_work).unwrap();
    println!("unary_map(x_seq) = {}", format_slice(&unary_map_y));

    let mut binary_zip_z = vec![0.0_f64; binary_zip_f_metadata.output_sizes[0]];
    let mut binary_zip_work = vec![0.0_f64; binary_zip_f_metadata.workspace_size];
    map_zip_kernel_binary_zip_f(&a_seq, &b_seq, &mut binary_zip_z, &mut binary_zip_work).unwrap();
    println!("binary_zip(a_seq, b_seq) = {}", format_slice(&binary_zip_z));

    let mut unary_map_jf_x_seq = vec![0.0_f64; unary_map_jf_metadata.output_sizes[0]];
    let mut unary_map_jf_x_seq_work = vec![0.0_f64; unary_map_jf_metadata.workspace_size];
    map_zip_kernel_unary_map_jf_x_seq(&x_seq, &mut unary_map_jf_x_seq, &mut unary_map_jf_x_seq_work)
        .unwrap();
    println!(
        "J_unary_map wrt x_seq = {}",
        format_slice(&unary_map_jf_x_seq)
    );

    let mut binary_zip_jf_a_seq = vec![0.0_f64; binary_zip_jf_a_seq_metadata.output_sizes[0]];
    let mut binary_zip_jf_a_seq_work = vec![0.0_f64; binary_zip_jf_a_seq_metadata.workspace_size];
    map_zip_kernel_binary_zip_jf_a_seq(
        &a_seq,
        &b_seq,
        &mut binary_zip_jf_a_seq,
        &mut binary_zip_jf_a_seq_work,
    )
    .unwrap();
    println!(
        "J_binary_zip wrt a_seq from binary_zip_jf_a_seq = {}",
        format_slice(&binary_zip_jf_a_seq)
    );

}
