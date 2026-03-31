use zip_3_kernel::*;

fn print_metadata(label: &str, metadata: FunctionMetadata) {
    println!("{label}: {metadata:#?}");
}

fn format_slice(values: &[f64]) -> String {
    let items: Vec<String> = values.iter().map(|value| format!("{value:.4}")).collect();
    format!("[{}]", items.join(", "))
}

fn main() {
    let zip3_f_metadata = zip_3_kernel_zip3_f_meta();
    let zip3_jf_a_metadata = zip_3_kernel_zip3_jf_a_seq_meta();

    print_metadata("zip3_f metadata", zip3_f_metadata);
    print_metadata("zip3_jf_a_seq metadata", zip3_jf_a_metadata);

    let a_seq = [1.0_f64, -0.2_f64, 1.5_f64, 0.05_f64, 2.0_f64, 0.3_f64];
    let b_seq = [2.0_f64, 1.75_f64, 1.5_f64];
    let c_seq = [0.1_f64, 0.1_f64, 0.5_f64];

    let mut y_seq = vec![0.0_f64; zip3_f_metadata.output_sizes[0]];
    let mut f_work = vec![0.0_f64; zip3_f_metadata.workspace_size];
    zip_3_kernel_zip3_f(&a_seq, &b_seq, &c_seq, &mut y_seq, &mut f_work).unwrap();
    println!("zip3(a_seq, b_seq, c_seq) = {}", format_slice(&y_seq));

    let mut jacobian_y = vec![0.0_f64; zip3_jf_a_metadata.output_sizes[0]];
    let mut jf_work = vec![0.0_f64; zip3_jf_a_metadata.workspace_size];
    zip_3_kernel_zip3_jf_a_seq(&a_seq, &b_seq, &c_seq, &mut jacobian_y, &mut jf_work).unwrap();
    println!("J_zip3 wrt a_seq = {}", format_slice(&jacobian_y));
}
