use reduce_map_kernel::*;

fn print_metadata(label: &str, metadata: FunctionMetadata) {
    println!("{label}: {metadata:#?}");
}

fn format_slice(values: &[f64]) -> String {
    let items: Vec<String> = values.iter().map(|value| format!("{value:.6}")).collect();
    format!("[{}]", items.join(", "))
}

fn main() {
    let mapped_f_meta = reduce_map_kernel_mapped_seq_f_meta();
    let mapped_jf_meta = reduce_map_kernel_mapped_seq_jf_x_seq_meta();
    let reduced_f_meta = reduce_map_kernel_reduced_scalar_f_meta();

    print_metadata("mapped_seq_f metadata", mapped_f_meta);
    print_metadata("mapped_seq_jf_x_seq metadata", mapped_jf_meta);
    print_metadata("reduced_scalar_f metadata", reduced_f_meta);

    let x_len = mapped_f_meta.input_sizes[0];
    let mut x_seq = vec![0.0_f64; x_len];
    for (index, value) in x_seq.iter_mut().enumerate() {
        let idx = index as f64;
        *value = 0.25_f64 + (0.2_f64 * idx) - (0.05_f64 * idx * idx);
    }
    let acc0 = [0.3_f64];

    let mut mapped_out = vec![0.0_f64; mapped_f_meta.output_sizes[0]];
    let mut mapped_work = vec![0.0_f64; mapped_f_meta.workspace_size];
    reduce_map_kernel_mapped_seq_f(&x_seq, &mut mapped_out, &mut mapped_work).unwrap();
    println!("mapped_seq(x_seq) = {}", format_slice(&mapped_out));

    let mut mapped_jac = vec![0.0_f64; mapped_jf_meta.output_sizes[0]];
    let mut mapped_jac_work = vec![0.0_f64; mapped_jf_meta.workspace_size];
    reduce_map_kernel_mapped_seq_jf_x_seq(&x_seq, &mut mapped_jac, &mut mapped_jac_work).unwrap();
    println!("J_mapped_seq wrt x_seq = {}", format_slice(&mapped_jac));

    let mut reduced_out = vec![0.0_f64; reduced_f_meta.output_sizes[0]];
    let mut reduced_work = vec![0.0_f64; reduced_f_meta.workspace_size];
    reduce_map_kernel_reduced_scalar_f(&acc0, &mapped_out, &mut reduced_out, &mut reduced_work)
        .unwrap();
    println!(
        "reduced_scalar(acc0, mapped_seq(x_seq)) = {}",
        format_slice(&reduced_out)
    );
}
