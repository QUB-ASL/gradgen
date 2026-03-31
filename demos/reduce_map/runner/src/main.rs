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
    let pipeline_f_meta = reduce_map_kernel_map_reduce_pipeline_f_meta();
    let pipeline_grad_acc0_meta = reduce_map_kernel_map_reduce_pipeline_grad_acc0_meta();
    let pipeline_grad_x_seq_meta = reduce_map_kernel_map_reduce_pipeline_grad_x_seq_meta();

    print_metadata("mapped_seq_f metadata", mapped_f_meta);
    print_metadata("mapped_seq_jf_x_seq metadata", mapped_jf_meta);
    print_metadata("reduced_scalar_f metadata", reduced_f_meta);
    print_metadata("map_reduce_pipeline_f metadata", pipeline_f_meta);
    print_metadata(
        "map_reduce_pipeline_grad_acc0 metadata",
        pipeline_grad_acc0_meta,
    );
    print_metadata(
        "map_reduce_pipeline_grad_x_seq metadata",
        pipeline_grad_x_seq_meta,
    );

    let x_len = pipeline_f_meta.input_sizes[1];
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

    let mut reduced_out = vec![0.0_f64; reduced_f_meta.output_sizes[0]];
    let mut reduced_work = vec![0.0_f64; reduced_f_meta.workspace_size];
    reduce_map_kernel_reduced_scalar_f(&acc0, &mapped_out, &mut reduced_out, &mut reduced_work)
        .unwrap();
    println!(
        "reduced_scalar(acc0, mapped_seq(x_seq)) = {}",
        format_slice(&reduced_out)
    );

    let mut pipeline_out = vec![0.0_f64; pipeline_f_meta.output_sizes[0]];
    let mut pipeline_work = vec![0.0_f64; pipeline_f_meta.workspace_size];
    reduce_map_kernel_map_reduce_pipeline_f(&acc0, &x_seq, &mut pipeline_out, &mut pipeline_work)
        .unwrap();
    println!(
        "map_reduce_pipeline(acc0, x_seq) = {}",
        format_slice(&pipeline_out)
    );

    let mut grad_acc0_out = vec![0.0_f64; pipeline_grad_acc0_meta.output_sizes[0]];
    let mut grad_acc0_work = vec![0.0_f64; pipeline_grad_acc0_meta.workspace_size];
    reduce_map_kernel_map_reduce_pipeline_grad_acc0(
        &acc0,
        &x_seq,
        &mut grad_acc0_out,
        &mut grad_acc0_work,
    )
    .unwrap();
    println!("dF/dacc0 = {}", format_slice(&grad_acc0_out));

    let mut grad_x_seq_out = vec![0.0_f64; pipeline_grad_x_seq_meta.output_sizes[0]];
    let mut grad_x_seq_work = vec![0.0_f64; pipeline_grad_x_seq_meta.workspace_size];
    reduce_map_kernel_map_reduce_pipeline_grad_x_seq(
        &acc0,
        &x_seq,
        &mut grad_x_seq_out,
        &mut grad_x_seq_work,
    )
    .unwrap();
    println!("dF/dx_seq = {}", format_slice(&grad_x_seq_out));
}
