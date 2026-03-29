use single_shooting_kernel::*;


fn print_metadata(label: &str, metadata: FunctionMetadata) {
    println!("{label}: {metadata:#?}");
}

fn format_slice(values: &[f64]) -> String {
    let items: Vec<String> = values.iter().map(|value| format!("{value:.4}")).collect();
    format!("[{}]", items.join(", "))
}

fn build_controls(control_len: usize) -> Vec<f64> {
    (0..control_len)
        .map(|stage_index| if stage_index % 2 == 0 { 0.2 } else { -0.1 })
        .collect()
}

fn build_parameters(parameter_len: usize) -> Vec<f64> {
    assert_eq!(
        parameter_len, 2,
        "the demo expects a shared 2D parameter vector"
    );
    vec![0.4, -1.2]
}

fn main() {
    let primal_metadata = single_shooting_kernel_mpc_cost_f_states_meta();
    let gradient_metadata = single_shooting_kernel_mpc_cost_grad_states_u_seq_meta();
    let joint_metadata = single_shooting_kernel_mpc_cost_f_grad_states_u_seq_meta();

    print_metadata("mpc_cost_f_states metadata", primal_metadata);
    print_metadata("mpc_cost_grad_states_u_seq metadata", gradient_metadata);
    print_metadata("mpc_cost_f_grad_states_u_seq metadata", joint_metadata);

    let x0 = [1.0_f64, -0.5_f64];
    let controls = build_controls(primal_metadata.input_sizes[1]);
    let parameters = build_parameters(primal_metadata.input_sizes[2]);

    let mut cost = vec![0.0_f64; primal_metadata.output_sizes[0]];
    let mut x_traj = vec![0.0_f64; primal_metadata.output_sizes[1]];
    let mut primal_work = vec![0.0_f64; primal_metadata.workspace_size];
    single_shooting_kernel_mpc_cost_f_states(
        &x0,
        &controls,
        &parameters,
        &mut cost,
        &mut x_traj,
        &mut primal_work,
    );
    println!("cost(x0, u_seq, p) = {}", format_slice(&cost));
    println!("x_traj(x0, u_seq, p) = {}", format_slice(&x_traj));

    let mut gradient = vec![0.0_f64; gradient_metadata.output_sizes[0]];
    let mut gradient_states = vec![0.0_f64; gradient_metadata.output_sizes[1]];
    let mut gradient_work = vec![0.0_f64; gradient_metadata.workspace_size];
    single_shooting_kernel_mpc_cost_grad_states_u_seq(
        &x0,
        &controls,
        &parameters,
        &mut gradient,
        &mut gradient_states,
        &mut gradient_work,
    );
    println!("grad cost(x0, u_seq, p) = {}", format_slice(&gradient));

    let mut joint_cost = vec![0.0_f64; joint_metadata.output_sizes[0]];
    let mut joint_gradient = vec![0.0_f64; joint_metadata.output_sizes[1]];
    let mut joint_states = vec![0.0_f64; joint_metadata.output_sizes[2]];
    let mut joint_work = vec![0.0_f64; joint_metadata.workspace_size];
    single_shooting_kernel_mpc_cost_f_grad_states_u_seq(
        &x0,
        &controls,
        &parameters,
        &mut joint_cost,
        &mut joint_gradient,
        &mut joint_states,
        &mut joint_work,
    );
    println!("joint cost(x0, u_seq, p) = {}", format_slice(&joint_cost));
    println!(
        "joint grad cost(x0, u_seq, p) = {}",
        format_slice(&joint_gradient)
    );
    println!(
        "joint x_traj(x0, u_seq, p) = {}",
        format_slice(&joint_states)
    );
}
