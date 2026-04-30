use single_shooting_penalty_kernel::*;

fn format_slice(values: &[f64]) -> String {
    let items: Vec<String> = values
        .iter()
        .map(|value| format!("{value:.4}"))
        .collect();
    format!("[{}]", items.join(", "))
}

fn alternating_values(len: usize, even_value: f64, odd_value: f64) -> Vec<f64> {
    (0..len)
        .map(|index| {
            if index % 2 == 0 {
                even_value
            } else {
                odd_value
            }
        })
        .collect()
}

fn main() {
    let primal_metadata =
        single_shooting_penalty_kernel_penalized_mpc_cost_f_states_meta();
    let gradient_metadata =
        single_shooting_penalty_kernel_penalized_mpc_cost_grad_states_u_seq_meta();
    let hvp_metadata =
        single_shooting_penalty_kernel_penalized_mpc_cost_hvp_states_u_seq_meta();
    let joint_metadata =
        single_shooting_penalty_kernel_penalized_mpc_cost_f_grad_states_u_seq_meta();

    println!("primal metadata: {primal_metadata:#?}");
    println!("gradient metadata: {gradient_metadata:#?}");
    println!("hvp metadata: {hvp_metadata:#?}");
    println!("joint metadata: {joint_metadata:#?}");

    let x0 = [1.0_f64, -0.5_f64];
    let controls = alternating_values(primal_metadata.input_sizes[1], 0.2, -0.1);
    let parameters = [0.4_f64, -1.2_f64];
    let penalty_weight = [10.0_f64];
    let control_direction =
        alternating_values(hvp_metadata.input_sizes[4], 0.5, -1.0);

    let mut cost = vec![0.0_f64; primal_metadata.output_sizes[0]];
    let mut x_traj = vec![0.0_f64; primal_metadata.output_sizes[1]];
    let mut primal_work = vec![0.0_f64; primal_metadata.workspace_size];
    single_shooting_penalty_kernel_penalized_mpc_cost_f_states(
        &x0,
        &controls,
        &parameters,
        &penalty_weight,
        &mut cost,
        &mut x_traj,
        &mut primal_work,
    )
    .unwrap();
    println!("penalized cost = {}", format_slice(&cost));
    println!("rollout states = {}", format_slice(&x_traj));

    let mut gradient = vec![0.0_f64; gradient_metadata.output_sizes[0]];
    let mut gradient_states = vec![0.0_f64; gradient_metadata.output_sizes[1]];
    let mut gradient_work = vec![0.0_f64; gradient_metadata.workspace_size];
    single_shooting_penalty_kernel_penalized_mpc_cost_grad_states_u_seq(
        &x0,
        &controls,
        &parameters,
        &penalty_weight,
        &mut gradient,
        &mut gradient_states,
        &mut gradient_work,
    )
    .unwrap();
    println!("control gradient = {}", format_slice(&gradient));

    let mut hvp = vec![0.0_f64; hvp_metadata.output_sizes[0]];
    let mut hvp_states = vec![0.0_f64; hvp_metadata.output_sizes[1]];
    let mut hvp_work = vec![0.0_f64; hvp_metadata.workspace_size];
    single_shooting_penalty_kernel_penalized_mpc_cost_hvp_states_u_seq(
        &x0,
        &controls,
        &parameters,
        &penalty_weight,
        &control_direction,
        &mut hvp,
        &mut hvp_states,
        &mut hvp_work,
    )
    .unwrap();
    println!("control HVP = {}", format_slice(&hvp));

    let mut joint_cost = vec![0.0_f64; joint_metadata.output_sizes[0]];
    let mut joint_gradient = vec![0.0_f64; joint_metadata.output_sizes[1]];
    let mut joint_states = vec![0.0_f64; joint_metadata.output_sizes[2]];
    let mut joint_work = vec![0.0_f64; joint_metadata.workspace_size];
    single_shooting_penalty_kernel_penalized_mpc_cost_f_grad_states_u_seq(
        &x0,
        &controls,
        &parameters,
        &penalty_weight,
        &mut joint_cost,
        &mut joint_gradient,
        &mut joint_states,
        &mut joint_work,
    )
    .unwrap();
    println!("joint cost = {}", format_slice(&joint_cost));
    println!("joint gradient = {}", format_slice(&joint_gradient));
    println!("joint states = {}", format_slice(&joint_states));
}
