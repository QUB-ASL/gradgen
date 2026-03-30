use std::hint::black_box;
use std::time::Instant;

use single_shooting_kernel::*;

fn print_metadata(label: &str, metadata: FunctionMetadata) {
    println!("{label}: {metadata:#?}");
}

fn format_slice(values: &[f64]) -> String {
    let items: Vec<String> = values.iter().map(|value| format!("{value:.3}")).collect();
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

fn build_control_direction(control_len: usize) -> Vec<f64> {
    (0..control_len)
        .map(|stage_index| if stage_index % 2 == 0 { 0.5 } else { -1.0 })
        .collect()
}

fn main() {
    let primal_metadata = single_shooting_kernel_mpc_cost_f_states_meta();
    let gradient_metadata = single_shooting_kernel_mpc_cost_grad_states_u_seq_meta();
    let hvp_metadata = single_shooting_kernel_mpc_cost_hvp_states_u_seq_meta();
    let joint_metadata = single_shooting_kernel_mpc_cost_f_grad_states_u_seq_meta();

    print_metadata("mpc_cost_f_states metadata", primal_metadata);
    print_metadata("mpc_cost_grad_states_u_seq metadata", gradient_metadata);
    print_metadata("mpc_cost_hvp_states_u_seq metadata", hvp_metadata);
    print_metadata("mpc_cost_f_grad_states_u_seq metadata", joint_metadata);

    let x0 = [1.0_f64, -0.5_f64];
    let controls = build_controls(primal_metadata.input_sizes[1]);
    let parameters = build_parameters(primal_metadata.input_sizes[2]);
    let control_direction = build_control_direction(hvp_metadata.input_sizes[3]);

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
    )
    .unwrap();
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
    )
    .unwrap();
    println!("grad cost(x0, u_seq, p) = {}", format_slice(&gradient));

    let mut hvp = vec![0.0_f64; hvp_metadata.output_sizes[0]];
    let mut hvp_states = vec![0.0_f64; hvp_metadata.output_sizes[1]];
    let mut hvp_work = vec![0.0_f64; hvp_metadata.workspace_size];
    single_shooting_kernel_mpc_cost_hvp_states_u_seq(
        &x0,
        &controls,
        &parameters,
        &control_direction,
        &mut hvp,
        &mut hvp_states,
        &mut hvp_work,
    )
    .unwrap();
    println!("hvp cost(x0, u_seq, p; v_u_seq) = {}", format_slice(&hvp));

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
    )
    .unwrap();
    println!("joint cost(x0, u_seq, p) = {}", format_slice(&joint_cost));
    println!(
        "joint grad cost(x0, u_seq, p) = {}",
        format_slice(&joint_gradient)
    );
    println!(
        "joint x_traj(x0, u_seq, p) = {}",
        format_slice(&joint_states)
    );

    let iterations = 1000_u32;
    let mut benchmark_cost = vec![0.0_f64; joint_metadata.output_sizes[0]];
    let mut benchmark_gradient = vec![0.0_f64; joint_metadata.output_sizes[1]];
    let mut benchmark_states = vec![0.0_f64; joint_metadata.output_sizes[2]];
    let mut benchmark_work = vec![0.0_f64; joint_metadata.workspace_size];
    let mut checksum = 0.0_f64;

    println!("\n---- BENCHMARKIGN gradient computation ----");

    let started = Instant::now();
    for _ in 0..iterations {
        single_shooting_kernel_mpc_cost_f_grad_states_u_seq(
            black_box(&x0),
            black_box(&controls),
            black_box(&parameters),
            black_box(&mut benchmark_cost),
            black_box(&mut benchmark_gradient),
            black_box(&mut benchmark_states),
            black_box(&mut benchmark_work),
        )
        .unwrap();
        checksum += benchmark_cost[0] + benchmark_gradient[0] + benchmark_states[0];
    }
    let elapsed = started.elapsed();
    let average_microseconds = (elapsed.as_secs_f64() * 1_000_000.0) / f64::from(iterations);

    println!(
        "benchmark: single_shooting_kernel_mpc_cost_f_grad_states_u_seq ran {} times",
        iterations
    );
    println!(
        "average runtime: {:.3} us per call (checksum = {:.4})",
        average_microseconds, checksum
    );

    println!("\n---- BENCHMARKIGN HVP computation ---------");

    let mut benchmark_hvp = vec![0.0_f64; hvp_metadata.output_sizes[0]];
    let mut benchmark_hvp_states = vec![0.0_f64; hvp_metadata.output_sizes[1]];
    let mut benchmark_hvp_work = vec![0.0_f64; hvp_metadata.workspace_size];
    let mut hvp_checksum = 0.0_f64;

    let hvp_started = Instant::now();
    for _ in 0..iterations {
        single_shooting_kernel_mpc_cost_hvp_states_u_seq(
            black_box(&x0),
            black_box(&controls),
            black_box(&parameters),
            black_box(&control_direction),
            black_box(&mut benchmark_hvp),
            black_box(&mut benchmark_hvp_states),
            black_box(&mut benchmark_hvp_work),
        )
        .unwrap();
        hvp_checksum += benchmark_hvp[0] + benchmark_hvp_states[0];
    }
    let hvp_elapsed = hvp_started.elapsed();
    let hvp_average_microseconds =
        (hvp_elapsed.as_secs_f64() * 1_000_000.0) / f64::from(iterations);

    println!(
        "benchmark: single_shooting_kernel_mpc_cost_hvp_states_u_seq ran {} times",
        iterations
    );
    println!(
        "average runtime: {:.3} us per call (checksum = {:.4})\n",
        hvp_average_microseconds, hvp_checksum
    );
}
