---
sidebar_position: 6
---

# Optimal control

Gradgen provides the class `SingleShootingProblem`, which facilitates the 
computation of the gradient of the total cost function of a finite horizon 
optimal control problem using the single shooting formulation. 

:::note

Although you can use [`Function`](/gradgen/docs/basics/functions) to construct 
the total cost function of your optimal control problem, and then you can use 
gradgen's [automatic differentiation](/gradgen/docs/basics/ad) to compute the 
gradient, `SingleShootingProblem` produces **staged** Rust code: the generated 
code exploits the structure of the optimal control problem. Instead, if you simply 
use `Function` you can end up with thousands of lines of generated code.

:::

## Problem statement

We can the discrete-time dynamical system 
$$x_{k+1} = f(x_k, u_k, p),$$
where $x_k \in \mathbb{R}^{n_x}$ is the state, $u_k\in \mathbb{R}^{n_u}$ is the input,
and $p \in \mathbb{R}^{n_p}$ is a parameter vector. 
We have a stage cost function 
$\ell: \mathbb{R}^{n_x} \times \mathbb{R}^{n_u} \to \mathbb{R}$
and a terminal cost function, 
$V_f: \mathbb{R}^{n_x} \times \mathbb{R}^{n_p} \to \mathbb{R}$. 

For a horizon $N$ we denote the sequence of control actions as 
$u_{\mathrm{seq}} = (u_0, \ldots, u_{N-1})$.

The total cost function with horizon $N$ is

$$V_N(x_0, u_{\mathrm{seq}}, p) = \sum_{t=0}^{N-1} \ell(x_t, u_t, p) + V_f(x_N, p),$$

where note that $x_t = x_t(x_0, u_0, \ldots, u_{t-1})$, so $x_t$ is a 
function of $x_0$ and $u_{\mathrm{seq}}$.

The objective is to generated code for $V_N$, its gradient with respect
to $u_{\mathrm{seq}}$, as well as Hessian-vector products.

The computation of the gradient and Hessian-vector products is particularly 
useful for solving optimal control problems numerically.

## Example

### System dynamics

As an example, consider a dynamical system with $n_x = 2$, 
$n_u=1$, and $n_p = 2$. The system dynamics is 
$$f(x, u, p) = \begin{bmatrix}
x_1 + p_1 x_2 + u \\\\
x_2 + p_2  u - 0.5 x_2
\end{bmatrix}.$$
Let us create the necessary symbols and let us define the system dynamics

[![Try it In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1QFMP-ZF3ZN3swWuab6v5neOmS0n_yKQ2?usp=sharing)

```python
nx, nu, np, N = 2, 1, 2, 5

x = SXVector.sym("x", nx)
u = SXVector.sym("u", nu)
p = SXVector.sym("p", np)

dynamics = Function(
    "dynamics",
    [x, u, p],
    [SXVector((x[0] + p[0] * x[1] + u[0], x[1] + p[1] * u[0] - 0.5 * x[0]))],
    input_names=["x", "u", "p"],
    output_names=["x_next"],
)
```

### Stage and terminal costs

Next, the stage cost function is 
$$\ell(x, u, p) = x_1^2 + 2 x_2^2 + 0.3 u^2.$$

[![Try it In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1QFMP-ZF3ZN3swWuab6v5neOmS0n_yKQ2?usp=sharing)

```python
stage_cost = Function(
    "stage_cost",
    [x, u, p],
    [x[0] * x[0] + 2.0 * x[1] * x[1] + 0.3 * u[0] * u[0]],
    input_names=["x", "u", "p"],
    output_names=["ell"],
)
```

Lastly, the terminal cost function is 
$$V_f(x, p) = 3 x_1^2 + 0.5 x_2^2.$$

[![Try it In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1QFMP-ZF3ZN3swWuab6v5neOmS0n_yKQ2?usp=sharing)

```python
terminal_cost = Function(
    "terminal_cost",
    [x, p],
    [3.0 * x[0] * x[0] + 0.5 * x[1] * x[1]],
    input_names=["x", "p"],
    output_names=["vf"],
)
```

### Single shooting total cost function

We can now construct an instance of `SingleShootingProblem`

[![Try it In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1QFMP-ZF3ZN3swWuab6v5neOmS0n_yKQ2?usp=sharing)

```python
problem = SingleShootingProblem(
    name="mpc_cost",
    horizon=N,
    dynamics=dynamics,
    stage_cost=stage_cost,
    terminal_cost=terminal_cost,
    initial_state_name="x0",
    control_sequence_name="u_seq",
    parameter_name="p",
)
```

### Code generation

We can now generate code for our `problem`. We will generate a Rust 
crate with a function that computes $V_N$, $\nabla V_N$ (with respect to the 
sequence of inputs), and Hessian-vector products, $\nabla^2 V_N^\intercal d$.
Alongside, we want this function to return the corresponding sequence of 
states (this is what `add_rollout_states` does). Here is an example:

[![Try it In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1QFMP-ZF3ZN3swWuab6v5neOmS0n_yKQ2?usp=sharing)

```python
builder = (
    CodeGenerationBuilder()
    .with_backend_config(
        RustBackendConfig()
        .with_crate_name("single_shooting_kernel")
        .with_backend_mode("no_std")
    )
    .for_function(problem)
        .add_joint(
            SingleShootingBundle()
            .add_cost()
            .add_gradient()
            .add_hvp()
            .add_rollout_states()
        )
        .done()
).build()
```

Instead of a single functions that computes the total cost, its gradient, and 
Hessian-vector products at the same time, we can generate separate functions
(see [docs](/gradgen/docs/basics/codegen) for details) using

[![Try it In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1QFMP-ZF3ZN3swWuab6v5neOmS0n_yKQ2?usp=sharing)

```python
.for_function(problem)
        .add_primal(include_states=True)
        .add_gradient(include_states=True)
        .add_hvp(include_states=True)

```

### Generated code

An excerpt of the generated code is shown below

```rust
// Computation of total cost and state trajectory
for stage_index in 0..5 {
        let u_t = &u_seq[stage_index..(stage_index + 1)];
        single_shooting_kernel_mpc_cost_stage_cost(current_state, u_t, p, scalar_buffer, stage_work);
        total_cost += scalar_buffer[0];
        single_shooting_kernel_mpc_cost_dynamics(current_state, u_t, p, next_state, stage_work);
        x_traj[((stage_index + 1) * 2)..((stage_index + 2) * 2)].copy_from_slice(next_state);
        core::mem::swap(&mut current_state, &mut next_state);
    }
```

Note the use of a for loop in the generated code. 

The automatically generated Rust code for the computation of the cost gradient is 
also structured code.

<details>

<summary>Generated gradient</summary>

This is how the generated code for $\nabla V_N(u_{{\rm seq}}, p)$ looks like (excerpt from auto-generated code):

```rust
// Forward pass
for stage_index in 0..5 {
    let u_t = &u_seq[stage_index..(stage_index + 1)];
    single_shooting_kernel_mpc_cost_dynamics(current_state, u_t, p, next_state, stage_work);
    state_history[(stage_index * 2)..((stage_index + 1) * 2)].copy_from_slice(next_state);
    core::mem::swap(&mut current_state, &mut next_state);
}
// Backward pass
single_shooting_kernel_mpc_cost_terminal_cost_grad_x(current_state, p, lambda_current, stage_work);
for stage_index in (1..5).rev() {
    let x_t = &state_history[((stage_index - 1) * 2)..(stage_index * 2)];
    let u_t = &u_seq[stage_index..(stage_index + 1)];
    let grad_u_t = &mut gradient_u_seq[stage_index..(stage_index + 1)];
    single_shooting_kernel_mpc_cost_stage_cost_grad_u(x_t, u_t, p, grad_u_t, stage_work);
    single_shooting_kernel_mpc_cost_dynamics_vjp_u(x_t, u_t, p, &lambda_current[..], temp_control, stage_work);
    grad_u_t[0] += temp_control[0];
    single_shooting_kernel_mpc_cost_stage_cost_grad_x(x_t, u_t, p, lambda_next, stage_work);
    single_shooting_kernel_mpc_cost_dynamics_vjp_x(x_t, u_t, p, &lambda_current[..], temp_state, stage_work);
    lambda_next[0] += temp_state[0];
    lambda_next[1] += temp_state[1];
    lambda_current.copy_from_slice(lambda_next);
}
```

</details>