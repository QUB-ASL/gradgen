---
sidebar_position: 5
---

# Optimal Control

For deterministic fixed-horizon optimal-control problems, `SingleShootingProblem`
keeps the rollout and adjoint structure explicit instead of expanding the whole
horizon into one very large symbolic expression.

The expected signatures are:

- dynamics: `f(x, u, p) -> x_next`
- stage cost: `ell(x, u, p) -> scalar`
- terminal cost: `Vf(x, p) -> scalar`

Here:

- `x` is the per-stage state vector
- `u` is the per-stage control vector
- `p` is one shared parameter vector used at every stage and in the terminal cost

If the horizon is `N`, the total cost is

$$V_N(x_0, u_{\mathrm{seq}}, p) = \sum_{t=0}^{N-1} \ell(x_t, u_t, p) + V_f(x_N, p),$$

with the state rollout defined by

$$x_{t+1} = f(x_t, u_t, p), \qquad t = 0, \dots, N-1.$$

The packed runtime control-sequence input is one flat vector `u_seq` laid out
stage-major over the horizon:

$$u_{\mathrm{seq}} = [u_0^\top, u_1^\top, \dots, u_{N-1}^\top]^\top.$$

You can generate:

- the total cost
- its gradient with respect to the packed control sequence
- its Hessian-vector product with respect to the packed control sequence
- any joint combination of those, optionally including the rollout states

```python
from gradgen import (
    CodeGenerationBuilder,
    Function,
    RustBackendConfig,
    SXVector,
    SingleShootingBundle,
    SingleShootingProblem,
)

nx = 2
nu = 1
np = 2
N = 5

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

stage_cost = Function(
    "stage_cost",
    [x, u, p],
    [x[0] * x[0] + 2.0 * x[1] * x[1] + 0.3 * u[0] * u[0]],
    input_names=["x", "u", "p"],
    output_names=["ell"],
)

terminal_cost = Function(
    "terminal_cost",
    [x, p],
    [3.0 * x[0] * x[0] + 0.5 * x[1] * x[1]],
    input_names=["x", "p"],
    output_names=["vf"],
)

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

builder = (
    CodeGenerationBuilder()
    .with_backend_config(
        RustBackendConfig()
        .with_crate_name("single_shooting_kernel")
        .with_backend_mode("no_std")
    )
    .for_function(problem)
        .add_primal(include_states=True)
        .add_gradient(include_states=True)
        .add_hvp(include_states=True)
        .add_joint(
            SingleShootingBundle()
            .add_cost()
            .add_gradient()
            .add_hvp()
            .add_rollout_states()
        )
        .with_simplification("medium")
        .done()
)

project = builder.build("./single_shooting_kernel")
```

The generated Rust uses forward and backward `for` loops over the horizon
instead of fully unrolling the dynamics and adjoint recursion. The public Rust
ABI uses:

- `x0`: initial state slice of length `nx`
- `u_seq`: packed control-sequence slice of length `N * nu`
- `p`: shared parameter slice of length `np`
- `v_u_seq`: packed HVP direction slice of length `N * nu` when HVP is requested

Typical generated kernels include:

- `single_shooting_kernel_mpc_cost_f_states`
- `single_shooting_kernel_mpc_cost_grad_states_u_seq`
- `single_shooting_kernel_mpc_cost_hvp_states_u_seq`
- `single_shooting_kernel_mpc_cost_f_grad_hvp_states_u_seq`