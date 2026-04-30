# Single-Shooting Residual Penalty Demo

This demo shows how to add vector-valued residual penalties to a deterministic
single-shooting optimal-control problem.

The generated total cost has the form

```text
sum_k ell(x_k, u_k, p) + c/2 * ||q(x_k, u_k, p)||_2^2
    + V_f(x_N, p) + c/2 * ||q_N(x_N, p)||_2^2
```

where:

- `ell(x, u, p)` is the scalar stage cost.
- `q(x, u, p)` is a vector-valued stage residual.
- `V_f(x_N, p)` is the scalar terminal cost.
- `q_N(x_N, p)` is a vector-valued terminal residual.
- `c` is a scalar runtime input passed to the generated Rust kernels.

Run the generator from the repository root with:

```bash
python demos/single_shooting_penalty/main.py --horizon 5
```

or through the demos Makefile:

```bash
make -C demos single_shooting_penalty PYTHON=../venv/bin/python
```

The script generates:

- `single_shooting_penalty_kernel/`: the Rust crate containing primal,
  gradient, HVP, and joint single-shooting kernels.
- `single_shooting_penalty_kernel_python/`: the PyO3 wrapper crate used for
  `import single_shooting_penalty_kernel` from Python.
- `runner/`: a small Rust binary that calls the generated kernels.

At the end of generation, `main.py` imports the generated Python module and
calls the joint cost, gradient, and rollout-states kernel from Python.

After generating the crate, run the binary demo with:

```bash
cargo run --manifest-path demos/single_shooting_penalty/runner/Cargo.toml
```
