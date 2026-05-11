# Nonlinear System Benchmark

This benchmark compares Gradgen and CasADi on a larger nonlinear
single-shooting optimal-control problem with 10 states and 5 control
inputs.

The model is intentionally more compute-heavy than the bicycle benchmark
so we can check whether the Gradgen-versus-CasADi gap changes when the
system dimension grows.

For each horizon, the script generates and runs native code for:

- a Gradgen Rust crate that evaluates the total cost and its gradient in
  one joint call with respect to the packed control sequence
- a small C runner that calls the generated CasADi joint function
  directly

The benchmark prints the number of non-empty lines in each generated
source file and reports the average runtime of one joint
cost-plus-gradient call after dividing the total time for all repeated
runs by `--num-runs`.

## Run

From the repository root, after activating your virtual environment:

```bash
python demos/benchmarks/nonlinear_system/main.py
```
