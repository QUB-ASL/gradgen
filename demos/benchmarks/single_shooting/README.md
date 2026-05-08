# Single Shooting Benchmark

This benchmark compares Gradgen and CasADi for the same single-shooting
optimal-control problem across horizons `N = 10, 20, ..., 100`.

You can choose the Gradgen scalar type with `--gradgen-scalar-type`
(`f64` by default, or `f32` for a smaller floating-point build).

The model is a bicycle-style kinematic system with:

- 4 state variables
- 2 control inputs
- quadratic tracking cost
- quadratic control regularization

For each horizon, the script generates and runs native code for:

- a Gradgen Rust crate that evaluates the total cost and its gradient with
  respect to the packed control sequence
- a small C runner that calls the generated CasADi functions directly

The benchmark prints the number of non-empty lines in each generated source
file, and it also reports the average runtime plus standard deviation for one
cost call plus one gradient call.

The generated runners live in `runners/`.

## Run

From the repository root, after activating your virtual environment:

```bash
python demos/benchmarks/single_shooting/main.py
```
