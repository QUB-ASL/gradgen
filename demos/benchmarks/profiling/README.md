# Profiling Demo

This demo profiles the generated Gradgen Rust kernel for a bicycle-model
single-shooting problem. It times the staged primal, gradient, and joint
paths separately, and it also measures a plain slice-copy loop so you can
estimate how much of the time comes from memory movement versus kernel
compute.

## Run

From the repository root, after activating your virtual environment:

```bash
python demos/benchmarks/profiling/main.py
```

Useful options:

- `--horizon`: horizon length used in the generated kernel
- `--num-runs`: number of timed repetitions
- `--gradgen-scalar-type`: `f64` or `f32`

The script prints the average runtime per call in microseconds together with
derived totals that help distinguish kernel math from slice-copy overhead.
