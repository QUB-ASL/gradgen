# Benchmarks

This folder contains small comparative benchmarks that focus on code
generation size and native runtime rather than end-user demos.

## Available benchmarks

- [single_shooting](./single_shooting/README.md): compare generated code
  size and native runtime for a bicycle-model single-shooting problem in
  Gradgen and CasADi. The Gradgen side can be generated in `f64` or `f32`.
- [profiling](./profiling/README.md): profile the generated Gradgen Rust
  kernel itself and separate kernel compute from slice-copy overhead.
- [nonlinear_system](./nonlinear_system/README.md): compare Gradgen and
  CasADi on a larger nonlinear single-shooting system with more states
  and inputs.

## Running the benchmarks

From the repository root, after activating your virtual environment:

```bash
make -C demos benchmarks
```
