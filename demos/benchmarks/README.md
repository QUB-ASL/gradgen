# Benchmarks

This folder contains small comparative benchmarks that focus on code
generation size and native runtime rather than end-user demos.

## Available benchmarks

- [single_shooting](./single_shooting/README.md): compare generated code
  size and native runtime for a bicycle-model single-shooting problem in
  Gradgen and CasADi. The Gradgen side can be generated in `f64` or `f32`.

## Running the benchmarks

From the repository root, after activating your virtual environment:

```bash
make -C demos benchmarks
```
