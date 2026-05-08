# Benchmarks

This folder contains small comparative benchmarks that focus on code
generation size and shape rather than end-user demos.

## Available benchmarks

- [single_shooting](./single_shooting/README.md): compare generated code
  size for a bicycle-model single-shooting problem in Gradgen and CasADi.

## Running the benchmarks

From the repository root, after activating your virtual environment:

```bash
make -C demos benchmarks
```
