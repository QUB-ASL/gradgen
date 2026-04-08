# Composed Runner

This binary crate depends on the generated `composed_kernel` crate by path and
calls the staged primal and Jacobian kernels, printing their outputs.

In this demo, the generated `composed_kernel` library is configured as
`no_std`. The `runner` crate itself is a normal `std` binary so it can use
`println!` and serve as a simple practical example of consuming the generated
library from Rust.

Run it from this directory with:

```bash
cargo run
```
