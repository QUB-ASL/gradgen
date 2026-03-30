# Single Shooting Runner

This binary crate depends on the generated `single_shooting_kernel` crate by
path and calls the generated single-shooting kernels, printing their metadata
and outputs.

In this demo, the generated `single_shooting_kernel` library is configured as
`no_std`. The `runner` crate itself is a normal `std` binary so it can use
`println!` and serve as a simple practical example of consuming the generated
library from Rust.

Run it from this directory with:

```bash
cargo run
```
