# Custom Function Runner

This binary crate depends on the generated `custom_function_kernel` crate by
path and calls the exported primal and derivative kernels, printing the
results.

In this demo, the generated `custom_function_kernel` library is configured as
`no_std`. The `runner` crate itself is a normal `std` binary so it can use
`println!` and serve as a simple practical example of consuming the generated
library from Rust.

Run it from this directory with:

```bash
cargo run
```
