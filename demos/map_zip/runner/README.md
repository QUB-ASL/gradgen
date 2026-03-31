# Map/Zip Runner

This binary crate depends on the generated `map_zip_kernel` crate by path and
calls map/zip-related kernels, printing their outputs.

In this demo, the generated `map_zip_kernel` library is configured as
`no_std`. The `runner` crate itself is a normal `std` binary so it can use
`println!` and serve as a simple practical example of consuming the generated
library from Rust.

Run it from this directory with:

```bash
cargo run
```
