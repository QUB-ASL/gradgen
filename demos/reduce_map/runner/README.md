# Map + Reduce Runner

This binary crate depends on the generated `reduce_map_kernel` crate by path and
calls map, reduce, and composed map+reduce kernels.

The generated kernel crate is configured as `no_std`, while this runner is a
normal `std` binary so it can print results.

Run from this directory:

```bash
cargo run
```
