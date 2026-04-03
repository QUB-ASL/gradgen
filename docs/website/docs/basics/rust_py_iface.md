---
sidebar_position: 5
---

# Rust-Python interface

Once you have generated a Rust crate for your functions, you can consume it directly from Python. 

```python
x = SXVector.sym("x", 2)
w = SXVector.sym("w", 1)

energy = Function(
    "energy",
    [x, w],
    [x.norm2sq() + w[0], x[0] + x[1]],
    input_names=["x", "w"],
    output_names=["cost", "state"],
)

project = create_rust_project(
    energy,
    args.output_dir,
    config=RustBackendConfig()
    .with_crate_name("foo")
    .with_enable_python_interface(True),
)
```