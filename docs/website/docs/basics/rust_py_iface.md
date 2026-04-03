---
sidebar_position: 5
---

# Rust-Python interface

Once you have generated a Rust crate for your functions, you can consume it directly from Python. 
This is done by using `.with_enable_python_interface()` as shown below

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

project = (
    CodeGenerationBuilder()
    .with_backend_config(
        RustBackendConfig()
        .with_crate_name("blah")
        .with_backend_mode("no_std")
        .with_scalar_type("f64")
        .with_enable_python_interface() # <-- this
    )
    # Specify what needs to be generated
    .for_function(energy)
        .add_primal()
        .done()
    .build("./blah")
)
```

This will generate a crate (in folder `./blah`) and a Python interface, 
which will be stored in the folder `./blah_python`.

You can now use this as a **python module**. To this end, 
from within `./blah_python` run 

```bash
pip install -e .
```


