---
sidebar_position: 5
---

# Rust-Python interface

[![Try it In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1xueMa6EcfOR_M9mPvCGdqhyCyyW8NI-_?usp=sharing)

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
    .build("./my_crates")
)
```

This will generate a crate in `./my_crates/blah` and a Python interface in
`./my_crates/blah_python`.

The generated Python wrapper uses its own `pyproject.toml` version. The first
time the interface is generated, that version is `0.1.0`. If you regenerate
the same interface later, Gradgen bumps the wrapper version to `0.2.0`, then
`0.3.0`, and so on. The low-level Cargo crate version does not change.

If you also want Gradgen to compile the generated Rust crate immediately after
writing it, enable:

```python
RustBackendConfig().with_build_crate()
```

That step is optional and defaults to off. When it is enabled, Gradgen runs
`cargo build` for the generated low-level crate and raises an informative error
if `cargo` is not installed.

You can now use this as a **python module**. You can then do

```python
import blah
print(blah.__version__)
print(blah.__all__)
```

If you want to get a list of all functions in `blah`, do

```python
print(blah.all_functions()) # prints ['energy']
```

## Calling functions

[![Try it In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1xueMa6EcfOR_M9mPvCGdqhyCyyW8NI-_?usp=sharing)

To call any function from the generated package, `blah`, you first need 
to create a workspace variable. This is the only time you will need to allocate 
memory. To allocate memory for the function `energy` do

```python
# Do this once
wspace = blah.workspace_for_function('energy')
```

and then

```python
x = [1., 2.]
w = [3.]
result = blah.energy(x, w, wspace)
```

This returns a dictionary with the output or outputs of the function - in this case 

```json
{
    'cost': 8.0, 
    'state': 3.0
}
```
