from gradgen import Function, SX, SXVector, RustBackendConfig, create_rust_project

x = SXVector.sym("x", 2)
u = SX.sym("u")

f = Function(
    "multi_out",
    [x, u],
    [
        x[0] + x[1],                     # output 1: scalar
        SXVector((x[0] * u, x[1] - u)),  # output 2: vector (length 2)
    ],
    input_names=["x", "u"],
    output_names=["sum_x", "affine_x"],
)

project = create_rust_project(
    f,
    "multi_out_kernel",
    config=(
        RustBackendConfig()
        .with_crate_name("asdf")
        .with_backend_mode("std")   # or "no_std"
        .with_scalar_type("f64")    # or "f32"
    ),
)
