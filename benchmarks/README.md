## for gradgen benchmark
- add `example.rs` file to gradgen rust src folder
- change name `example.rs` to `main.rs`
- terminal: `cargo build; cargo run`

## for casadi benchmark
- replace start and end of `interface.c` in open icasadi extern folder with `example.c`
- terminal: `clang -fPIC -shared auto_casadi_cost.c auto_casadi_grad.c auto_casadi_mapping_f1.c auto_casadi_mapping_f2.c auto_preconditioning_functions.c -o casadi_functions.so`
- terminal: `clang -DTEST_INTERFACE -o test interface.c casadi_functions.so -lm; ./test`
