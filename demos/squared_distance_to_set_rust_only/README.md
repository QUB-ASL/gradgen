# Rust-Only Squared Distance To Set

This demo shows how to register a `SquaredDistanceToSet` using only Rust
snippets for the primal distance and the projection map.

The object can still be used symbolically inside gradgen expressions, but it
cannot be evaluated numerically in Python. Rust code generation still works,
which makes this a useful path when the implementation already lives on the
Rust side.

Run the Python generator from this directory:

```bash
python main.py
```

Then run the generated Rust runner:

```bash
cargo run --manifest-path runner/Cargo.toml
```
