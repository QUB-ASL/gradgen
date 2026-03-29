# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).


## Unreleased

### Added

- Added many new scalar elementary functions to `SX`, evaluation, simplification, and Rust code generation:
  `atan2`, `hypot`, `asinh`, `acosh`, `atanh`, `cbrt`, `erf`, `erfc`,
  `floor`, `ceil`, `round`, `trunc`, `fract`, `signum`, and scalar `minimum`.
- Added first-class vector norm operations:
  `norm2`, `norm2sq`, `norm1`, `norm_inf`, `norm_p`, and `norm_p_to_p`.
- Added vector reduction operations:
  `sum`, `prod`, `max`, `min`, and `mean`.
- Added `SXVector` slicing support so `x[i]` returns an `SX` and `x[a:b]`
  returns an `SXVector`.
- Added module-level Rust helper emission for vector norms, reductions, and
  special functions such as `erf` and `erfc`.
- Added support for registering user-defined elementary functions with
  `register_elementary_function(...)` for scalar-input and fixed-size
  vector-input custom primitives.
- Added support for parameter vectors on custom primitives via
  `parameter_dimension` and `w=[...]`.
- Added optional custom Python `hvp` callbacks for user-defined elementary
  functions.
- Added runnable demos in `demos/codegen` and `demos/custom_function`.
- Added substantial regression and runtime coverage for symbolic math, AD, and
  generated Rust crates.

### Changed

- Improved Rust workspace allocation by reusing slots based on value lifetimes
  instead of assigning one slot per intermediate.
- Improved generated Rust formatting for metadata helpers so emitted code is
  closer to `cargo fmt` style.
- Updated zero-workspace generated Rust functions to use `_work` instead of
  `work` to avoid unused-variable warnings.
- Updated Rust reduction helpers to use collision-safe names such as
  `vec_sum`, `vec_prod`, `vec_max`, `vec_min`, and `vec_mean`.
- Updated singleton `SXVector` behavior so length-1 vectors can participate more
  naturally in scalar expressions.
- Updated custom primitive derivative APIs so Python Jacobian, Hessian, and HVP
  callbacks may return plain Python containers or NumPy arrays.
- Updated custom primitive derivative handling so Jacobian, Hessian, and HVP
  callbacks are treated as opaque numeric Python callbacks during registration,
  evaluation, and simplification.
- Updated vector Hessian APIs so `Function.hessian(...)` returns a single flat
  row-major output vector instead of one output row per variable.
- Updated custom vector Rust derivative helpers so Jacobian, HVP, and Hessian
  helpers operate on full output slices.

### Fixed

- Fixed generation of useless assertions like `assert!(work.len() >= 0);` for
  kernels that do not require workspace.
- Fixed Rust norm helper emission so generated iterator expressions are valid in
  runtime-compiled crates.
- Fixed repeated emission of local `norm2` helpers by moving shared helper
  functions to module scope.
- Fixed custom-function registration so NumPy-based opaque derivative callbacks
  are no longer evaluated on `SX` symbolic objects.
- Fixed generated Rust custom Hessian emission so flat helper calls are used
  directly where possible, with compatibility wrappers retained for internal
  per-entry access.

### Automatic Differentiation

- Added AD support where mathematically appropriate for:
  `atan2`, `hypot`, `asinh`, `acosh`, `atanh`, `cbrt`, `erf`, `erfc`,
  `norm2`, `norm2sq`, `norm_p`, `norm_p_to_p`, `sum`, `prod`, and `mean`.
- Added explicit AD errors for nonsmooth or unsupported cases including:
  `min`, `floor`, `ceil`, `round`, `trunc`, `fract`, `signum`, `norm1`,
  `norm_inf`, vector `max`, and vector `min`.
- Added explicit validation for `norm_p` and `norm_p_to_p` so AD raises when
  `p == 1` or when `p` is not a constant.

### Documentation

- Updated the main `README.md` to document the expanded elementary function
  support, vector norms, `SXVector` slicing, AD behavior, and shared Rust
  helper generation.
- Added and refined documentation for the new codegen and custom-function
  demos, including virtual-environment guidance.
