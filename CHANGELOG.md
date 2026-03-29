# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).


## 0.3.0 - Unreleased

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
- Added support for generating one Rust crate from multiple source functions
  through `CodeGenerationBuilder().for_function(...)`.
- Added runnable demos in `demos/codegen`, `demos/custom_function`, and
  `demos/multi_function`.
- Added staged composition support through `ComposedFunction` with
  `.then(...)`, `.chain(...)`, `.repeat(...)`, and `.finish(...)`.
- Added staged gradient code generation for composed functions, including
  loop-preserving Rust emission for repeated stages.
- Added runtime-seeded `vjp` kernels to the Rust code generation workflow.
- Added dedicated demos in `demos/composed_function` and `demos/vjp`.
- Added `runner` crates for every demo to show how generated Rust crates can be
  consumed from ordinary Rust binaries.
- Added substantial regression and runtime coverage for symbolic math, AD, and
  generated Rust crates.
- Added CI coverage for demo generation, demo runner execution, and Rust
  `cargo clippy` checks on all demo runners.
- Added early Rust-facing name validation for explicit crate and function
  names, including Rust keyword rejection and detection of sanitized-name
  collisions.

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
- Updated `CodeGenerationBuilder` so it can manage one or more source
  `Function`s in the same generated crate while preserving
  `CodeGenerationBuilder(f)` as the single-function shorthand.
- Updated `CodeGenerationBuilder` so it can also accept `ComposedFunction` and
  `ComposedGradientFunction` sources directly in multi-function crates.
- Updated builder-generated Rust naming so multi-function crates automatically
  include the source function name in generated entrypoints to avoid collisions.
- Updated vector Hessian APIs so `Function.hessian(...)` returns a single flat
  row-major output vector instead of one output row per variable.
- Updated custom vector Rust derivative helpers so Jacobian, HVP, and Hessian
  helpers operate on full output slices.
- Updated Jacobian blocks for vector-valued outputs so generated Rust returns a
  single flat row-major output slice instead of one row output per variable.
- Updated all demos so their generated Rust crates target `no_std`, while the
  demo `runner` crates remain ordinary `std` binaries for ease of use.
- Updated demo runners to print all generated `FunctionMetadata` first and to
  format numeric results to four decimal places.
- Refactored the Rust code generation and custom-elementary internals into
  smaller private submodules for improved maintainability.

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
- Fixed multi-function staged code generation so one crate can contain both
  loop-based composed primal and composed gradient kernels.
- Fixed runner examples to allocate workspaces from generated metadata instead
  of relying on hand-written sizes.
- Fixed repository noise from generated Rust crates by ignoring common Cargo
  artifacts such as `Cargo.lock`, `target/`, backup files, and local `.cargo/`
  directories.

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
  support, vector norms, `SXVector` slicing, AD behavior, shared Rust helper
  generation, and the new multi-function `CodeGenerationBuilder` workflow.
- Updated the main `README.md` with Rust naming rules, composed-function
  guidance, and a live GitHub Actions CI badge.
- Added and refined documentation for the new codegen and custom-function
  demos, including virtual-environment guidance.
- Added a dedicated multi-function demo documenting how several source
  functions can share the same generated Rust crate.
- Updated all demo READMEs and runner READMEs to document the `runner` crates
  and the `no_std` generated-library setup.

## 0.2.0 - 28-03-2026

### Changed

- Introducing an entirely new structure that supports automatic differentiation without casadi and generates Rust crates. For the time being, only a few elementary functions and operators are supported
