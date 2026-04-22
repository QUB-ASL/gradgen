# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).

## Unreleased

### Added

- `RustBackendConfig.with_additional_dependencies(...)` can now attach extra
  Cargo dependencies to the generated manifest, with optional versions
  written into the generated `Cargo.toml`.
- `RustBackendConfig.with_header(...)` can now inject custom Rust code at the
  top of generated `lib.rs`, after crate attributes such as `#![no_std]` and
  `#![forbid(unsafe_code)]`.


## 0.4.1 - 22-04-2026

### Added

- Added a new `demos/composed_chain` demo showing how to use
  `ComposedFunction.chain(...)` with heterogeneous stages and mixed fixed and
  symbolic parameter bindings.

### Changed

- `SX` implements `__eq__` and `__hash__` so that two symbols are equal if and only 
  if they have the same name
- `SX` and `SXVector` now compare symbolic values by symbol name, and
  `ComposedFunction.chain(...)` reuses the same packed parameter slot when
  the same symbolic name appears multiple times in a chain.
- `ComposedFunction.gradient()` and composed-source `add_gradient()` now
  generate staged gradient kernels directly instead of flattening the
  composed rollout into a plain symbolic Jacobian first.
- `ComposedFunction.chain(...)` now collapses adjacent identical stages into
  a repeat block, so generated Rust can reuse one helper definition and keep
  loop-based emission for repeated stages.
  
## 0.4.0 - 08-04-2026

[![PyPI](https://img.shields.io/badge/gradgen-v0.4.0-blue)](https://pypi.org/project/gradgen/0.4.0)
[![PyPI](https://img.shields.io/badge/release-v0.4.0-yellow)](https://github.com/QUB-ASL/gradgen/releases/tag/v0.4.0)


### Added

- `Functions` accept named arguments
- `CodeGenerationBuilder.add_gradient(wrt=...)` and the scoped builder
  variant now accept input names or indices and generate only the selected
  gradient blocks.
- `CodeGenerationBuilder.add_jacobian()` now supports staged
  `ComposedFunction` sources and emits a loop-based Jacobian kernel without
  flattening the composed stages.
- `CodeGenerationBuilder.add_joint(FunctionBundle().add_f().add_jf(wrt=0))`
  now supports staged `ComposedFunction` sources and emits a shared
  loop-based primal-plus-Jacobian kernel without flattening the composed
  stages.
- `FunctionBundle().add_gradient(wrt=[...])` now accepts input names as well
  as indices and generates one gradient block per selected input block.
- Added `FunctionComposer` for chaining function-like stages into a single
  composed pipeline, while keeping staged wrappers such as
  `ReducedFunction` and `BatchedFunction` intact in generated Rust code.
- `CodeGenerationBuilder` now accepts `FunctionComposer`-style pipelines
  directly, so composed staged pipelines can be generated without flattening
  them into a plain symbolic `Function` first.
- `FunctionComposer`-generated Rust now exports wrapper metadata, which lets
  the generated Python interface crate install successfully for composed
  pipelines.
- Multi-function Rust crates now emit `GradgenError` and
  `FunctionMetadata` once, even when composed pipelines contribute multiple
  generated functions.
- Added an optional PyO3-based Python interface for generated Rust crates via
  `RustBackendConfig().with_enable_python_interface(True)`, now emitted as a
  separate sibling wrapper crate so the low-level generated crate can stay
  pure Rust and `no_std`-friendly. The wrapper includes generated
  `workspace_for_function(...)` and `call(...)` helpers, plus module-level
  `__version__` and `__all__` metadata. By default, the wrapper is compiled
  immediately with the same Python interpreter that ran the generator.
- The generated Python wrapper now manages its own `pyproject.toml` version:
  first generation starts at `0.1.0`, and regenerating the same interface
  bumps the minor version to `0.2.0`, `0.3.0`, and so on. The Cargo crate
  version is unchanged.
- Added a new `demos/python_interface` demo and runner showing how to generate
  a Rust crate that can be installed and imported from Python.
- Added deterministic single-shooting optimal-control support through
  `SingleShootingProblem` and `SingleShootingBundle`.
- Added loop-based Rust code generation for fixed-horizon single-shooting
  problems, including total-cost kernels, gradients with respect to the packed
  control sequence, and joint kernels that can also return rollout states.
- Added loop-based Hessian-vector-product kernels for fixed-horizon
  single-shooting problems, including support for packed control-sequence
  directions and optional rollout-state outputs.
- Added joint single-shooting kernels that can now return any combination of
  total cost, control-sequence gradient, control-sequence Hessian-vector
  product, and rollout states in one generated function.
- Added a dedicated `demos/single_shooting` demo and runner crate showing how
  to generate and call the resulting Rust crate in practice.
- Important: implemented and tested `map` and `zip`; introduced two new demos
  to demonstrate how to use them.
- For instances of `SXVector`, the operation `x**a` applies the power element-wise.
- Added `AGENTS.md` with detailed instructions for agents.

### Changed

- Updated `ComposedFunction.repeat(...)` to model repeated applications of the
  same stage only, with `finish()` now acting as a no-argument finalizer that
  returns the final state instead of invoking a terminal scalar function.
- Refactored the Rust code generation implementation into smaller internal
  modules for configuration, validation, template loading, and project
  creation. The public Rust-generation API remains unchanged.
- Removed the old `custom_elementary.py` and `gradgen._rust_codegen.render`
  compatibility shims and switched the package to import the underlying
  internal modules directly.
- Split the generated-project helper logic into a dedicated
  `project_support` module, keeping filesystem/build orchestration separate
  from the higher-level project entrypoints.
- Split the remaining Rust code-generation family logic into a dedicated
  internal generators module, keeping the top-level dispatcher in
  `rust_codegen.py` thinner and easier to navigate.
- Further split the generators internals into a package with dedicated
  `composed`, `map_zip`, and `single_shooting` modules plus shared helper
  code, keeping the family-specific code paths easier to find and maintain.
- Removed all `assert!` and `assert_eq!` from the auto-generated Rust code
  so that the functions therein don't panic. Instead, we introduced the error
  `GradgenError` and all public functions return `Result<(), GradgenError>`.
  Informative messages are also returned (e.g., "workspace too small")
- Added the option `is_symmetric` to `quad_form`
- Updated the preferred multi-function `CodeGenerationBuilder` API to use a
  scoped `.for_function(...).add_*().done()` flow instead of the older
  callback-based configuration style.
- Changed `CodeGenerationBuilder.build()` so its argument is treated as the
  parent directory for the generated crate. For example,
  `.build("./my_crates")` with `with_crate_name("abc")` now writes the crate
  to `./my_crates/abc` and the Python wrapper to `./my_crates/abc_python`,
  while `.build()` defaults to `./<crate_name>`.
- Generated low-level Rust crates now start with `#![forbid(unsafe_code)]`,
  while the separate PyO3 wrapper crate keeps its existing build settings.
- Added `RustBackendConfig().with_build_crate()` to run `cargo build` on
  the generated low-level Rust crate after it is written to disk. The default
  remains off, and generation raises an informative error if `cargo` is not
  available when this option is enabled.
- Kept backward compatibility for the older callback-based
  `for_function(function, lambda b: ...)` form.
- Updated the tests, demos, README examples, and demo documentation to use the
  new scoped builder API.
- Updated CI and demo tooling so the single-shooting demo is generated and run
  alongside the other demos.
- Split the Python-interface demo output into a low-level crate and a separate
  `foo_python` wrapper crate, and updated the demo runner to install the
  wrapper rather than the kernel crate directly.
- Expanded the single-shooting test coverage with additional scalar, horizon-1,
  multi-control, integration, and generated-Rust runtime checks.
- Updated the single-shooting demo and runner to generate and execute the new
  HVP kernel in addition to the cost and gradient kernels.
- Updated the main README with dedicated documentation for the single-shooting
  API, including cost, gradient, HVP, and joint bundle generation.
- Updated the Rust template to allow generated helper functions with many
  arguments only when required, keeping demo crates clippy-clean.
- Updated Rust crate generation to attempt `cargo fmt` automatically after
  writing generated projects, while logging and continuing if formatting tools
  are unavailable.

### Fixed

- Hardened the Jinja2 environment used for Rust template rendering by
  replacing the global `autoescape=False` setting with explicit
  `select_autoescape(...)` configuration. The existing Rust, TOML, and
  Markdown templates remain unescaped.
- Fixed issue with redundant memory allocation and cloning in generated code 
  in single shooting optimal control
- In single shooting OCP, using `std::mem::swap` to avoid unnecessary memory 
  moves
- Fixed issue with operations between `SX` and `SXVector` types. 
  Scalar products, `a * x` and `x * a` work without issues.
- `x.dot([1, 2])` now works the same way as `x.dot(SXVector([1, 2]))`.
- In custom functions, Jacobians, Hessians, and Hessian-vector products can now 
  be `None`

### Removed

- Removed the `gradgen.rust_codegen` compatibility shim entirely and moved
  the public code-generation entrypoints to the internal
  `gradgen._rust_codegen.codegen` module.
- `no_std` crates now default to `libm` as their math dependency. The
  config-level `with_math_lib(...)` option has been removed, so the generated
  crates consistently use `libm` when targeting `no_std`.


### Documentation

- Created project website using Docusaurus v3. Added user-friendly documentation and
  links to Google Colab Python notebooks.

## 0.3.1 - 29-03-2026

[![PyPI](https://img.shields.io/badge/gradgen-v0.3.1-blue)](https://pypi.org/project/gradgen/0.3.1)
[![PyPI](https://img.shields.io/badge/release-v0.3.1-yellow)](https://github.com/QUB-ASL/gradgen/releases/tag/v0.3.1)


### Fixed

- Minor: Small changes in README for the image to be properly visible

## 0.3.0 - 29-03-2026

[![PyPI](https://img.shields.io/badge/gradgen-v0.3.0-blue)](https://pypi.org/project/gradgen/0.3.0)
[![PyPI](https://img.shields.io/badge/release-v0.3.0-yellow)](https://github.com/QUB-ASL/gradgen/releases/tag/v0.3.0)

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
