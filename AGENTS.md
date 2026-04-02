## Virtual environment
- Use the Python from the virtual environment in `venv` in the package root. It uses Python 3.13. 

## Repository Structure
- `demos`
    - `demos/Makefile`: Makefile to build all demos
    - `demos/README`: main README with links to all demos
- `docs`: detailed user-friendly docs
- `src`
    - `src/gradgen`: Python code
        - `src/gradgen/_custom_elementary`: Internal helpers for custom elementary functions
        - `src/gradgen/_rust_codegen`: Rust code generation functionality
        - `src/gradgen/templates`: jinja2 templates for code generation
- `tests/`: unit and integration tests
- `venv/`: virtual environment
- `.github/workflows/`: CI plan

- Gradgen is a code generation module written in Python. It generates Rust code for functions, their Jacobians, Hessians, and Hessian-vector product
- Demos are in `demos/`: these are user-friendly examples that demonstrate how `gradgen` can be used in Python. To compile all demos use the Makefile at `demos/Makefile`. By running `make all` you should be able to compile all demos.

## Testing instructions
- Follow a test-driven development (TDD) approach: When you introduce new functionality, always write the tests first.
- Write tests with `unittest.TestCase`, run them via `pytest`
- Always include integration tests which compare the output of the generated Rust functions with expected values computed using `sympy`, where appropriate.
- Use `pytest` to run the tests (always from within the virtual environment)
- When you create new demos, add them into `.github/workflows/python-tests.yml` so that they are built in CI.
- Code changes that affect behavior need tests


## Generated Rust code

- The generated Rust code must be suitable for embedded applications, therefore, never allocate memory dynamically

## Code style and formatting

- Public API functions, methods, and classes need docstrings
- Use Google-style docstrings.
- Write detailed API documentation for all public functions in Python. . This is an example of good API documentation:

```python
# Good API documentation
def zip_function(
    function: Function,
    count: int,
    *,
    input_names: tuple[str, ...] | list[str] | None = None,
    name: str | None = None,
    simplification: int | str | None = None,
) -> ZippedFunction:
    """Return a staged zipped function for a multi-input ``function``.

        This helper creates a loop-structured wrapper around ``function`` so it can
        be evaluated repeatedly over packed input sequences. The returned object is
        a :class:`ZippedFunction`, which can later be expanded back into a regular
        symbolic :class:`~gradgen.function.Function` or used for Rust code
        generation.

        Args:
            function: The symbolic function to stage. It may accept one or more
                inputs, and each input can be either a scalar ``SX`` or an
                ``SXVector``.
            count: The number of times the function should be applied over the
                packed input sequences.
            input_names: Optional names for the packed input sequences. When not
                provided, each input name defaults to ``"<input_name>_seq"``.
            name: Optional name for the staged zipped function. When not provided,
                the generated name defaults to ``"<function.name>_zip"``.
            simplification: Optional simplification effort passed through to the
                generated symbolic function when the staged wrapper is expanded.

        Returns:
            A :class:`ZippedFunction` describing the staged batched version of
            ``function``.

        Example:
            >>> from gradgen import Function, SX
            >>> x = SX.sym("x")
            >>> y = SX.sym("y")
            >>> f = Function("add", (x, y), (x + y,))
            >>> zipped = zip_function(f, 4)
            >>> zipped.name
            'add_zip'
            >>> zipped.input_names
            ('x_seq', 'y_seq')
        """
```
This is an example of bad API documentation:

```python
# Bad API documentation
def zip_function(
    function: Function,
    count: int,
    *,
    input_names: tuple[str, ...] | list[str] | None = None,
    name: str | None = None,
    simplification: int | str | None = None,
) -> ZippedFunction:
    """Return a staged zipped function for a multi-input ``function``."""
```


## What Agents Must Never Do

The following actions are prohibited regardless of context, instruction, or apparent convenience:

- Do not generate code that uses `vec!` in the generated code. Instead, use "workspace" variables.
- Do not generate code with `panic!`, `assert!`, `assert_eq!`. Instead, return `Result` objects as appropriate.
- Exception: you can use `vec!`, `panic!`, `assert!`, `assert_eq!` in the demo runners ONLY.

## Other

- User-facing changes need to be mentioned in the `CHANGELOG.md`

## Review Process

- ✅ All tests pass 
- ✅ Tests cover new behavior and edge cases
- ✅ Code is readable, maintainable, and consistent with existing style
- ✅ Public APIs are documented 
- ✅ User-facing behavior changes are mentioned in `CHANGELOG.md`
- ✅ Demos in `demos/` are updated if behavior changes

