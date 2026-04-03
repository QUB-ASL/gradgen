# Python Interface Runner

This runner installs the generated `foo_python` wrapper crate into an isolated
temporary Python virtual environment and then imports the module to call:

- `all_functions()`
- `function_info("energy")`
- `workspace_for_function("energy")`
- `energy([1.0, 2.0], [3.0], workspace)`

Run it from this directory with:

```bash
python main.py
```
