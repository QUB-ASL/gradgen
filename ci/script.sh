#!/bin/bash
set -euxo pipefail


regular_test() {
    # Run Python tests
    # ------------------------------------

    # --- create virtual environment
    export PYTHONPATH=.

    # --- install virtualenv
    pip install virtualenv

    # --- create virtualenv
    virtualenv -p python3.8 venv

    # --- activate venv
    source venv/bin/activate

    # --- upgrade pip within venv
    pip install --upgrade pip

    # --- install opengen
    pip install .

    # --- run the tests
    cd gradgen/test
    export PYTHONPATH=.
    python test_gradgen.py
    cargo test --manifest-path codegenz/quadcopter_test/Cargo.toml
    cargo test --manifest-path codegenz/quadcopter_test/casadi_quadcopter_test/Cargo.toml
}

main() {
    regular_test
}

main
