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
    python gradgen/test/test_gradgen.py
    cd gradgen/test/codegenz/quadcopter_test/casadi_quadcopter_test
    cargo test
    cd ..
    cargo test
}

main() {
    regular_test
}

main
