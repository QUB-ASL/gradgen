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
    export PYTHONPATH=.
    python main.py
    
}

main() {
    regular_test
}

main
