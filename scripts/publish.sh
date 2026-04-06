#!/bin/sh
set -eu

# This script facilitates releasing a new version of GradGen to PyPI.
# It expects a local virtual environment at ./venv with publishing tools.
current_branch=$(git rev-parse --abbrev-ref HEAD)

echo "[GradGen] Cleaning previous build artifacts"
rm -rf ./build ./dist ./src/gradgen.egg-info
find . -type f -name '*.py[co]' -delete -o -type d -name __pycache__ -delete

echo "[GradGen] Activating virtual environment"
. venv/bin/activate

echo "[GradGen] Installing packaging tools"
python -m pip install --upgrade pip build twine

echo "[GradGen] Building source and wheel distributions"
python -m build

echo "[GradGen] Checking distributions with twine"
python -m twine check dist/*

echo "[GradGen] You are about to publish a new version from branch '$current_branch'."
printf "Are you sure? [y/N] "
read -r response
case "$response" in
    [yY][eE][sS]|[yY])
        echo "[GradGen] Uploading to PyPI now"
        python -m twine upload dist/*
        ;;
    *)
        echo "[GradGen] Upload cancelled"
        ;;
esac

echo "[GradGen] Don't forget to create a tag!"
