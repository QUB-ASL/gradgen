#!/bin/bash
set -euo pipefail

build_api_dox=false

if [ "$#" -gt 1 ]; then
    echo "Usage: $0 [--build-api-dox]" >&2
    exit 1
fi

if [ "$#" -eq 1 ]; then
    if [ "$1" = "--build-api-dox" ]; then
        build_api_dox=true
    else
        echo "Unknown argument: $1" >&2
        echo "Usage: $0 [--build-api-dox]" >&2
        exit 1
    fi
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../.." && pwd)"
tmp_dir="$(mktemp -d)"
local_api_dox_dir="$repo_root/docs/sphinx/build/html"

cleanup() {
    rm -rf "$tmp_dir"
}

trap cleanup EXIT

cd "$repo_root"
git fetch origin gh-pages:gh-pages || :

if [ "$build_api_dox" = true ]; then
    python -m pip install sphinx sphinx-rtd-theme
    python -m pip install .
    python -m sphinx.ext.apidoc -f -o docs/sphinx/source/api src/gradgen
    make -C docs/sphinx html
fi

if git cat-file -e gh-pages:api-dox 2>/dev/null; then
    git archive gh-pages api-dox | tar -x -C "$tmp_dir"
fi

cd "$script_dir"
yarn build

if [ -f "$local_api_dox_dir/index.html" ]; then
    rm -rf build/api-dox
    mkdir -p build/api-dox
    cp -R "$local_api_dox_dir"/. build/api-dox/
elif [ -d "$tmp_dir/api-dox" ]; then
    rm -rf build/api-dox
    cp -R "$tmp_dir/api-dox" build/api-dox
else
    echo "No generated API docs found." >&2
    echo "Run $0 --build-api-dox, or publish API docs to gh-pages first." >&2
    exit 1
fi

touch build/.nojekyll

GIT_USER="${GIT_USER:-QUB-ASL}" \
  CURRENT_BRANCH="${CURRENT_BRANCH:-main}" \
  USE_SSH="${USE_SSH:-true}" \
  yarn deploy --skip-build
