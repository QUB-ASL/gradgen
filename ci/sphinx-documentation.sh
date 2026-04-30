#!/bin/bash
set -euxo pipefail

repo_root="${GITHUB_WORKSPACE:-$(pwd)}"
default_branch="main"
magic_docs_keyword="[docit]"

cd "$repo_root"

current_branch="$(git rev-parse --abbrev-ref HEAD)"
commit_message="$(git log -1 --pretty=format:%s)"
commit_hash="$(git rev-parse --short HEAD)"
api_build_dir="$(mktemp -d)"
pages_dir="$(mktemp -d)"

cleanup() {
    rm -rf "$api_build_dir" "$pages_dir"
    rm -rf docs/sphinx/source/api
}

trap cleanup EXIT

echo "CURRENT BRANCH: $current_branch"
echo "COMMIT MESSAGE: $commit_message"
echo "COMMIT HASH   : $commit_hash"

if [[ "$commit_message" != *"$magic_docs_keyword"* ]] && \
    [ "$current_branch" != "$default_branch" ]; then
    echo "Skipping API docs build outside $default_branch without $magic_docs_keyword."
    exit 0
fi

python -m pip install --upgrade pip
python -m pip install sphinx sphinx-rtd-theme
python -m pip install .

python -m sphinx.ext.apidoc -f -o docs/sphinx/source/api src/gradgen
make -C docs/sphinx html
cp -R docs/sphinx/build/html/. "$api_build_dir"

git config --global user.name "github-actions"
git config --global user.email "actions@github.com"

git fetch origin gh-pages
git worktree add "$pages_dir" origin/gh-pages

rm -rf "$pages_dir/api-dox"
mkdir -p "$pages_dir/api-dox"
cp -R "$api_build_dir"/. "$pages_dir/api-dox/"
touch "$pages_dir/.nojekyll"

pushd "$pages_dir"
git add api-dox .nojekyll

if git diff --cached --quiet; then
    echo "No API documentation changes to publish."
    popd
    exit 0
fi

git commit -m "documentation for $commit_hash"
git push origin HEAD:gh-pages
popd
