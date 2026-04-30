#!/bin/bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../.." && pwd)"
source_dir="$repo_root/docs/sphinx/build/html"
target_dir="$script_dir/static/api-dox"
legacy_target_dir="$script_dir/static/api-ref"

rm -rf "$target_dir"
rm -rf "$legacy_target_dir"
mkdir -p "$target_dir"

if [ -f "$source_dir/index.html" ]; then
    cp -R "$source_dir"/. "$target_dir/"
    exit 0
fi

cat > "$target_dir/index.html" <<'EOF'
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Gradgen API Docs</title>
    <style>
      :root {
        color-scheme: light dark;
        font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI",
          sans-serif;
      }
      body {
        margin: 0;
        min-height: 100vh;
        display: grid;
        place-items: center;
        background: #f6f8fb;
        color: #1f2937;
      }
      main {
        width: min(42rem, calc(100vw - 2rem));
        padding: 2rem;
        border-radius: 12px;
        background: white;
        box-shadow: 0 18px 40px rgba(15, 23, 42, 0.08);
      }
      h1 {
        margin-top: 0;
      }
      code {
        padding: 0.15rem 0.35rem;
        border-radius: 6px;
        background: #eef2ff;
      }
      pre {
        overflow-x: auto;
        padding: 1rem;
        border-radius: 10px;
        background: #0f172a;
        color: #e2e8f0;
      }
    </style>
  </head>
  <body>
    <main>
      <h1>API docs are not built yet</h1>
      <p>
        The Docusaurus dev server can serve the Sphinx API reference at this
        path, but the Sphinx HTML output does not exist yet in
        <code>docs/sphinx/build/html</code>.
      </p>
      <p>Build the API docs first, then restart <code>yarn start</code>:</p>
      <pre><code>python -m pip install sphinx sphinx-rtd-theme
python -m pip install .
python -m sphinx.ext.apidoc -f -o docs/sphinx/source/api src/gradgen
make -C docs/sphinx html</code></pre>
    </main>
  </body>
</html>
EOF
