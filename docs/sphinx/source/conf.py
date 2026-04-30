"""Sphinx configuration for the generated Gradgen API docs."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version as pkg_version
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

project = "Gradgen API"
copyright = "2026, Gradgen contributors"
author = "Gradgen contributors"

try:
    release = pkg_version("gradgen")
except PackageNotFoundError:
    release = "dev"

version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_typehints = "description"
napoleon_google_docstring = True
napoleon_numpy_docstring = False
root_doc = "index"
exclude_patterns = ["_build"]
html_theme = "sphinx_rtd_theme"
html_title = f"{project} {release}"
