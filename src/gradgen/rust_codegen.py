"""Compatibility wrapper for Rust code generation."""

import shutil
import subprocess

from ._rust_codegen.codegen import *  # noqa: F401,F403
from ._rust_codegen.codegen import _gradgen_version  # noqa: F401
