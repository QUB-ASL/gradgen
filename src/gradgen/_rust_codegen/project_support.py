"""Shared helpers for generated Rust project creation."""

from __future__ import annotations

from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version as package_version
import json
import logging
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tomllib

from ..function import Function
from .config import RustBackendConfig
from .models import (
    RustCodegenResult,
    RustPythonInterfaceProjectResult,
)
from .naming import sanitize_ident
from .templates import _get_template, _render_custom_rust_header
from .validation import validate_unique_rust_names


_LOGGER = logging.getLogger(__name__)


def _try_run_cargo_fmt(project_dir: Path) -> None:
    """Run ``cargo fmt`` for a generated crate when available."""
    if shutil.which("cargo") is None:
        _LOGGER.info(
            "Skipping cargo fmt for %s because cargo is not installed.",
            project_dir)
        return
    try:
        subprocess.run(
            ["cargo", "fmt"],
            cwd=project_dir,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        _LOGGER.info(
            "Skipping cargo fmt for %s as cargo is not installed.",
            project_dir)
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        stdout = (exc.stdout or "").strip()
        details = stderr or stdout
        _LOGGER.warning(
            "Skipping cargo fmt for %s as it could not be run successfully%s",
            project_dir,
            f": {details}" if details else ".",
        )


def _run_python_interface_build(project_dir: Path) -> None:
    """
    Install a generated Python wrapper crate into the 
    active Python environment.
    """
    if shutil.which("cargo") is None:
        raise RuntimeError(
            "cargo is required to compile the generated Python interface")

    env = os.environ.copy()
    env["PYO3_PYTHON"] = sys.executable
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-e", str(project_dir)],
            cwd=project_dir,
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "cargo is required to compile the generated Python interface") \
                from exc
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        stdout = (exc.stdout or "").strip()
        details = stderr or stdout
        raise RuntimeError(
            "failed to install the generated Python interface"
            + (f": {details}" if details else "")
        ) from exc


def _next_python_interface_version(wrapper_project_dir: Path) -> str:
    """Return the next Python wrapper version for ``wrapper_project_dir``."""
    pyproject = wrapper_project_dir / "pyproject.toml"
    if not pyproject.exists():
        return "0.1.0"

    try:
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise RuntimeError(
            f"failed to read Python interface version from {pyproject}"
        ) from exc

    project_section = data.get("project")
    if not isinstance(project_section, dict):
        raise RuntimeError(
            f"existing Python interface metadata at {pyproject} doesn't "
            "define [project]"
        )

    version = project_section.get("version")
    if not isinstance(version, str):
        raise RuntimeError(
            f"existing Python interface metadata at {pyproject} "
            "does not define a string version"
        )

    match = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)", version)
    if match is None:
        raise RuntimeError(
            f"existing Python interface version {version!r} in {pyproject} "
            "is not a semantic version"
        )

    major, minor, _patch = (int(group) for group in match.groups())
    return f"{major}.{minor + 1}.0"


def _run_cargo_build(project_dir: Path) -> None:
    """Compile a generated Rust crate with Cargo."""
    if shutil.which("cargo") is None:
        raise RuntimeError(
            "cargo is required to build the generated Rust crate")
    try:
        subprocess.run(
            ["cargo", "build"],
            cwd=project_dir,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "cargo is required to build the generated Rust crate") from exc
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        stdout = (exc.stdout or "").strip()
        details = stderr or stdout
        raise RuntimeError(
            "failed to build the generated Rust crate"
            + (f": {details}" if details else "")
        ) from exc


def _render_cargo_dependency_lines(
    math_library: str | None,
    math_library_version: str | None,
    additional_dependencies: tuple[tuple[str, str | None], ...],
) -> tuple[str, ...]:
    """Return rendered Cargo dependency lines for a generated crate."""
    lines: list[str] = []
    seen: set[str] = set()

    def add_dependency(name: str, version: str | None) -> None:
        if name in seen:
            raise ValueError(f"duplicate Cargo dependency {name!r}")
        seen.add(name)
        rendered_version = version if version is not None else "*"
        lines.append(f'{name} = {json.dumps(rendered_version)}')

    if math_library is not None:
        add_dependency(math_library, math_library_version)
    for dependency_name, dependency_version in additional_dependencies:
        add_dependency(dependency_name, dependency_version)
    return tuple(lines)


def _derive_python_function_name(function_name: str,
                                 crate_name: str | None) -> str:
    """Derive the public Python name from a generated Rust symbol name."""
    candidate = function_name
    if crate_name:
        crate_prefix = sanitize_ident(crate_name)
        prefix = f"{crate_prefix}_"
        if candidate.startswith(prefix):
            candidate = candidate[len(prefix):]
    if candidate.endswith("_f"):
        candidate = candidate[:-2]
    return candidate or function_name


def _create_python_interface_project(
    *,
    low_level_project_dir: Path,
    low_level_crate_name: str,
    codegens: tuple[RustCodegenResult, ...],
    build_python_interface: bool,
) -> RustPythonInterfaceProjectResult:
    """Create a sibling PyO3 wrapper crate for generated Rust kernels."""
    validate_unique_rust_names(
        [(codegen.function_name, codegen.python_name) for codegen in codegens],
        label="generated Python function",
    )
    wrapper_project_dir = low_level_project_dir.parent \
        / f"{low_level_crate_name}_python"
    wrapper_crate_name = wrapper_project_dir.name
    module_name = low_level_crate_name

    wrapper_project_dir.mkdir(parents=True, exist_ok=True)
    src_dir = wrapper_project_dir / "src"
    src_dir.mkdir(parents=True, exist_ok=True)

    cargo_toml = wrapper_project_dir / "Cargo.toml"
    pyproject = wrapper_project_dir / "pyproject.toml"
    readme = wrapper_project_dir / "README.md"
    lib_rs = src_dir / "lib.rs"
    project_version = _next_python_interface_version(wrapper_project_dir)

    dependency_path = Path("..") / low_level_project_dir.name
    cargo_toml.write_text(
        _get_template("python_interface_Cargo.toml.j2").render(
            crate_name=wrapper_crate_name,
            module_name=module_name,
            low_level_crate_name=low_level_crate_name,
            low_level_crate_path=dependency_path.as_posix(),
        ),
        encoding="utf-8",
    )
    pyproject.write_text(
        _get_template("python_interface_pyproject.toml.j2").render(
            crate_name=wrapper_crate_name,
            module_name=module_name,
            low_level_crate_name=low_level_crate_name,
            project_version=project_version,
        ),
        encoding="utf-8",
    )
    readme.write_text(
        _get_template("python_interface_README.md.j2").render(
            crate_name=wrapper_crate_name,
            module_name=module_name,
            low_level_crate_name=low_level_crate_name,
            low_level_project_dir=low_level_project_dir.name,
            codegens=codegens,
        ),
        encoding="utf-8",
    )
    lib_rs.write_text(
        _render_python_interface_source(
            codegens,
            low_level_crate_name=low_level_crate_name,
            module_name=module_name,
            project_version=project_version,
        )
        + "\n",
        encoding="utf-8",
    )

    _try_run_cargo_fmt(wrapper_project_dir)
    if build_python_interface:
        _run_python_interface_build(wrapper_project_dir)

    return RustPythonInterfaceProjectResult(
        project_dir=wrapper_project_dir,
        cargo_toml=cargo_toml,
        pyproject=pyproject,
        readme=readme,
        lib_rs=lib_rs,
        module_name=module_name,
        low_level_crate_name=low_level_crate_name,
    )


def _render_python_interface_source(
    codegens: tuple[RustCodegenResult, ...],
    low_level_crate_name: str,
    module_name: str,
    project_version: str,
) -> str:
    """Render the PyO3 Python interface for one or more generated functions."""
    return _get_template("python_interface.rs.j2").render(
        codegens=codegens,
        module_name=module_name,
        low_level_crate_name=low_level_crate_name,
        project_version=project_version,
        scalar_type=codegens[0].scalar_type,
    ).rstrip()


def _private_helper_section_key(section: str) -> str | None:
    """Return a stable deduplication key for a private helper section."""
    match = re.search(r"(?m)^fn\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", section)
    if match is None:
        return None
    function_name = match.group(1)
    if function_name.endswith("_meta"):
        return None
    if re.search(r"(?m)^pub\s+fn\s+", section) is not None:
        return None
    return section


def _generated_module_section_key(section: str) -> str | None:
    """Return a stable key for a shared generated-module section."""
    stripped = section.lstrip()
    if stripped.startswith("#!["):
        return "module_header"
    if "pub enum GradgenError" in section:
        return "gradgen_error"
    if "pub struct FunctionMetadata" in section:
        return "function_metadata"
    return None


def _split_module_sections(source: str) -> list[str]:
    """Split generated Rust source into normalized top-level sections."""
    if source.startswith("#![no_std]\n\n"):
        source = source[len("#![no_std]\n\n"):]
    elif source.startswith("#![no_std]\n"):
        source = source[len("#![no_std]\n"):]
    return [part.rstrip() for part in source.split("\n\n") if part.strip()]


def _normalize_header_sections(
    sections: list[str],
    header_sections: tuple[str, ...],
    *,
    header_emitted: bool,
) -> tuple[list[str], bool]:
    """Remove or keep the configured custom header at the front."""
    if not sections or not header_sections:
        return sections, header_emitted

    header_start = 0
    if _generated_module_section_key(sections[0]) == "module_header":
        header_start = 1
    header_end = header_start + len(header_sections)
    if tuple(sections[header_start:header_end]) != header_sections:
        return sections, header_emitted
    if header_emitted:
        return (
            sections[:header_start] + sections[header_end:],
            True,
        )
    return sections, True


def _should_include_module_section(
    section: str,
    *,
    seen_private_helpers: set[str],
    seen_generated_module_sections: set[str],
) -> bool:
    """Return whether a multi-function module section should be emitted."""
    module_key = _generated_module_section_key(section)
    if module_key is not None:
        if module_key in seen_generated_module_sections:
            return False
        seen_generated_module_sections.add(module_key)

    helper_key = _private_helper_section_key(section)
    if helper_key is not None:
        if helper_key in seen_private_helpers:
            return False
        seen_private_helpers.add(helper_key)

    return True


def _render_multi_function_lib(
    codegens: tuple[RustCodegenResult, ...],
    config: RustBackendConfig,
) -> str:
    """Render a crate source file containing many generated functions."""
    sections: list[str] = []
    seen_private_helpers: set[str] = set()
    seen_generated_module_sections: set[str] = set()
    rendered_header = _render_custom_rust_header(
        config.header,
        backend_mode=config.backend_mode,
        scalar_type=config.scalar_type,
        math_library="libm" if config.backend_mode == "no_std" else None,
        emit_metadata_helpers=config.emit_metadata_helpers,
    )
    header_sections = tuple(
        _split_module_sections(rendered_header)
        if rendered_header else ()
    )
    header_emitted = False
    if config.backend_mode == "no_std":
        sections.append("#![no_std]")

    for codegen in codegens:
        codegen_sections = _split_module_sections(codegen.source)
        codegen_sections, header_emitted = _normalize_header_sections(
            codegen_sections,
            header_sections,
            header_emitted=header_emitted,
        )
        for section in codegen_sections:
            if _should_include_module_section(
                section,
                seen_private_helpers=seen_private_helpers,
                seen_generated_module_sections=seen_generated_module_sections,
            ):
                sections.append(section)

    rendered = "\n\n".join(section for section in sections if section)
    return rendered if rendered.endswith("\n") else f"{rendered}\n"


def _render_metadata_json(crate_name: str,
                          codegens: tuple[RustCodegenResult, ...]) -> str:
    """Render the crate metadata JSON file."""
    payload = {
        "crate_name": crate_name,
        "created_at": _metadata_created_at(),
        "gradgen_version": _gradgen_version(),
        "functions": [
            {
                "function_name": codegen.function_name,
                "workspace_size": codegen.workspace_size,
                "input_names": list(codegen.input_names),
                "input_sizes": list(codegen.input_sizes),
                "output_names": list(codegen.output_names),
                "output_sizes": list(codegen.output_sizes),
            }
            for codegen in codegens
        ],
    }
    return json.dumps(payload, indent=2) + "\n"


def _maybe_simplify_derivative_function(
        function: Function,
        simplify_derivatives: int | str | None):
    """
    Return ``function`` simplified when a simplification level is requested.
    """
    if simplify_derivatives is None:
        return function
    return function.simplify(
        max_effort=simplify_derivatives,
        name=function.name)


def _metadata_created_at() -> str:
    """Return the UTC timestamp used in generated metadata files."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _gradgen_version() -> str:
    """Return the installed gradgen package version when available."""
    try:
        return package_version("gradgen")
    except PackageNotFoundError:
        pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
        match = re.search(r'^version = "([^"]+)"$',
                          pyproject.read_text(encoding="utf-8"),
                          re.MULTILINE)
        if match is None:
            raise RuntimeError(
                "could not determine gradgen version for metadata.json")
        return match.group(1)
