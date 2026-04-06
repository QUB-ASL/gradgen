"""Backend configuration for Rust code generation."""

from __future__ import annotations

from dataclasses import dataclass, replace

from .naming import validate_rust_ident
from .validation import (
    validate_backend_mode,
    validate_crate_name,
    validate_scalar_type
)


RustBackendMode = str
RustScalarType = str


@dataclass(frozen=True, slots=True)
class RustBackendConfig:
    """Configuration for generated Rust source and project layout."""

    backend_mode: RustBackendMode = "std"
    scalar_type: RustScalarType = "f64"
    crate_name: str | None = None
    function_name: str | None = None
    emit_metadata_helpers: bool = True
    enable_python_interface: bool = False
    build_python_interface: bool = True
    build_crate: bool = False

    def with_backend_mode(self,
                          backend_mode: RustBackendMode) \
            -> RustBackendConfig:
        """Return a copy with a different Rust backend mode."""
        validate_backend_mode(backend_mode)
        return replace(self, backend_mode=backend_mode)

    def with_scalar_type(self, scalar_type: RustScalarType) \
            -> RustBackendConfig:
        """Return a copy with a different generated Rust scalar type."""
        validate_scalar_type(scalar_type)
        return replace(self, scalar_type=scalar_type)

    def with_crate_name(self, crate_name: str | None) -> RustBackendConfig:
        """Return a copy with a different generated crate name."""
        validate_crate_name(crate_name)
        return replace(self, crate_name=crate_name)

    def with_function_name(self, function_name: str | None) \
            -> RustBackendConfig:
        """Return a copy with a different generated Rust function name."""
        validate_rust_ident(function_name, label="function_name")
        return replace(self, function_name=function_name)

    def with_emit_metadata_helpers(self,
                                   emit_metadata_helpers: bool) \
            -> RustBackendConfig:
        """Return a copy with metadata helper emission enabled or disabled."""
        return replace(self, emit_metadata_helpers=emit_metadata_helpers)

    def with_enable_python_interface(self,
                                     enable_python_interface: bool = True) \
            -> RustBackendConfig:
        """
        Return a copy with optional PyO3-based Python bindings enabled.
        """
        return replace(self, enable_python_interface=enable_python_interface)

    def with_build_python_interface(self,
                                    build_python_interface: bool = True) \
            -> RustBackendConfig:
        """
        Return a copy with Python wrapper compilation enabled or disabled.
        """
        return replace(self, build_python_interface=build_python_interface)

    def with_build_crate(self, build_crate: bool = True) -> RustBackendConfig:
        """
        Return a copy with low-level crate compilation enabled or disabled.
        """
        return replace(self, build_crate=build_crate)
