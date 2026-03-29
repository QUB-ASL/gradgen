"""Builder API for creating multi-kernel Rust crates."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Protocol

from ..composed_function import ComposedFunction, ComposedGradientFunction
from ..function import Function
from .naming import sanitize_ident


BuilderSource = Function | ComposedFunction | ComposedGradientFunction


class RustBuilderConfigLike(Protocol):
    """Protocol describing the config surface used by the builder."""

    crate_name: str | None
    function_name: str | None

    def with_crate_name(self, crate_name: str | None) -> RustBuilderConfigLike:
        """Return a copy with an updated crate name."""


@dataclass(frozen=True, slots=True)
class _BuilderRequest:
    """A requested generated kernel kind."""

    kind: str
    components: tuple[str, ...] = ()
    bundle: FunctionBundle | None = None


@dataclass(frozen=True, slots=True)
class _BundleItem:
    """One artifact entry inside a ``FunctionBundle``."""

    kind: str
    wrt_indices: tuple[int, ...] | None = None


@dataclass(frozen=True, slots=True)
class FunctionBundle:
    """Fluent description of artifacts to compute together in one joint kernel."""

    items: tuple[_BundleItem, ...] = ()

    def add_f(self) -> FunctionBundle:
        """Include the primal outputs."""
        return self._add_item("f")

    def add_gradient(self, *, wrt: int | list[int] | tuple[int, ...]) -> FunctionBundle:
        """Include gradient blocks for one or more input indices."""
        return self._add_item("grad", wrt)

    def add_jf(self, *, wrt: int | list[int] | tuple[int, ...]) -> FunctionBundle:
        """Include Jacobian blocks for one or more input indices."""
        return self._add_item("jf", wrt)

    def add_hessian(self, *, wrt: int | list[int] | tuple[int, ...]) -> FunctionBundle:
        """Include Hessian blocks for one or more input indices."""
        return self._add_item("hessian", wrt)

    def add_hvp(self, *, wrt: int | list[int] | tuple[int, ...]) -> FunctionBundle:
        """Include Hessian-vector-product blocks for one or more input indices."""
        return self._add_item("hvp", wrt)

    def _add_item(
        self,
        kind: str,
        wrt: int | list[int] | tuple[int, ...] | None = None,
    ) -> FunctionBundle:
        wrt_indices = None if wrt is None else _normalize_wrt_indices(wrt)
        candidate = _BundleItem(kind, wrt_indices)
        if any(item == candidate for item in self.items):
            return self
        return replace(self, items=(*self.items, candidate))


@dataclass(frozen=True, slots=True)
class _BuilderFunctionSpec:
    """One source function plus the kernels requested for it."""

    function: BuilderSource
    requests: tuple[_BuilderRequest, ...] = ()
    simplification: int | str | None = None


@dataclass(frozen=True, slots=True)
class CodeGenerationBuilder:
    """Fluent builder for a generated Rust crate with one or more functions."""

    function: BuilderSource | None = None
    config: RustBuilderConfigLike | None = None
    requests: tuple[_BuilderRequest, ...] = ()
    simplification: int | str | None = None
    functions: tuple[_BuilderFunctionSpec, ...] = ()

    def with_backend_config(self, config: RustBuilderConfigLike) -> CodeGenerationBuilder:
        """Return a copy using ``config`` for generated Rust code."""
        return replace(self, config=config)

    def with_simplification(self, max_effort: int | str | None) -> CodeGenerationBuilder:
        """Return a copy applying ``max_effort`` simplification to generated kernels."""
        return replace(self, simplification=max_effort)

    def add_primal(self) -> CodeGenerationBuilder:
        """Include the primal function in the generated crate."""
        return self._add_request("primal")

    def add_gradient(self) -> CodeGenerationBuilder:
        """Include gradient kernels for scalar-output functions."""
        return self._add_request("gradient")

    def add_jacobian(self) -> CodeGenerationBuilder:
        """Include Jacobian kernels for all input blocks."""
        return self._add_request("jacobian")

    def add_vjp(self) -> CodeGenerationBuilder:
        """Include runtime-seeded vector-Jacobian-product kernels for input blocks."""
        return self._add_request("vjp")

    def add_joint(self, bundle: FunctionBundle) -> CodeGenerationBuilder:
        """Include kernels that compute bundled artifacts together."""
        return self._add_request("joint", bundle=bundle)

    def add_hessian(self) -> CodeGenerationBuilder:
        """Include Hessian kernels for scalar-output functions."""
        return self._add_request("hessian")

    def add_hvp(self) -> CodeGenerationBuilder:
        """Include Hessian-vector product kernels for scalar-output functions."""
        return self._add_request("hvp")

    def for_function(
        self,
        function: BuilderSource,
        configure: Callable[[CodeGenerationBuilder], CodeGenerationBuilder] | None = None,
    ) -> CodeGenerationBuilder:
        """Add ``function`` to the crate, optionally configuring its kernels."""
        scoped_builder = CodeGenerationBuilder(function=function, config=self.config)
        configured_builder = configure(scoped_builder) if configure is not None else scoped_builder
        if configured_builder.function is None:
            raise ValueError("configured function builder must target a function")
        entry = _BuilderFunctionSpec(
            function=configured_builder.function,
            requests=configured_builder.requests,
            simplification=configured_builder.simplification,
        )
        return replace(self, functions=(*self.functions, entry))

    def build(self, path: str | Path):
        """Generate a single Rust crate containing all requested kernels."""
        from ..rust_codegen import RustBackendConfig, create_multi_function_rust_project

        resolved_config = self.config or RustBackendConfig()
        if resolved_config.crate_name is None:
            resolved_config = resolved_config.with_crate_name(
                sanitize_ident(Path(path).expanduser().resolve().name)
            )
        functions = resolve_builder_function_specs(
            self._resolved_function_specs(),
            resolved_config,
        )
        return create_multi_function_rust_project(
            functions,
            path,
            config=resolved_config,
        )

    def _add_request(
        self,
        kind: str,
        *,
        components: tuple[str, ...] = (),
        bundle: FunctionBundle | None = None,
    ) -> CodeGenerationBuilder:
        if self.function is None:
            raise ValueError(
                "no source function is selected; initialize CodeGenerationBuilder(function) "
                "or use for_function(...)"
            )
        candidate = _BuilderRequest(kind, components, bundle)
        if any(request == candidate for request in self.requests):
            return self
        return replace(self, requests=(*self.requests, candidate))

    def _resolved_function_specs(self) -> tuple[_BuilderFunctionSpec, ...]:
        """Return all source-function entries tracked by the builder."""
        root_specs: tuple[_BuilderFunctionSpec, ...] = ()
        if self.function is not None and self.requests:
            root_specs = (
                _BuilderFunctionSpec(
                    function=self.function,
                    requests=self.requests,
                    simplification=self.simplification,
                ),
            )
        specs = (*root_specs, *self.functions)
        if not specs:
            raise ValueError(
                "no kernels were requested; call add_primal() or another add_* method first"
            )
        return specs


def resolve_builder_function_specs(
    specs: tuple[_BuilderFunctionSpec, ...],
    config: RustBuilderConfigLike,
) -> tuple[BuilderSource, ...]:
    """Expand builder function specs into concrete symbolic functions."""
    if len(specs) > 1 and config.function_name is not None:
        raise ValueError(
            "RustBackendConfig.function_name is only supported when the builder targets "
            "a single source function"
        )

    include_base_name = True
    resolved: list[Function] = []
    for spec in specs:
        base_name = config.function_name if len(specs) == 1 else None
        resolved.extend(
            _resolve_builder_functions(
                spec.function,
                config,
                spec.requests,
                spec.simplification,
                include_base_name=include_base_name,
                function_name=base_name,
            )
        )
    return tuple(resolved)


def _resolve_builder_functions(
    function: BuilderSource,
    config: RustBuilderConfigLike,
    requests: tuple[_BuilderRequest, ...],
    simplification: int | str | None,
    *,
    include_base_name: bool = False,
    function_name: str | None = None,
) -> tuple[BuilderSource, ...]:
    """Expand builder requests into concrete symbolic functions."""
    if not requests:
        raise ValueError("no kernels were requested; call add_primal() or another add_* method first")

    if isinstance(function, (ComposedFunction, ComposedGradientFunction)):
        return _resolve_builder_composed_sources(
            function,
            config,
            requests,
            include_base_name=include_base_name,
            function_name=function_name,
        )

    base_function = _apply_builder_base_name(function, function_name)
    crate_prefix = sanitize_ident(config.crate_name or base_function.name)
    resolved: list[Function] = []

    for request in requests:
        if request.kind == "primal":
            resolved.append(
                _rename_generated_function(
                    _maybe_simplify_generated_function(base_function, simplification),
                    _builder_function_name(
                        crate_prefix,
                        "f",
                        base_name=base_function.name,
                        include_base_name=include_base_name,
                    ),
                )
            )
            continue
        if request.kind == "gradient":
            resolved.extend(
                _rename_generated_function(
                    _maybe_simplify_generated_function(base_function.gradient(index), simplification),
                    _builder_function_name(
                        crate_prefix,
                        "grad",
                        base_name=base_function.name,
                        include_base_name=include_base_name,
                        input_name=base_function.input_names[index],
                        include_input_name=len(base_function.inputs) > 1,
                    ),
                )
                for index in range(len(base_function.inputs))
            )
            continue
        if request.kind == "jacobian":
            resolved.extend(
                _rename_generated_function(
                    _maybe_simplify_generated_function(block, simplification),
                    _builder_function_name(
                        crate_prefix,
                        "jf",
                        base_name=base_function.name,
                        include_base_name=include_base_name,
                        input_name=base_function.input_names[index],
                        include_input_name=len(base_function.inputs) > 1,
                    ),
                )
                for index, block in enumerate(base_function.jacobian_blocks())
            )
            continue
        if request.kind == "vjp":
            resolved.extend(
                _rename_generated_function(
                    _maybe_simplify_generated_function(block, simplification),
                    _builder_function_name(
                        crate_prefix,
                        "vjp",
                        base_name=base_function.name,
                        include_base_name=include_base_name,
                        input_name=base_function.input_names[index],
                        include_input_name=len(base_function.inputs) > 1,
                    ),
                )
                for index, block in enumerate(base_function.vjp_blocks())
            )
            continue
        if request.kind == "joint":
            if request.bundle is None:
                raise ValueError("joint builder requests require a FunctionBundle")
            resolved.extend(
                _rename_generated_function(
                    _maybe_simplify_generated_function(
                        base_function.joint(components, index),
                        simplification,
                    ),
                    _builder_function_name(
                        crate_prefix,
                        *_builder_joint_labels(components),
                        base_name=base_function.name,
                        include_base_name=include_base_name,
                        input_name=base_function.input_names[index],
                        include_input_name=len(base_function.inputs) > 1,
                    ),
                )
                for index, components in _resolve_function_bundle(
                    request.bundle,
                    len(base_function.inputs),
                )
            )
            continue
        if request.kind == "hessian":
            resolved.extend(
                _rename_generated_function(
                    _maybe_simplify_generated_function(block, simplification),
                    _builder_function_name(
                        crate_prefix,
                        "hessian",
                        base_name=base_function.name,
                        include_base_name=include_base_name,
                        input_name=base_function.input_names[index],
                        include_input_name=len(base_function.inputs) > 1,
                    ),
                )
                for index, block in enumerate(base_function.hessian_blocks())
            )
            continue
        if request.kind == "hvp":
            resolved.extend(
                _rename_generated_function(
                    _maybe_simplify_generated_function(block, simplification),
                    _builder_function_name(
                        crate_prefix,
                        "hvp",
                        base_name=base_function.name,
                        include_base_name=include_base_name,
                        input_name=base_function.input_names[index],
                        include_input_name=len(base_function.inputs) > 1,
                    ),
                )
                for index, block in enumerate(base_function.hvp_blocks())
            )
            continue
        raise ValueError(f"unsupported builder request kind {request.kind!r}")

    return tuple(resolved)


def _resolve_builder_composed_sources(
    function: ComposedFunction | ComposedGradientFunction,
    config: RustBuilderConfigLike,
    requests: tuple[_BuilderRequest, ...],
    *,
    include_base_name: bool = False,
    function_name: str | None = None,
) -> tuple[BuilderSource, ...]:
    """Expand builder requests for staged composed sources."""
    crate_prefix = sanitize_ident(config.crate_name or function.name)
    base_name = function_name or function.name
    resolved: list[BuilderSource] = []

    for request in requests:
        if request.kind == "primal":
            resolved.append(
                _rename_builder_source(
                    function,
                    _builder_function_name(
                        crate_prefix,
                        "f",
                        base_name=base_name,
                        include_base_name=include_base_name,
                    ),
                )
            )
            continue
        if request.kind == "gradient" and isinstance(function, ComposedFunction):
            resolved.append(
                _rename_builder_source(
                    function.gradient(),
                    _builder_function_name(
                        crate_prefix,
                        "grad",
                        base_name=base_name,
                        include_base_name=include_base_name,
                        input_name=function.input_name,
                        include_input_name=True,
                    ),
                )
            )
            continue
        raise ValueError(
            "staged composed sources currently support only add_primal() and "
            "ComposedFunction.add_gradient() in CodeGenerationBuilder"
        )

    return tuple(resolved)


def _apply_builder_base_name(function: Function, function_name: str | None) -> Function:
    if function_name is None or function_name == function.name:
        return function
    return Function(
        function_name,
        function.inputs,
        function.outputs,
        input_names=function.input_names,
        output_names=function.output_names,
    )


def _rename_generated_function(function: Function, name: str) -> Function:
    if function.name == name:
        return function
    return Function(
        name,
        function.inputs,
        function.outputs,
        input_names=function.input_names,
        output_names=function.output_names,
    )


def _rename_builder_source(
    function: BuilderSource,
    name: str,
) -> BuilderSource:
    """Return a source object with an updated generated Rust function name."""
    if function.name == name:
        return function
    if isinstance(function, Function):
        return _rename_generated_function(function, name)
    return replace(function, name=name)


def _builder_joint_labels(components: tuple[str, ...]) -> tuple[str, ...]:
    mapping = {
        "f": "f",
        "grad": "grad",
        "jf": "jf",
        "hessian": "hessian",
        "hvp": "hvp",
    }
    return tuple(mapping[component] for component in components)


def _builder_function_name(
    crate_prefix: str,
    *labels: str,
    base_name: str | None = None,
    include_base_name: bool = False,
    input_name: str | None = None,
    include_input_name: bool = False,
) -> str:
    parts = [crate_prefix]
    if include_base_name and base_name is not None:
        parts.append(sanitize_ident(base_name))
    parts.extend(labels)
    if include_input_name and input_name is not None:
        parts.append(sanitize_ident(input_name))
    return "_".join(parts)


def _normalize_wrt_indices(wrt: int | list[int] | tuple[int, ...]) -> tuple[int, ...]:
    if isinstance(wrt, int):
        return (wrt,)
    return tuple(wrt)


def _resolve_function_bundle(
    bundle: FunctionBundle,
    input_count: int,
) -> tuple[tuple[int, tuple[str, ...]], ...]:
    if not bundle.items:
        raise ValueError("FunctionBundle must contain at least one item")

    has_primal = any(item.kind == "f" for item in bundle.items)
    grouped: dict[int, list[str]] = {}

    for item in bundle.items:
        if item.kind == "f":
            continue
        if item.wrt_indices is None:
            raise ValueError(f"FunctionBundle item {item.kind!r} requires a wrt specification")
        for index in item.wrt_indices:
            if not 0 <= index < input_count:
                raise IndexError("wrt_index is out of range")
            grouped.setdefault(index, [])
            if has_primal and "f" not in grouped[index]:
                grouped[index].append("f")
            if item.kind not in grouped[index]:
                grouped[index].append(item.kind)

    if not grouped:
        raise ValueError("FunctionBundle must include at least one derivative-producing item")

    resolved: list[tuple[int, tuple[str, ...]]] = []
    for index in sorted(grouped):
        components = tuple(grouped[index])
        if len(components) < 2:
            raise ValueError("joint functions require at least two components")
        resolved.append((index, components))
    return tuple(resolved)


def _maybe_simplify_generated_function(
    function: Function,
    simplification: int | str | None,
) -> Function:
    if simplification is None:
        return function
    return function.simplify(max_effort=simplification, name=function.name)
