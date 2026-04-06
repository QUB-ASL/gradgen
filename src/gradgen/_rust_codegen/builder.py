"""Builder API for creating multi-kernel Rust crates."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Protocol

from ..composed_function import ComposedFunction, ComposedGradientFunction
from ..function import Function
from ..map_zip import ZippedFunction, ZippedJacobianFunction, ReducedFunction
from ..single_shooting import (
    SingleShootingBundle,
    SingleShootingGradientFunction,
    SingleShootingHvpFunction,
    SingleShootingJointFunction,
    SingleShootingPrimalFunction,
    SingleShootingProblem,
)
from .naming import sanitize_ident

if TYPE_CHECKING:
    from ..composer import FunctionComposition


BuilderSource = (
    Function
    | ComposedFunction
    | ComposedGradientFunction
    | ZippedFunction
    | ZippedJacobianFunction
    | ReducedFunction
    | SingleShootingProblem
    | SingleShootingPrimalFunction
    | SingleShootingGradientFunction
    | SingleShootingHvpFunction
    | SingleShootingJointFunction
)


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
    bundle: FunctionBundle | SingleShootingBundle | None = None


@dataclass(frozen=True, slots=True)
class _BundleItem:
    """One artifact entry inside a ``FunctionBundle``."""

    kind: str
    wrt_indices: tuple[int, ...] | None = None


@dataclass(frozen=True, slots=True)
class FunctionBundle:
    """
    Fluent description of artifacts to compute together in one joint kernel.
    """

    items: tuple[_BundleItem, ...] = ()

    def add_f(self) -> FunctionBundle:
        """Include the primal outputs."""
        return self._add_item("f")

    def add_gradient(self,
                     *,
                     wrt: int | list[int] | tuple[int, ...]) \
            -> FunctionBundle:
        """Include gradient blocks for one or more input indices."""
        return self._add_item("grad", wrt)

    def add_jf(self,
               *,
               wrt: int | list[int] | tuple[int, ...]) \
            -> FunctionBundle:
        """Include Jacobian blocks for one or more input indices."""
        return self._add_item("jf", wrt)

    def add_hessian(self, *, wrt: int | list[int] | tuple[int, ...]) \
            -> FunctionBundle:
        """Include Hessian blocks for one or more input indices."""
        return self._add_item("hessian", wrt)

    def add_hvp(self, *, wrt: int | list[int] | tuple[int, ...]) \
            -> FunctionBundle:
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


@dataclass(frozen=True,
           slots=True)
class _BuilderFunctionSpec:
    """One source function plus the kernels requested for it."""

    function: BuilderSource
    requests: tuple[_BuilderRequest, ...] = ()
    simplification: int | str | None = None


@dataclass(frozen=True,
           slots=True)
class CodeGenerationBuilder:
    """Fluent builder for a generated Rust crate with one or more functions."""

    function: BuilderSource | None = None
    config: RustBuilderConfigLike | None = None
    requests: tuple[_BuilderRequest, ...] = ()
    simplification: int | str | None = None
    functions: tuple[_BuilderFunctionSpec, ...] = ()

    def with_backend_config(self,
                            config: RustBuilderConfigLike) \
            -> CodeGenerationBuilder:
        """Return a copy using ``config`` for generated Rust code."""
        return replace(self, config=config)

    def with_simplification(self,
                            max_effort: int | str | None) \
            -> CodeGenerationBuilder:
        """Return a copy applying ``max_effort`` simplification to generated kernels."""
        return replace(self, simplification=max_effort)

    def add_primal(self,
                   *,
                   include_states: bool = False) \
            -> CodeGenerationBuilder:
        """Include the primal function in the generated crate."""
        components = ("states",) if include_states else ()
        return self._add_request("primal", components=components)

    def add_gradient(self, *, include_states: bool = False) -> CodeGenerationBuilder:
        """Include gradient kernels for scalar-output functions."""
        components = ("states",) if include_states else ()
        return self._add_request("gradient", components=components)

    def add_jacobian(self) -> CodeGenerationBuilder:
        """Include Jacobian kernels for all input blocks."""
        return self._add_request("jacobian")

    def add_vjp(self) -> CodeGenerationBuilder:
        """Include runtime-seeded vector-Jacobian-product kernels for input blocks."""
        return self._add_request("vjp")

    def add_joint(self, bundle: FunctionBundle | SingleShootingBundle) -> CodeGenerationBuilder:
        """Include kernels that compute bundled artifacts together."""
        return self._add_request("joint", bundle=bundle)

    def add_hessian(self) -> CodeGenerationBuilder:
        """Include Hessian kernels for scalar-output functions."""
        return self._add_request("hessian")

    def add_hvp(self, *, include_states: bool = False) -> CodeGenerationBuilder:
        """Include Hessian-vector product kernels for scalar-output functions."""
        components = ("states",) if include_states else ()
        return self._add_request("hvp", components=components)

    def for_function(
        self,
        function: BuilderSource,
        configure: Callable[[CodeGenerationBuilder], CodeGenerationBuilder] | None = None,
    ) -> CodeGenerationBuilder | FunctionCodegenBuilder:
        """Start configuring kernels for ``function`` or apply a legacy callback."""
        scoped_builder = FunctionCodegenBuilder(parent=self, function=function)
        if configure is None:
            return scoped_builder
        configured_builder = configure(
            CodeGenerationBuilder(function=function, config=self.config)
        )
        return _append_builder_function_spec(
            self,
            _BuilderFunctionSpec(
                function=_require_builder_function(configured_builder),
                requests=configured_builder.requests,
                simplification=configured_builder.simplification,
            ),
        )

    def build(self, path: str | Path = "."):
        """Generate a single Rust crate inside ``path``.

        The argument is treated as the parent directory that will contain the
        generated crate. The crate directory itself is named from
        ``with_crate_name(...)`` when provided, or otherwise from the first
        generated source function.
        """
        from .codegen import create_multi_function_rust_project
        from .config import RustBackendConfig

        resolved_config = self.config or RustBackendConfig()
        raw_path = Path(path).expanduser()
        if resolved_config.crate_name is not None:
            crate_name = sanitize_ident(resolved_config.crate_name)
        else:
            if raw_path != Path("."):
                crate_name = sanitize_ident(raw_path.name)
                resolved_config = resolved_config.with_crate_name(crate_name)
            else:
                crate_name = None
        functions = resolve_builder_function_specs(
            self._resolved_function_specs(),
            resolved_config,
        )
        if crate_name is None:
            crate_name = sanitize_ident(functions[0].name)
            resolved_config = resolved_config.with_crate_name(crate_name)
            functions = resolve_builder_function_specs(
                self._resolved_function_specs(),
                resolved_config,
            )
        if raw_path == Path("."):
            project_dir = raw_path.resolve() / crate_name
        else:
            resolved_path = raw_path.resolve()
            project_dir = (
                resolved_path
                if resolved_path.name == crate_name
                else resolved_path / crate_name
            )
        return create_multi_function_rust_project(
            functions,
            project_dir,
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


@dataclass(frozen=True, slots=True)
class FunctionCodegenBuilder:
    """Scoped builder used to configure kernels for one source function."""

    parent: CodeGenerationBuilder
    function: BuilderSource
    requests: tuple[_BuilderRequest, ...] = ()
    simplification: int | str | None = None

    def with_simplification(self, max_effort: int | str | None) -> FunctionCodegenBuilder:
        """Return a copy applying ``max_effort`` simplification to this function."""
        return replace(self, simplification=max_effort)

    def add_primal(self, *, include_states: bool = False) -> FunctionCodegenBuilder:
        """Include the primal function in the generated crate."""
        components = ("states",) if include_states else ()
        return self._add_request("primal", components=components)

    def add_gradient(self, *, include_states: bool = False) -> FunctionCodegenBuilder:
        """Include gradient kernels for scalar-output functions."""
        components = ("states",) if include_states else ()
        return self._add_request("gradient", components=components)

    def add_jacobian(self) -> FunctionCodegenBuilder:
        """Include Jacobian kernels for all input blocks."""
        return self._add_request("jacobian")

    def add_vjp(self) -> FunctionCodegenBuilder:
        """Include runtime-seeded vector-Jacobian-product kernels for input blocks."""
        return self._add_request("vjp")

    def add_joint(self, bundle: FunctionBundle | SingleShootingBundle) -> FunctionCodegenBuilder:
        """Include kernels that compute bundled artifacts together."""
        return self._add_request("joint", bundle=bundle)

    def add_hessian(self) -> FunctionCodegenBuilder:
        """Include Hessian kernels for scalar-output functions."""
        return self._add_request("hessian")

    def add_hvp(self, *, include_states: bool = False) -> FunctionCodegenBuilder:
        """Include Hessian-vector product kernels for scalar-output functions."""
        components = ("states",) if include_states else ()
        return self._add_request("hvp", components=components)

    def done(self) -> CodeGenerationBuilder:
        """Commit this scoped function configuration back to the parent builder."""
        if not self.requests:
            raise ValueError(
                "no kernels were requested for this function; call add_primal() or another add_* method first"
            )
        return _append_builder_function_spec(
            self.parent,
            _BuilderFunctionSpec(
                function=self.function,
                requests=self.requests,
                simplification=self.simplification,
            ),
        )

    def _add_request(
        self,
        kind: str,
        *,
        components: tuple[str, ...] = (),
        bundle: FunctionBundle | None = None,
    ) -> FunctionCodegenBuilder:
        candidate = _BuilderRequest(kind, components, bundle)
        if any(request == candidate for request in self.requests):
            return self
        return replace(self, requests=(*self.requests, candidate))


def _append_builder_function_spec(
    builder: CodeGenerationBuilder,
    spec: _BuilderFunctionSpec,
) -> CodeGenerationBuilder:
    """Return ``builder`` with one resolved function specification appended."""
    return replace(builder, functions=(*builder.functions, spec))


def _require_builder_function(builder: CodeGenerationBuilder) -> BuilderSource:
    """Return the selected source function or raise for misconfigured callback builders."""
    if builder.function is None:
        raise ValueError("configured function builder must target a function")
    return builder.function


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
        raise ValueError(
            "no kernels were requested; call add_primal() or another "
            "add_* method first"
        )

    if isinstance(function, (ComposedFunction, ComposedGradientFunction)):
        return _resolve_builder_composed_sources(
            function,
            config,
            requests,
            simplification,
            include_base_name=include_base_name,
            function_name=function_name,
        )
    from ..composer import FunctionComposition

    if isinstance(function, FunctionComposition):
        return _resolve_builder_function_composition_sources(
            function,
            config,
            requests,
            simplification,
            include_base_name=include_base_name,
            function_name=function_name,
        )
    if isinstance(
        function,
        (ZippedFunction, ZippedJacobianFunction, ReducedFunction),
    ):
        return _resolve_builder_zipped_sources(
            function,
            config,
            requests,
            simplification,
            include_base_name=include_base_name,
            function_name=function_name,
        )
    if isinstance(
        function,
        (
            SingleShootingProblem,
            SingleShootingPrimalFunction,
            SingleShootingGradientFunction,
            SingleShootingHvpFunction,
            SingleShootingJointFunction,
        ),
    ):
        return _resolve_builder_single_shooting_sources(
            function,
            config,
            requests,
            simplification,
            include_base_name=include_base_name,
            function_name=function_name,
        )

    base_function = _apply_builder_base_name(function, function_name)
    crate_prefix = sanitize_ident(config.crate_name or base_function.name)
    resolved: list[BuilderSource] = []

    for request in requests:
        resolved.extend(
            _resolve_builder_base_function_request(
                base_function,
                request,
                crate_prefix=crate_prefix,
                simplification=simplification,
                include_base_name=include_base_name,
            )
        )

    return tuple(resolved)


def _resolve_builder_base_function_request(
    base_function: Function,
    request: _BuilderRequest,
    *,
    crate_prefix: str,
    simplification: int | str | None,
    include_base_name: bool,
) -> tuple[BuilderSource, ...]:
    """Expand one builder request for a plain symbolic function."""
    if request.kind == "primal":
        if request.components:
            raise ValueError(
                "include_states is only supported for "
                "SingleShootingProblem sources"
            )
        return (
            _rename_generated_function(
                _maybe_simplify_generated_function(
                    base_function,
                    simplification,
                ),
                _builder_function_name(
                    crate_prefix,
                    "f",
                    base_name=base_function.name,
                    include_base_name=include_base_name,
                ),
            ),
        )
    if request.kind == "gradient":
        if request.components:
            raise ValueError(
                "include_states is only supported for "
                "SingleShootingProblem sources"
            )
        return tuple(
            _rename_generated_function(
                _maybe_simplify_generated_function(
                    base_function.gradient(index),
                    simplification,
                ),
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
    if request.kind == "jacobian":
        return tuple(
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
    if request.kind == "vjp":
        return tuple(
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
    if request.kind == "joint":
        if request.bundle is None:
            raise ValueError("joint builder requests require a FunctionBundle")
        return tuple(
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
    if request.kind == "hessian":
        return tuple(
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
    if request.kind == "hvp":
        if request.components:
            raise ValueError(
                "include_states is only supported for "
                "SingleShootingProblem sources"
            )
        return tuple(
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
    raise ValueError(f"unsupported builder request kind {request.kind!r}")


def _resolve_builder_composed_sources(
    function: ComposedFunction | ComposedGradientFunction,
    config: RustBuilderConfigLike,
    requests: tuple[_BuilderRequest, ...],
    simplification: int | str | None,
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
            if request.components:
                raise ValueError("include_states is only supported for SingleShootingProblem sources")
            resolved.append(
                _rename_builder_source(
                    replace(function, simplification=simplification),
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
            if request.components:
                raise ValueError("include_states is only supported for SingleShootingProblem sources")
            simplified_function = replace(function, simplification=simplification)
            resolved.append(
                _rename_builder_source(
                    simplified_function.gradient(),
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


def _resolve_builder_function_composition_sources(
    function: FunctionComposition,
    config: RustBuilderConfigLike,
    requests: tuple[_BuilderRequest, ...],
    simplification: int | str | None,
    *,
    include_base_name: bool = False,
    function_name: str | None = None,
) -> tuple[BuilderSource, ...]:
    """Expand builder requests for staged composer pipelines."""
    crate_prefix = sanitize_ident(config.crate_name or function.name)
    base_name = function_name or function.name
    resolved: list[BuilderSource] = []

    for request in requests:
        if request.kind == "primal":
            if request.components:
                raise ValueError(
                    "include_states is not supported for composer pipelines"
                )
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

        symbolic_function = function.to_function(name=base_name)
        resolved.extend(
            _resolve_builder_base_function_request(
                symbolic_function,
                request,
                crate_prefix=crate_prefix,
                simplification=simplification,
                include_base_name=include_base_name,
            )
        )

    return tuple(resolved)


def _resolve_builder_zipped_sources(
    function: ZippedFunction | ZippedJacobianFunction | ReducedFunction,
    config: RustBuilderConfigLike,
    requests: tuple[_BuilderRequest, ...],
    simplification: int | str | None,
    *,
    include_base_name: bool = False,
    function_name: str | None = None,
) -> tuple[BuilderSource, ...]:
    """Expand builder requests for staged map/zip/reduce sources."""
    crate_prefix = sanitize_ident(config.crate_name or function.name)
    base_name = function_name or function.name
    resolved: list[BuilderSource] = []

    if isinstance(function, ZippedJacobianFunction):
        for request in requests:
            if request.kind != "primal":
                raise ValueError(
                    "explicit staged Jacobian map/zip sources in CodeGenerationBuilder "
                    "currently support only add_primal()"
                )
            if request.components:
                raise ValueError("include_states is not supported for map/zip/reduce sources")
            resolved.append(
                _rename_builder_source(
                    replace(function, simplification=simplification),
                    _builder_function_name(
                        crate_prefix,
                        "f",
                        base_name=base_name,
                        include_base_name=include_base_name,
                    ),
                )
            )
        return tuple(resolved)

    simplified_function = replace(function, simplification=simplification)
    for request in requests:
        if request.kind == "primal":
            if request.components:
                raise ValueError("include_states is not supported for map/zip/reduce sources")
            resolved.append(
                _rename_builder_source(
                    simplified_function,
                    _builder_function_name(
                        crate_prefix,
                        "f",
                        base_name=base_name,
                        include_base_name=include_base_name,
                    ),
                )
            )
            continue
        if request.kind == "jacobian":
            if request.components:
                raise ValueError("include_states is not supported for map/zip/reduce sources")
            if not isinstance(simplified_function, ZippedFunction):
                raise ValueError("add_jacobian() is only supported for map/zip sources")
            for input_index, input_name in enumerate(simplified_function.input_names):
                resolved.append(
                    _rename_builder_source(
                        simplified_function.jacobian(input_index),
                        _builder_function_name(
                            crate_prefix,
                            "jf",
                            base_name=base_name,
                            include_base_name=include_base_name,
                            input_name=input_name,
                            include_input_name=True,
                        ),
                    )
                )
            continue
        raise ValueError(
            "map/zip/reduce sources currently support only add_primal(), and add_jacobian() only for map/zip"
        )

    return tuple(resolved)


def _resolve_builder_single_shooting_sources(
    function: (
        SingleShootingProblem
        | SingleShootingPrimalFunction
        | SingleShootingGradientFunction
        | SingleShootingJointFunction
    ),
    config: RustBuilderConfigLike,
    requests: tuple[_BuilderRequest, ...],
    simplification: int | str | None,
    *,
    include_base_name: bool = False,
    function_name: str | None = None,
) -> tuple[BuilderSource, ...]:
    """Expand builder requests for single-shooting sources."""
    crate_prefix = sanitize_ident(config.crate_name or function.name)
    base_name = function_name or function.name
    resolved: list[BuilderSource] = []

    if isinstance(
        function,
        (
            SingleShootingPrimalFunction,
            SingleShootingGradientFunction,
            SingleShootingHvpFunction,
            SingleShootingJointFunction,
        ),
    ):
        for request in requests:
            if request.kind != "primal":
                raise ValueError(
                    "explicit single-shooting staged sources in CodeGenerationBuilder "
                    "currently support only add_primal()"
                )
            if request.components:
                raise ValueError("include_states is configured on the single-shooting staged source itself")
            resolved.append(
                _rename_builder_source(
                    replace(function, simplification=simplification),
                    _builder_function_name(
                        crate_prefix,
                        "f",
                        base_name=base_name,
                        include_base_name=include_base_name,
                    ),
                )
            )
        return tuple(resolved)

    simplified_problem = replace(function, simplification=simplification)
    for request in requests:
        if request.kind == "primal":
            include_states = "states" in request.components
            labels = ("f", "states") if include_states else ("f",)
            resolved.append(
                _rename_builder_source(
                    simplified_problem.primal(include_states=include_states),
                    _builder_function_name(
                        crate_prefix,
                        *labels,
                        base_name=base_name,
                        include_base_name=include_base_name,
                    ),
                )
            )
            continue
        if request.kind == "gradient":
            include_states = "states" in request.components
            gradient_source = simplified_problem.gradient(include_states=include_states)
            labels = ("grad", "states") if include_states else ("grad",)
            resolved.append(
                _rename_builder_source(
                    gradient_source,
                    _builder_function_name(
                        crate_prefix,
                        *labels,
                        base_name=base_name,
                        include_base_name=include_base_name,
                        input_name=simplified_problem.control_sequence_name,
                        include_input_name=True,
                    ),
                )
            )
            continue
        if request.kind == "hvp":
            include_states = "states" in request.components
            hvp_source = simplified_problem.hvp(include_states=include_states)
            labels = ("hvp", "states") if include_states else ("hvp",)
            resolved.append(
                _rename_builder_source(
                    hvp_source,
                    _builder_function_name(
                        crate_prefix,
                        *labels,
                        base_name=base_name,
                        include_base_name=include_base_name,
                        input_name=simplified_problem.control_sequence_name,
                        include_input_name=True,
                    ),
                )
            )
            continue
        if request.kind == "joint":
            if request.bundle is None or not isinstance(request.bundle, SingleShootingBundle):
                raise ValueError(
                    "joint single-shooting requests require a SingleShootingBundle"
                )
            labels: list[str] = []
            if request.bundle.include_cost:
                labels.append("f")
            if request.bundle.include_gradient:
                labels.append("grad")
            if request.bundle.include_hvp:
                labels.append("hvp")
            if request.bundle.include_states:
                labels.append("states")
            resolved.append(
                _rename_builder_source(
                    simplified_problem.joint(request.bundle),
                    _builder_function_name(
                        crate_prefix,
                        *labels,
                        base_name=base_name,
                        include_base_name=include_base_name,
                        input_name=simplified_problem.control_sequence_name,
                        include_input_name=request.bundle.include_gradient or request.bundle.include_hvp,
                    ),
                )
            )
            continue
        raise ValueError(
            "single-shooting sources currently support only add_primal(), "
            "add_gradient(), add_hvp(), and add_joint()"
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
