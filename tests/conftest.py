from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from gradgen._rust_codegen.config import RustBackendConfig


@pytest.fixture(autouse=True)
def _force_debug_build_profile(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default Rust crate builds in tests to the debug profile.

    This keeps crate compilation fast and makes the test suite behave the
    same way regardless of whether callers construct a backend config
    explicitly or rely on the default ``RustBackendConfig()`` path.
    ``test_config.py`` is excluded so the configuration defaults remain
    covered by dedicated assertions there.
    """
    if Path(str(request.node.fspath)).name == "test_config.py":
        yield
        return

    original_init = RustBackendConfig.__init__
    signature = inspect.signature(original_init)

    def patched_init(self, *args, **kwargs):
        bound = signature.bind_partial(self, *args, **kwargs)
        if "build_profile" not in bound.arguments:
            kwargs = dict(kwargs)
            kwargs["build_profile"] = "debug"
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(RustBackendConfig, "__init__", patched_init)
    yield
