import pytest

from gradgen import SX, SXVector
from gradgen._custom_elementary import callbacks
from gradgen._custom_elementary.model import RegisteredElementaryFunction


def test_invoke_custom_callback_omitted_params_calls_single_arg():
    def cb_single(x):
        return ("single", x)

    out = callbacks.invoke_custom_callback(cb_single, 5, (), 0)
    assert out == ("single", 5)


def test_invoke_custom_callback_with_two_params_calls_with_w():
    def cb_two(x, w):
        return ("two", x, w)

    out = callbacks.invoke_custom_callback(cb_two, 1, (2, 3), 0)
    assert out == ("two", 1, (2, 3))


def test_invoke_custom_hvp_callback_two_arg_and_orders():
    def cb_vt(v, t):
        return ("vt", v, t)

    assert callbacks.invoke_custom_hvp_callback(cb_vt, 1, 2, (), 0) == ("vt", 1, 2)

    def cb_three_reorder(value, w, v_arg):
        return ("reordered", value, w, v_arg)

    # third positional name startswith 'v' -> callback(value, w, tangent)
    assert callbacks.invoke_custom_hvp_callback(cb_three_reorder, "x", "tan", "w", 0) == (
        "reordered",
        "x",
        "w",
        "tan",
    )

    def cb_order(value, t, w):
        return ("order", value, t, w)

    # default fallback ordering -> callback(value, tangent, w)
    assert callbacks.invoke_custom_hvp_callback(cb_order, "x", "tval", "wval", 1) == (
        "order",
        "x",
        "tval",
        "wval",
    )


def test_coerce_symbolic_scalar_and_errors():
    assert callbacks.coerce_symbolic_scalar(3, "err").value == 3.0
    sx = SX.const(4.0)
    assert callbacks.coerce_symbolic_scalar(sx, "err").value == 4.0

    class HasItem:
        def __init__(self, v):
            self._v = v

        def item(self):
            return self._v

    inner = HasItem(SX.const(2.0))
    out = callbacks.coerce_symbolic_scalar(inner, "err")
    assert isinstance(out, SX) and out.value == 2.0

    with pytest.raises(TypeError):
        callbacks.coerce_symbolic_scalar([1, 2], "err")


def test_coerce_symbolic_vector_and_matrix():
    v = callbacks.coerce_symbolic_vector((1, 2), 2, "err")
    assert isinstance(v, SXVector) and len(v) == 2

    with pytest.raises(TypeError):
        callbacks.coerce_symbolic_vector("nope", 1, "err")

    with pytest.raises(TypeError):
        callbacks.coerce_symbolic_vector((1, 2, 3), 2, "err")

    # matrix coercion
    mat = callbacks.coerce_symbolic_matrix(((1, 0), (0, 2)), 2, "err")
    assert isinstance(mat, tuple) and len(mat) == 2

    with pytest.raises(TypeError):
        callbacks.coerce_symbolic_matrix("bad", 1, "err")


def test_coerce_numeric_scalar_vector_matrix():
    assert callbacks.coerce_numeric_scalar(1, "err") == 1.0
    assert callbacks.coerce_numeric_scalar(SX.const(3.0), "err") == 3.0

    with pytest.raises(TypeError):
        callbacks.coerce_numeric_scalar(SX.sym("x"), "err")

    vec = callbacks.coerce_numeric_vector((1, 2), 2, "err")
    assert vec == (1.0, 2.0)

    with pytest.raises(TypeError):
        callbacks.coerce_numeric_vector("s", 1, "err")

    mat = callbacks.coerce_numeric_matrix(((1, 0), (0, 2)), 2, "err")
    assert mat == ((1.0, 0.0), (0.0, 2.0))

    with pytest.raises(TypeError):
        callbacks.coerce_numeric_matrix("bad", 1, "err")


def test_build_custom_hvp_from_hessian_scalar_and_vector():
    # scalar case
    spec_scalar = RegisteredElementaryFunction(
        name="s",
        input_dimension=1,
        parameter_dimension=0,
        parameter_defaults=(),
        eval_python=None,
        jacobian=lambda x, w: SX.const(2.0),
        hessian=lambda x, w: SX.const(3.0),
        hvp=None,
        rust_primal=None,
        rust_jacobian=None,
        rust_hvp=None,
        rust_hessian=None,
    )

    # correct SX tangent multiplies
    out = callbacks._build_custom_hvp_from_hessian(spec_scalar, SX.const(1.0), SX.const(4.0), ())
    assert isinstance(out, SX) and out.op == "mul"

    # wrong tangent type raises
    with pytest.raises(TypeError):
        callbacks._build_custom_hvp_from_hessian(spec_scalar, SX.const(1.0), (1.0,), ())

    # vector case
    spec_vec = RegisteredElementaryFunction(
        name="v",
        input_dimension=2,
        parameter_dimension=0,
        parameter_defaults=(),
        eval_python=None,
        jacobian=lambda x, w: SXVector((SX.const(1.0), SX.const(1.0))),
        hessian=lambda x, w: ((SXVector((SX.const(1.0), SX.const(0.0)))), (SXVector((SX.const(0.0), SX.const(2.0))))) ,
        hvp=None,
        rust_primal=None,
        rust_jacobian=None,
        rust_hvp=None,
        rust_hessian=None,
    )

    tangent = SXVector((SX.const(3.0), SX.const(4.0)))
    hvp = callbacks._build_custom_hvp_from_hessian(spec_vec, tangent, tangent, ())
    assert isinstance(hvp, SXVector) and len(hvp) == 2
