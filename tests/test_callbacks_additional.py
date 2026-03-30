import pytest

from gradgen import SX, SXNode, SXVector, Function
from gradgen._custom_elementary import callbacks
from gradgen._custom_elementary.model import RegisteredElementaryFunction


def test_invoke_custom_callback_fallback_signature(monkeypatch):
    # Make inspect.signature raise to exercise the fallback path
    def cb_expected(x, w):
        return ("fallback", x, w)

    monkeypatch.setattr(callbacks.inspect, "signature", lambda *_: (_ for _ in ()).throw(ValueError()))
    out = callbacks.invoke_custom_callback(cb_expected, 7, (9,), 0)
    assert out == ("fallback", 7, (9,))


def test_invoke_custom_hvp_callback_signature_and_ordering(monkeypatch):
    # When inspect.signature raises, the code should call (value, tangent, w)
    def cb_fallback(value, tangent, w):
        return ("fb", value, tangent, w)

    monkeypatch.setattr(callbacks.inspect, "signature", lambda *_: (_ for _ in ()).throw(TypeError()))
    out = callbacks.invoke_custom_hvp_callback(cb_fallback, "x", "t", "w", 1)
    assert out == ("fb", "x", "t", "w")


def test_coerce_symbolic_vector_and_matrix_stronger_checks():
    v = callbacks.coerce_symbolic_vector((1, 2), 2, "err")
    assert isinstance(v, SXVector) and len(v) == 2
    assert all(isinstance(elem, SX) for elem in v)
    assert tuple(elem.value for elem in v) == (1.0, 2.0)

    # SXVector input path and mismatched length
    sv = SXVector((SX.const(1.0), SX.const(2.0)))
    assert callbacks.coerce_symbolic_vector(sv, 2, "err") == sv
    with pytest.raises(TypeError):
        callbacks.coerce_symbolic_vector(sv, 1, "err")

    mat = callbacks.coerce_symbolic_matrix(((1, 0), (0, 2)), 2, "err")
    assert isinstance(mat, tuple) and len(mat) == 2
    assert all(isinstance(row, SXVector) for row in mat)
    assert tuple(tuple(e.value for e in row) for row in mat) == ((1.0, 0.0), (0.0, 2.0))


def test_coerce_numeric_vector_handles_sxvector_and_item_edge():
    # SXVector of consts
    sv = SXVector((SX.const(3.0), SX.const(4.0)))
    out = callbacks.coerce_numeric_vector(sv, 2, "err")
    assert out == (3.0, 4.0)

    # object with item() returning non-scalar should raise via coerce_numeric_scalar
    class BadItem:
        def item(self):
            return object()

    with pytest.raises(TypeError):
        callbacks.coerce_numeric_scalar(BadItem(), "err")


def test_build_custom_hvp_from_hessian_numeric_evaluation():
    # scalar case: hessian=3.0, tangent=4.0 -> hvp=12.0
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

    hvp_expr = callbacks._build_custom_hvp_from_hessian(spec_scalar, SX.const(1.0), SX.const(4.0), ())
    # hvp_expr should evaluate to 12.0 regardless of input, so create a dummy Function
    x = SX.sym("x")
    f = Function("f", [x], [hvp_expr])
    assert f(0.0) == 12.0

    # vector case: use simple diagonal Hessian [[1,0],[0,2]] and tangent [3,4]
    spec_vec = RegisteredElementaryFunction(
        name="v",
        input_dimension=2,
        parameter_dimension=0,
        parameter_defaults=(),
        eval_python=None,
        jacobian=lambda x, w: SXVector((SX.const(1.0), SX.const(1.0))),
        hessian=lambda x, w: (SXVector((SX.const(1.0), SX.const(0.0))), SXVector((SX.const(0.0), SX.const(2.0)))),
        hvp=None,
        rust_primal=None,
        rust_jacobian=None,
        rust_hvp=None,
        rust_hessian=None,
    )

    tangent = SXVector((SX.const(3.0), SX.const(4.0)))
    hvp_vec = callbacks._build_custom_hvp_from_hessian(spec_vec, tangent, tangent, ())
    # evaluate via a Function with one vector input
    v = SXVector.sym("v", 2)
    f2 = Function("f2", [v], [hvp_vec])
    assert f2([0.0, 0.0]) == (3.0, 8.0)
