import pytest

from gradgen import SX, SXVector, Function
from gradgen.single_shooting import (
    _slice_packed_sequence,
    _extract_single_output,
    _extract_scalar_output,
    _flatten_rollout_states,
    _validate_single_shooting_bundle,
    _single_shooting_primal_name,
    _single_shooting_joint_name,
    _single_shooting_bundle_output_names,
    SingleShootingBundle,
    SingleShootingProblem,
)


def _build_minimal_scalar_problem():
    # state is 2-D, control is scalar, parameter is 3-D
    x = SXVector.sym("x", 2)
    u = SX.sym("u")
    p = SXVector.sym("p", 3)

    dynamics = Function(
        "d",
        [x, u, p],
        [SXVector((0.5 * x[0] + p[0] * u, 0.5 * x[1] + p[1] * u))],
    )
    stage_cost = Function(
        "s",
        [x, u, p],
        [x[0] * x[0] + x[1] * x[1] + u * u + p[2] * u],
    )
    terminal_cost = Function(
        "t",
        [x, p],
        [x[0] * x[0] + x[1] * x[1] + p[2] * x[1]],
    )
    return SingleShootingProblem(
        "sp",
        horizon=2,
        dynamics=dynamics,
        stage_cost=stage_cost,
        terminal_cost=terminal_cost,
    )


def test_slice_packed_sequence_scalar_and_vector():
    seq = SXVector.sym("U", 4)
    # scalar formal -> should return a single SX element equal to sequence[start]
    sx_formal = SX.sym("u")
    first = _slice_packed_sequence(seq, 0, 1, sx_formal)
    assert isinstance(first, SX)
    assert first.name == seq[0].name

    # vector formal -> returns SXVector slice of block_size
    vec_formal = SXVector.sym("uvec", 2)
    slice_vec = _slice_packed_sequence(seq, 1, 2, vec_formal)
    assert isinstance(slice_vec, SXVector)
    assert len(slice_vec) == 2
    assert slice_vec[0].name == seq[2].name


def test_extract_single_and_scalar_output_errors_and_types():
    # tuple of length !=1 -> ValueError
    with pytest.raises(ValueError):
        _extract_single_output((SX.const(1.0), SX.const(2.0)))

    # non-SX and non-tuple -> TypeError
    with pytest.raises(TypeError):
        _extract_single_output(123)

    # scalar extractor rejects SXVector
    with pytest.raises(ValueError):
        _extract_scalar_output(SXVector((SX.const(1.0), SX.const(2.0))))

    # valid scalar path
    assert _extract_scalar_output(SX.const(3.0)).value == 3.0


def test_flatten_rollout_states_mixes_scalar_and_vector():
    s1 = SX.const(1.0)
    v = SXVector((SX.const(2.0), SX.const(3.0)))
    s2 = SX.const(4.0)
    packed = _flatten_rollout_states([s1, v, s2])
    assert isinstance(packed, SXVector)
    assert tuple(elem.value for elem in packed) == (1.0, 2.0, 3.0, 4.0)


def test_validate_single_shooting_bundle_errors_and_name_helpers():
    # empty bundle must raise
    empty = SingleShootingBundle()
    with pytest.raises(ValueError):
        _validate_single_shooting_bundle(empty)

    # single flag only must raise
    only_cost = SingleShootingBundle(include_cost=True)
    with pytest.raises(ValueError):
        _validate_single_shooting_bundle(only_cost)

    # name helpers
    assert _single_shooting_primal_name("base", False) == "base"
    assert _single_shooting_primal_name("base", True) == "base_with_states"

    bundle = SingleShootingBundle().add_cost().add_gradient()
    name = _single_shooting_joint_name("prob", bundle, "U")
    assert "cost" in name and "gradient_U" in name
    assert _single_shooting_bundle_output_names(_build_minimal_scalar_problem(), bundle) == ("cost", "gradient_U")


def test_primal_to_function_and_staged_wrappers():
    problem = _build_minimal_scalar_problem()
    primal = problem.primal(include_states=True, name="myp")
    func = primal.to_function()
    # inputs and outputs names should match problem
    assert func.input_names == problem.input_names
    assert func.output_names == ("cost", "x_traj")
