import math

import pytest
import sympy as sp

from algorithms.fixed_point import FixedPointAlgorithm, FixedPointParams

x = sp.Symbol("x")
y = sp.Symbol("y")


@pytest.fixture
def algorithm():
    return FixedPointAlgorithm()


def test_validate_raises_on_wrong_parameters_type(algorithm):
    with pytest.raises(TypeError, match="FixedPointParams"):
        algorithm._validate(object())


def test_validate_raises_when_a_is_not_less_than_b(algorithm):
    params = FixedPointParams(f=x, p0=0, a=1, b=1)
    with pytest.raises(ValueError, match="a must be less than b"):
        algorithm._validate(params)


def test_validate_raises_when_p0_is_outside_interval(algorithm):
    params = FixedPointParams(f=x - 1, p0=5, a=0, b=2)
    with pytest.raises(ValueError, match="p0 must lie within"):
        algorithm._validate(params)


def test_validate_raises_when_err_is_not_positive(algorithm):
    params = FixedPointParams(f=x - 1, p0=0.5, a=0, b=2, err=0)
    with pytest.raises(ValueError, match="err must be positive"):
        algorithm._validate(params)


def test_validate_raises_when_max_iter_is_not_positive(algorithm):
    params = FixedPointParams(f=x - 1, p0=0.5, a=0, b=2, max_iter=0)
    with pytest.raises(ValueError, match="max_iter must be positive"):
        algorithm._validate(params)


def test_validate_raises_when_f_is_not_a_valid_expression(algorithm):
    params = FixedPointParams(f="not a math expr $$", p0=0.5, a=0, b=2)
    with pytest.raises(ValueError, match="f must be a valid sympy expression"):
        algorithm._validate(params)


def test_validate_raises_when_f_uses_a_variable_other_than_x(algorithm):
    params = FixedPointParams(f=y - 1, p0=0.5, a=0, b=2)
    with pytest.raises(ValueError, match="only be expressed in terms of the variable x"):
        algorithm._validate(params)


def test_validate_raises_when_g_is_not_continuous_on_interval(algorithm):
    params = FixedPointParams(f=1 / (x - 1), p0=2, a=0, b=3)
    with pytest.raises(ValueError, match="not continuous"):
        algorithm._validate(params)


def test_validate_raises_when_g_range_escapes_interval(algorithm):
    params = FixedPointParams(f=sp.cos(x) - x, p0=0.5, a=0, b=1)
    with pytest.raises(ValueError, match=r"is not contained in \[a, b\]"):
        algorithm._validate(params)


def test_validate_raises_when_g_differentiability_cannot_be_determined(algorithm):
    params = FixedPointParams(f=x - sp.Abs(x), p0=0.1, a=-1, b=1)
    with pytest.raises(ValueError, match="cannot determine whether g is continuous"):
        algorithm._validate(params)


def test_validate_raises_when_g_is_not_a_contraction(algorithm):
    params = FixedPointParams(f=sp.Integer(0), p0=1, a=0, b=2)
    with pytest.raises(ValueError, match="not a contraction"):
        algorithm._validate(params)


def test_execute_converges_to_root_for_linear_f(algorithm):
    params = FixedPointParams(f=sp.Rational(2, 3) * x - 2, p0=1.0, a=0.0, b=5.0, err=1e-8)
    result = algorithm.run(params)

    assert result.converged is True
    assert math.isclose(result.root, 3, abs_tol=1e-6)


def test_execute_converges_to_root_for_cosine_f(algorithm):
    params = FixedPointParams(f=x - sp.cos(x), p0=0.5, a=0.0, b=1.0, err=1e-8)
    result = algorithm.run(params)

    assert result.converged is True
    assert math.isclose(result.root, 0.7390851332151607, abs_tol=1e-6)


def test_execute_converges_to_root_for_sqrt_f(algorithm):
    params = FixedPointParams(f=x - sp.sqrt(x + 2), p0=1.0, a=0.0, b=2.0, err=1e-8)
    result = algorithm.run(params)

    assert result.converged is True
    assert math.isclose(result.root, 2, abs_tol=1e-6)


def test_execute_does_not_converge_with_too_few_iterations(algorithm):
    params = FixedPointParams(f=sp.Rational(2, 3) * x - 2, p0=1.0, a=0.0, b=5.0, err=1e-8, max_iter=1)
    result = algorithm.run(params)

    assert result.converged is False
    assert result.iterations == 1

def test_validate_accepts_float_interval_bounds_matching_cli_input(algorithm):
    params = FixedPointParams(f="x - cos(x)", p0=0.5, a=0.0, b=1.0, err=1e-8)
    algorithm._validate(params)
