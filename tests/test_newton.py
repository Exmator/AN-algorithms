import math

import pytest
import sympy as sp

from algorithms.newton import NewtonAlgorithm, NewtonParams

x = sp.Symbol("x")
y = sp.Symbol("y")


@pytest.fixture
def algorithm():
    return NewtonAlgorithm()


def test_validate_raises_on_wrong_parameters_type(algorithm):
    with pytest.raises(TypeError, match="NewtonParams"):
        algorithm._validate(object())


def test_validate_raises_when_err_is_not_positive(algorithm):
    params = NewtonParams(f=x - 0.5, x=1, err=0)
    with pytest.raises(ValueError, match="err must be positive"):
        algorithm._validate(params)


def test_validate_raises_when_max_iter_is_not_positive(algorithm):
    params = NewtonParams(f=x - 0.5, x=1, max_iter=0)
    with pytest.raises(ValueError, match="max_iter must be positive"):
        algorithm._validate(params)


def test_validate_raises_when_f_is_not_a_valid_expression(algorithm):
    params = NewtonParams(f="not a math expr $$", x=1)
    with pytest.raises(ValueError, match="f must be a valid sympy expression"):
        algorithm._validate(params)


def test_validate_raises_when_f_uses_a_variable_other_than_x(algorithm):
    params = NewtonParams(f=y - 0.5, x=1)
    with pytest.raises(ValueError, match="only be expressed in terms of the variable x"):
        algorithm._validate(params)


def test_validate_raises_when_f_is_not_continuous(algorithm):
    params = NewtonParams(f=1 / (x - 1), x=2)
    with pytest.raises(ValueError, match="not continuous"):
        algorithm._validate(params)


def test_validate_raises_when_f_is_not_differentiable(algorithm):
    params = NewtonParams(f=sp.Abs(x), x=1)
    with pytest.raises(ValueError, match="differentiable"):
        algorithm._validate(params)


def test_execute_does_not_converge_with_too_few_iterations(algorithm):
    params = NewtonParams(f=x**2 - 2, x=1.0, err=1e-12, max_iter=1)
    result = algorithm.run(params)

    assert result.converged is False
    assert result.iterations == 1


def test_execute_converges_to_root_for_polynomial_f(algorithm):
    params = NewtonParams(f=x**2 - 2, x=1.0, err=1e-8, max_iter=100)
    result = algorithm.run(params)

    assert result.converged is True
    assert math.isclose(result.root, math.sqrt(2), abs_tol=1e-6)


def test_execute_converges_to_root_for_transcendental_f(algorithm):
    params = NewtonParams(f=sp.cos(x) - x, x=0.5, err=1e-8, max_iter=100)
    result = algorithm.run(params)

    assert result.converged is True
    assert math.isclose(result.root, 0.7390851332151607, abs_tol=1e-6)


def test_execute_converges_to_root_for_exponential_f(algorithm):
    params = NewtonParams(f=sp.exp(x) - 3 * x, x=1.0, err=1e-8, max_iter=100)
    result = algorithm.run(params)

    assert result.converged is True
    assert math.isclose(result.root, 0.6190612867359450, abs_tol=1e-6)


def test_execute_raises_when_derivative_is_zero(algorithm):
    params = NewtonParams(f=x**2 - 2, x=0.0, err=1e-8, max_iter=100)
    with pytest.raises(ValueError, match="f'\\(x\\) = 0"):
        algorithm.run(params)
