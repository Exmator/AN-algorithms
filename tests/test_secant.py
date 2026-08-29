import math

import pytest
import sympy as sp

from algorithms.secant import SecantAlgorithm, SecantParams

x = sp.Symbol("x")
y = sp.Symbol("y")


@pytest.fixture
def algorithm():
    return SecantAlgorithm()


def test_validate_raises_on_wrong_parameters_type(algorithm):
    with pytest.raises(TypeError, match="SecantParams"):
        algorithm._validate(object())


def test_validate_raises_when_err_is_not_positive(algorithm):
    params = SecantParams(f=x - 0.5, a=0, b=1, err=0)
    with pytest.raises(ValueError, match="err must be positive"):
        algorithm._validate(params)


def test_validate_raises_when_max_iter_is_not_positive(algorithm):
    params = SecantParams(f=x - 0.5, a=0, b=1, max_iter=0)
    with pytest.raises(ValueError, match="max_iter must be positive"):
        algorithm._validate(params)


def test_validate_raises_when_a_equals_b(algorithm):
    params = SecantParams(f=x - 0.5, a=1, b=1)
    with pytest.raises(ValueError, match="a must not be the same as b"):
        algorithm._validate(params)


def test_validate_raises_when_f_is_not_a_valid_expression(algorithm):
    params = SecantParams(f="not a math expr $$", a=0, b=1)
    with pytest.raises(ValueError, match="f must be a valid sympy expression"):
        algorithm._validate(params)


def test_validate_raises_when_f_uses_a_variable_other_than_x(algorithm):
    params = SecantParams(f=y - 0.5, a=0, b=1)
    with pytest.raises(ValueError, match="only be expressed in terms of the variable x"):
        algorithm._validate(params)


def test_validate_raises_when_f_is_not_continuous(algorithm):
    params = SecantParams(f=1 / (x - 1), a=0, b=2)
    with pytest.raises(ValueError, match="not continuous"):
        algorithm._validate(params)


def test_execute_does_not_converge_with_too_few_iterations(algorithm):
    params = SecantParams(f=x**2 - 2, a=0.0, b=2.0, err=1e-12, max_iter=1)
    result = algorithm.run(params)

    assert result.converged is False
    assert result.iterations == 1


def test_execute_raises_when_f_a_equals_f_b(algorithm):
    params = SecantParams(f=x**2 - 2, a=-1.0, b=1.0, err=1e-8, max_iter=100)
    with pytest.raises(ValueError, match="cannot continue"):
        algorithm.run(params)


def test_execute_converges_to_root_for_polynomial_f(algorithm):
    params = SecantParams(f=x**2 - 2, a=0.0, b=2.0, err=1e-8, max_iter=100)
    result = algorithm.run(params)

    assert result.converged is True
    assert math.isclose(result.root, math.sqrt(2), abs_tol=1e-6)


def test_execute_converges_to_root_for_transcendental_f(algorithm):
    params = SecantParams(f=sp.cos(x) - x, a=0.0, b=1.0, err=1e-8, max_iter=100)
    result = algorithm.run(params)

    assert result.converged is True
    assert math.isclose(result.root, 0.7390851332151607, abs_tol=1e-6)


def test_execute_converges_to_root_for_exponential_f(algorithm):
    params = SecantParams(f=sp.exp(x) - 3 * x, a=0.0, b=1.0, err=1e-8, max_iter=100)
    result = algorithm.run(params)

    assert result.converged is True
    assert math.isclose(result.root, 0.6190612867359450, abs_tol=1e-6)


def test_execute_converges_even_when_f_a_and_f_b_have_the_same_sign(algorithm):
    params = SecantParams(f=x**2 - 2, a=1.5, b=2.0, err=1e-8, max_iter=100)
    assert math.copysign(1, 1.5**2 - 2) == math.copysign(1, 2.0**2 - 2)

    result = algorithm.run(params)

    assert result.converged is True
    assert math.isclose(result.root, math.sqrt(2), abs_tol=1e-6)
