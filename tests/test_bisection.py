import math

import pytest

from algorithms.bisection import BisectionAlgorithm, BisectionParams


@pytest.fixture
def algorithm():
    return BisectionAlgorithm()


def test_validate_raises_on_wrong_parameters_type(algorithm):
    with pytest.raises(TypeError, match="BisectionParams"):
        algorithm._validate(object())


def test_validate_raises_when_f_is_not_callable(algorithm):
    params = BisectionParams(f="not callable", a=0, b=1)
    with pytest.raises(ValueError, match="f must be a function"):
        algorithm._validate(params)


def test_validate_raises_when_a_is_not_less_than_b(algorithm):
    params = BisectionParams(f=lambda x: x, a=1, b=1)
    with pytest.raises(ValueError, match="a must be less than b"):
        algorithm._validate(params)


def test_validate_raises_when_tol_is_not_positive(algorithm):
    params = BisectionParams(f=lambda x: x - 0.5, a=0, b=1, tol=0)
    with pytest.raises(ValueError, match="tol must be positive"):
        algorithm._validate(params)


def test_validate_raises_when_max_iter_is_not_positive(algorithm):
    params = BisectionParams(f=lambda x: x - 0.5, a=0, b=1, max_iter=0)
    with pytest.raises(ValueError, match="max_iter must be positive"):
        algorithm._validate(params)


def test_validate_raises_when_signs_do_not_change(algorithm):
    params = BisectionParams(f=lambda x: x**2 + 1, a=-1, b=1)
    with pytest.raises(ValueError, match="opposite signs"):
        algorithm._validate(params)


def test_execute_does_not_converge_with_too_few_iterations(algorithm):
    params = BisectionParams(f=lambda x: x**2 - 2, a=0, b=2, tol=1e-12, max_iter=1)
    result = algorithm.run(params)

    assert result.converged is False
    assert result.iterations == 1


def test_execute_converges_fast_with_a_loose_tolerance(algorithm):
    params = BisectionParams(f=lambda x: x**2 - 2, a=0, b=2, tol=0.5, max_iter=100)
    result = algorithm.run(params)

    assert result.converged is True
    assert result.iterations < 5


def test_execute_does_not_converge_with_an_unreachable_tolerance(algorithm):
    params = BisectionParams(f=lambda x: x**2 - 2, a=0, b=2, tol=1e-300, max_iter=100)
    result = algorithm.run(params)

    assert result.converged is False
    assert result.iterations == 100


def test_execute_converges_to_root(algorithm):
    params = BisectionParams(f=lambda x: x**2 - 2, a=0, b=2, tol=1e-8, max_iter=100)
    result = algorithm.run(params)

    assert result.converged is True
    assert math.isclose(result.root, math.sqrt(2), abs_tol=1e-6)
