import math

import pytest

from algorithms.back_substitution import (
    BackSubstitutionAlgorithm,
    BackSubstitutionParams,
)


@pytest.fixture
def algorithm():
    return BackSubstitutionAlgorithm()


def test_validate_raises_on_wrong_parameters_type(algorithm):
    with pytest.raises(TypeError, match="BackSubstitutionParams"):
        algorithm._validate(object())


def test_validate_raises_when_matrix_is_not_square(algorithm):
    params = BackSubstitutionParams(A=[[1, 2, 3], [0, 4, 5]], b=[6, 7])

    with pytest.raises(ValueError, match="square matrix"):
        algorithm.run(params)


def test_validate_raises_when_b_has_wrong_length(algorithm):
    params = BackSubstitutionParams(A=[[2, 1], [0, 3]], b=[5])

    with pytest.raises(ValueError, match="same length"):
        algorithm.run(params)


def test_validate_raises_when_matrix_is_not_upper_triangular(algorithm):
    params = BackSubstitutionParams(A=[[2, 1], [1, 3]], b=[5, 6])

    with pytest.raises(ValueError, match="upper triangular"):
        algorithm.run(params)


def test_validate_raises_when_diagonal_contains_zero(algorithm):
    params = BackSubstitutionParams(A=[[2, 1], [0, 0]], b=[5, 6])

    with pytest.raises(ValueError, match="nonzero diagonal"):
        algorithm.run(params)


def test_execute_solves_two_by_two_system(algorithm):
    params = BackSubstitutionParams(A=[[2, 1], [0, 3]], b=[5, 6])

    result = algorithm.run(params)

    assert math.isclose(result.x[0], 1.5)
    assert math.isclose(result.x[1], 2.0)


def test_execute_solves_three_by_three_system(algorithm):
    params = BackSubstitutionParams(
        A=[[1, 2, 1], [0, 1, 1], [0, 0, 2]],
        b=[8, 5, 4],
    )

    result = algorithm.run(params)

    assert result.x == pytest.approx([0, 3, 2])


def test_execute_accepts_matrix_and_vector_as_cli_strings(algorithm):
    params = BackSubstitutionParams(A="2,1;0,3", b="5,6")

    result = algorithm.run(params)

    assert result.x == pytest.approx([1.5, 2.0])


def test_result_has_readable_string(algorithm):
    params = BackSubstitutionParams(A=[[2, 1], [0, 3]], b=[5, 6])

    result = algorithm.run(params)

    assert str(result) == "Solution: x = (1.5, 2)"
