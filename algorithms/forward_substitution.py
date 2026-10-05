"""Forward substitution for lower-triangular linear systems."""

from dataclasses import dataclass
import math
from typing import Sequence

from algorithm_interface import Algorithm, Parameters, Result
from registry import AlgorithmRegistry


@dataclass
class ForwardSubstitutionParams(Parameters):
    """Input matrix and right-hand side for ``A x = b``.

    In the CLI, enter rows separated by semicolons and entries by commas,
    for example: ``2,0;1,3``.
    """

    A: Sequence[Sequence[float]] | str
    b: Sequence[float] | str


@dataclass
class ForwardSubstitutionResult(Result):
    x: list[float]

    def __str__(self) -> str:
        values = ", ".join(f"{value:.6g}" for value in self.x)
        return f"Solution: x = ({values})"


def _parse_matrix(value: Sequence[Sequence[float]] | str) -> list[list[float]]:
    if isinstance(value, str):
        try:
            rows = [row.strip() for row in value.split(";")]
            if not rows or any(not row for row in rows):
                raise ValueError
            matrix = [[float(item.strip()) for item in row.split(",")] for row in rows]
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "A must be a numeric matrix; enter rows separated by ';' "
                "and entries by ',' (for example: 2,0;1,3)"
            ) from exc
    else:
        try:
            matrix = [[float(item) for item in row] for row in value]
        except (TypeError, ValueError) as exc:
            raise ValueError("A must be a numeric matrix") from exc

    if not matrix or any(not row for row in matrix):
        raise ValueError("A must not be empty")
    if any(not math.isfinite(item) for row in matrix for item in row):
        raise ValueError("A must contain only finite numbers")
    if any(len(row) != len(matrix) for row in matrix):
        raise ValueError("A must be a square matrix")
    return matrix


def _parse_vector(value: Sequence[float] | str) -> list[float]:
    if isinstance(value, str):
        try:
            vector = [float(item.strip()) for item in value.split(",")]
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "b must be a numeric vector; enter its entries separated by ',' "
                "(for example: 4,7)"
            ) from exc
    else:
        try:
            vector = [float(item) for item in value]
        except (TypeError, ValueError) as exc:
            raise ValueError("b must be a numeric vector") from exc

    if not vector:
        raise ValueError("b must not be empty")
    if any(not math.isfinite(item) for item in vector):
        raise ValueError("b must contain only finite numbers")
    return vector


@AlgorithmRegistry.register(
    name="Forward Substitution",
    params_cls=ForwardSubstitutionParams,
    description="Solve A x = b when A is lower triangular",
)
class ForwardSubstitutionAlgorithm(Algorithm):
    def _validate(self, p: ForwardSubstitutionParams) -> None:
        if not isinstance(p, ForwardSubstitutionParams):
            raise TypeError("parameters must be a ForwardSubstitutionParams instance")

        matrix = _parse_matrix(p.A)
        vector = _parse_vector(p.b)
        n = len(matrix)

        if len(vector) != n:
            raise ValueError("b must have the same length as the number of rows in A")
        for i in range(n):
            if matrix[i][i] == 0:
                raise ValueError(f"A must have nonzero diagonal entries; A[{i},{i}] is zero")
            if any(matrix[i][j] != 0 for j in range(i + 1, n)):
                raise ValueError("A must be lower triangular")

    def _execute(self, p: ForwardSubstitutionParams) -> ForwardSubstitutionResult:
        matrix = _parse_matrix(p.A)
        vector = _parse_vector(p.b)
        n = len(matrix)
        solution = [0.0] * n

        for i in range(n):
            known_terms = sum(matrix[i][j] * solution[j] for j in range(i))
            solution[i] = (vector[i] - known_terms) / matrix[i][i]

        return ForwardSubstitutionResult(x=solution)
