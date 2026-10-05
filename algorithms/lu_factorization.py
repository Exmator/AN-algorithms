"""LU factorization without pivoting (Doolittle/Gaussian elimination)."""

from dataclasses import dataclass
import math
from typing import Sequence

from algorithm_interface import Algorithm, Parameters, Result
from registry import AlgorithmRegistry


@dataclass
class LUFactorizationParams(Parameters):
    """Square matrix to factor as ``A = L U``.

    In the CLI, enter rows separated by semicolons and entries by commas,
    for example: ``4,3;6,3``.
    """

    A: Sequence[Sequence[float]] | str


@dataclass
class LUFactorizationResult(Result):
    L: list[list[float]]
    U: list[list[float]]

    def __str__(self) -> str:
        def format_matrix(matrix: list[list[float]]) -> str:
            return "\n".join(
                "[" + ", ".join(f"{value:.6g}" for value in row) + "]"
                for row in matrix
            )

        return f"L (lower triangular):\n{format_matrix(self.L)}\nU (upper triangular):\n{format_matrix(self.U)}"


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
                "and entries by ',' (for example: 4,3;6,3)"
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


@AlgorithmRegistry.register(
    name="LU Factorization",
    params_cls=LUFactorizationParams,
    description="Factor A into a lower-triangular L and upper-triangular U (without pivoting)",
)
class LUFactorizationAlgorithm(Algorithm):
    def _validate(self, p: LUFactorizationParams) -> None:
        if not isinstance(p, LUFactorizationParams):
            raise TypeError("parameters must be a LUFactorizationParams instance")
        _parse_matrix(p.A)

    def _execute(self, p: LUFactorizationParams) -> LUFactorizationResult:
        matrix = _parse_matrix(p.A)
        n = len(matrix)
        lower = [[0.0] * n for _ in range(n)]
        upper = [row[:] for row in matrix]

        for i in range(n):
            lower[i][i] = 1.0

        for i in range(n):
            pivot = upper[i][i]
            if pivot == 0:
                raise ValueError(
                    f"Cannot factor A = L U without pivoting: zero pivot at A[{i},{i}]"
                )

            for j in range(i + 1, n):
                multiplier = upper[j][i] / pivot
                lower[j][i] = multiplier
                upper[j][i] = 0.0
                for k in range(i + 1, n):
                    upper[j][k] -= multiplier * upper[i][k]

        return LUFactorizationResult(L=lower, U=upper)
