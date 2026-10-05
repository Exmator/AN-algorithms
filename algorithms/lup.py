"""Solve a linear system with LU factorization and partial pivoting."""

from dataclasses import dataclass
import math
from typing import Sequence

from algorithm_interface import Algorithm, Parameters, Result
from registry import AlgorithmRegistry


@dataclass
class LUPParams(Parameters):
    """Matrix and right-hand side for ``A x = b``.

    In the CLI, enter matrix rows separated by semicolons and entries by
    commas (for example, ``0,2;1,3``). Enter ``b`` as comma-separated values.
    """

    A: Sequence[Sequence[float]] | str
    b: Sequence[float] | str


@dataclass
class LUPResult(Result):
    x: list[float]
    P: list[int]
    L: list[list[float]]
    U: list[list[float]]

    def __str__(self) -> str:
        def format_matrix(matrix: list[list[float]]) -> str:
            return "\n".join(
                "[" + ", ".join(f"{value:.6g}" for value in row) + "]"
                for row in matrix
            )

        values = ", ".join(f"{value:.6g}" for value in self.x)
        return (
            f"Solution: x = ({values})\n"
            f"Permutation vector (1-based): {self.P}\n"
            f"L:\n{format_matrix(self.L)}\n"
            f"U:\n{format_matrix(self.U)}"
        )


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
                "and entries by ',' (for example: 0,2;1,3)"
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
    name="LUP Factorization",
    params_cls=LUPParams,
    description="Solve A x = b using LU factorization with partial pivoting",
)
class LUPAlgorithm(Algorithm):
    def _validate(self, p: LUPParams) -> None:
        if not isinstance(p, LUPParams):
            raise TypeError("parameters must be a LUPParams instance")
        matrix = _parse_matrix(p.A)
        vector = _parse_vector(p.b)
        if len(vector) != len(matrix):
            raise ValueError("b must have the same length as the number of rows in A")

    def _execute(self, p: LUPParams) -> LUPResult:
        matrix = _parse_matrix(p.A)
        rhs = _parse_vector(p.b)
        n = len(matrix)

        upper = [row[:] for row in matrix]
        lower = [[0.0] * n for _ in range(n)]
        permutation = list(range(n))
        for i in range(n):
            lower[i][i] = 1.0

        # Gaussian elimination with partial pivoting, producing P A = L U.
        for k in range(n):
            pivot_row = max(range(k, n), key=lambda i: abs(upper[i][k]))
            if upper[pivot_row][k] == 0:
                raise ValueError(
                    "A is singular; the system does not have a unique solution"
                )

            if pivot_row != k:
                upper[k], upper[pivot_row] = upper[pivot_row], upper[k]
                rhs[k], rhs[pivot_row] = rhs[pivot_row], rhs[k]
                permutation[k], permutation[pivot_row] = (
                    permutation[pivot_row],
                    permutation[k],
                )
                # Swap the already computed part of these rows in L as well.
                for j in range(k):
                    lower[k][j], lower[pivot_row][j] = (
                        lower[pivot_row][j],
                        lower[k][j],
                    )

            for i in range(k + 1, n):
                multiplier = upper[i][k] / upper[k][k]
                lower[i][k] = multiplier
                upper[i][k] = 0.0
                for j in range(k + 1, n):
                    upper[i][j] -= multiplier * upper[k][j]
                rhs[i] -= multiplier * rhs[k]

        # Back substitution on the upper-triangular system U x = rhs.
        solution = [0.0] * n
        for i in range(n - 1, -1, -1):
            remaining = sum(upper[i][j] * solution[j] for j in range(i + 1, n))
            solution[i] = (rhs[i] - remaining) / upper[i][i]

        return LUPResult(
            x=solution,
            P=[index + 1 for index in permutation],
            L=lower,
            U=upper,
        )
