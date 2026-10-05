"""Solve a linear system A x = b using LU factorization and substitutions."""

from dataclasses import dataclass
import math
from typing import Sequence

from algorithm_interface import Algorithm, Parameters, Result
from algorithms.back_substitution import BackSubstitutionAlgorithm, BackSubstitutionParams
from algorithms.forward_substitution import (
    ForwardSubstitutionAlgorithm,
    ForwardSubstitutionParams,
)
from algorithms.lu_factorization import LUFactorizationAlgorithm, LUFactorizationParams
from registry import AlgorithmRegistry


@dataclass
class LUSolveParams(Parameters):
    """Matrix and right-hand side for the system ``A x = b``.

    In the CLI, enter rows separated by semicolons and entries by commas,
    for example: ``4,3;6,3``. Enter ``b`` as comma-separated entries.
    """

    A: Sequence[Sequence[float]] | str
    b: Sequence[float] | str


@dataclass
class LUSolveResult(Result):
    x: list[float]

    def __str__(self) -> str:
        values = ", ".join(f"{value:.6g}" for value in self.x)
        return f"Unique solution: x = ({values})"


def _parse_vector(value: Sequence[float] | str) -> list[float]:
    if isinstance(value, str):
        try:
            vector = [float(item.strip()) for item in value.split(",")]
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "b must be a numeric vector; enter its entries separated by ',' "
                "(for example: 5,6)"
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
    name="Solve System with LU",
    params_cls=LUSolveParams,
    description="Solve A x = b by LU factorization and forward/back substitution",
)
class LUSolveAlgorithm(Algorithm):
    def _validate(self, p: LUSolveParams) -> None:
        if not isinstance(p, LUSolveParams):
            raise TypeError("parameters must be a LUSolveParams instance")

        # The LU factorization validates that A is a finite, square matrix.
        LUFactorizationAlgorithm()._validate(LUFactorizationParams(A=p.A))
        vector = _parse_vector(p.b)
        n = len(p.A.split(";")) if isinstance(p.A, str) else len(p.A)
        if len(vector) != n:
            raise ValueError("b must have the same length as the number of rows in A")

    def _execute(self, p: LUSolveParams) -> LUSolveResult:
        vector = _parse_vector(p.b)

        try:
            factors = LUFactorizationAlgorithm().run(LUFactorizationParams(A=p.A))
            intermediate = ForwardSubstitutionAlgorithm().run(
                ForwardSubstitutionParams(A=factors.L, b=vector)
            )
            solution = BackSubstitutionAlgorithm().run(
                BackSubstitutionParams(A=factors.U, b=intermediate.x)
            )
        except ValueError as exc:
            raise ValueError(
                "Could not find a unique solution using LU without pivoting: "
                f"{exc}"
            ) from exc

        return LUSolveResult(x=solution.x)
