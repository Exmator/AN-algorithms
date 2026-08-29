from dataclasses import dataclass

import sympy as sp

from algorithm_interface import Parameters, Result, Algorithm
from registry import AlgorithmRegistry
from symbolic import (
    ExpressionType,
    parse_univariate_expression,
    require_continuous,
    to_numeric_function,
)

@dataclass
class SecantParams(Parameters):
    f: ExpressionType
    a: float
    b: float
    err: float = 1e-6
    max_iter: int = 100

@dataclass
class SecantResult(Result):
    root: float
    iterations: int
    converged: bool

    def __str__(self) -> str:
        if self.converged:
            return f"Root found: {self.root:.6f} (after {self.iterations} iterations)"
        return f"Did not converge after {self.iterations} iterations (last estimate: {self.root:.6f})"

@AlgorithmRegistry.register(
    name="Secant",
    params_cls=SecantParams,
    description="Secant method to find the root of f from a and b",
)
class SecantAlgorithm(Algorithm):
    def _validate(self, p: SecantParams) -> None:
        if not isinstance(p, SecantParams):
            raise TypeError("parameters must be a SecantParams instance")
        if p.err <= 0:
            raise ValueError("err must be positive")
        if p.max_iter <= 0:
            raise ValueError("max_iter must be positive")
        if p.a == p.b:
            raise ValueError("a must not be the same as b")

        # Secant method hypothesis: f continuous on the real line
        f = parse_univariate_expression(p.f)
        require_continuous(f, sp.S.Reals, subject="f", interval_label="the real line")

    def _execute(self, p: SecantParams) -> SecantResult:
        f_func = to_numeric_function(parse_univariate_expression(p.f))

        a = p.a
        b = p.b

        for i in range(p.max_iter):
            Fa = f_func(a)
            Fb = f_func(b)
            if Fa == Fb:
                raise ValueError(f"f({a}) = f({b}) = {Fa}; Secant's method cannot continue, try a different starting point a,b")
            x = (a*Fb-b*Fa)/(Fb-Fa)
            if abs(f_func(x)) < p.err:
                return SecantResult(root=x, iterations=i + 1, converged=True)
            a = b
            b = x
        return SecantResult(root=x, iterations=p.max_iter, converged=False)