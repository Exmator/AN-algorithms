from dataclasses import dataclass

import sympy as sp

from algorithm_interface import Parameters, Result, Algorithm
from registry import AlgorithmRegistry
from symbolic import (
    ExpressionType,
    X,
    parse_univariate_expression,
    require_continuous,
    to_numeric_function,
)

@dataclass
class NewtonParams(Parameters):
    f: ExpressionType
    x: float
    err: float = 1e-6
    max_iter: int = 100

@dataclass
class NewtonResult(Result):
    root: float
    iterations: int
    converged: bool

    def __str__(self) -> str:
        if self.converged:
            return f"Root found: {self.root:.6f} (after {self.iterations} iterations)"
        return f"Did not converge after {self.iterations} iterations (last estimate: {self.root:.6f})"

@AlgorithmRegistry.register(
    name="Newton",
    params_cls=NewtonParams,
    description="Newton-Raphson method to find the root of f from x",
)
class NewtonAlgorithm(Algorithm):
    def _validate(self, p: NewtonParams) -> None:
        if not isinstance(p, NewtonParams):
            raise TypeError("parameters must be a NewtonParams instance")
        if p.err <= 0:
            raise ValueError("err must be positive")
        if p.max_iter <= 0:
            raise ValueError("max_iter must be positive")

        f = parse_univariate_expression(p.f)

        # Newton's method hypothesis: f continuous and differentiable on the real line
        require_continuous(f, sp.S.Reals, subject="f", interval_label="the real line")
        df = sp.diff(f, X)
        require_continuous(
            df, sp.S.Reals, subject="f", interval_label="the real line", property_name="differentiable"
        )

    def _execute(self, p: NewtonParams) -> NewtonResult:
        f_expr = parse_univariate_expression(p.f)
        f_func = to_numeric_function(f_expr)
        f_func_der = to_numeric_function(sp.diff(f_expr, X))

        x = p.x

        for i in range(p.max_iter):
            derivative = f_func_der(x)
            if derivative == 0:
                raise ValueError(
                    f"f'(x) = 0 at x={x}; Newton's method cannot continue, try a different starting x"
                )
            x = x - (f_func(x) / derivative)
            if abs(f_func(x)) < p.err:
                return NewtonResult(root=x, iterations=i + 1, converged=True)
        return NewtonResult(root=x, iterations=p.max_iter, converged=False)
