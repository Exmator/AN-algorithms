from dataclasses import dataclass

import sympy as sp

from algorithms.algorithm_interface import Parameters, Result, Algorithm
from algorithms.registry import AlgorithmRegistry

@dataclass
class FixedPointParams(Parameters):
    f: sp.Expr | str
    p0: float
    a: float
    b: float
    err: float = 1e-6
    max_iter: int = 100

@dataclass
class FixedPointResult(Result):
    root: float
    iterations: int
    converged: bool

    def __str__(self) -> str:
        if self.converged:
            return f"Root found: {self.root:.6f} (after {self.iterations} iterations)"
        return f"Did not converge after {self.iterations} iterations (last estimate: {self.root:.6f})"

@AlgorithmRegistry.register(
    name="Fixed Point",
    params_cls=FixedPointParams,
    description="Fixed Point method to find the root of f in [a, b]",
)
class FixedPointAlgorithm(Algorithm):
    def _validate(self, p: FixedPointParams) -> None:
        if not isinstance(p, FixedPointParams):
            raise TypeError("parameters must be a FixedPointParams instance")
        if p.a >= p.b:
            raise ValueError("a must be less than b")
        if not (p.a <= p.p0 <= p.b):
            raise ValueError("p0 must lie within [a, b]")
        if p.err <= 0:
            raise ValueError("err must be positive")
        if p.max_iter <= 0:
            raise ValueError("max_iter must be positive")

        try:
            f = sp.sympify(p.f)
        except (sp.SympifyError, TypeError) as exc:
            raise ValueError("f must be a valid sympy expression") from exc
        if f.free_symbols - {sp.Symbol("x")}:
            raise ValueError("f must only be expressed in terms of the variable x")

        g = sp.Symbol("x") - f

        # Use exact rationals instead of the raw floats: sympy's containment
        # checks on Float-bounded intervals can be undecidable (return None)
        # due to limited precision, which `not None` would misread as False.
        a_exact = sp.Rational(str(p.a))
        b_exact = sp.Rational(str(p.b))
        closed_interval = sp.Interval(a_exact, b_exact)
        open_interval = sp.Interval.open(a_exact, b_exact)

        # Existence: g continuous on [a, b] and g([a, b]) subseteq [a, b]
        try:
            domain = sp.calculus.util.continuous_domain(g, sp.Symbol("x"), closed_interval)
            if not closed_interval.is_subset(domain):
                raise ValueError("g is not continuous on [a, b]")

            g_range = sp.calculus.util.function_range(g, sp.Symbol("x"), closed_interval)
        except NotImplementedError as exc:
            raise ValueError("cannot determine whether g is continuous on [a, b] for the given f") from exc
        if not g_range.is_subset(closed_interval):
            raise ValueError(
                f"g([a, b]) = {g_range} is not contained in [a, b]: existence of a fixed point is not guaranteed"
            )

        # Uniqueness/convergence: g differentiable on (a, b) with sup|g'| < 1
        dg = sp.diff(g, sp.Symbol("x"))
        try:
            dg_domain = sp.calculus.util.continuous_domain(dg, sp.Symbol("x"), open_interval)
            if not open_interval.is_subset(dg_domain):
                raise ValueError("g is not differentiable on (a, b)")

            dg_range = sp.calculus.util.function_range(dg, sp.Symbol("x"), open_interval)
        except NotImplementedError as exc:
            raise ValueError("cannot determine whether g is differentiable on (a, b) for the given f") from exc
        k = sp.Max(sp.Abs(dg_range.inf), sp.Abs(dg_range.sup))
        if k >= 1:
            raise ValueError(
                f"g is not a contraction on (a, b): sup|g'(x)| = {k} >= 1, "
                "uniqueness of the fixed point is not guaranteed"
            )

    def _execute(self, p: FixedPointParams) -> FixedPointResult:
        f = sp.sympify(p.f)
        g = sp.Symbol("x") - f
        g_func = sp.lambdify(sp.Symbol("x"), g, "math")

        p_curr = p.p0
        for i in range(p.max_iter):
            p_next = g_func(p_curr)
            if abs(p_next - p_curr) < p.err:
                return FixedPointResult(root=p_next, iterations=i + 1, converged=True)
            p_curr = p_next

        return FixedPointResult(root=p_curr, iterations=p.max_iter, converged=False)
