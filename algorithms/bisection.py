from dataclasses import dataclass

from algorithms.algorithm_interface import Parameters, Result, Algorithm
from algorithms.registry import AlgorithmRegistry

@dataclass
class BisectionParams(Parameters):
    f: callable
    a: float
    b: float
    tol: float = 1e-6
    max_iter: int = 100

@dataclass
class BisectionResult(Result):
    root: float
    iterations: int
    converged: bool

    def __str__(self) -> str:
        if self.converged:
            return f"Root found: {self.root:.6f} (after {self.iterations} iterations)"
        return f"Did not converge after {self.iterations} iterations (last estimate: {self.root:.6f})"

@AlgorithmRegistry.register(
    name="Bisection",
    params_cls=BisectionParams,
    description="Bisection method to find the root of f in [a, b]",
)
class BisectionAlgorithm(Algorithm):
    def _validate(self, p: BisectionParams) -> None:
        if not isinstance(p, BisectionParams):
            raise TypeError("parameters must be a BisectionParams instance")
        if not callable(p.f):
            raise ValueError("f must be a function")
        if p.a >= p.b:
            raise ValueError("a must be less than b")
        if p.tol <= 0:
            raise ValueError("tol must be positive")
        if p.max_iter <= 0:
            raise ValueError("max_iter must be positive")
        if p.f(p.a) * p.f(p.b) >= 0:
            raise ValueError("f(a) and f(b) must have opposite signs")

    def _execute(self, p: BisectionParams) -> BisectionResult:
        a = p.a
        b = p.b
        Fa = p.f(a)

        for i in range(p.max_iter):
            mean = (a + b) / 2
            Fp = p.f(mean)
            if abs(Fp) < p.tol:
                return BisectionResult(root=mean, iterations=i + 1, converged=True)
            if Fa * Fp > 0:
                a = mean
                Fa = Fp
            else:
                b = mean

        return BisectionResult(root=mean, iterations=p.max_iter, converged=False)