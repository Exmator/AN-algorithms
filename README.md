# AN Algorithms

Numerical methods for a college Numerical Analysis course, built around a common interface so each method validates its own assumptions and exposes a consistent way to run it.

## What This Does

The project implements approximation algorithms for mathematically hard problems (root-finding, and more to come as the course progresses). Every algorithm shares the same contract — validate its input assumptions, then execute — but each defines its own parameters and result type, so adding a new method never requires touching existing code.

Functions are entered as plain math expressions (e.g. `x**2 - 2`), parsed with `sympy`. Before running, each algorithm symbolically verifies the actual mathematical hypotheses it depends on — not just type/range checks — and reports which one fails in plain terms.

## Project Structure

```
AN-algorithms/
├── main.py                       # CLI entry point
├── algorithm_interface.py        # Parameters, Result and Algorithm base classes
├── registry.py                   # AlgorithmRegistry: name -> algorithm/params lookup
├── symbolic.py                   # shared sympy helpers used by every algorithm
├── algorithms/
│   ├── __init__.py               # auto-discovers and registers every algorithm module
│   ├── bisection.py              # Bisection method implementation
│   └── fixed_point.py            # Fixed Point method implementation
└── tests/
    ├── test_bisection.py
    └── test_fixed_point.py
```

### How it fits together

- **`Algorithm`** (`algorithm_interface.py`) is an abstract base class with `_validate` and `_execute`. `run()` calls both, catching any validation/execution error.
- **`Parameters`** / **`Result`** are marker base classes. Each algorithm defines its own dataclass for input parameters and for its result (e.g. `BisectionParams`, `BisectionResult`), so every method can require exactly the inputs it needs.
- **`AlgorithmRegistry`** (`registry.py`) lets an algorithm register itself with `@AlgorithmRegistry.register(name=..., params_cls=..., description=...)`. `algorithms/__init__.py` auto-imports every module in the `algorithms/` package, so the registry is fully populated just by dropping a new file there.
- **`symbolic.py`** is the shared sympy layer every algorithm's `_validate`/`_execute` builds on: parsing/validating a user-given expression (`parse_univariate_expression`), converting float bounds to exact `Rational`s (`exact_bounds` — sympy's containment checks on raw `Float`-bounded intervals can be undecidable due to limited precision, which this sidesteps), verifying continuity on an interval with a clear error when sympy can't decide (`require_continuous`), computing an expression's range over an interval (`function_range`), and turning an expression into a plain numeric function for the iteration loop (`to_numeric_function`).
- **`algorithm_interface.py`**, **`registry.py`** and **`symbolic.py`** live at the project root, not inside `algorithms/`, because they're infrastructure every algorithm depends on — `algorithms/` holds only the algorithms themselves.
- **`main.py`** only depends on these abstractions — it lists whatever is registered, builds the right `Parameters` via introspection (prompting for an expression whenever a field is typed `ExpressionType`, i.e. `sp.Expr | str`), and runs it. It never needs to change when a new algorithm is added. It `import`s `algorithms` once, for the side effect of triggering registration.

### Adding a new algorithm

1. Create `algorithms/<name>.py`.
2. Define a `Parameters` dataclass, a `Result` dataclass (with a `__str__` for readable output), and an `Algorithm` subclass implementing `_validate`/`_execute`.
3. Decorate the class with `@AlgorithmRegistry.register(...)`.
4. If the algorithm takes a function as input, type that field as `symbolic.ExpressionType` and reuse `symbolic.py`'s helpers to parse it and verify whatever mathematical hypotheses the method relies on.

That's it — no other file needs to be touched; the CLI and tests can pick it up on their own.

## Getting Started

Requires Python 3.10+.

```bash
pip install pytest sympy
```

### Run the CLI

```bash
python main.py
```

You'll get a menu of every registered algorithm. Pick one, fill in its parameters (functions are entered as an expression in `x`, e.g. `x**2 - 2`), and the result prints back. If the expression doesn't satisfy the method's hypotheses on `[a, b]` (not continuous, wrong sign, not a contraction, etc.), you'll get a specific error. Choose `0` to exit.

### Run the tests

```bash
python -m pytest
```

`pyproject.toml` pins the pytest rootdir, so this works from anywhere inside the repo.
