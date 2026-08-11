# AN Algorithms

Numerical methods for a college Numerical Analysis course, built around a common interface so each method validates its own assumptions and exposes a consistent way to run it.

## What This Does

The project implements approximation algorithms for mathematically hard problems (root-finding, and more to come as the course progresses). Every algorithm shares the same contract — validate its input assumptions, then execute — but each defines its own parameters and result type, so adding a new method never requires touching existing code.

## Project Structure

```
AN-algorithms/
├── main.py                       # CLI entry point
├── algorithms/
│   ├── __init__.py               # auto-discovers and registers every algorithm module
│   ├── algorithm_interface.py    # Parameters, Result and Algorithm base classes
│   ├── registry.py               # AlgorithmRegistry: name -> algorithm/params lookup
│   └── bisection.py               # Bisection method implementation
└── tests/
    └── test_bisection.py
```

### How it fits together

- **`Algorithm`** (`algorithm_interface.py`) is an abstract base class with `_validate` and `_execute`. `run()` calls both, catching any validation/execution error.
- **`Parameters`** / **`Result`** are marker base classes. Each algorithm defines its own dataclass for input parameters and for its result (e.g. `BisectionParams`, `BisectionResult`), so every method can require exactly the inputs it needs.
- **`AlgorithmRegistry`** lets an algorithm register itself with `@AlgorithmRegistry.register(name=..., params_cls=..., description=...)`. `algorithms/__init__.py` auto-imports every module in the package, so the registry is fully populated just by dropping a new file in `algorithms/`.
- **`main.py`** only depends on these abstractions — it lists whatever is registered, builds the right `Parameters` via introspection, and runs it. It never needs to change when a new algorithm is added.

### Adding a new algorithm

1. Create `algorithms/<name>.py`.
2. Define a `Parameters` dataclass, a `Result` dataclass (with a `__str__` for readable output), and an `Algorithm` subclass implementing `_validate`/`_execute`.
3. Decorate the class with `@AlgorithmRegistry.register(...)`.

That's it — no other file needs to be touched; the CLI and tests can pick it up on their own.

## Getting Started

Requires Python 3.10+.

```bash
pip install pytest
```

### Run the CLI

```bash
python main.py
```

You'll get a menu of every registered algorithm. Pick one, fill in its parameters (functions are entered as an expression in `x`, e.g. `x**2 - 2`), and the result prints back. Choose `0` to exit.

### Run the tests

```bash
python -m pytest
```

`pyproject.toml` pins the pytest rootdir, so this works from anywhere inside the repo.
