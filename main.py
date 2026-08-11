import dataclasses
import math

from algorithms.registry import AlgorithmEntry, AlgorithmRegistry


def _prompt_field(field: dataclasses.Field):
    has_default = field.default is not dataclasses.MISSING
    label = field.name if not has_default else f"{field.name} [{field.default}]"

    if field.type is callable:
        raw = input(f"{label} (expression in x, e.g. x**2 - 2): ").strip()
        return eval(f"lambda x: {raw}", {"math": math})

    raw = input(f"{label}: ").strip()
    if raw == "" and has_default:
        return field.default

    if field.type is float:
        return float(raw)
    if field.type is int:
        return int(raw)
    return raw


def prompt_for_params(params_cls):
    kwargs = {f.name: _prompt_field(f) for f in dataclasses.fields(params_cls)}
    return params_cls(**kwargs)


def choose_algorithm(entries: list[AlgorithmEntry]) -> AlgorithmEntry | None:
    print("Available algorithms:")
    for i, entry in enumerate(entries, start=1):
        print(f"  {i}. {entry.name} - {entry.description}")
    print("  0. Exit")

    choice = int(input("Select an algorithm: "))
    if choice == 0:
        return None
    return entries[choice - 1]


def main():
    entries = AlgorithmRegistry.list()
    if not entries:
        print("No algorithms registered.")
        return

    while True:
        entry = choose_algorithm(entries)
        if entry is None:
            break

        params = prompt_for_params(entry.params_cls)
        algorithm = entry.algorithm_cls()
        result = algorithm.run(params)
        print(result)
        print()


if __name__ == "__main__":
    main()
