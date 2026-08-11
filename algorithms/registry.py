from dataclasses import dataclass
from typing import Type

from algorithms.algorithm_interface import Algorithm, Parameters


@dataclass(frozen=True)
class AlgorithmEntry:
    name: str
    algorithm_cls: Type[Algorithm]
    params_cls: Type[Parameters]
    description: str = ""


class AlgorithmRegistry:
    _entries: dict[str, AlgorithmEntry] = {}

    @classmethod
    def register(cls, name: str, params_cls: Type[Parameters], description: str = ""):
        def decorator(algorithm_cls: Type[Algorithm]) -> Type[Algorithm]:
            if name in cls._entries:
                raise ValueError(f"an algorithm named '{name}' is already registered")
            cls._entries[name] = AlgorithmEntry(name, algorithm_cls, params_cls, description)
            return algorithm_cls
        return decorator

    @classmethod
    def get(cls, name: str) -> AlgorithmEntry:
        return cls._entries[name]

    @classmethod
    def list(cls) -> list[AlgorithmEntry]:
        return list(cls._entries.values())
