"""
Generic strategy registry used by every module (ingestion, retrieval, generation).

Each module defines its own subclass of Registry (or just instantiates Registry directly)
and strategy implementations register themselves with a decorator, e.g.:

    parser_registry = Registry[Parser]("parser")

    @parser_registry.register("pdf_pymupdf")
    class PyMuPdfParser(Parser):
        ...

The UI and pipeline code then just does `parser_registry.get("pdf_pymupdf")()` without
needing to import each strategy module directly.
"""
from __future__ import annotations

from collections.abc import Callable


class StrategyNotFoundError(KeyError):
    pass


class Registry[T]:
    def __init__(self, kind: str):
        self.kind = kind
        self._strategies: dict[str, type[T]] = {}
        self._descriptions: dict[str, str] = {}

    def register(self, name: str, description: str = "") -> Callable[[type[T]], type[T]]:
        def _wrap(cls: type[T]) -> type[T]:
            if name in self._strategies:
                raise ValueError(f"{self.kind} strategy '{name}' already registered")
            self._strategies[name] = cls
            self._descriptions[name] = description or (cls.__doc__ or "").strip()
            return cls

        return _wrap

    def get(self, name: str) -> type[T]:
        try:
            return self._strategies[name]
        except KeyError as e:
            raise StrategyNotFoundError(
                f"Unknown {self.kind} strategy '{name}'. Available: {self.names()}"
            ) from e

    def create(self, name: str, **kwargs) -> T:
        return self.get(name)(**kwargs)

    def names(self) -> list[str]:
        return sorted(self._strategies.keys())

    def description(self, name: str) -> str:
        return self._descriptions.get(name, "")

    def as_choices(self) -> dict[str, str]:
        """name -> description, handy for populating a Streamlit selectbox with help text."""
        return {name: self._descriptions.get(name, "") for name in self.names()}
