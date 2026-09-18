"""Sampling validation and conservative loop detection. No MLX dependency."""
from __future__ import annotations

import inspect
import math
from dataclasses import asdict, dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class GenerationOptions:
    # Prism's non-thinking sampling values; output limit/window/guard are app choices.
    max_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    repetition_context_size: int = 256
    presence_penalty: float = 1.5
    presence_context_size: int = 256
    frequency_penalty: float = 0.0
    frequency_context_size: int = 256
    enable_thinking: bool = False
    loop_guard: bool = True

    @classmethod
    def from_request(cls, data: Any) -> GenerationOptions:
        if not isinstance(data, dict):
            raise ValueError("Request body must be a JSON object")
        defaults = cls()
        values: dict[str, Any] = {}
        limits = {
            "max_tokens": (1, 8192, True),
            "temperature": (0, 2, False),
            "top_p": (0.05, 1, False),
            "top_k": (0, 1000, True),
            "min_p": (0, 1, False),
            "repetition_penalty": (1, 1.3, False),
            "repetition_context_size": (32, 4096, True),
            "presence_penalty": (0, 2, False),
            "presence_context_size": (32, 4096, True),
            "frequency_penalty": (0, 2, False),
            "frequency_context_size": (32, 4096, True),
        }
        for name, (low, high, integer) in limits.items():
            raw = data.get(name, getattr(defaults, name))
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                raise ValueError(f"'{name}' must be a number")
            try:
                value = float(raw)
            except (ValueError, OverflowError):
                raise ValueError(f"'{name}' must be finite") from None
            if not math.isfinite(value) or not low <= value <= high:
                raise ValueError(f"'{name}' must be between {low} and {high}")
            if integer and not value.is_integer():
                raise ValueError(f"'{name}' must be an integer")
            values[name] = int(value) if integer else value
        for name in ("enable_thinking", "loop_guard"):
            value = data.get(name, getattr(defaults, name))
            if not isinstance(value, bool):
                raise ValueError(f"'{name}' must be true or false")
            values[name] = value
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def explicit_parameters(function: Callable[..., Any]) -> set[str]:
    """**kwargs alone is NOT evidence that a setting is implemented."""
    return {
        name for name, parameter in inspect.signature(function).parameters.items()
        if parameter.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    }


def sampling_kwargs(options: GenerationOptions, supported: set[str]) -> dict[str, Any]:
    """Send native MLX-VLM settings, refusing unsupported non-neutral values.

    Older MLX-VLM builds may expose **kwargs but not implement presence penalties.
    Do not pretend a control worked by silently passing it into the model.
    """
    values = options.to_dict()
    values.pop("enable_thinking")  # A chat-template option, not a sampling knob.
    values.pop("loop_guard")       # Implemented in the app, not the model.
    neutral = {
        "top_k": 0,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
    }
    contexts = {
        "repetition_context_size": ("repetition_penalty", 1.0),
        "presence_context_size": ("presence_penalty", 0.0),
        "frequency_context_size": ("frequency_penalty", 0.0),
    }
    result: dict[str, Any] = {}
    for name, value in values.items():
        if name in contexts:
            penalty, off = contexts[name]
            if values[penalty] == off:
                continue
        if name in supported:
            result[name] = value
        elif name in neutral and value == neutral[name]:
            continue
        else:
            raise ValueError(
                f"Installed mlx-vlm does not expose '{name}' in generate_step. "
                "Disable that control or use a compatible runtime from the model pack. "
                "The setting has not been silently ignored."
            )
    return result


@dataclass(frozen=True)
class LoopMatch:
    unit_characters: int
    repetitions: int
    excerpt: str


class RepetitionGuard:
    """Stop on three consecutive exact copies of a substantial text fragment.

    Detection is bounded to a recent character window. Small-period repetitions
    of punctuation, braces, indentation or short identifiers are excluded.
    This is a heuristic, not a correctness check. Deliberately repeated long
    blocks can also trigger it. No generated source is edited or deduplicated.
    """

    def __init__(
        self, *, min_unit: int = 80, max_unit: int = 2048,
        repetitions: int = 3, check_interval: int = 32,
    ) -> None:
        if min_unit < 1 or max_unit < min_unit or repetitions < 2 or check_interval < 1:
            raise ValueError("Invalid loop-guard configuration")
        self.min_unit = min_unit
        self.max_unit = max_unit
        self.repetitions = repetitions
        self.check_interval = check_interval
        self._buffer = ""
        self._unchecked = 0
        self.match: LoopMatch | None = None

    @staticmethod
    def _has_short_period(text: str) -> bool:
        # Reject a long unit that is itself merely `std::`/braces/etc. repeated.
        for period in range(1, min(40, len(text)) + 1):
            if text[period:] == text[:-period]:
                return True
        return False

    def feed(self, delta: str, *, force: bool = False) -> LoopMatch | None:
        if self.match:
            return self.match
        self._buffer = (self._buffer + delta)[-self.max_unit * self.repetitions:]
        self._unchecked += len(delta)
        if not force and self._unchecked < self.check_interval:
            return None
        self._unchecked = 0
        tail = self._buffer
        upper = min(self.max_unit, len(tail) // self.repetitions)
        for size in range(self.min_unit, upper + 1):
            unit = tail[-size:]
            if tail[-size * self.repetitions:] != unit * self.repetitions:
                continue
            if sum(ch.isalnum() for ch in unit) < 32 or self._has_short_period(unit):
                continue
            self.match = LoopMatch(size, self.repetitions, unit[:160])
            return self.match
        return None
