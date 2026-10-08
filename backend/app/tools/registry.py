"""Tool registry: every tool an agent may call is registered here.

The call wrapper guarantees three things to the agent loop (spec §3, §20, §21):
1. outputs are validated against the tool's declared Pydantic schema;
2. every call is logged (name, redacted arguments, duration, success);
3. failures surface only as ToolValidationError or ToolExecutionError,
   never as bare exceptions.

Tools are deterministic Python/ML/database functions. The LLM never computes
what a tool computes (spec §4).
"""

import asyncio
import inspect
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable

from pydantic import BaseModel, ValidationError

from app.core.logging import get_logger, log_event, redact_text

logger = get_logger("tools")


class ToolError(Exception):
    def __init__(self, tool_name: str, message: str):
        super().__init__(f"{tool_name}: {message}")
        self.tool_name = tool_name


class ToolValidationError(ToolError):
    """The tool's input or output did not match its declared schema."""


class ToolExecutionError(ToolError):
    """The tool raised, timed out, or is not registered."""


@dataclass(frozen=True)
class ToolSpec:
    name: str
    func: Callable[..., Any]
    output_schema: type[BaseModel]
    input_schema: type[BaseModel] | None
    description: str


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(
        self,
        name: str,
        *,
        output_schema: type[BaseModel],
        input_schema: type[BaseModel] | None = None,
        description: str = "",
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            if name in self._tools:
                raise ValueError(f"tool '{name}' is already registered")
            doc = description or (func.__doc__ or "").strip()
            self._tools[name] = ToolSpec(name, func, output_schema, input_schema, doc)
            return func

        return decorator

    def has(self, name: str) -> bool:
        return name in self._tools

    def names(self) -> list[str]:
        return sorted(self._tools)

    def get(self, name: str) -> ToolSpec:
        if name not in self._tools:
            raise ToolExecutionError(name, "tool is not registered")
        return self._tools[name]

    async def call(self, name: str, arguments: dict[str, Any], timeout: float | None = None) -> BaseModel:
        """Run one tool call and return its validated output, or raise a ToolError."""
        start = time.perf_counter()
        try:
            spec = self.get(name)
            kwargs = self._validate_input(spec, arguments)
            raw = await self._execute(spec, kwargs, timeout)
            output = self._validate_output(spec, raw)
        except ToolError as exc:
            log_event(logger, "tool_call", logging.WARNING, tool_name=name, arguments=arguments,
                      duration_ms=_ms(start), success=False, error_type=type(exc).__name__)
            raise
        log_event(logger, "tool_call", tool_name=name, arguments=arguments, duration_ms=_ms(start), success=True)
        return output

    @staticmethod
    def _validate_input(spec: ToolSpec, arguments: dict[str, Any]) -> dict[str, Any]:
        if spec.input_schema is None:
            return dict(arguments)
        try:
            return spec.input_schema.model_validate(arguments).model_dump()
        except ValidationError as exc:
            raise ToolValidationError(spec.name, f"invalid input ({exc.error_count()} error(s))") from exc

    @staticmethod
    async def _execute(spec: ToolSpec, kwargs: dict[str, Any], timeout: float | None) -> Any:
        if inspect.iscoroutinefunction(spec.func):
            call = spec.func(**kwargs)
        else:
            # A timed-out sync tool keeps running in its worker thread; the agent still moves on.
            call = asyncio.to_thread(spec.func, **kwargs)
        try:
            return await asyncio.wait_for(call, timeout)
        except TimeoutError as exc:
            raise ToolExecutionError(spec.name, f"timed out after {timeout}s") from exc
        except ToolError:
            raise
        except Exception as exc:
            raise ToolExecutionError(spec.name, f"{type(exc).__name__}: {redact_text(str(exc))}") from exc

    @staticmethod
    def _validate_output(spec: ToolSpec, raw: Any) -> BaseModel:
        try:
            if isinstance(raw, BaseModel):
                raw = raw.model_dump()  # re-validate: instances may be built without validation
            return spec.output_schema.model_validate(raw)
        except ValidationError as exc:
            raise ToolValidationError(
                spec.name, f"output failed {spec.output_schema.__name__} validation ({exc.error_count()} error(s))"
            ) from exc


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 3)


default_registry = ToolRegistry()
register_tool = default_registry.register
