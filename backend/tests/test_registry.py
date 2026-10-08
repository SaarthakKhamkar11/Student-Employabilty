import asyncio
import logging

import pytest
from pydantic import BaseModel

from app.tools.registry import ToolExecutionError, ToolRegistry, ToolValidationError


class Num(BaseModel):
    value: int


class AddIn(BaseModel):
    x: int


@pytest.fixture
def registry() -> ToolRegistry:
    reg = ToolRegistry()

    @reg.register("add_one", output_schema=Num, input_schema=AddIn)
    def add_one(x: int):
        return {"value": x + 1}

    @reg.register("async_double", output_schema=Num)
    async def async_double(x: int):
        return Num(value=x * 2)

    @reg.register("bad_output", output_schema=Num)
    def bad_output():
        return {"wrong": "shape"}

    @reg.register("boom", output_schema=Num)
    def boom():
        raise RuntimeError("database down")

    @reg.register("slow", output_schema=Num)
    async def slow():
        await asyncio.sleep(5)

    @reg.register("echo_email", output_schema=Num)
    def echo_email(email: str, x: int):
        return {"value": x}

    return reg


def call(reg, name, args=None, timeout=None):
    return asyncio.run(reg.call(name, args or {}, timeout=timeout))


def test_valid_output_is_returned_as_the_declared_model(registry):
    assert call(registry, "add_one", {"x": 1}) == Num(value=2)
    assert call(registry, "async_double", {"x": 3}) == Num(value=6)


def test_output_not_matching_schema_raises_tool_validation_error(registry):
    with pytest.raises(ToolValidationError):
        call(registry, "bad_output")


def test_invalid_input_raises_tool_validation_error(registry):
    with pytest.raises(ToolValidationError):
        call(registry, "add_one", {"x": "not-a-number"})


def test_tool_exception_is_wrapped_never_bare(registry):
    with pytest.raises(ToolExecutionError) as info:
        call(registry, "boom")
    assert isinstance(info.value.__cause__, RuntimeError)


def test_unknown_tool_raises_tool_execution_error(registry):
    with pytest.raises(ToolExecutionError):
        call(registry, "does_not_exist")


def test_timeout_raises_tool_execution_error(registry):
    with pytest.raises(ToolExecutionError, match="timed out"):
        call(registry, "slow", timeout=0.05)


def test_duplicate_registration_is_rejected(registry):
    with pytest.raises(ValueError):
        registry.register("add_one", output_schema=Num)(lambda: None)


def test_every_call_is_logged_with_redacted_arguments(registry, caplog):
    caplog.set_level(logging.INFO, logger="employability")
    call(registry, "echo_email", {"email": "jane.doe@example.com", "x": 4})
    (record,) = [r for r in caplog.records if r.getMessage() == "tool_call"]
    assert record.fields["tool_name"] == "echo_email"
    assert record.fields["success"] is True
    assert record.fields["arguments"] == {"email": "[REDACTED]", "x": 4}
    assert "duration_ms" in record.fields
