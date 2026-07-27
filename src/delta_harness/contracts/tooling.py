from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Literal, Protocol

from pydantic import Field, model_validator

from delta_harness.contracts.transcript import (
    ImageSegment,
    TextSegment,
    StrictModel,
)
from delta_harness.contracts.values import JValue


Parallelism = Literal["parallel", "sequential"]
InputShaper = Callable[[object], Mapping[str, JValue]]
ProgressNotifier = Callable[["ToolOutcome"], None]


class CancelToken(Protocol):
    def is_cancelled(self) -> bool: ...


class InvocationPrinter(Protocol):
    def __call__(
        self,
        arguments: Mapping[str, JValue],
    ) -> str | None: ...


class OutcomePresenter(Protocol):
    def __call__(
        self,
        result: "ToolOutcome",
        *,
        expanded: bool,
    ) -> str | None: ...


class RunHandler(Protocol):
    def __call__(
        self,
        tool_call_id: str,
        arguments: Mapping[str, JValue],
        signal: CancelToken | None = None,
        on_update: ProgressNotifier | None = None,
    ) -> Awaitable["ToolOutcome"]: ...


class ToolOutcome(StrictModel):
    content: list[TextSegment | ImageSegment] = Field(default_factory=list)
    details: JValue = None
    terminate: bool | None = None
    added_tool_names: list[str] | None = None

    @model_validator(mode="before")
    @classmethod
    def coerce_content(cls, value):
        if not isinstance(value, dict):
            return value

        data = dict(value)

        if isinstance(data.get("content"), str):
            text = data["content"]
            data["content"] = [TextSegment(text=text)] if text else []

        return data

    @property
    def text(self) -> str:
        return "".join(
            block.text
            for block in self.content
            if isinstance(block, TextSegment)
        )


@dataclass(slots=True, frozen=True)
class ToolSpec:
    name: str
    label: str
    description: str
    parameters: Mapping[str, JValue]
    run: RunHandler

    parallelism: Parallelism = "parallel"

    prompt_snippet: str | None = None
    prompt_guidelines: tuple[str, ...] = ()

    input_shaper: InputShaper | None = None

    format_call: InvocationPrinter | None = None
    format_result: OutcomePresenter | None = None

    @property
    def input_schema(self) -> Mapping[str, JValue]:
        return self.parameters

    async def execute(
        self,
        tool_call_id: str,
        arguments: Mapping[str, JValue],
        signal: CancelToken | None = None,
        on_update: ProgressNotifier | None = None,
    ) -> ToolOutcome:
        return await self.run(
            tool_call_id,
            arguments,
            signal,
            on_update,
        )


__all__ = [
    "CancelToken",
    "InputShaper",
    "InvocationPrinter",
    "OutcomePresenter",
    "Parallelism",
    "ProgressNotifier",
    "RunHandler",
    "ToolOutcome",
    "ToolSpec",
]