from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from types import UnionType
from typing import Any, Union, get_args, get_origin

from .types import ToolContext


JsonDict = dict[str, Any]


@dataclass(slots=True)
class BrainTool:
    name: str
    description: str
    func: Callable[..., Any]
    parameters: JsonDict
    strict: bool = False
    timeout_seconds: float | None = None

    def schema(self) -> JsonDict:
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "strict": self.strict,
        }


def _annotation_to_schema(annotation: Any) -> JsonDict:
    if annotation is inspect.Signature.empty or annotation is Any:
        return {"type": "string"}
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin in (Union, UnionType):
        non_none = [arg for arg in args if arg is not type(None)]
        if len(non_none) == 1:
            schema = _annotation_to_schema(non_none[0])
            schema["nullable"] = True
            return schema
        return {"anyOf": [_annotation_to_schema(arg) for arg in non_none]}
    if annotation is str:
        return {"type": "string"}
    if annotation is int:
        return {"type": "integer"}
    if annotation is float:
        return {"type": "number"}
    if annotation is bool:
        return {"type": "boolean"}
    if origin in (list, Sequence):
        item_type = args[0] if args else Any
        return {"type": "array", "items": _annotation_to_schema(item_type)}
    if origin is dict:
        return {"type": "object"}
    if hasattr(annotation, "model_json_schema"):
        return annotation.model_json_schema()
    return {"type": "string"}


def _parameters_from_function(func: Callable[..., Any]) -> JsonDict:
    signature = inspect.signature(func)
    properties: JsonDict = {}
    required: list[str] = []
    for name, parameter in signature.parameters.items():
        if name == "context":
            continue
        properties[name] = _annotation_to_schema(parameter.annotation)
        if parameter.default is inspect.Signature.empty:
            required.append(name)
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, BrainTool] = {}

    def register(
        self,
        func: Callable[..., Any] | None = None,
        *,
        name: str | None = None,
        description: str | None = None,
        parameters: JsonDict | None = None,
        strict: bool = False,
        timeout_seconds: float | None = None,
    ):
        def decorator(target: Callable[..., Any]) -> Callable[..., Any]:
            tool_name = name or target.__name__
            doc = inspect.getdoc(target) or ""
            self._tools[tool_name] = BrainTool(
                name=tool_name,
                description=description or doc or tool_name,
                func=target,
                parameters=parameters or _parameters_from_function(target),
                strict=strict,
                timeout_seconds=timeout_seconds,
            )
            return target

        if func is not None:
            return decorator(func)
        return decorator

    def get(self, name: str) -> BrainTool:
        return self._tools[name]

    def schemas(self, names: Sequence[str] | None = None) -> list[JsonDict]:
        selected = names or list(self._tools)
        return [self._tools[name].schema() for name in selected if name in self._tools]

    async def execute(
        self,
        name: str,
        arguments_json: str | dict[str, Any],
        *,
        context: ToolContext,
        timeout_seconds: float | None = None,
    ) -> Any:
        tool = self.get(name)
        if isinstance(arguments_json, str):
            arguments = json.loads(arguments_json or "{}")
        else:
            arguments = dict(arguments_json)
        if "context" in inspect.signature(tool.func).parameters:
            arguments["context"] = context
        async def run() -> Any:
            result = tool.func(**arguments)
            if isinstance(result, Awaitable):
                return await result
            if inspect.iscoroutine(result):
                return await result
            return result

        timeout = tool.timeout_seconds if tool.timeout_seconds is not None else timeout_seconds
        if timeout is not None and timeout > 0:
            return await asyncio.wait_for(run(), timeout=timeout)
        return await run()

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)
