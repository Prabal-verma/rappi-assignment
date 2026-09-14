"""The tool registry — everything the agent is allowed to touch.

Two deliberate design choices:

1. **The model only gets read tools.** It investigates with them freely.
   State-changing operations are never exposed as callable functions;
   they are *proposed* as structured data in the decision, then executed
   by the act node only after the constraint engine and the autonomy
   policy have both cleared them. A model cannot talk its way into a
   write, because there is no write to call.

2. **Every tool declares the evidence slot it fills.** That turns
   "did the agent gather what it needed?" into a checkable property:
   the graph refuses to leave the investigation phase until the required
   slots for the case type are populated.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Optional

from sqlalchemy.orm import Session

from app.domain.schemas import EvidenceSlot


@dataclass
class ToolContext:
    """Everything a tool needs that is not one of its declared arguments."""

    session: Session
    run_id: str
    # Case defaults, so the model does not have to repeat sku/node on every call.
    sku: Optional[str] = None
    node_id: Optional[str] = None


@dataclass
class ToolResult:
    ok: bool
    data: dict[str, Any]
    slot: Optional[EvidenceSlot] = None
    # False for anything a third party authored, e.g. supplier email text.
    trusted: bool = True
    error: Optional[str] = None
    latency_ms: int = 0


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[..., ToolResult]
    slot: Optional[EvidenceSlot] = None
    required: list[str] = field(default_factory=list)

    def to_function_declaration(self) -> dict[str, Any]:
        """Provider-neutral function schema.

        Kept to the intersection of what Gemini, Anthropic and OpenAI all
        accept, so provider adapters need no per-tool special cases.
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": self.parameters,
                "required": self.required,
            },
        }


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        slot: Optional[EvidenceSlot] = None,
        required: Optional[list[str]] = None,
    ) -> Callable:
        def decorator(fn: Callable[..., ToolResult]) -> Callable[..., ToolResult]:
            self._tools[name] = Tool(
                name=name,
                description=description,
                parameters=parameters,
                handler=fn,
                slot=slot,
                required=required or [],
            )
            return fn

        return decorator

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def all(self) -> list[Tool]:
        return list(self._tools.values())

    def names(self) -> list[str]:
        return sorted(self._tools)

    def function_declarations(self) -> list[dict[str, Any]]:
        return [t.to_function_declaration() for t in self._tools.values()]

    def invoke(self, name: str, ctx: ToolContext, arguments: dict[str, Any]) -> ToolResult:
        """Call a tool by name with model-supplied arguments.

        Argument handling is forgiving by design — a model that omits `sku`
        when the case already fixes it should not cost a turn. Anything
        genuinely wrong comes back as a structured error the model can read
        and correct, never as an exception that kills the run.
        """
        tool = self.get(name)
        if tool is None:
            return ToolResult(
                ok=False,
                data={},
                error=(
                    f"No tool named '{name}'. Available tools: {', '.join(self.names())}."
                ),
            )

        args = dict(arguments or {})
        # Fill case defaults for the two identifiers that appear everywhere.
        if "sku" in tool.parameters and not args.get("sku") and ctx.sku:
            args["sku"] = ctx.sku
        if "node_id" in tool.parameters and not args.get("node_id") and ctx.node_id:
            args["node_id"] = ctx.node_id

        unknown = [k for k in args if k not in tool.parameters]
        for k in unknown:
            args.pop(k)

        missing = [r for r in tool.required if not args.get(r)]
        if missing:
            return ToolResult(
                ok=False,
                data={},
                error=f"Tool '{name}' requires {missing}, which were not supplied.",
            )

        started = time.perf_counter()
        try:
            result = tool.handler(ctx, **args)
        except Exception as exc:  # noqa: BLE001 — surfaced to the model, not swallowed
            return ToolResult(
                ok=False,
                data={},
                error=f"{type(exc).__name__}: {exc}",
                latency_ms=int((time.perf_counter() - started) * 1000),
            )

        result.latency_ms = int((time.perf_counter() - started) * 1000)
        if result.slot is None:
            result.slot = tool.slot
        return result


registry = ToolRegistry()
