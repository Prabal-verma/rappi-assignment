"""Graph state and the run tracer.

State is a plain dict so the graph nodes stay framework-independent — each
one is an ordinary function from state to a state update, testable without
LangGraph in the loop.

The tracer writes every node entry, tool call, model turn, guardrail verdict
and validation result to the database as it happens. That trace is the
product, not a debug aid: it is what a buyer reads to decide whether to
trust the decision, and what an engineer reads when a decision was wrong.
"""

from __future__ import annotations

import time
from typing import Any, TypedDict

from sqlalchemy.orm import Session

from app.db.models import AgentStep, EvidenceRecord


class AgentState(TypedDict, total=False):
    # ---- identity ----
    run_id: str
    scenario_id: str
    case_type: str
    case: dict[str, Any]

    # ---- the subject of the case ----
    sku: str
    node_id: str
    supplier_id: str
    po_id: str
    recommended_units: int

    # ---- conversation with the model ----
    turns: list[Any]
    tool_calls_made: int

    # ---- what has been learned ----
    evidence: dict[str, Any]
    missing_slots: list[str]
    untrusted_flags: list[str]

    # ---- the decision and its fate ----
    decision: dict[str, Any]
    authorization: dict[str, Any]
    action_results: list[dict[str, Any]]
    validation: dict[str, Any]

    # ---- loop control ----
    attempts: int
    corrective_guidance: str
    status: str
    degraded: bool
    degradation_reasons: list[str]


class Tracer:
    """Append-only writer for a run's observable steps."""

    def __init__(self, session: Session, run_id: str) -> None:
        self.session = session
        self.run_id = run_id
        self._seq = 0
        self.total_tokens = 0
        self._t0 = time.perf_counter()

    def _next(self) -> int:
        self._seq += 1
        return self._seq

    def step(
        self,
        graph_node: str,
        kind: str,
        title: str,
        payload_in: dict | None = None,
        payload_out: dict | None = None,
        tool_name: str | None = None,
        latency_ms: int = 0,
        tokens: int = 0,
    ) -> None:
        self.total_tokens += tokens
        self.session.add(
            AgentStep(
                run_id=self.run_id,
                seq=self._next(),
                graph_node=graph_node,
                kind=kind,
                title=title[:256],
                tool_name=tool_name,
                payload_in=payload_in,
                payload_out=payload_out,
                latency_ms=latency_ms,
                tokens=tokens,
            )
        )
        self.session.flush()

    def evidence(self, slot: str, tool_name: str, value: dict, trusted: bool = True) -> None:
        self.session.add(
            EvidenceRecord(
                run_id=self.run_id,
                slot=slot,
                tool_name=tool_name,
                value=value,
                trusted=trusted,
            )
        )
        self.session.flush()

    @property
    def elapsed_ms(self) -> int:
        return int((time.perf_counter() - self._t0) * 1000)
