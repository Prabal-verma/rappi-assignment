"""Agent-facing API: scenarios, runs, traces, approvals, evaluations."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.agent.graph import AgentRunner
from app.agent.graph import run_case
from app.config import settings
from app.db.models import AgentRun, AgentStep, Approval, EvidenceRecord
from app.db.seed import SCENARIOS, seed_all
from app.db.session import get_session, reset_db
from app.domain import constraints as cons
from app.domain.schemas import (
    ActionResult,
    Decision,
    RunRequest,
    RunView,
    StepView,
    ValidationVerdict,
)
from app.tools.registry import registry
from app.tools.write_tools import execute_action

router = APIRouter()


# ─────────────────────────────────────────────────────────────────────
# Meta
# ─────────────────────────────────────────────────────────────────────


@router.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "llm_provider": settings.llm_provider,
        "llm_model": settings.llm_model,
        "llm_live": settings.llm_is_live,
        "autonomy_ceiling_usd": settings.autonomy_auto_approve_max_value_usd,
        "max_replan_attempts": settings.agent_max_replan_attempts,
    }


@router.get("/tools")
def list_tools() -> dict[str, Any]:
    """The agent's read surface, as the model sees it."""
    return {
        "count": len(registry.all()),
        "tools": [
            {
                "name": t.name,
                "description": t.description,
                "evidence_slot": t.slot.value if t.slot else None,
                "parameters": sorted(t.parameters),
            }
            for t in registry.all()
        ],
        "note": (
            "Read-only by design. State changes are proposed as structured data and executed "
            "by the graph after the constraint engine and autonomy policy clear them, so no "
            "write operation is reachable from the model."
        ),
    }


@router.get("/scenarios")
def list_scenarios() -> dict[str, Any]:
    return {"scenarios": list(SCENARIOS.values())}


@router.post("/admin/reset")
def reset(session: Session = Depends(get_session)) -> dict[str, Any]:
    """Drop, recreate and reseed. Every scenario starts from the same world."""
    reset_db()
    seed_all(session)
    session.commit()
    return {"status": "reset", "scenarios": sorted(SCENARIOS)}


# ─────────────────────────────────────────────────────────────────────
# Runs
# ─────────────────────────────────────────────────────────────────────


def _steps(run: AgentRun) -> list[StepView]:
    return [
        StepView(
            seq=s.seq,
            graph_node=s.graph_node,
            kind=s.kind,
            title=s.title,
            tool_name=s.tool_name,
            payload_in=s.payload_in,
            payload_out=s.payload_out,
            latency_ms=s.latency_ms,
            created_at=s.created_at.isoformat(),
        )
        for s in sorted(run.steps, key=lambda s: s.seq)
    ]


def _view(run: AgentRun, include_steps: bool = True) -> RunView:
    return RunView(
        run_id=run.run_id,
        scenario_id=run.scenario_id,
        case_type=run.case_type,
        status=run.status,
        provider=run.provider,
        model=run.model,
        attempts=run.attempts,
        sku=run.sku,
        node_id=run.node_id,
        created_at=run.created_at.isoformat(),
        completed_at=run.completed_at.isoformat() if run.completed_at else None,
        duration_ms=run.duration_ms,
        result=run.result,
        steps=_steps(run) if include_steps else [],
    )


@router.post("/runs", response_model=RunView)
def create_run(request: RunRequest, session: Session = Depends(get_session)) -> RunView:
    scenario = SCENARIOS.get(request.scenario_id)
    if scenario is None:
        raise HTTPException(404, f"Unknown scenario {request.scenario_id}")

    case = {**scenario, **request.overrides}
    run_id = run_case(session, request.scenario_id, case, request.provider, request.model)

    run = session.get(AgentRun, run_id)
    if run is None:
        raise HTTPException(500, "Run disappeared after execution.")
    return _view(run)


@router.get("/runs")
def list_runs(limit: int = 40, session: Session = Depends(get_session)) -> dict[str, Any]:
    runs = list(
        session.execute(
            select(AgentRun).order_by(AgentRun.created_at.desc()).limit(limit)
        ).scalars()
    )
    return {"runs": [_view(r, include_steps=False).model_dump() for r in runs]}


@router.get("/runs/{run_id}", response_model=RunView)
def get_run(run_id: str, session: Session = Depends(get_session)) -> RunView:
    run = session.get(AgentRun, run_id)
    if run is None:
        raise HTTPException(404, f"Unknown run {run_id}")
    return _view(run)


@router.get("/runs/{run_id}/evidence")
def get_evidence(run_id: str, session: Session = Depends(get_session)) -> dict[str, Any]:
    """Every fact the decision rested on, with its source."""
    rows = list(
        session.execute(
            select(EvidenceRecord).where(EvidenceRecord.run_id == run_id).order_by(EvidenceRecord.id)
        ).scalars()
    )
    return {
        "run_id": run_id,
        "count": len(rows),
        "evidence": [
            {
                "slot": r.slot,
                "tool": r.tool_name,
                "trusted": r.trusted,
                "collected_at": r.collected_at.isoformat(),
                "value": r.value,
            }
            for r in rows
        ],
    }


# ─────────────────────────────────────────────────────────────────────
# Approvals — the human in the loop
# ─────────────────────────────────────────────────────────────────────


class ApprovalDecision(BaseModel):
    approve: bool
    decided_by: str = "buyer"
    note: str = ""


@router.get("/approvals")
def list_approvals(status: str | None = None, session: Session = Depends(get_session)) -> dict[str, Any]:
    stmt = select(Approval).order_by(Approval.requested_at.desc())
    if status:
        stmt = stmt.where(Approval.status == status)
    rows = list(session.execute(stmt).scalars())
    return {
        "approvals": [
            {
                "id": a.id,
                "run_id": a.run_id,
                "status": a.status,
                "risk_level": a.risk_level,
                "reason": a.reason,
                "proposed_action": a.proposed_action,
                "requested_at": a.requested_at.isoformat(),
                "decided_at": a.decided_at.isoformat() if a.decided_at else None,
                "decided_by": a.decided_by,
                "decision_note": a.decision_note,
            }
            for a in rows
        ]
    }


@router.post("/approvals/{approval_id}/decide")
def decide_approval(
    approval_id: int, body: ApprovalDecision, session: Session = Depends(get_session)
) -> dict[str, Any]:
    """Approve or reject a queued action.

    On approval the stored action is executed **verbatim** — it is not
    re-derived from the model. The approver agreed to a specific purchase
    order, and that is the one that runs. The same validation pass then
    applies, so a human-approved action is held to the same standard as an
    autonomous one.
    """
    approval = session.get(Approval, approval_id)
    if approval is None:
        raise HTTPException(404, f"Unknown approval {approval_id}")
    if approval.status != "pending":
        raise HTTPException(409, f"Approval {approval_id} is already {approval.status}.")

    from app.domain import clock

    approval.decided_at = clock.now()
    approval.decided_by = body.decided_by
    approval.decision_note = body.note

    if not body.approve:
        approval.status = "rejected"
        run = session.get(AgentRun, approval.run_id)
        if run is not None:
            run.status = "rejected_by_human"
            run.result = {**(run.result or {}), "outcome": "rejected_by_human", "human_note": body.note}
        session.commit()
        return {"status": "rejected", "approval_id": approval_id}

    approval.status = "approved"
    decision = Decision.model_validate(approval.proposed_action["decision"])
    run = session.get(AgentRun, approval.run_id)
    runner_tracer_run_id = approval.run_id

    from app.agent.state import Tracer

    tracer = Tracer(session, runner_tracer_run_id)
    tracer._seq = max((s.seq for s in run.steps), default=0) if run else 0

    results: list[ActionResult] = []
    for action in decision.proposed_actions:
        result = execute_action(session, runner_tracer_run_id, action)
        session.flush()
        results.append(result)
        tracer.step(
            graph_node="act",
            kind="node",
            title=f"[human-approved] {action.action_type.value} -> {result.outcome}",
            payload_in=action.model_dump(mode="json"),
            payload_out=result.model_dump(mode="json"),
        )

    po_ids = [r.entity_id for r in results if r.entity_id and r.entity_id.startswith("PO")]
    if po_ids and run is not None:
        report = cons.validate_purchase_order(
            session, po_ids[-1], decision.expectation, run.sku, run.node_id
        )
    elif run is not None:
        report = cons.validate_no_action(session, run.sku, run.node_id, decision.expectation)
    else:
        report = None

    if report is not None:
        tracer.step(
            graph_node="validate",
            kind="validation",
            title=f"Post-approval validation: {report.verdict.value.upper()}",
            payload_out=report.model_dump(mode="json"),
        )

    if run is not None:
        run.status = (
            "completed"
            if report is not None and report.verdict == ValidationVerdict.PASS
            else f"approved_with_{report.verdict.value}" if report else "approved"
        )
        run.result = {
            **(run.result or {}),
            "outcome": run.status,
            "approved_by": body.decided_by,
            "action_results": [r.model_dump(mode="json") for r in results],
            "validation": report.model_dump(mode="json") if report else None,
        }

    session.commit()
    return {
        "status": "approved",
        "approval_id": approval_id,
        "action_results": [r.model_dump(mode="json") for r in results],
        "validation": report.model_dump(mode="json") if report else None,
    }
