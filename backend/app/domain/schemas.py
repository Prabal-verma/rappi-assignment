"""Shared vocabulary for the agent, the tools, the API and the UI.

The important type here is `ActionExpectation`. Before the agent is allowed
to touch the world it must state, numerically, what it expects the world to
look like afterwards. Validation then compares that claim against ground
truth re-read from the database. An agent that cannot predict the outcome of
its own action does not get to keep the action.
"""

from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────────────────────────────


class CaseType(str, Enum):
    RECOMMENDATION_REVIEW = "recommendation_review"
    SUPPLIER_SHORTFALL = "supplier_shortfall"
    DEMAND_SHIFT = "demand_shift"
    CONSTRAINED_PURCHASE = "constrained_purchase"


class DecisionType(str, Enum):
    ACCEPT = "accept"
    MODIFY = "modify"
    REJECT = "reject"
    INVESTIGATE_FURTHER = "investigate_further"
    ESCALATE = "escalate"


class ActionType(str, Enum):
    CREATE_PO = "create_purchase_order"
    MODIFY_PO = "modify_purchase_order"
    CANCEL_PO = "cancel_purchase_order"
    CREATE_TRANSFER = "create_transfer_order"
    NO_ACTION = "no_action"
    ESCALATE = "escalate_to_human"


class Severity(str, Enum):
    BLOCK = "block"
    WARN = "warn"
    INFO = "info"


class ValidationVerdict(str, Enum):
    PASS = "pass"
    # Action succeeded but the world does not match what the agent predicted.
    DRIFT = "drift"
    # A hard business constraint is broken by the persisted state.
    VIOLATION = "violation"
    # The action itself did not complete.
    FAILED = "failed"


class EvidenceSlot(str, Enum):
    INVENTORY = "inventory_position"
    DEMAND = "demand_forecast"
    SALES = "sales_actuals"
    OPEN_POS = "open_purchase_orders"
    SUPPLIER_TERMS = "supplier_terms"
    BUDGET = "budget"
    STORAGE = "storage_capacity"
    POLICY = "policy_guidance"
    PROMO = "promo_calendar"
    SUPPLIER_AVAILABILITY = "supplier_availability"
    ALTERNATE_SUPPLIERS = "alternate_suppliers"
    ANOMALY = "demand_anomaly"
    REPLENISHMENT_MATH = "replenishment_analysis"


# Which facts a case type may not be decided without. Enforced by the graph,
# not by the prompt — see agent/nodes/investigate.py.
REQUIRED_SLOTS: dict[CaseType, list[EvidenceSlot]] = {
    CaseType.RECOMMENDATION_REVIEW: [
        EvidenceSlot.INVENTORY,
        EvidenceSlot.DEMAND,
        EvidenceSlot.OPEN_POS,
        EvidenceSlot.SUPPLIER_TERMS,
        EvidenceSlot.BUDGET,
        EvidenceSlot.STORAGE,
        EvidenceSlot.REPLENISHMENT_MATH,
    ],
    CaseType.SUPPLIER_SHORTFALL: [
        EvidenceSlot.INVENTORY,
        EvidenceSlot.DEMAND,
        EvidenceSlot.OPEN_POS,
        EvidenceSlot.SUPPLIER_AVAILABILITY,
        EvidenceSlot.ALTERNATE_SUPPLIERS,
        EvidenceSlot.REPLENISHMENT_MATH,
    ],
    CaseType.DEMAND_SHIFT: [
        EvidenceSlot.INVENTORY,
        EvidenceSlot.SALES,
        EvidenceSlot.DEMAND,
        EvidenceSlot.ANOMALY,
        EvidenceSlot.PROMO,
        EvidenceSlot.OPEN_POS,
        EvidenceSlot.REPLENISHMENT_MATH,
    ],
    CaseType.CONSTRAINED_PURCHASE: [
        EvidenceSlot.INVENTORY,
        EvidenceSlot.DEMAND,
        EvidenceSlot.BUDGET,
        EvidenceSlot.STORAGE,
        EvidenceSlot.SUPPLIER_TERMS,
        EvidenceSlot.ALTERNATE_SUPPLIERS,
        EvidenceSlot.REPLENISHMENT_MATH,
    ],
}


# ─────────────────────────────────────────────────────────────────────
# Replenishment analysis (deterministic — computed, never inferred)
# ─────────────────────────────────────────────────────────────────────


class ReplenishmentAnalysis(BaseModel):
    sku: str
    node_id: str
    supplier_id: Optional[str] = None

    on_hand_units: int
    reserved_units: int
    available_units: int
    in_transit_units: int
    inventory_position_units: int

    daily_demand_mean: float
    daily_demand_std: float
    demand_source: str = Field(description="Which signal the mean came from")
    horizon_days: int
    lead_time_days: int
    review_period_days: int

    safety_stock_units: int
    target_position_units: int
    net_requirement_units: int

    days_of_cover_now: float
    projected_stockout_in_days: Optional[float]
    shelf_life_days: int
    max_useful_units: int = Field(
        description="Units above which stock would expire before it sells"
    )

    recommended_order_units: int
    recommended_order_rounded: int
    rounding_note: str = ""
    unit_cost_usd: float = 0.0
    estimated_cost_usd: float = 0.0
    notes: list[str] = Field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────
# Constraints
# ─────────────────────────────────────────────────────────────────────


class ConstraintViolation(BaseModel):
    code: str
    severity: Severity
    message: str
    observed: Optional[float] = None
    limit: Optional[float] = None
    unit: str = ""
    remedy_hint: str = ""


class ConstraintReport(BaseModel):
    passed: bool
    blocking: list[ConstraintViolation] = Field(default_factory=list)
    warnings: list[ConstraintViolation] = Field(default_factory=list)
    checks_run: list[str] = Field(default_factory=list)

    @property
    def summary(self) -> str:
        if self.passed and not self.warnings:
            return f"All {len(self.checks_run)} constraint checks passed."
        parts = []
        if self.blocking:
            parts.append(f"{len(self.blocking)} blocking")
        if self.warnings:
            parts.append(f"{len(self.warnings)} warning(s)")
        return f"{len(self.checks_run)} checks run: " + ", ".join(parts)


# ─────────────────────────────────────────────────────────────────────
# Decisions and actions
# ─────────────────────────────────────────────────────────────────────


class ProposedAction(BaseModel):
    action_type: ActionType
    sku: Optional[str] = None
    node_id: Optional[str] = None
    supplier_id: Optional[str] = None
    po_id: Optional[str] = None
    units: Optional[int] = None
    from_node_id: Optional[str] = None
    reason: str = ""

    def signature(self) -> str:
        """Stable identity for idempotency."""
        return "|".join(
            str(x)
            for x in [
                self.action_type.value,
                self.sku,
                self.node_id,
                self.supplier_id,
                self.po_id,
                self.units,
                self.from_node_id,
            ]
        )


class ActionExpectation(BaseModel):
    """What the agent asserts will be true once its actions have run.

    Deliberately numeric. `validate` re-derives each of these from the
    database and diffs them; a mismatch outside tolerance is DRIFT and
    sends the agent back to replan.
    """

    expected_units_secured: int = 0
    expected_spend_usd: float = 0.0
    expected_inventory_position_units: int = 0
    expected_days_of_cover: float = 0.0
    expected_stockout_risk: Literal["low", "medium", "high"] = "low"
    expected_constraint_violations: int = 0
    notes: str = ""


class Decision(BaseModel):
    """The structured output the reasoning node must produce."""

    decision_type: DecisionType
    recommended_units: Optional[int] = None
    rationale: str
    key_factors: list[str] = Field(default_factory=list)
    # Facts the agent knows it is missing. Non-empty + INVESTIGATE_FURTHER
    # is a legitimate, well-behaved outcome.
    information_gaps: list[str] = Field(default_factory=list)
    proposed_actions: list[ProposedAction] = Field(default_factory=list)
    expectation: ActionExpectation = Field(default_factory=ActionExpectation)
    confidence: float = 0.5
    risk_level: Literal["low", "medium", "high"] = "medium"


# ─────────────────────────────────────────────────────────────────────
# Authorization (guardrail node output)
# ─────────────────────────────────────────────────────────────────────


class AuthorizationOutcome(BaseModel):
    mode: Literal["autonomous", "needs_approval", "blocked"]
    reasons: list[str] = Field(default_factory=list)
    policy_refs: list[str] = Field(default_factory=list)
    constraint_report: Optional[ConstraintReport] = None
    risk_level: Literal["low", "medium", "high"] = "medium"


# ─────────────────────────────────────────────────────────────────────
# Execution + validation
# ─────────────────────────────────────────────────────────────────────


class ActionResult(BaseModel):
    action: ProposedAction
    succeeded: bool
    entity_id: Optional[str] = None
    # What the downstream system actually did, which may differ from the ask.
    outcome: str = ""
    confirmed_units: Optional[int] = None
    detail: dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None


class ExpectationDiff(BaseModel):
    field: str
    expected: float
    actual: float
    tolerance: float
    within_tolerance: bool

    @property
    def delta(self) -> float:
        return round(self.actual - self.expected, 2)


class ValidationReport(BaseModel):
    verdict: ValidationVerdict
    constraint_report: ConstraintReport
    expectation_diffs: list[ExpectationDiff] = Field(default_factory=list)
    observed_state: dict[str, Any] = Field(default_factory=dict)
    findings: list[str] = Field(default_factory=list)
    # Instruction handed back into the graph when the verdict is not PASS.
    corrective_guidance: str = ""


# ─────────────────────────────────────────────────────────────────────
# API surface
# ─────────────────────────────────────────────────────────────────────


class RunRequest(BaseModel):
    scenario_id: str
    # Overrides let the UI and the evals perturb a scenario without editing seeds.
    overrides: dict[str, Any] = Field(default_factory=dict)
    provider: Optional[str] = None
    model: Optional[str] = None


class StepView(BaseModel):
    seq: int
    graph_node: str
    kind: str
    title: str
    tool_name: Optional[str]
    payload_in: Optional[dict[str, Any]]
    payload_out: Optional[dict[str, Any]]
    latency_ms: int
    created_at: str


class RunView(BaseModel):
    run_id: str
    scenario_id: str
    case_type: str
    status: str
    provider: str
    model: str
    attempts: int
    sku: Optional[str]
    node_id: Optional[str]
    created_at: str
    completed_at: Optional[str]
    duration_ms: int
    result: Optional[dict[str, Any]]
    steps: list[StepView] = Field(default_factory=list)
