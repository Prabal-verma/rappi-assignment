"""Autonomy policy — what the agent may do by itself.

The separation that matters: *can this action be executed* is a constraint
question, answered in constraints.py. *Should a human look at it first* is a
policy question, answered here. Both run before any write, and neither is
expressed in a prompt, so neither can be talked out of by the model.

The policy is deliberately conservative in the directions where being wrong
is expensive and asymmetric: large spend, big deviations from the planning
system, unreliable suppliers, and perishable overbuys.
"""

from __future__ import annotations

from sqlalchemy.orm import Session

from app.config import settings
from app.db.models import Product, Supplier
from app.domain.schemas import (
    ActionType,
    AuthorizationOutcome,
    ConstraintReport,
    Decision,
    DecisionType,
    ProposedAction,
)

# Supplier reliability below which a sizeable order wants a human eye.
RELIABILITY_REVIEW_THRESHOLD = 0.85
RELIABILITY_REVIEW_MIN_VALUE_USD = 1000.0


def order_value_usd(session: Session, action: ProposedAction) -> float:
    if not action.units or not action.sku:
        return 0.0
    from app.db.models import SupplierProduct
    from sqlalchemy import select

    price = None
    if action.supplier_id:
        sp = session.execute(
            select(SupplierProduct).where(
                SupplierProduct.supplier_id == action.supplier_id,
                SupplierProduct.sku == action.sku,
            )
        ).scalar_one_or_none()
        price = sp.unit_price_usd if sp else None
    if price is None:
        product = session.get(Product, action.sku)
        price = product.unit_cost_usd if product else 0.0
    return round(action.units * price, 2)


def authorize(
    session: Session,
    decision: Decision,
    constraint_report: ConstraintReport,
    system_recommendation_units: int | None = None,
) -> AuthorizationOutcome:
    """Decide whether the agent executes, asks a human, or must replan."""
    reasons: list[str] = []
    policy_refs: list[str] = []

    # ---- hard constraints win outright ----------------------------
    if not constraint_report.passed:
        return AuthorizationOutcome(
            mode="blocked",
            reasons=[f"[{v.code}] {v.message}" for v in constraint_report.blocking],
            policy_refs=["POL-CONSTRAINT-01"],
            constraint_report=constraint_report,
            risk_level="high",
        )

    write_actions = [
        a
        for a in decision.proposed_actions
        if a.action_type in {ActionType.CREATE_PO, ActionType.MODIFY_PO, ActionType.CANCEL_PO, ActionType.CREATE_TRANSFER}
    ]

    # ---- nothing to execute ----------------------------------------
    if not write_actions:
        if decision.decision_type == DecisionType.ESCALATE:
            return AuthorizationOutcome(
                mode="needs_approval",
                reasons=["The agent escalated: the situation needs a human decision."],
                policy_refs=["POL-ESCALATE-01"],
                constraint_report=constraint_report,
                risk_level=decision.risk_level,
            )
        return AuthorizationOutcome(
            mode="autonomous",
            reasons=["No state-changing action proposed."],
            constraint_report=constraint_report,
            risk_level="low",
        )

    total_value = sum(order_value_usd(session, a) for a in write_actions)
    needs_approval = False

    # ---- value ceiling ----------------------------------------------
    if total_value > settings.autonomy_auto_approve_max_value_usd:
        needs_approval = True
        reasons.append(
            f"Order value ${total_value:,.2f} exceeds the ${settings.autonomy_auto_approve_max_value_usd:,.0f} "
            f"autonomous execution ceiling."
        )
        policy_refs.append("POL-AUTONOMY-01")

    # ---- deviation from the planning system --------------------------
    if system_recommendation_units:
        proposed_units = sum(a.units or 0 for a in write_actions if a.action_type != ActionType.CANCEL_PO)
        deviation = abs(proposed_units - system_recommendation_units) / max(system_recommendation_units, 1)
        if deviation > settings.autonomy_max_deviation_without_approval:
            needs_approval = True
            reasons.append(
                f"Proposed {proposed_units} units against a system recommendation of "
                f"{system_recommendation_units} — a {deviation:.0%} deviation, above the "
                f"{settings.autonomy_max_deviation_without_approval:.0%} threshold for "
                f"unattended execution."
            )
            policy_refs.append("POL-AUTONOMY-02")

    # ---- supplier reliability ----------------------------------------
    for action in write_actions:
        if not action.supplier_id:
            continue
        supplier = session.get(Supplier, action.supplier_id)
        if supplier is None:
            continue
        if (
            supplier.reliability_score < RELIABILITY_REVIEW_THRESHOLD
            and order_value_usd(session, action) > RELIABILITY_REVIEW_MIN_VALUE_USD
        ):
            needs_approval = True
            reasons.append(
                f"{supplier.name} has a reliability score of {supplier.reliability_score:.2f} "
                f"(fill rate {supplier.fill_rate_90d:.0%}, on-time {supplier.on_time_rate_90d:.0%}), "
                f"below the {RELIABILITY_REVIEW_THRESHOLD:.2f} threshold for orders of this size."
            )
            policy_refs.append("POL-SUPPLIER-03")

    # ---- perishable overbuy -------------------------------------------
    for action in write_actions:
        if not action.sku:
            continue
        product = session.get(Product, action.sku)
        if product and product.is_perishable and constraint_report.warnings:
            cover_warnings = [w for w in constraint_report.warnings if w.code.startswith("C08")]
            if cover_warnings:
                needs_approval = True
                reasons.append(
                    f"{product.name} is perishable and the order approaches its shelf-life "
                    f"limit; a buyer should confirm the waste risk is acceptable."
                )
                policy_refs.append("POL-WASTE-02")

    # ---- agent's own uncertainty ---------------------------------------
    if decision.confidence < 0.6:
        needs_approval = True
        reasons.append(
            f"The agent reported low confidence ({decision.confidence:.2f}) in this decision."
        )
        policy_refs.append("POL-AUTONOMY-04")

    if decision.risk_level == "high":
        needs_approval = True
        reasons.append("The decision is self-assessed as high risk.")
        policy_refs.append("POL-AUTONOMY-05")

    if needs_approval:
        return AuthorizationOutcome(
            mode="needs_approval",
            reasons=reasons,
            policy_refs=sorted(set(policy_refs)),
            constraint_report=constraint_report,
            risk_level="high" if decision.risk_level == "high" else "medium",
        )

    return AuthorizationOutcome(
        mode="autonomous",
        reasons=[
            f"Order value ${total_value:,.2f} is within the autonomous ceiling, all "
            f"{len(constraint_report.checks_run)} constraint checks passed, and the "
            f"deviation from the system recommendation is inside policy."
        ],
        policy_refs=["POL-AUTONOMY-01"],
        constraint_report=constraint_report,
        risk_level="low",
    )
