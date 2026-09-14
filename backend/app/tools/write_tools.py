"""State-changing operations.

These are deliberately **not** registered as model-callable tools. The agent
proposes actions as structured data; the constraint engine and the autonomy
policy vet them; only then does the act node call into this module. A model
cannot invoke a write directly because no write is exposed to it — which
means no amount of clever prompting, or injected text in a supplier email,
can produce an unauthorised purchase order.

Every function here is idempotent on an idempotency key, writes a domain
event, and keeps the budget commitment in step with reality.
"""

from __future__ import annotations

import hashlib
from datetime import date

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db.models import (
    Budget,
    DomainEvent,
    InventoryPosition,
    Product,
    PurchaseOrder,
    PurchaseOrderLine,
    SupplierProduct,
    TransferOrder,
)
from app.domain import clock
from app.domain import constraints as cons
from app.domain.schemas import ActionResult, ActionType, ProposedAction
from app.services import supplier_sim


def _next_id(session: Session, prefix: str, model, column) -> str:
    count = session.execute(select(func.count()).select_from(model)).scalar_one()
    return f"{prefix}-{9000 + count + 1}"


def make_idempotency_key(run_id: str, action: ProposedAction) -> str:
    """Stable across retries of the same action within a run.

    A replan that proposes a *different* quantity gets a different key and
    is allowed through; a blind retry of the same action is not.
    """
    raw = f"{run_id}:{action.signature()}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


def record_event(
    session: Session,
    entity_type: str,
    entity_id: str,
    event_type: str,
    run_id: str | None,
    payload: dict,
    actor: str = "agent",
) -> None:
    session.add(
        DomainEvent(
            entity_type=entity_type,
            entity_id=entity_id,
            event_type=event_type,
            actor=actor,
            run_id=run_id,
            payload=payload,
        )
    )


def _adjust_budget(session: Session, node_id: str, sku: str, delta_usd: float, run_id: str | None) -> None:
    product = session.get(Product, sku)
    if product is None:
        return
    budget = cons.find_budget(session, node_id, product.category)
    if budget is None:
        return
    budget.committed_usd = round(max(0.0, budget.committed_usd + delta_usd), 2)
    record_event(
        session,
        "budget",
        f"{node_id}:{product.category}",
        "committed_changed",
        run_id,
        {"delta_usd": round(delta_usd, 2), "committed_usd": budget.committed_usd},
    )


# ─────────────────────────────────────────────────────────────────────
# Purchase orders
# ─────────────────────────────────────────────────────────────────────


def create_purchase_order(
    session: Session,
    run_id: str,
    sku: str,
    node_id: str,
    supplier_id: str,
    units: int,
    reason: str = "",
    idempotency_key: str | None = None,
    submit: bool = True,
) -> ActionResult:
    """Create a PO and submit it to the supplier.

    Submission is where reality answers back: the supplier may confirm in
    full, confirm part, or reject. The returned result carries what the
    supplier actually committed to, which is frequently not what was asked.
    """
    action = ProposedAction(
        action_type=ActionType.CREATE_PO,
        sku=sku,
        node_id=node_id,
        supplier_id=supplier_id,
        units=units,
        reason=reason,
    )

    if idempotency_key:
        existing = session.execute(
            select(PurchaseOrder).where(PurchaseOrder.idempotency_key == idempotency_key)
        ).scalar_one_or_none()
        if existing is not None:
            line = next((l for l in existing.lines if l.sku == sku), None)
            return ActionResult(
                action=action,
                succeeded=True,
                entity_id=existing.po_id,
                outcome="already_exists",
                confirmed_units=line.effective_units if line else None,
                detail={"note": "Identical action already executed in this run; not duplicated."},
            )

    sp = session.execute(
        select(SupplierProduct).where(
            SupplierProduct.supplier_id == supplier_id, SupplierProduct.sku == sku
        )
    ).scalar_one_or_none()
    if sp is None:
        return ActionResult(
            action=action,
            succeeded=False,
            outcome="rejected",
            error=f"{supplier_id} does not carry {sku}.",
        )

    po_id = _next_id(session, "PO", PurchaseOrder, PurchaseOrder.po_id)
    eta: date = clock.days_from_today(sp.lead_time_days)

    po = PurchaseOrder(
        po_id=po_id,
        supplier_id=supplier_id,
        node_id=node_id,
        status="draft",
        expected_delivery_date=eta,
        created_by=f"agent:{run_id}",
        source_run_id=run_id,
        idempotency_key=idempotency_key,
        notes=reason,
    )
    po.lines.append(
        PurchaseOrderLine(
            po_id=po_id, sku=sku, ordered_units=units, unit_price_usd=sp.unit_price_usd
        )
    )
    session.add(po)
    session.flush()

    record_event(
        session,
        "purchase_order",
        po_id,
        "created",
        run_id,
        {"sku": sku, "node_id": node_id, "supplier_id": supplier_id, "units": units, "reason": reason},
    )

    if not submit:
        _adjust_budget(session, node_id, sku, units * sp.unit_price_usd, run_id)
        return ActionResult(
            action=action,
            succeeded=True,
            entity_id=po_id,
            outcome="draft",
            confirmed_units=None,
            detail={"status": "draft", "expected_delivery_date": eta.isoformat()},
        )

    po.status = "submitted"
    confirmation = supplier_sim.confirm_order(session, supplier_id, sku, units)
    line = po.lines[0]
    line.confirmed_units = confirmation.confirmed_units
    po.status = confirmation.status

    committed_value = confirmation.confirmed_units * sp.unit_price_usd
    _adjust_budget(session, node_id, sku, committed_value, run_id)

    record_event(
        session,
        "purchase_order",
        po_id,
        f"supplier_{confirmation.status}",
        run_id,
        {
            "ordered_units": units,
            "confirmed_units": confirmation.confirmed_units,
            "reason": confirmation.reason,
        },
    )

    return ActionResult(
        action=action,
        succeeded=confirmation.confirmed_units > 0,
        entity_id=po_id,
        outcome=confirmation.status,
        confirmed_units=confirmation.confirmed_units,
        detail={
            "ordered_units": units,
            "confirmed_units": confirmation.confirmed_units,
            "shortfall_units": max(0, units - confirmation.confirmed_units),
            "unit_price_usd": sp.unit_price_usd,
            "committed_value_usd": round(committed_value, 2),
            "expected_delivery_date": eta.isoformat(),
            "supplier_response": confirmation.reason,
        },
        error=None if confirmation.confirmed_units > 0 else confirmation.reason,
    )


def modify_purchase_order(
    session: Session,
    run_id: str,
    po_id: str,
    sku: str,
    units: int,
    reason: str = "",
) -> ActionResult:
    """Change the quantity on an existing PO and re-confirm with the supplier."""
    action = ProposedAction(
        action_type=ActionType.MODIFY_PO, po_id=po_id, sku=sku, units=units, reason=reason
    )
    po = session.get(PurchaseOrder, po_id)
    if po is None:
        return ActionResult(
            action=action, succeeded=False, outcome="not_found", error=f"No purchase order {po_id}."
        )
    if not po.is_open:
        return ActionResult(
            action=action,
            succeeded=False,
            outcome="not_modifiable",
            error=f"{po_id} is {po.status} and can no longer be modified.",
        )

    line = next((l for l in po.lines if l.sku == sku), None)
    if line is None:
        return ActionResult(
            action=action, succeeded=False, outcome="not_found", error=f"{po_id} has no line for {sku}."
        )

    previous_committed = line.effective_units * line.unit_price_usd
    previous_units = line.ordered_units
    line.ordered_units = units

    confirmation = supplier_sim.confirm_order(session, po.supplier_id, sku, units)
    line.confirmed_units = confirmation.confirmed_units
    po.status = confirmation.status
    po.notes = (po.notes or "") + f"\n[{clock.today().isoformat()}] {reason}".rstrip()
    po.source_run_id = run_id

    new_committed = confirmation.confirmed_units * line.unit_price_usd
    _adjust_budget(session, po.node_id, sku, new_committed - previous_committed, run_id)

    record_event(
        session,
        "purchase_order",
        po_id,
        "modified",
        run_id,
        {
            "previous_units": previous_units,
            "new_units": units,
            "confirmed_units": confirmation.confirmed_units,
            "reason": reason,
        },
    )

    return ActionResult(
        action=action,
        succeeded=confirmation.confirmed_units > 0,
        entity_id=po_id,
        outcome=confirmation.status,
        confirmed_units=confirmation.confirmed_units,
        detail={
            "previous_units": previous_units,
            "requested_units": units,
            "confirmed_units": confirmation.confirmed_units,
            "shortfall_units": max(0, units - confirmation.confirmed_units),
            "committed_value_usd": round(new_committed, 2),
            "supplier_response": confirmation.reason,
        },
        error=None if confirmation.confirmed_units > 0 else confirmation.reason,
    )


def cancel_purchase_order(
    session: Session, run_id: str, po_id: str, reason: str = ""
) -> ActionResult:
    action = ProposedAction(action_type=ActionType.CANCEL_PO, po_id=po_id, reason=reason)
    po = session.get(PurchaseOrder, po_id)
    if po is None:
        return ActionResult(
            action=action, succeeded=False, outcome="not_found", error=f"No purchase order {po_id}."
        )
    if not po.is_open:
        return ActionResult(
            action=action,
            succeeded=False,
            outcome="not_cancellable",
            error=f"{po_id} is already {po.status}.",
        )

    released = po.committed_value_usd
    for line in po.lines:
        _adjust_budget(session, po.node_id, line.sku, -(line.effective_units * line.unit_price_usd), run_id)
    po.status = "cancelled"
    po.source_run_id = run_id

    record_event(
        session, "purchase_order", po_id, "cancelled", run_id, {"reason": reason, "released_usd": released}
    )
    return ActionResult(
        action=action,
        succeeded=True,
        entity_id=po_id,
        outcome="cancelled",
        detail={"released_budget_usd": round(released, 2), "reason": reason},
    )


def create_transfer_order(
    session: Session,
    run_id: str,
    sku: str,
    from_node_id: str,
    to_node_id: str,
    units: int,
    reason: str = "",
) -> ActionResult:
    """Move stock between nodes — no spend, but it must not create a
    stockout at the source."""
    action = ProposedAction(
        action_type=ActionType.CREATE_TRANSFER,
        sku=sku,
        node_id=to_node_id,
        from_node_id=from_node_id,
        units=units,
        reason=reason,
    )

    source = session.execute(
        select(InventoryPosition).where(
            InventoryPosition.sku == sku, InventoryPosition.node_id == from_node_id
        )
    ).scalar_one_or_none()
    if source is None or source.available_units < units:
        have = source.available_units if source else 0
        return ActionResult(
            action=action,
            succeeded=False,
            outcome="insufficient_stock",
            error=f"{from_node_id} has {have} available units of {sku}, cannot release {units}.",
        )

    to_id = _next_id(session, "TO", TransferOrder, TransferOrder.to_id)
    session.add(
        TransferOrder(
            to_id=to_id,
            from_node_id=from_node_id,
            to_node_id=to_node_id,
            sku=sku,
            units=units,
            status="in_transit",
            transit_days=1,
            source_run_id=run_id,
        )
    )
    source.on_hand_units -= units
    session.flush()

    record_event(
        session,
        "transfer_order",
        to_id,
        "created",
        run_id,
        {"sku": sku, "from": from_node_id, "to": to_node_id, "units": units, "reason": reason},
    )
    return ActionResult(
        action=action,
        succeeded=True,
        entity_id=to_id,
        outcome="in_transit",
        confirmed_units=units,
        detail={
            "transfer_order_id": to_id,
            "units": units,
            "from_node_id": from_node_id,
            "to_node_id": to_node_id,
            "transit_days": 1,
        },
    )


# ─────────────────────────────────────────────────────────────────────
# Dispatch
# ─────────────────────────────────────────────────────────────────────


def execute_action(session: Session, run_id: str, action: ProposedAction) -> ActionResult:
    """Route a vetted action to its implementation."""
    key = make_idempotency_key(run_id, action)

    if action.action_type == ActionType.CREATE_PO:
        if not (action.sku and action.node_id and action.supplier_id and action.units):
            return ActionResult(
                action=action,
                succeeded=False,
                outcome="invalid",
                error="create_purchase_order needs sku, node_id, supplier_id and units.",
            )
        return create_purchase_order(
            session, run_id, action.sku, action.node_id, action.supplier_id,
            action.units, action.reason, idempotency_key=key,
        )

    if action.action_type == ActionType.MODIFY_PO:
        if not (action.po_id and action.sku and action.units is not None):
            return ActionResult(
                action=action,
                succeeded=False,
                outcome="invalid",
                error="modify_purchase_order needs po_id, sku and units.",
            )
        return modify_purchase_order(
            session, run_id, action.po_id, action.sku, action.units, action.reason
        )

    if action.action_type == ActionType.CANCEL_PO:
        if not action.po_id:
            return ActionResult(
                action=action, succeeded=False, outcome="invalid", error="cancel needs po_id."
            )
        return cancel_purchase_order(session, run_id, action.po_id, action.reason)

    if action.action_type == ActionType.CREATE_TRANSFER:
        if not (action.sku and action.from_node_id and action.node_id and action.units):
            return ActionResult(
                action=action,
                succeeded=False,
                outcome="invalid",
                error="create_transfer_order needs sku, from_node_id, node_id and units.",
            )
        return create_transfer_order(
            session, run_id, action.sku, action.from_node_id, action.node_id,
            action.units, action.reason,
        )

    if action.action_type in (ActionType.NO_ACTION, ActionType.ESCALATE):
        return ActionResult(
            action=action, succeeded=True, outcome=action.action_type.value, detail={}
        )

    return ActionResult(
        action=action,
        succeeded=False,
        outcome="unsupported",
        error=f"No executor for action type {action.action_type}.",
    )
