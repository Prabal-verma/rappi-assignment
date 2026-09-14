"""Read views onto the mock ERP, for the dashboard.

These exist so a reviewer can see the world the agent acted on — and see it
change after a run — rather than taking the agent's word for it.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import (
    Budget,
    DomainEvent,
    InventoryPosition,
    Node,
    Product,
    PurchaseOrder,
    Supplier,
    SupplierProduct,
    TransferOrder,
)
from app.db.session import get_session
from app.domain import calculations as calc
from app.domain import clock
from app.domain import constraints as cons

router = APIRouter()


@router.get("/products")
def products(session: Session = Depends(get_session)) -> dict[str, Any]:
    rows = list(session.execute(select(Product)).scalars())
    return {
        "products": [
            {
                "sku": p.sku,
                "name": p.name,
                "category": p.category,
                "unit_cost_usd": p.unit_cost_usd,
                "retail_price_usd": p.retail_price_usd,
                "unit_margin_usd": p.unit_margin_usd,
                "case_pack": p.case_pack,
                "unit_volume_m3": p.unit_volume_m3,
                "shelf_life_days": p.shelf_life_days,
                "is_perishable": p.is_perishable,
            }
            for p in rows
        ]
    }


@router.get("/nodes")
def nodes(session: Session = Depends(get_session)) -> dict[str, Any]:
    rows = list(session.execute(select(Node)).scalars())
    out = []
    for n in rows:
        state = cons.node_storage_state(session, n.node_id)
        out.append(
            {
                "node_id": n.node_id,
                "name": n.name,
                "country": n.country,
                "city": n.city,
                "capacity_m3": state.capacity_m3,
                "used_m3": state.used_m3,
                "inbound_m3": state.inbound_m3,
                "free_m3": state.free_m3,
                "utilisation_pct": round(
                    100 * (state.used_m3 + state.inbound_m3) / max(state.capacity_m3, 0.001), 1
                ),
                "review_period_days": n.review_period_days,
                "service_level_target": n.service_level_target,
            }
        )
    return {"nodes": out}


@router.get("/inventory")
def inventory(session: Session = Depends(get_session)) -> dict[str, Any]:
    rows = session.execute(
        select(InventoryPosition, Product).join(Product, Product.sku == InventoryPosition.sku)
    ).all()
    out = []
    for inv, product in rows:
        shipments = calc.inbound_shipments(session, inv.sku, inv.node_id)
        in_transit = sum(s.units for s in shipments)
        out.append(
            {
                "sku": inv.sku,
                "name": product.name,
                "node_id": inv.node_id,
                "on_hand_units": inv.on_hand_units,
                "reserved_units": inv.reserved_units,
                "available_units": inv.available_units,
                "in_transit_units": in_transit,
                "inventory_position_units": inv.available_units + in_transit,
            }
        )
    return {"inventory": out}


@router.get("/suppliers")
def suppliers(session: Session = Depends(get_session)) -> dict[str, Any]:
    rows = list(session.execute(select(Supplier)).scalars())
    catalogue = session.execute(select(SupplierProduct)).scalars()
    by_supplier: dict[str, list[dict]] = {}
    for sp in catalogue:
        by_supplier.setdefault(sp.supplier_id, []).append(
            {
                "sku": sp.sku,
                "unit_price_usd": sp.unit_price_usd,
                "moq_units": sp.moq_units,
                "case_pack": sp.case_pack,
                "lead_time_days": sp.lead_time_days,
                "max_weekly_units": sp.max_weekly_units,
            }
        )
    return {
        "suppliers": [
            {
                "supplier_id": s.supplier_id,
                "name": s.name,
                "country": s.country,
                "status": s.status,
                "status_reason": s.status_reason,
                "lead_time_days": s.lead_time_days,
                "fill_rate_90d": s.fill_rate_90d,
                "on_time_rate_90d": s.on_time_rate_90d,
                "reliability_score": s.reliability_score,
                "allows_split_delivery": s.allows_split_delivery,
                "catalogue": by_supplier.get(s.supplier_id, []),
            }
            for s in rows
        ]
    }


@router.get("/purchase-orders")
def purchase_orders(session: Session = Depends(get_session)) -> dict[str, Any]:
    rows = list(
        session.execute(select(PurchaseOrder).order_by(PurchaseOrder.created_at.desc())).scalars()
    )
    today = clock.today()
    return {
        "purchase_orders": [
            {
                "po_id": po.po_id,
                "supplier_id": po.supplier_id,
                "node_id": po.node_id,
                "status": po.status,
                "created_by": po.created_by,
                "source_run_id": po.source_run_id,
                "expected_delivery_date": po.expected_delivery_date.isoformat(),
                "eta_days": (po.expected_delivery_date - today).days,
                "ordered_value_usd": po.ordered_value_usd,
                "committed_value_usd": po.committed_value_usd,
                "notes": po.notes,
                "lines": [
                    {
                        "sku": line.sku,
                        "ordered_units": line.ordered_units,
                        "confirmed_units": line.confirmed_units,
                        "received_units": line.received_units,
                        "unit_price_usd": line.unit_price_usd,
                    }
                    for line in po.lines
                ],
            }
            for po in rows
        ]
    }


@router.get("/transfer-orders")
def transfer_orders(session: Session = Depends(get_session)) -> dict[str, Any]:
    rows = list(session.execute(select(TransferOrder)).scalars())
    return {
        "transfer_orders": [
            {
                "to_id": t.to_id,
                "sku": t.sku,
                "from_node_id": t.from_node_id,
                "to_node_id": t.to_node_id,
                "units": t.units,
                "status": t.status,
                "transit_days": t.transit_days,
                "source_run_id": t.source_run_id,
            }
            for t in rows
        ]
    }


@router.get("/budgets")
def budgets(session: Session = Depends(get_session)) -> dict[str, Any]:
    rows = list(session.execute(select(Budget)).scalars())
    return {
        "budgets": [
            {
                "node_id": b.node_id,
                "category": b.category,
                "period_start": b.period_start.isoformat(),
                "period_end": b.period_end.isoformat(),
                "allocated_usd": b.allocated_usd,
                "committed_usd": b.committed_usd,
                "spent_usd": b.spent_usd,
                "available_usd": b.available_usd,
            }
            for b in rows
        ]
    }


@router.get("/events")
def events(limit: int = 60, session: Session = Depends(get_session)) -> dict[str, Any]:
    rows = list(
        session.execute(
            select(DomainEvent).order_by(DomainEvent.created_at.desc()).limit(limit)
        ).scalars()
    )
    return {
        "events": [
            {
                "id": e.id,
                "entity_type": e.entity_type,
                "entity_id": e.entity_id,
                "event_type": e.event_type,
                "actor": e.actor,
                "run_id": e.run_id,
                "payload": e.payload,
                "created_at": e.created_at.isoformat(),
            }
            for e in rows
        ]
    }


@router.get("/analysis/{sku}/{node_id}")
def analysis(sku: str, node_id: str, session: Session = Depends(get_session)) -> dict[str, Any]:
    """The computed replenishment view for any SKU/node, outside a run."""
    return calc.analyze_replenishment(session, sku, node_id).model_dump(mode="json")
