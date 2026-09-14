"""Supplier simulator — the part of the world that answers back.

A purchasing agent that only ever gets what it asked for has never been
tested. This module is the external system whose response may differ from
the request: it confirms in full, confirms partially, or rejects outright,
and it does so deterministically from a seed so that evaluation runs are
reproducible.

It is also the *authoritative* source for supplier availability. A supplier
email claiming "we can only ship 250" is a claim; this API is the record.
Policy requires the agent to corroborate the former with the latter.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.db.models import Supplier, SupplierProduct


@dataclass
class AvailabilityAnswer:
    supplier_id: str
    sku: str
    requested_units: int
    available_units: int
    can_fulfil_in_full: bool
    lead_time_days: int
    unit_price_usd: float
    note: str

    def to_dict(self) -> dict:
        return {
            "supplier_id": self.supplier_id,
            "sku": self.sku,
            "requested_units": self.requested_units,
            "available_units": self.available_units,
            "can_fulfil_in_full": self.can_fulfil_in_full,
            "shortfall_units": max(0, self.requested_units - self.available_units),
            "lead_time_days": self.lead_time_days,
            "unit_price_usd": self.unit_price_usd,
            "source": "supplier_availability_api (authoritative)",
            "note": self.note,
        }


@dataclass
class ConfirmationAnswer:
    confirmed_units: int
    status: str  # confirmed | partially_confirmed | rejected
    reason: str


def _jitter(supplier_id: str, sku: str) -> float:
    """Stable pseudo-random value in [0, 1) for a supplier/SKU pair."""
    digest = hashlib.sha256(
        f"{settings.supplier_sim_seed}:{supplier_id}:{sku}".encode()
    ).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def check_availability(
    session: Session, supplier_id: str, sku: str, requested_units: int
) -> AvailabilityAnswer:
    supplier = session.get(Supplier, supplier_id)
    sp = session.execute(
        select(SupplierProduct).where(
            SupplierProduct.supplier_id == supplier_id, SupplierProduct.sku == sku
        )
    ).scalar_one_or_none()

    if supplier is None or sp is None:
        return AvailabilityAnswer(
            supplier_id=supplier_id,
            sku=sku,
            requested_units=requested_units,
            available_units=0,
            can_fulfil_in_full=False,
            lead_time_days=0,
            unit_price_usd=0.0,
            note=f"{supplier_id} does not carry {sku}.",
        )

    if supplier.status != "active":
        return AvailabilityAnswer(
            supplier_id=supplier_id,
            sku=sku,
            requested_units=requested_units,
            available_units=0,
            can_fulfil_in_full=False,
            lead_time_days=sp.lead_time_days,
            unit_price_usd=sp.unit_price_usd,
            note=(
                f"{supplier.name} is {supplier.status}"
                + (f" ({supplier.status_reason})" if supplier.status_reason else "")
                + " and cannot accept orders."
            ),
        )

    if sp.available_units_override is not None:
        available = sp.available_units_override
        note = "Supplier reports a constrained position for this SKU."
    else:
        available = sp.max_weekly_units
        note = "Supplier reports normal availability."

    return AvailabilityAnswer(
        supplier_id=supplier_id,
        sku=sku,
        requested_units=requested_units,
        available_units=available,
        can_fulfil_in_full=available >= requested_units,
        lead_time_days=sp.lead_time_days,
        unit_price_usd=sp.unit_price_usd,
        note=note,
    )


def confirm_order(
    session: Session, supplier_id: str, sku: str, ordered_units: int
) -> ConfirmationAnswer:
    """What the supplier actually commits to when a PO is submitted.

    Confirmation is bounded by real availability and by the supplier's
    simulated fill ratio, then snapped down to a whole number of cases —
    suppliers ship cases, not units.
    """
    sp = session.execute(
        select(SupplierProduct).where(
            SupplierProduct.supplier_id == supplier_id, SupplierProduct.sku == sku
        )
    ).scalar_one_or_none()
    supplier = session.get(Supplier, supplier_id)

    if sp is None or supplier is None:
        return ConfirmationAnswer(0, "rejected", f"{supplier_id} does not carry {sku}.")
    if supplier.status != "active":
        return ConfirmationAnswer(
            0, "rejected", f"{supplier.name} is {supplier.status} and rejected the order."
        )

    ceiling = sp.available_units_override if sp.available_units_override is not None else sp.max_weekly_units
    confirmable = min(ordered_units, ceiling)
    confirmable = int(math.floor(confirmable * sp.simulated_fill_ratio))

    if sp.case_pack > 1 and confirmable > 0:
        confirmable = (confirmable // sp.case_pack) * sp.case_pack

    if confirmable <= 0:
        return ConfirmationAnswer(
            0,
            "rejected",
            f"{supplier.name} has no stock of {sku} available against this order.",
        )
    if confirmable < ordered_units:
        return ConfirmationAnswer(
            confirmable,
            "partially_confirmed",
            (
                f"{supplier.name} confirmed {confirmable} of {ordered_units} units. "
                f"Availability for this SKU is capped at {ceiling} units this cycle."
            ),
        )
    return ConfirmationAnswer(
        confirmable, "confirmed", f"{supplier.name} confirmed the order in full."
    )
