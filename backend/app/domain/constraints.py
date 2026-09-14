"""The constraint engine — the agent's independent critic.

Nothing here is a prompt. These are the hard rules of the purchasing
domain, expressed in code, and they run in two places:

  * pre-flight, against a *proposed* purchase, before anything is written;
  * post-hoc, against the *persisted* state, after the world has answered.

The post-hoc pass is the one that matters. It re-reads ground truth from
the database rather than trusting anything the agent said, and diffs the
result against the expectation the agent committed to before acting. That
diff is what closes the feedback loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import (
    Budget,
    InventoryPosition,
    Node,
    Product,
    PurchaseOrder,
    PurchaseOrderLine,
    Supplier,
    SupplierProduct,
)
from app.domain import calculations as calc
from app.domain import clock
from app.domain.schemas import (
    ActionExpectation,
    ConstraintReport,
    ConstraintViolation,
    ExpectationDiff,
    Severity,
    ValidationReport,
    ValidationVerdict,
)

# Policy ceiling on how much forward cover a single order may create.
MAX_COVER_DAYS = 21


@dataclass
class ProposedPurchase:
    """A purchase the agent wants to make, before it exists anywhere."""

    sku: str
    node_id: str
    supplier_id: str
    units: int
    unit_price_usd: Optional[float] = None
    expected_delivery_date: Optional[date] = None
    # Set when modifying an existing PO, so its current value is not
    # double-counted against the budget.
    replaces_po_id: Optional[str] = None
    system_recommendation_units: Optional[int] = None


@dataclass
class StorageState:
    capacity_m3: float
    used_m3: float
    inbound_m3: float

    @property
    def free_m3(self) -> float:
        return round(self.capacity_m3 - self.used_m3 - self.inbound_m3, 4)


def node_storage_state(session: Session, node_id: str) -> StorageState:
    """Volume already consumed at a node, including stock still inbound.

    Inbound counts: capacity that will be occupied when a confirmed PO
    lands is not free capacity today.
    """
    node = session.get(Node, node_id)
    if node is None:
        raise ValueError(f"Unknown node {node_id}")

    rows = session.execute(
        select(InventoryPosition, Product)
        .join(Product, Product.sku == InventoryPosition.sku)
        .where(InventoryPosition.node_id == node_id)
    ).all()
    used = node.baseline_used_m3 + sum(
        inv.on_hand_units * prod.unit_volume_m3 for inv, prod in rows
    )

    open_rows = session.execute(
        select(PurchaseOrder, PurchaseOrderLine, Product)
        .join(PurchaseOrderLine, PurchaseOrderLine.po_id == PurchaseOrder.po_id)
        .join(Product, Product.sku == PurchaseOrderLine.sku)
        .where(
            PurchaseOrder.node_id == node_id,
            PurchaseOrder.status.in_(
                ["submitted", "confirmed", "partially_confirmed", "pending_approval"]
            ),
        )
    ).all()
    inbound = sum(
        max(0, line.effective_units - line.received_units) * prod.unit_volume_m3
        for _po, line, prod in open_rows
    )

    return StorageState(
        capacity_m3=round(node.storage_capacity_m3, 4),
        used_m3=round(used, 4),
        inbound_m3=round(inbound, 4),
    )


def find_budget(session: Session, node_id: str, category: str) -> Optional[Budget]:
    today = clock.today()
    return session.execute(
        select(Budget).where(
            Budget.node_id == node_id,
            Budget.category == category,
            Budget.period_start <= today,
            Budget.period_end >= today,
        )
    ).scalar_one_or_none()


# ─────────────────────────────────────────────────────────────────────
# Pre-flight evaluation
# ─────────────────────────────────────────────────────────────────────


def evaluate_purchase(session: Session, purchase: ProposedPurchase) -> ConstraintReport:
    """Run every hard rule against a proposed purchase.

    Blocking violations mean the purchase must not be executed as
    specified. Warnings are real findings the agent must acknowledge in its
    rationale but which do not by themselves stop the order.
    """
    blocking: list[ConstraintViolation] = []
    warnings: list[ConstraintViolation] = []
    checks: list[str] = []

    product = session.get(Product, purchase.sku)
    node = session.get(Node, purchase.node_id)
    supplier = session.get(Supplier, purchase.supplier_id)

    if product is None or node is None:
        blocking.append(
            ConstraintViolation(
                code="C00_REFERENCE",
                severity=Severity.BLOCK,
                message=f"Unknown sku or node: {purchase.sku} @ {purchase.node_id}.",
                remedy_hint="Check the identifiers before ordering.",
            )
        )
        return ConstraintReport(passed=False, blocking=blocking, checks_run=["C00_REFERENCE"])

    # ---- C01 supplier status --------------------------------------
    checks.append("C01_SUPPLIER_STATUS")
    if supplier is None:
        blocking.append(
            ConstraintViolation(
                code="C01_SUPPLIER_STATUS",
                severity=Severity.BLOCK,
                message=f"Supplier {purchase.supplier_id} does not exist.",
                remedy_hint="Pick a supplier from get_alternate_suppliers.",
            )
        )
        return ConstraintReport(passed=False, blocking=blocking, checks_run=checks)
    if supplier.status != "active":
        blocking.append(
            ConstraintViolation(
                code="C01_SUPPLIER_STATUS",
                severity=Severity.BLOCK,
                message=(
                    f"Supplier {supplier.name} is {supplier.status}"
                    + (f": {supplier.status_reason}" if supplier.status_reason else "")
                    + ". Purchasing from a non-active supplier is not permitted."
                ),
                remedy_hint="Source from an active supplier or escalate to sourcing.",
            )
        )

    # ---- C02 supplier carries the sku ------------------------------
    checks.append("C02_SUPPLIER_CARRIES_SKU")
    sp = session.execute(
        select(SupplierProduct).where(
            SupplierProduct.supplier_id == purchase.supplier_id,
            SupplierProduct.sku == purchase.sku,
        )
    ).scalar_one_or_none()
    if sp is None:
        blocking.append(
            ConstraintViolation(
                code="C02_SUPPLIER_CARRIES_SKU",
                severity=Severity.BLOCK,
                message=f"{purchase.supplier_id} does not list {purchase.sku} in its catalogue.",
                remedy_hint="Use get_alternate_suppliers to find one that carries this SKU.",
            )
        )
        return ConstraintReport(passed=False, blocking=blocking, warnings=warnings, checks_run=checks)

    unit_price = purchase.unit_price_usd or sp.unit_price_usd
    order_value = round(purchase.units * unit_price, 2)

    # ---- C03 minimum order quantity --------------------------------
    checks.append("C03_MOQ")
    if purchase.units > 0 and sp.moq_units and purchase.units < sp.moq_units:
        blocking.append(
            ConstraintViolation(
                code="C03_MOQ",
                severity=Severity.BLOCK,
                message=(
                    f"{purchase.units} units is below {supplier.name}'s minimum order "
                    f"quantity of {sp.moq_units} for this SKU."
                ),
                observed=purchase.units,
                limit=sp.moq_units,
                unit="units",
                remedy_hint=(
                    "Raise the quantity to the MOQ if the extra stock is usable, "
                    "or source from a supplier with a lower MOQ."
                ),
            )
        )

    # ---- C04 case pack ---------------------------------------------
    checks.append("C04_CASE_PACK")
    if purchase.units > 0 and sp.case_pack > 1 and purchase.units % sp.case_pack != 0:
        blocking.append(
            ConstraintViolation(
                code="C04_CASE_PACK",
                severity=Severity.BLOCK,
                message=(
                    f"{purchase.units} units is not a multiple of the case pack "
                    f"({sp.case_pack}). The supplier cannot ship a partial case."
                ),
                observed=purchase.units,
                limit=sp.case_pack,
                unit="units",
                remedy_hint=f"Round to the nearest multiple of {sp.case_pack}.",
            )
        )

    # ---- C05 supplier weekly capacity ------------------------------
    checks.append("C05_SUPPLIER_CAPACITY")
    if purchase.units > sp.max_weekly_units:
        blocking.append(
            ConstraintViolation(
                code="C05_SUPPLIER_CAPACITY",
                severity=Severity.BLOCK,
                message=(
                    f"{purchase.units} units exceeds {supplier.name}'s stated weekly "
                    f"capacity of {sp.max_weekly_units} for this SKU."
                ),
                observed=purchase.units,
                limit=sp.max_weekly_units,
                unit="units",
                remedy_hint="Split the volume across suppliers or across two weeks.",
            )
        )

    # ---- C06 budget -------------------------------------------------
    checks.append("C06_BUDGET")
    budget = find_budget(session, purchase.node_id, product.category)
    if budget is None:
        warnings.append(
            ConstraintViolation(
                code="C06_BUDGET",
                severity=Severity.WARN,
                message=(
                    f"No active budget found for {product.category} at {purchase.node_id}; "
                    f"spend cannot be verified."
                ),
                remedy_hint="Confirm with finance before committing.",
            )
        )
    else:
        # A modification releases the value it replaces.
        released = 0.0
        if purchase.replaces_po_id:
            old = session.get(PurchaseOrder, purchase.replaces_po_id)
            if old is not None:
                released = old.committed_value_usd
        effective_available = round(budget.available_usd + released, 2)
        if order_value > effective_available:
            blocking.append(
                ConstraintViolation(
                    code="C06_BUDGET",
                    severity=Severity.BLOCK,
                    message=(
                        f"Order value ${order_value:,.2f} exceeds the available "
                        f"{product.category} budget of ${effective_available:,.2f} at "
                        f"{purchase.node_id} for the current period."
                    ),
                    observed=order_value,
                    limit=effective_available,
                    unit="USD",
                    remedy_hint=(
                        f"Reduce to at most {int(effective_available // unit_price)} units, "
                        f"or request a budget exception with the cost of the shortfall."
                    ),
                )
            )
        elif order_value > effective_available * 0.85:
            warnings.append(
                ConstraintViolation(
                    code="C06_BUDGET",
                    severity=Severity.WARN,
                    message=(
                        f"Order consumes {order_value / max(effective_available, 0.01):.0%} of "
                        f"the remaining {product.category} budget, leaving little room for "
                        f"the rest of the period."
                    ),
                    observed=order_value,
                    limit=effective_available,
                    unit="USD",
                )
            )

    # ---- C07 storage capacity ---------------------------------------
    checks.append("C07_STORAGE")
    storage = node_storage_state(session, purchase.node_id)
    order_volume = round(purchase.units * product.unit_volume_m3, 4)
    if order_volume > storage.free_m3:
        max_units = int(max(0.0, storage.free_m3) // product.unit_volume_m3) if product.unit_volume_m3 else 0
        blocking.append(
            ConstraintViolation(
                code="C07_STORAGE",
                severity=Severity.BLOCK,
                message=(
                    f"{purchase.units} units need {order_volume:.2f} m³ but only "
                    f"{storage.free_m3:.2f} m³ is free at {purchase.node_id} "
                    f"(capacity {storage.capacity_m3:.1f} m³, {storage.used_m3:.1f} m³ on hand, "
                    f"{storage.inbound_m3:.1f} m³ already inbound)."
                ),
                observed=order_volume,
                limit=storage.free_m3,
                unit="m³",
                remedy_hint=(
                    f"At most {max_units} units fit. Consider a phased delivery, a smaller "
                    f"order, or a transfer to a node with space."
                ),
            )
        )
    elif order_volume > storage.free_m3 * 0.9:
        warnings.append(
            ConstraintViolation(
                code="C07_STORAGE",
                severity=Severity.WARN,
                message=(
                    f"This order fills {order_volume / max(storage.free_m3, 0.001):.0%} of the "
                    f"remaining space at {purchase.node_id}."
                ),
                observed=order_volume,
                limit=storage.free_m3,
                unit="m³",
            )
        )

    # ---- C08 shelf life / maximum cover ------------------------------
    checks.append("C08_COVER_AND_SHELF_LIFE")
    analysis = calc.analyze_replenishment(
        session, purchase.sku, purchase.node_id, purchase.supplier_id
    )
    resulting_position = analysis.inventory_position_units + purchase.units
    daily = analysis.daily_demand_mean
    resulting_cover = calc.days_of_cover(resulting_position, daily)
    cover_limit = min(product.shelf_life_days, MAX_COVER_DAYS) if product.shelf_life_days else MAX_COVER_DAYS

    if daily > 0 and resulting_cover > cover_limit:
        excess_units = int(resulting_position - daily * cover_limit)
        binding = "shelf life" if product.shelf_life_days < MAX_COVER_DAYS else "maximum cover policy"
        violation = ConstraintViolation(
            code="C08_COVER_AND_SHELF_LIFE",
            severity=Severity.BLOCK if product.is_perishable else Severity.WARN,
            message=(
                f"Ordering {purchase.units} units puts the position at {resulting_position} "
                f"units — {resulting_cover:.1f} days of cover against a {cover_limit}-day "
                f"limit set by {binding}. Roughly {excess_units} units would not sell in time."
            ),
            observed=resulting_cover,
            limit=cover_limit,
            unit="days",
            remedy_hint=f"Cap the order near {max(0, analysis.max_useful_units - analysis.inventory_position_units)} units.",
        )
        (blocking if product.is_perishable else warnings).append(violation)

    # ---- C09 lead time feasibility -----------------------------------
    checks.append("C09_LEAD_TIME")
    eta = purchase.expected_delivery_date or clock.days_from_today(sp.lead_time_days)
    eta_days = (eta - clock.today()).days
    if analysis.projected_stockout_in_days is not None and eta_days > analysis.projected_stockout_in_days:
        gap = round(eta_days - analysis.projected_stockout_in_days, 1)
        warnings.append(
            ConstraintViolation(
                code="C09_LEAD_TIME",
                severity=Severity.WARN,
                message=(
                    f"Stock runs out in {analysis.projected_stockout_in_days} days but this "
                    f"order arrives in {eta_days} days — a {gap}-day gap on shelf."
                ),
                observed=eta_days,
                limit=analysis.projected_stockout_in_days,
                unit="days",
                remedy_hint=(
                    "Use a faster supplier for a bridging quantity, or pull stock from a "
                    "nearby node with a transfer order."
                ),
            )
        )

    # ---- C10 duplicate coverage --------------------------------------
    checks.append("C10_DUPLICATE_COVERAGE")
    inbound = calc.inbound_shipments(session, purchase.sku, purchase.node_id)
    if analysis.net_requirement_units == 0 and purchase.units > 0 and inbound:
        blocking.append(
            ConstraintViolation(
                code="C10_DUPLICATE_COVERAGE",
                severity=Severity.BLOCK,
                message=(
                    f"Open purchase orders already cover the requirement "
                    f"({analysis.in_transit_units} units inbound, target position "
                    f"{analysis.target_position_units}, current position "
                    f"{analysis.inventory_position_units}). A further order duplicates cover."
                ),
                observed=purchase.units,
                limit=0,
                unit="units",
                remedy_hint="Reject the recommendation, or modify the existing PO instead.",
            )
        )

    # ---- C11 deviation from the system recommendation -----------------
    if purchase.system_recommendation_units:
        checks.append("C11_DEVIATION_FROM_RECOMMENDATION")
        base = purchase.system_recommendation_units
        deviation = abs(purchase.units - base) / max(base, 1)
        if deviation > 0.5:
            warnings.append(
                ConstraintViolation(
                    code="C11_DEVIATION_FROM_RECOMMENDATION",
                    severity=Severity.WARN,
                    message=(
                        f"Proposed {purchase.units} units deviates {deviation:.0%} from the "
                        f"system recommendation of {base}. Large deviations need an explicit "
                        f"justification a buyer can audit."
                    ),
                    observed=purchase.units,
                    limit=base,
                    unit="units",
                )
            )

    return ConstraintReport(
        passed=len(blocking) == 0,
        blocking=blocking,
        warnings=warnings,
        checks_run=checks,
    )


# ─────────────────────────────────────────────────────────────────────
# Post-hoc validation — the feedback loop
# ─────────────────────────────────────────────────────────────────────

# How far reality may drift from the agent's prediction before we make it
# think again. Chosen to be loose enough to tolerate case-pack rounding and
# tight enough to catch a supplier halving an order.
TOLERANCES = {
    "units_secured": 0.05,
    "spend_usd": 0.05,
    "days_of_cover": 1.5,  # absolute, in days
}


def validate_purchase_order(
    session: Session,
    po_id: str,
    expectation: ActionExpectation,
    sku: str,
    node_id: str,
) -> ValidationReport:
    """Re-derive the world from the database and hold it against the claim.

    Deliberately takes only identifiers plus the expectation: it does not
    receive, and cannot be influenced by, the agent's reasoning. Whatever
    the agent believed it did is irrelevant here.
    """
    findings: list[str] = []
    po = session.get(PurchaseOrder, po_id)

    if po is None:
        return ValidationReport(
            verdict=ValidationVerdict.FAILED,
            constraint_report=ConstraintReport(passed=False, checks_run=[]),
            findings=[f"Purchase order {po_id} does not exist after the action."],
            corrective_guidance=(
                "The write did not persist. Re-read state and decide whether to retry "
                "the order or escalate."
            ),
        )

    line = next((l for l in po.lines if l.sku == sku), None)
    if line is None:
        return ValidationReport(
            verdict=ValidationVerdict.FAILED,
            constraint_report=ConstraintReport(passed=False, checks_run=[]),
            findings=[f"{po_id} exists but has no line for {sku}."],
            corrective_guidance="Recreate the order line or escalate.",
        )

    # ---- what actually happened -------------------------------------
    units_secured = line.effective_units
    spend = round(units_secured * line.unit_price_usd, 2)

    report = evaluate_purchase(
        session,
        ProposedPurchase(
            sku=sku,
            node_id=node_id,
            supplier_id=po.supplier_id,
            units=0,  # the PO is already committed; re-check the resulting world
            unit_price_usd=line.unit_price_usd,
            expected_delivery_date=po.expected_delivery_date,
        ),
    )
    analysis = calc.analyze_replenishment(session, sku, node_id, po.supplier_id)

    observed = {
        "po_id": po.po_id,
        "po_status": po.status,
        "supplier_id": po.supplier_id,
        "ordered_units": line.ordered_units,
        "confirmed_units": line.confirmed_units,
        "units_secured": units_secured,
        "spend_usd": spend,
        "inventory_position_units": analysis.inventory_position_units,
        "days_of_cover": analysis.days_of_cover_now,
        "projected_stockout_in_days": analysis.projected_stockout_in_days,
        "target_position_units": analysis.target_position_units,
        "residual_requirement_units": analysis.net_requirement_units,
        "expected_delivery_date": po.expected_delivery_date.isoformat(),
    }

    # ---- diff against the agent's stated expectation -----------------
    diffs: list[ExpectationDiff] = []

    def _rel(field: str, expected: float, actual: float, tol: float) -> None:
        allowed = abs(expected) * tol
        diffs.append(
            ExpectationDiff(
                field=field,
                expected=round(expected, 2),
                actual=round(actual, 2),
                tolerance=round(allowed, 2),
                within_tolerance=abs(actual - expected) <= max(allowed, 1e-6),
            )
        )

    _rel("units_secured", expectation.expected_units_secured, units_secured, TOLERANCES["units_secured"])
    _rel("spend_usd", expectation.expected_spend_usd, spend, TOLERANCES["spend_usd"])
    diffs.append(
        ExpectationDiff(
            field="days_of_cover",
            expected=round(expectation.expected_days_of_cover, 2),
            actual=round(analysis.days_of_cover_now, 2),
            tolerance=TOLERANCES["days_of_cover"],
            within_tolerance=abs(analysis.days_of_cover_now - expectation.expected_days_of_cover)
            <= TOLERANCES["days_of_cover"],
        )
    )

    out_of_tolerance = [d for d in diffs if not d.within_tolerance]

    # ---- verdict ------------------------------------------------------
    if report.blocking:
        verdict = ValidationVerdict.VIOLATION
        findings.extend(v.message for v in report.blocking)
        guidance = (
            "The persisted state breaks hard constraints: "
            + "; ".join(f"[{v.code}] {v.remedy_hint or v.message}" for v in report.blocking)
            + " Amend or cancel the order so the resulting state is compliant."
        )
    elif out_of_tolerance:
        verdict = ValidationVerdict.DRIFT
        for d in out_of_tolerance:
            findings.append(
                f"{d.field}: expected {d.expected}, actual {d.actual} "
                f"(delta {d.delta}, tolerance ±{d.tolerance})."
            )
        shortfall = expectation.expected_units_secured - units_secured
        if shortfall > 0:
            guidance = (
                f"The action completed but secured {shortfall} fewer units than planned. "
                f"Residual requirement is {analysis.net_requirement_units} units and stock "
                f"runs out in {analysis.projected_stockout_in_days} days. Decide whether to "
                f"close the gap from another source, accept the shortfall, or escalate."
            )
        else:
            guidance = (
                "The outcome differs from the prediction. Re-read the position and decide "
                "whether the difference is material."
            )
    else:
        verdict = ValidationVerdict.PASS
        findings.append(
            f"{po.po_id} secured {units_secured} units for ${spend:,.2f}; position now "
            f"{analysis.inventory_position_units} units "
            f"({analysis.days_of_cover_now:.1f} days of cover). All constraints satisfied."
        )
        guidance = ""

    if report.warnings:
        findings.extend(f"Warning [{w.code}]: {w.message}" for w in report.warnings)

    return ValidationReport(
        verdict=verdict,
        constraint_report=report,
        expectation_diffs=diffs,
        observed_state=observed,
        findings=findings,
        corrective_guidance=guidance,
    )


def preflight_validate(
    session: Session,
    sku: str,
    node_id: str,
    supplier_id: str,
    units: int,
    expectation: ActionExpectation,
    replaces_po_id: str | None = None,
) -> ValidationReport:
    """Validate a proposal *before* it executes, for the approval queue.

    An approver should not be asked "is this order acceptable?" without
    being shown what it does. This runs the same constraint engine against
    the proposed purchase and projects the resulting position, so the human
    sees the consequence rather than just the quantity. The verdict is
    provisional — the post-execution pass still runs after approval, because
    the supplier gets a say between the two.
    """
    report = evaluate_purchase(
        session,
        ProposedPurchase(
            sku=sku,
            node_id=node_id,
            supplier_id=supplier_id,
            units=units,
            replaces_po_id=replaces_po_id,
        ),
    )
    analysis = calc.analyze_replenishment(session, sku, node_id, supplier_id)
    projected_position = analysis.inventory_position_units + units
    projected_cover = calc.days_of_cover(projected_position, analysis.daily_demand_mean)

    observed = {
        "stage": "pre-execution projection",
        "proposed_units": units,
        "supplier_id": supplier_id,
        "current_inventory_position_units": analysis.inventory_position_units,
        "projected_inventory_position_units": projected_position,
        "projected_days_of_cover": projected_cover,
        "target_position_units": analysis.target_position_units,
        "projected_stockout_in_days": analysis.projected_stockout_in_days,
        "estimated_cost_usd": round(units * analysis.unit_cost_usd, 2),
    }

    diffs = [
        ExpectationDiff(
            field="projected_days_of_cover",
            expected=round(expectation.expected_days_of_cover, 2),
            actual=round(projected_cover, 2),
            tolerance=TOLERANCES["days_of_cover"],
            within_tolerance=abs(projected_cover - expectation.expected_days_of_cover)
            <= TOLERANCES["days_of_cover"],
        )
    ]

    if report.blocking:
        return ValidationReport(
            verdict=ValidationVerdict.VIOLATION,
            constraint_report=report,
            expectation_diffs=diffs,
            observed_state=observed,
            findings=[v.message for v in report.blocking],
            corrective_guidance=(
                "The proposal breaks hard constraints and must not be queued for approval as "
                "it stands: "
                + "; ".join(f"[{v.code}] {v.remedy_hint or v.message}" for v in report.blocking)
            ),
        )

    findings = [
        f"Pre-flight: {units} units from {supplier_id} passes all "
        f"{len(report.checks_run)} constraint checks. Position would move from "
        f"{analysis.inventory_position_units} to {projected_position} units "
        f"({projected_cover:.1f} days of cover), costing about "
        f"${observed['estimated_cost_usd']:,.2f}."
    ]
    findings.extend(f"Warning [{w.code}]: {w.message}" for w in report.warnings)

    return ValidationReport(
        verdict=ValidationVerdict.PASS,
        constraint_report=report,
        expectation_diffs=diffs,
        observed_state=observed,
        findings=findings,
        corrective_guidance="",
    )


def validate_no_action(
    session: Session, sku: str, node_id: str, expectation: ActionExpectation
) -> ValidationReport:
    """Validate a decision to do nothing.

    Rejecting a recommendation is still a decision with consequences: if the
    position genuinely runs out inside the lead time, 'no action' is wrong
    and the agent needs to know that.
    """
    findings: list[str] = []
    checks = ["C09_LEAD_TIME", "C10_DUPLICATE_COVERAGE"]
    blocking: list[ConstraintViolation] = []

    # How long the node is genuinely exposed before stock can be replaced.
    # That is the *fastest* supplier who could actually take an order, not
    # the incumbent: a buyer with a one-day emergency source is not exposed
    # for the incumbent's seven days. Using the incumbent here would condemn
    # a correct "no further action" as a stockout risk.
    #
    # The requirement is then priced against that same supplier. Mixing the
    # two bases produces an audit line that contradicts itself — "no action
    # required, net requirement 302 units" — which is worse than being
    # wrong, because it is unreadable.
    fastest = session.execute(
        select(SupplierProduct.supplier_id, SupplierProduct.lead_time_days)
        .join(Supplier, Supplier.supplier_id == SupplierProduct.supplier_id)
        .where(SupplierProduct.sku == sku, Supplier.status == "active")
        .order_by(SupplierProduct.lead_time_days)
    ).first()

    analysis = calc.analyze_replenishment(
        session, sku, node_id, fastest[0] if fastest else None
    )
    stockout = analysis.projected_stockout_in_days
    lead = fastest[1] if fastest else analysis.lead_time_days
    exposure = lead + analysis.review_period_days

    if stockout is not None and stockout < exposure:
        blocking.append(
            ConstraintViolation(
                code="C09_LEAD_TIME",
                severity=Severity.BLOCK,
                message=(
                    f"Taking no action leaves stock running out in {stockout} days, inside "
                    f"the {exposure}-day replenishment exposure window (fastest available "
                    f"lead time {lead} + review period {analysis.review_period_days})."
                ),
                observed=stockout,
                limit=exposure,
                unit="days",
                remedy_hint="Order a bridging quantity or arrange a transfer.",
            )
        )

    report = ConstraintReport(passed=not blocking, blocking=blocking, checks_run=checks)
    if blocking:
        return ValidationReport(
            verdict=ValidationVerdict.VIOLATION,
            constraint_report=report,
            observed_state=analysis.model_dump(mode="json"),
            findings=[v.message for v in blocking],
            corrective_guidance=(
                "Doing nothing creates a stockout inside the exposure window. Reconsider: a "
                "smaller bridging order or a transfer may be justified."
            ),
        )

    findings.append(
        f"No action required: position {analysis.inventory_position_units} units covers "
        f"{analysis.days_of_cover_now:.1f} days against a {exposure}-day exposure window "
        f"(fastest replacement lead time {lead}d + {analysis.review_period_days}d review). "
        f"Residual requirement on that basis is {analysis.net_requirement_units} units."
    )
    return ValidationReport(
        verdict=ValidationVerdict.PASS,
        constraint_report=report,
        observed_state=analysis.model_dump(mode="json"),
        findings=findings,
    )
