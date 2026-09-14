"""Replenishment mathematics.

Every number the agent reasons about is produced here, deterministically,
from the database. The language model is never asked to do arithmetic — it
is asked to choose between options whose consequences have already been
quantified. This removes the single largest source of silent error in an
LLM-driven planning system.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Iterable, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import (
    DemandForecast,
    InventoryPosition,
    Node,
    Product,
    PromoCalendar,
    PurchaseOrder,
    PurchaseOrderLine,
    SalesActual,
    SupplierProduct,
    TransferOrder,
)
from app.domain import clock
from app.domain.schemas import ReplenishmentAnalysis

# Service-level -> z multiplier for the safety stock formula.
_Z_TABLE: dict[float, float] = {
    0.80: 0.8416,
    0.85: 1.0364,
    0.90: 1.2816,
    0.95: 1.6449,
    0.975: 1.9600,
    0.98: 2.0537,
    0.99: 2.3263,
}


def z_for_service_level(service_level: float) -> float:
    nearest = min(_Z_TABLE, key=lambda k: abs(k - service_level))
    return _Z_TABLE[nearest]


# ─────────────────────────────────────────────────────────────────────
# Primitives
# ─────────────────────────────────────────────────────────────────────


def safety_stock_units(
    sigma_daily: float, lead_time_days: int, review_period_days: int, service_level: float
) -> int:
    """Classic (R, S) safety stock over the protection interval L + R."""
    protection_days = max(1, lead_time_days + review_period_days)
    z = z_for_service_level(service_level)
    return int(math.ceil(z * sigma_daily * math.sqrt(protection_days)))


def target_position_units(
    daily_demand: float, lead_time_days: int, review_period_days: int, safety_stock: int
) -> int:
    """Order-up-to level S."""
    protection_days = max(1, lead_time_days + review_period_days)
    return int(math.ceil(daily_demand * protection_days + safety_stock))


def round_to_constraints(units: int, case_pack: int, moq: int) -> tuple[int, str]:
    """Snap a raw requirement onto the supplier's orderable grid.

    Rounds the case pack *up* so the order never lands short of the
    requirement, then lifts to MOQ if the supplier demands it. Returns the
    quantity and a human-readable note, because 'why is it 264 and not 260'
    is the first question a buyer asks.
    """
    if units <= 0:
        return 0, "No requirement — nothing to order."

    notes: list[str] = []
    qty = units

    if case_pack > 1:
        packs = math.ceil(qty / case_pack)
        rounded = packs * case_pack
        if rounded != qty:
            notes.append(f"rounded {qty} up to {rounded} to fit case pack of {case_pack}")
        qty = rounded

    if moq and qty < moq:
        lifted = moq
        if case_pack > 1:
            lifted = math.ceil(moq / case_pack) * case_pack
        notes.append(f"lifted {qty} to supplier MOQ of {moq} (ordered {lifted})")
        qty = lifted

    return qty, "; ".join(notes) if notes else "exact multiple — no rounding needed"


def days_of_cover(position_units: int, daily_demand: float) -> float:
    if daily_demand <= 0:
        return float("inf") if position_units > 0 else 0.0
    return round(position_units / daily_demand, 2)


def max_useful_units(daily_demand: float, shelf_life_days: int, max_cover_days: int) -> int:
    """Units above which stock expires or ties up capital beyond policy.

    Bounded by whichever bites first: the product's shelf life or the
    maximum days of cover the replenishment policy allows.
    """
    horizon = min(shelf_life_days, max_cover_days) if shelf_life_days > 0 else max_cover_days
    return int(math.floor(daily_demand * horizon))


# ─────────────────────────────────────────────────────────────────────
# Data gathering
# ─────────────────────────────────────────────────────────────────────


@dataclass
class InboundShipment:
    po_id: str
    units: int
    eta: date
    status: str
    supplier_id: str

    @property
    def days_out(self) -> int:
        return max(0, (self.eta - clock.today()).days)


def inbound_shipments(session: Session, sku: str, node_id: str) -> list[InboundShipment]:
    """Open purchase orders that will land stock at this node.

    Uses `effective_units` — once a supplier has confirmed a reduced
    quantity, the reduced quantity is what is actually inbound. Treating
    ordered units as inbound is exactly the mistake that hides a shortfall.
    """
    rows = session.execute(
        select(PurchaseOrder, PurchaseOrderLine)
        .join(PurchaseOrderLine, PurchaseOrderLine.po_id == PurchaseOrder.po_id)
        .where(
            PurchaseOrder.node_id == node_id,
            PurchaseOrderLine.sku == sku,
            PurchaseOrder.status.in_(
                ["submitted", "confirmed", "partially_confirmed", "pending_approval"]
            ),
        )
    ).all()

    shipments: list[InboundShipment] = []
    for po, line in rows:
        outstanding = line.effective_units - line.received_units
        if outstanding <= 0:
            continue
        shipments.append(
            InboundShipment(
                po_id=po.po_id,
                units=outstanding,
                eta=po.expected_delivery_date,
                status=po.status,
                supplier_id=po.supplier_id,
            )
        )

    # Stock moving in from another node is inbound cover just as much as a
    # purchase order is. Counting only POs makes a transfer look like it did
    # nothing, which sends the agent back to buy stock it has already moved.
    transfers = session.execute(
        select(TransferOrder).where(
            TransferOrder.sku == sku,
            TransferOrder.to_node_id == node_id,
            TransferOrder.status.in_(["draft", "in_transit"]),
        )
    ).scalars()
    for transfer in transfers:
        shipments.append(
            InboundShipment(
                po_id=transfer.to_id,
                units=transfer.units,
                eta=clock.today() + timedelta(days=transfer.transit_days),
                status=f"transfer:{transfer.status}",
                supplier_id=f"node:{transfer.from_node_id}",
            )
        )

    return sorted(shipments, key=lambda s: s.eta)


def recent_sales(session: Session, sku: str, node_id: str, lookback_days: int = 28) -> list[SalesActual]:
    since = clock.today() - timedelta(days=lookback_days)
    return list(
        session.execute(
            select(SalesActual)
            .where(
                SalesActual.sku == sku,
                SalesActual.node_id == node_id,
                SalesActual.sale_date >= since,
            )
            .order_by(SalesActual.sale_date)
        ).scalars()
    )


def forward_forecast(
    session: Session, sku: str, node_id: str, horizon_days: int = 21
) -> list[DemandForecast]:
    start = clock.today()
    end = start + timedelta(days=horizon_days)
    return list(
        session.execute(
            select(DemandForecast)
            .where(
                DemandForecast.sku == sku,
                DemandForecast.node_id == node_id,
                DemandForecast.forecast_date >= start,
                DemandForecast.forecast_date < end,
            )
            .order_by(DemandForecast.forecast_date)
        ).scalars()
    )


def _stdev(values: Iterable[float]) -> float:
    vals = list(values)
    if len(vals) < 2:
        return 0.0
    return statistics.stdev(vals)


# ─────────────────────────────────────────────────────────────────────
# Demand anomaly detection
# ─────────────────────────────────────────────────────────────────────


@dataclass
class DemandAnomaly:
    detected: bool
    uplift_factor: float
    recent_mean: float
    baseline_mean: float
    recent_days: int
    consecutive_days_above: int
    significant: bool
    censored_by_stockout: bool
    explanation: str

    def to_dict(self) -> dict:
        return {
            "detected": self.detected,
            "uplift_factor": round(self.uplift_factor, 3),
            "recent_mean_units_per_day": round(self.recent_mean, 2),
            "baseline_mean_units_per_day": round(self.baseline_mean, 2),
            "recent_window_days": self.recent_days,
            "consecutive_days_above_threshold": self.consecutive_days_above,
            "statistically_significant": self.significant,
            "demand_censored_by_stockout": self.censored_by_stockout,
            "explanation": self.explanation,
        }


def detect_demand_anomaly(
    session: Session,
    sku: str,
    node_id: str,
    recent_days: int = 7,
    baseline_days: int = 28,
    threshold: float = 1.25,
) -> DemandAnomaly:
    """Compare the recent selling rate against the preceding baseline.

    Significance uses a two-sigma rule on the baseline distribution rather
    than the raw ratio, so a single spike day cannot move the forecast.
    """
    sales = recent_sales(session, sku, node_id, lookback_days=baseline_days + recent_days)
    if len(sales) < recent_days + 3:
        return DemandAnomaly(
            False, 1.0, 0.0, 0.0, recent_days, 0, False, False,
            "Insufficient sales history to assess an anomaly.",
        )

    recent = sales[-recent_days:]
    baseline = sales[:-recent_days]
    recent_units = [s.units_sold for s in recent]
    baseline_units = [s.units_sold for s in baseline]

    recent_mean = statistics.fmean(recent_units)
    baseline_mean = statistics.fmean(baseline_units) if baseline_units else 0.0
    baseline_sd = _stdev(baseline_units)

    uplift = (recent_mean / baseline_mean) if baseline_mean > 0 else 1.0
    consecutive = 0
    for s in reversed(recent):
        if baseline_mean > 0 and s.units_sold > baseline_mean * threshold:
            consecutive += 1
        else:
            break

    significant = bool(
        baseline_sd > 0
        and recent_mean > baseline_mean + 2 * baseline_sd
        and consecutive >= 3
    )
    censored = any(s.stockout_hours > 0 for s in recent)

    if uplift >= threshold and significant:
        explanation = (
            f"Selling {recent_mean:.0f}/day over the last {recent_days} days versus a "
            f"{baseline_mean:.0f}/day baseline ({uplift:.2f}x), above two baseline "
            f"standard deviations on {consecutive} consecutive days."
        )
    elif uplift >= threshold:
        explanation = (
            f"Recent rate is {uplift:.2f}x baseline but the lift is not statistically "
            f"separated from normal variation ({consecutive} consecutive days above "
            f"threshold). Treat as noise until it persists."
        )
    else:
        explanation = f"Recent rate {uplift:.2f}x baseline — within normal variation."

    if censored:
        explanation += (
            " Note: the SKU was out of stock for part of the window, so true demand is "
            "censored and the real uplift may be higher."
        )

    return DemandAnomaly(
        detected=uplift >= threshold and significant,
        uplift_factor=uplift,
        recent_mean=recent_mean,
        baseline_mean=baseline_mean,
        recent_days=recent_days,
        consecutive_days_above=consecutive,
        significant=significant,
        censored_by_stockout=censored,
        explanation=explanation,
    )


def active_promotions(session: Session, sku: str, node_id: str) -> list[PromoCalendar]:
    """Promotions overlapping the next three weeks.

    A demand spike with a promotion behind it is temporary; extrapolating
    it into a standing order is how buyers end up with dead stock.
    """
    today = clock.today()
    horizon = today + timedelta(days=21)
    rows = session.execute(
        select(PromoCalendar).where(
            PromoCalendar.sku == sku,
            PromoCalendar.end_date >= today - timedelta(days=7),
            PromoCalendar.start_date <= horizon,
        )
    ).scalars()
    return [p for p in rows if p.node_id in (None, node_id)]


def blended_daily_demand(
    session: Session,
    sku: str,
    node_id: str,
    horizon_days: int,
    anomaly: DemandAnomaly | None = None,
) -> tuple[float, float, str, list[str]]:
    """Best available estimate of demand per day over the horizon.

    Precedence:
      1. A statistically significant, non-promotional uplift overrides the
         published forecast (the forecast is stale by definition).
      2. A promotional uplift is applied only for the days the promotion is
         actually live, then decays to a residual halo.
      3. Otherwise trust the published forecast.
    """
    notes: list[str] = []
    forecasts = forward_forecast(session, sku, node_id, horizon_days)
    sales = recent_sales(session, sku, node_id, lookback_days=28)

    forecast_mean = statistics.fmean([f.forecast_units for f in forecasts]) if forecasts else 0.0
    sales_units = [s.units_sold for s in sales]
    sales_sd = _stdev(sales_units)
    recent_mean = statistics.fmean(sales_units[-7:]) if len(sales_units) >= 7 else forecast_mean

    if anomaly is None:
        anomaly = detect_demand_anomaly(session, sku, node_id)

    promos = active_promotions(session, sku, node_id)
    today = clock.today()
    live_promos = [p for p in promos if p.start_date <= today <= p.end_date]

    if anomaly.detected and live_promos:
        promo = live_promos[0]
        promo_days_left = max(0, (promo.end_date - today).days + 1)
        # Elevated while the promo runs, then a residual halo — new
        # customers who tried the product stay, most of the lift does not.
        halo = 1.0 + (anomaly.uplift_factor - 1.0) * 0.25
        elevated_days = min(promo_days_left, horizon_days)
        normal_days = max(0, horizon_days - elevated_days)
        blended = (
            anomaly.recent_mean * elevated_days + anomaly.baseline_mean * halo * normal_days
        ) / max(1, horizon_days)
        notes.append(
            f"Uplift of {anomaly.uplift_factor:.2f}x coincides with promotion "
            f"'{promo.description or promo.promo_type}' ending in {promo_days_left} day(s). "
            f"Demand modelled as {anomaly.recent_mean:.0f}/day for {elevated_days} day(s), "
            f"then a {halo:.2f}x residual halo — not extrapolated at full uplift."
        )
        return round(blended, 2), max(sales_sd, 1.0), "promo_adjusted_blend", notes

    if anomaly.detected:
        notes.append(
            f"Sustained non-promotional uplift of {anomaly.uplift_factor:.2f}x. Published "
            f"forecast of {forecast_mean:.0f}/day is stale; using the observed rate."
        )
        return round(anomaly.recent_mean, 2), max(sales_sd, 1.0), "observed_sales_override", notes

    if forecasts:
        if recent_mean and forecast_mean and abs(recent_mean - forecast_mean) / max(forecast_mean, 1) > 0.15:
            notes.append(
                f"Forecast ({forecast_mean:.0f}/day) and recent actuals ({recent_mean:.0f}/day) "
                f"disagree by more than 15%, but the gap is not statistically significant."
            )
        return round(forecast_mean, 2), max(sales_sd, 1.0), "published_forecast", notes

    notes.append("No forward forecast available; falling back to trailing sales.")
    return round(recent_mean, 2), max(sales_sd, 1.0), "trailing_sales_fallback", notes


# ─────────────────────────────────────────────────────────────────────
# The headline analysis
# ─────────────────────────────────────────────────────────────────────


def projected_stockout_days(
    available_units: int, shipments: list[InboundShipment], daily_demand: float, horizon: int = 60
) -> Optional[float]:
    """Simulate the stock curve day by day until it crosses zero.

    Walks inbound deliveries in as they land, which is the only way to see
    that a PO arriving on day 7 does not help a stockout on day 4.
    """
    if daily_demand <= 0:
        return None
    stock = float(available_units)
    by_day: dict[int, int] = {}
    for s in shipments:
        by_day[s.days_out] = by_day.get(s.days_out, 0) + s.units

    for day in range(0, horizon + 1):
        stock += by_day.get(day, 0)
        stock -= daily_demand
        if stock < 0:
            # Interpolate within the day for a slightly finer answer.
            overshoot = -stock / daily_demand
            return round(max(0.0, day + 1 - overshoot), 2)
    return None


def analyze_replenishment(
    session: Session,
    sku: str,
    node_id: str,
    supplier_id: Optional[str] = None,
    horizon_days: Optional[int] = None,
    max_cover_days: int = 21,
) -> ReplenishmentAnalysis:
    product = session.get(Product, sku)
    node = session.get(Node, node_id)
    if product is None or node is None:
        raise ValueError(f"Unknown sku/node combination: {sku} @ {node_id}")

    inv = session.execute(
        select(InventoryPosition).where(
            InventoryPosition.sku == sku, InventoryPosition.node_id == node_id
        )
    ).scalar_one_or_none()
    on_hand = inv.on_hand_units if inv else 0
    reserved = inv.reserved_units if inv else 0
    damaged = inv.damaged_units if inv else 0
    available = max(0, on_hand - reserved - damaged)

    shipments = inbound_shipments(session, sku, node_id)
    in_transit = sum(s.units for s in shipments)
    position = available + in_transit

    sp: SupplierProduct | None = None
    if supplier_id:
        sp = session.execute(
            select(SupplierProduct).where(
                SupplierProduct.supplier_id == supplier_id, SupplierProduct.sku == sku
            )
        ).scalar_one_or_none()
    if sp is None:
        sp = session.execute(
            select(SupplierProduct)
            .where(SupplierProduct.sku == sku)
            .order_by(SupplierProduct.is_primary.desc(), SupplierProduct.unit_price_usd)
        ).scalars().first()

    lead_time = sp.lead_time_days if sp else 7
    review_period = node.review_period_days
    horizon = horizon_days or (lead_time + review_period)

    anomaly = detect_demand_anomaly(session, sku, node_id)
    daily_demand, sigma, source, notes = blended_daily_demand(
        session, sku, node_id, horizon, anomaly
    )

    ss = safety_stock_units(sigma, lead_time, review_period, node.service_level_target)
    target = target_position_units(daily_demand, lead_time, review_period, ss)
    net_req = max(0, target - position)

    cover_now = days_of_cover(position, daily_demand)
    stockout_in = projected_stockout_days(available, shipments, daily_demand)
    useful_cap = max_useful_units(daily_demand, product.shelf_life_days, max_cover_days)

    # Never order into guaranteed waste: cap the raw requirement so the
    # resulting position stays inside the useful window.
    capped = net_req
    if position + net_req > useful_cap:
        capped = max(0, useful_cap - position)
        if capped < net_req:
            limit_name = (
                f"{product.shelf_life_days}-day shelf life"
                if product.shelf_life_days < max_cover_days
                else f"{max_cover_days}-day maximum cover policy"
            )
            notes.append(
                f"Requirement trimmed from {net_req} to {capped} units: ordering the full "
                f"amount would push cover beyond the {limit_name}."
            )

    unit_cost = sp.unit_price_usd if sp else product.unit_cost_usd
    case_pack = sp.case_pack if sp else product.case_pack
    moq = sp.moq_units if sp else 0
    rounded, rounding_note = round_to_constraints(capped, case_pack, moq)

    if shipments:
        notes.append(
            "Open inbound: "
            + ", ".join(
                f"{s.po_id} {s.units}u in {s.days_out}d ({s.status})" for s in shipments
            )
        )
    if stockout_in is not None:
        notes.append(
            f"Projected to run out in {stockout_in} days at {daily_demand:.0f} units/day, "
            f"including inbound deliveries."
        )

    return ReplenishmentAnalysis(
        sku=sku,
        node_id=node_id,
        supplier_id=sp.supplier_id if sp else None,
        on_hand_units=on_hand,
        reserved_units=reserved,
        available_units=available,
        in_transit_units=in_transit,
        inventory_position_units=position,
        daily_demand_mean=daily_demand,
        daily_demand_std=round(sigma, 2),
        demand_source=source,
        horizon_days=horizon,
        lead_time_days=lead_time,
        review_period_days=review_period,
        safety_stock_units=ss,
        target_position_units=target,
        net_requirement_units=net_req,
        days_of_cover_now=cover_now,
        projected_stockout_in_days=stockout_in,
        shelf_life_days=product.shelf_life_days,
        max_useful_units=useful_cap,
        recommended_order_units=capped,
        recommended_order_rounded=rounded,
        rounding_note=rounding_note,
        unit_cost_usd=unit_cost,
        estimated_cost_usd=round(rounded * unit_cost, 2),
        notes=notes,
    )


def estimated_lost_margin(units_short: int, unit_margin: float, substitution_rate: float = 0.3) -> float:
    """Cost of not buying — the number that makes an escalation actionable.

    Some shoppers buy something else instead, so only the non-substituted
    share is genuinely lost.
    """
    return round(units_short * unit_margin * (1 - substitution_rate), 2)
