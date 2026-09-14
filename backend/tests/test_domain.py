"""Unit tests for the decision core.

These cover the parts where a silent error would be expensive and invisible:
the replenishment arithmetic, the constraint engine, and the validator that
diffs reality against what the agent predicted. The agent graph is exercised
end to end by the evaluation suite instead — see `evals/`.
"""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.models import Base, PurchaseOrder, PurchaseOrderLine
from app.db.seed import seed_all
from app.domain import calculations as calc
from app.domain import clock
from app.domain import constraints as cons
from app.domain.schemas import ActionExpectation, Severity, ValidationVerdict


@pytest.fixture()
def session():
    engine = create_engine("sqlite+pysqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, expire_on_commit=False)
    with Session() as s:
        seed_all(s)
        s.commit()
        yield s


# ─────────────────────────────────────────────────────────────────────
# Replenishment arithmetic
# ─────────────────────────────────────────────────────────────────────


def test_safety_stock_scales_with_protection_interval():
    """Doubling the protection interval raises safety stock by sqrt(2), not 2."""
    short = calc.safety_stock_units(10.0, 2, 2, 0.95)
    long = calc.safety_stock_units(10.0, 6, 2, 0.95)
    assert long > short
    assert long == pytest.approx(short * (8 / 4) ** 0.5, rel=0.05)


def test_safety_stock_rises_with_service_level():
    assert calc.safety_stock_units(10.0, 3, 7, 0.99) > calc.safety_stock_units(10.0, 3, 7, 0.90)


def test_rounding_never_lands_short_of_the_requirement():
    """Case-pack rounding goes up. Rounding down would under-order by design."""
    qty, note = calc.round_to_constraints(294, case_pack=12, moq=240)
    assert qty == 300
    assert qty % 12 == 0
    assert qty >= 294
    assert "case pack" in note


def test_moq_is_applied_after_case_pack():
    qty, note = calc.round_to_constraints(30, case_pack=25, moq=100)
    assert qty == 100
    assert qty % 25 == 0
    assert "MOQ" in note


def test_zero_requirement_orders_nothing():
    qty, _ = calc.round_to_constraints(0, case_pack=12, moq=240)
    assert qty == 0


def test_stockout_projection_accounts_for_arrival_timing():
    """A delivery that lands after stock hits zero does not prevent the stockout."""
    late = calc.projected_stockout_days(
        available_units=100,
        shipments=[calc.InboundShipment("PO-1", 500, clock.days_from_today(10), "confirmed", "S")],
        daily_demand=50,
    )
    early = calc.projected_stockout_days(
        available_units=100,
        shipments=[calc.InboundShipment("PO-1", 500, clock.days_from_today(1), "confirmed", "S")],
        daily_demand=50,
    )
    assert late is not None and late < 3, "should run out before the late delivery arrives"
    assert early is None or early > late


def test_inbound_counts_confirmed_not_ordered(session):
    """PO-2001 was raised for 500 and confirmed at 250.

    Counting ordered units here is precisely the error that hides a
    shortfall, so the position must reflect 250.
    """
    shipments = calc.inbound_shipments(session, "SKU-3007", "CO-BOG-01")
    assert sum(s.units for s in shipments) == 250


def test_analysis_excludes_reserved_stock(session):
    analysis = calc.analyze_replenishment(session, "SKU-1001", "MX-CDMX-01")
    assert analysis.on_hand_units == 420
    assert analysis.reserved_units == 30
    assert analysis.available_units == 390
    assert analysis.inventory_position_units == 390 + analysis.in_transit_units


def test_shelf_life_caps_useful_quantity(session):
    """Milk expires in 12 days, so no amount of demand justifies 30 days of cover."""
    analysis = calc.analyze_replenishment(session, "SKU-1001", "MX-CDMX-01")
    assert analysis.max_useful_units <= analysis.daily_demand_mean * 12 + 1


# ─────────────────────────────────────────────────────────────────────
# Demand anomalies
# ─────────────────────────────────────────────────────────────────────


def test_promotional_spike_is_not_extrapolated_at_full_strength(session):
    """SC3: sales are ~2.5x, but the promotion ends in three days.

    The blended estimate must sit well below the observed rate, or the
    agent would buy stock for demand that is about to disappear.
    """
    anomaly = calc.detect_demand_anomaly(session, "SKU-6021", "BR-SP-01")
    assert anomaly.detected, "a sustained 2.5x lift should be detected"

    blended, _sigma, source, notes = calc.blended_daily_demand(
        session, "SKU-6021", "BR-SP-01", horizon_days=9, anomaly=anomaly
    )
    assert blended < anomaly.recent_mean, "must not extrapolate the full promotional uplift"
    assert blended > anomaly.baseline_mean, "but must still reflect a real lift"
    assert source == "promo_adjusted_blend"
    assert any("promotion" in n.lower() for n in notes)


def test_stable_demand_is_not_treated_as_an_anomaly(session):
    anomaly = calc.detect_demand_anomaly(session, "SKU-1001", "MX-CDMX-01")
    assert not anomaly.detected


# ─────────────────────────────────────────────────────────────────────
# Constraint engine
# ─────────────────────────────────────────────────────────────────────


def test_blocked_supplier_cannot_be_ordered_from(session):
    report = cons.evaluate_purchase(
        session,
        cons.ProposedPurchase(
            sku="SKU-3007", node_id="CO-BOG-01", supplier_id="SUP-DELTA", units=250
        ),
    )
    assert not report.passed
    assert any(v.code == "C01_SUPPLIER_STATUS" for v in report.blocking)


def test_case_pack_violation_is_blocking(session):
    report = cons.evaluate_purchase(
        session,
        cons.ProposedPurchase(
            sku="SKU-3007", node_id="CO-BOG-01", supplier_id="SUP-GAMMA", units=263
        ),
    )
    assert any(v.code == "C04_CASE_PACK" for v in report.blocking)


def test_order_below_moq_is_blocking(session):
    report = cons.evaluate_purchase(
        session,
        cons.ProposedPurchase(
            sku="SKU-4012", node_id="MX-CDMX-02", supplier_id="SUP-ALPHA", units=100
        ),
    )
    assert any(v.code == "C03_MOQ" for v in report.blocking)


def test_budget_overrun_is_blocking_and_suggests_a_feasible_quantity(session):
    report = cons.evaluate_purchase(
        session,
        cons.ProposedPurchase(
            sku="SKU-4012", node_id="MX-CDMX-02", supplier_id="SUP-ALPHA", units=2000
        ),
    )
    budget_violations = [v for v in report.blocking if v.code == "C06_BUDGET"]
    assert budget_violations
    assert "units" in budget_violations[0].remedy_hint

def test_storage_overrun_is_blocking(session):
    report = cons.evaluate_purchase(
        session,
        cons.ProposedPurchase(
            sku="SKU-4012", node_id="MX-CDMX-02", supplier_id="SUP-GAMMA", units=5000
        ),
    )
    assert any(v.code == "C07_STORAGE" for v in report.blocking)


def test_perishable_overbuy_is_blocking(session):
    """SC1: 800 units of 12-day milk on top of an inbound PO is waste."""
    report = cons.evaluate_purchase(
        session,
        cons.ProposedPurchase(
            sku="SKU-1001", node_id="MX-CDMX-01", supplier_id="SUP-ALPHA", units=800
        ),
    )
    cover = [v for v in report.blocking if v.code == "C08_COVER_AND_SHELF_LIFE"]
    assert cover, "ordering 800 units of a 12-day perishable must be blocked"
    assert cover[0].severity == Severity.BLOCK


def test_the_defensible_quantity_passes_every_check(session):
    """The counterpart to the test above: ~300 units is allowed."""
    analysis = calc.analyze_replenishment(session, "SKU-1001", "MX-CDMX-01", "SUP-ALPHA")
    report = cons.evaluate_purchase(
        session,
        cons.ProposedPurchase(
            sku="SKU-1001",
            node_id="MX-CDMX-01",
            supplier_id="SUP-ALPHA",
            units=analysis.recommended_order_rounded,
        ),
    )
    assert report.passed, f"blocked by {[v.code for v in report.blocking]}"


def test_storage_counts_stock_already_inbound(session):
    """Capacity that an open PO will occupy is not free capacity today."""
    state = cons.node_storage_state(session, "MX-CDMX-01")
    assert state.inbound_m3 > 0
    assert state.free_m3 == pytest.approx(
        state.capacity_m3 - state.used_m3 - state.inbound_m3, abs=0.001
    )


# ─────────────────────────────────────────────────────────────────────
# Validation — the feedback loop
# ─────────────────────────────────────────────────────────────────────


def _make_po(session, po_id, sku, node_id, supplier, ordered, confirmed, price):
    po = PurchaseOrder(
        po_id=po_id,
        supplier_id=supplier,
        node_id=node_id,
        status="partially_confirmed" if confirmed < ordered else "confirmed",
        expected_delivery_date=clock.days_from_today(2),
        source_run_id="run_test",
    )
    po.lines.append(
        PurchaseOrderLine(
            po_id=po_id, sku=sku, ordered_units=ordered, confirmed_units=confirmed, unit_price_usd=price
        )
    )
    session.add(po)
    session.flush()
    return po


def test_validation_passes_when_reality_matches_the_prediction(session):
    _make_po(session, "PO-TEST1", "SKU-3007", "CO-BOG-01", "SUP-GAMMA", 200, 200, 1.28)
    report = cons.validate_purchase_order(
        session,
        "PO-TEST1",
        ActionExpectation(
            expected_units_secured=200,
            expected_spend_usd=256.0,
            expected_days_of_cover=calc.analyze_replenishment(
                session, "SKU-3007", "CO-BOG-01"
            ).days_of_cover_now,
        ),
        "SKU-3007",
        "CO-BOG-01",
    )
    assert report.verdict == ValidationVerdict.PASS


def test_short_shipment_is_detected_as_drift_and_produces_guidance(session):
    """The agent asked for 500 and the supplier confirmed 175.

    This must be caught by measurement, and the guidance handed back must
    name the residual so the next decision starts from reality.
    """
    _make_po(session, "PO-TEST2", "SKU-3007", "CO-BOG-01", "SUP-GAMMA", 500, 175, 1.28)
    report = cons.validate_purchase_order(
        session,
        "PO-TEST2",
        ActionExpectation(
            expected_units_secured=500, expected_spend_usd=640.0, expected_days_of_cover=15.0
        ),
        "SKU-3007",
        "CO-BOG-01",
    )
    assert report.verdict == ValidationVerdict.DRIFT
    diffs = {d.field: d for d in report.expectation_diffs}
    assert not diffs["units_secured"].within_tolerance
    assert diffs["units_secured"].actual == 175
    assert report.corrective_guidance
    assert "fewer units" in report.corrective_guidance


def test_validation_reports_failure_when_the_order_does_not_exist(session):
    report = cons.validate_purchase_order(
        session, "PO-NOPE", ActionExpectation(), "SKU-3007", "CO-BOG-01"
    )
    assert report.verdict == ValidationVerdict.FAILED


def test_validator_ignores_agent_claims_and_reads_the_database(session):
    """A wildly optimistic expectation cannot talk the validator round."""
    _make_po(session, "PO-TEST3", "SKU-3007", "CO-BOG-01", "SUP-GAMMA", 100, 100, 1.28)
    report = cons.validate_purchase_order(
        session,
        "PO-TEST3",
        ActionExpectation(expected_units_secured=9999, expected_spend_usd=0.01),
        "SKU-3007",
        "CO-BOG-01",
    )
    assert report.verdict == ValidationVerdict.DRIFT
    assert report.observed_state["units_secured"] == 100


def test_preflight_blocks_a_proposal_that_breaches_constraints(session):
    report = cons.preflight_validate(
        session,
        sku="SKU-1001",
        node_id="MX-CDMX-01",
        supplier_id="SUP-ALPHA",
        units=800,
        expectation=ActionExpectation(expected_units_secured=800),
    )
    assert report.verdict == ValidationVerdict.VIOLATION
    assert report.corrective_guidance
