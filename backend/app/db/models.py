"""The mock ERP schema the agent reasons over.

Three groups of tables:

1. Operational system of record — products, nodes, inventory, forecasts,
   sales, suppliers, purchase orders, budgets, storage. This is the
   "truth" the agent must read and write through tools.
2. Supplier communication — inbound messages that assert things which may
   or may not be true. Treated as untrusted input, never as instruction.
3. Agent observability — runs, steps, evidence, approvals, domain events.
   This is what makes a decision auditable after the fact.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any, Optional

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


# ─────────────────────────────────────────────────────────────────────
# 1. Operational system of record
# ─────────────────────────────────────────────────────────────────────


class Product(Base):
    __tablename__ = "products"

    sku: Mapped[str] = mapped_column(String(32), primary_key=True)
    name: Mapped[str] = mapped_column(String(128))
    category: Mapped[str] = mapped_column(String(64), index=True)
    unit_cost_usd: Mapped[float] = mapped_column(Float)
    retail_price_usd: Mapped[float] = mapped_column(Float)
    case_pack: Mapped[int] = mapped_column(Integer, default=1)
    unit_volume_m3: Mapped[float] = mapped_column(Float)
    shelf_life_days: Mapped[int] = mapped_column(Integer, default=365)
    is_perishable: Mapped[bool] = mapped_column(Boolean, default=False)

    @property
    def unit_margin_usd(self) -> float:
        return round(self.retail_price_usd - self.unit_cost_usd, 4)


class Node(Base):
    """A dark store / micro-fulfilment centre."""

    __tablename__ = "nodes"

    node_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    name: Mapped[str] = mapped_column(String(128))
    country: Mapped[str] = mapped_column(String(8), index=True)
    city: Mapped[str] = mapped_column(String(64))
    storage_capacity_m3: Mapped[float] = mapped_column(Float)
    # Volume consumed by SKUs outside this simulation. Keeps capacity
    # realistic without seeding the entire assortment.
    baseline_used_m3: Mapped[float] = mapped_column(Float, default=0.0)
    review_period_days: Mapped[int] = mapped_column(Integer, default=7)
    service_level_target: Mapped[float] = mapped_column(Float, default=0.95)


class InventoryPosition(Base):
    __tablename__ = "inventory_positions"
    __table_args__ = (UniqueConstraint("sku", "node_id", name="uq_inventory_sku_node"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    sku: Mapped[str] = mapped_column(ForeignKey("products.sku"), index=True)
    node_id: Mapped[str] = mapped_column(ForeignKey("nodes.node_id"), index=True)
    on_hand_units: Mapped[int] = mapped_column(Integer, default=0)
    # Allocated to customer orders already placed but not yet picked.
    reserved_units: Mapped[int] = mapped_column(Integer, default=0)
    damaged_units: Mapped[int] = mapped_column(Integer, default=0)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    @property
    def available_units(self) -> int:
        return max(0, self.on_hand_units - self.reserved_units - self.damaged_units)


class DemandForecast(Base):
    __tablename__ = "demand_forecasts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    sku: Mapped[str] = mapped_column(ForeignKey("products.sku"), index=True)
    node_id: Mapped[str] = mapped_column(ForeignKey("nodes.node_id"), index=True)
    forecast_date: Mapped[date] = mapped_column(Date, index=True)
    forecast_units: Mapped[float] = mapped_column(Float)
    p90_units: Mapped[float] = mapped_column(Float)
    model_version: Mapped[str] = mapped_column(String(32), default="baseline-v3")
    # Set when the forecast was generated; a stale forecast is itself a
    # finding the agent should surface.
    generated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class SalesActual(Base):
    __tablename__ = "sales_actuals"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    sku: Mapped[str] = mapped_column(ForeignKey("products.sku"), index=True)
    node_id: Mapped[str] = mapped_column(ForeignKey("nodes.node_id"), index=True)
    sale_date: Mapped[date] = mapped_column(Date, index=True)
    units_sold: Mapped[int] = mapped_column(Integer)
    # Hours the SKU was unavailable — censors demand and matters when
    # comparing actuals against forecast.
    stockout_hours: Mapped[float] = mapped_column(Float, default=0.0)


class Supplier(Base):
    __tablename__ = "suppliers"

    supplier_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    name: Mapped[str] = mapped_column(String(128))
    country: Mapped[str] = mapped_column(String(8))
    lead_time_days: Mapped[int] = mapped_column(Integer)
    # Observed, not promised: share of confirmed units actually delivered.
    fill_rate_90d: Mapped[float] = mapped_column(Float, default=1.0)
    on_time_rate_90d: Mapped[float] = mapped_column(Float, default=1.0)
    status: Mapped[str] = mapped_column(String(24), default="active")  # active|on_hold|blocked
    status_reason: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    allows_split_delivery: Mapped[bool] = mapped_column(Boolean, default=False)
    payment_terms_days: Mapped[int] = mapped_column(Integer, default=30)

    @property
    def reliability_score(self) -> float:
        return round(0.6 * self.fill_rate_90d + 0.4 * self.on_time_rate_90d, 4)


class SupplierProduct(Base):
    """What a specific supplier will sell of a specific SKU, on what terms."""

    __tablename__ = "supplier_products"
    __table_args__ = (UniqueConstraint("supplier_id", "sku", name="uq_supplier_sku"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    supplier_id: Mapped[str] = mapped_column(ForeignKey("suppliers.supplier_id"), index=True)
    sku: Mapped[str] = mapped_column(ForeignKey("products.sku"), index=True)
    unit_price_usd: Mapped[float] = mapped_column(Float)
    moq_units: Mapped[int] = mapped_column(Integer, default=0)
    case_pack: Mapped[int] = mapped_column(Integer, default=1)
    lead_time_days: Mapped[int] = mapped_column(Integer)
    # Ceiling on what the supplier can physically ship in a week.
    max_weekly_units: Mapped[int] = mapped_column(Integer, default=100000)
    is_primary: Mapped[bool] = mapped_column(Boolean, default=False)

    # --- supplier simulator controls -------------------------------
    # Units the supplier can actually ship right now. None means "no
    # constraint beyond max_weekly_units". This is what the availability
    # API reports, and it is the authoritative answer against which an
    # emailed claim gets corroborated.
    available_units_override: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    # Share of an order this supplier will confirm when it is submitted.
    # 1.0 confirms in full; 0.5 confirms half — the partial-fulfilment case.
    simulated_fill_ratio: Mapped[float] = mapped_column(Float, default=1.0)


class PurchaseOrder(Base):
    __tablename__ = "purchase_orders"

    po_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    supplier_id: Mapped[str] = mapped_column(ForeignKey("suppliers.supplier_id"), index=True)
    node_id: Mapped[str] = mapped_column(ForeignKey("nodes.node_id"), index=True)
    # draft | pending_approval | submitted | confirmed | partially_confirmed
    # | rejected | cancelled | received
    status: Mapped[str] = mapped_column(String(32), default="draft", index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    expected_delivery_date: Mapped[date] = mapped_column(Date)
    created_by: Mapped[str] = mapped_column(String(64), default="system")
    # Links the PO back to the agent run that produced it — the audit trail.
    source_run_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    # Guards against duplicate POs when a step is retried.
    idempotency_key: Mapped[Optional[str]] = mapped_column(
        String(128), nullable=True, unique=True, index=True
    )
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    lines: Mapped[list["PurchaseOrderLine"]] = relationship(
        back_populates="po", cascade="all, delete-orphan", lazy="selectin"
    )

    @property
    def ordered_value_usd(self) -> float:
        return round(sum(line.ordered_units * line.unit_price_usd for line in self.lines), 2)

    @property
    def committed_value_usd(self) -> float:
        """Value of what the supplier actually committed to, not what we asked for."""
        return round(sum(line.effective_units * line.unit_price_usd for line in self.lines), 2)

    @property
    def is_open(self) -> bool:
        return self.status in {"draft", "pending_approval", "submitted", "confirmed", "partially_confirmed"}


class PurchaseOrderLine(Base):
    __tablename__ = "purchase_order_lines"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    po_id: Mapped[str] = mapped_column(ForeignKey("purchase_orders.po_id"), index=True)
    sku: Mapped[str] = mapped_column(ForeignKey("products.sku"), index=True)
    ordered_units: Mapped[int] = mapped_column(Integer)
    confirmed_units: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    received_units: Mapped[int] = mapped_column(Integer, default=0)
    unit_price_usd: Mapped[float] = mapped_column(Float)

    po: Mapped[PurchaseOrder] = relationship(back_populates="lines")

    @property
    def effective_units(self) -> int:
        """Confirmed quantity once the supplier has responded, else ordered."""
        return self.confirmed_units if self.confirmed_units is not None else self.ordered_units


class TransferOrder(Base):
    """Stock movement between nodes — the non-purchase lever available to a buyer."""

    __tablename__ = "transfer_orders"

    to_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    from_node_id: Mapped[str] = mapped_column(ForeignKey("nodes.node_id"))
    to_node_id: Mapped[str] = mapped_column(ForeignKey("nodes.node_id"))
    sku: Mapped[str] = mapped_column(ForeignKey("products.sku"), index=True)
    units: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(32), default="draft")
    transit_days: Mapped[int] = mapped_column(Integer, default=1)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    source_run_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)


class Budget(Base):
    """Purchasing budget, scoped to a node + category + period."""

    __tablename__ = "budgets"
    __table_args__ = (
        UniqueConstraint("node_id", "category", "period_start", name="uq_budget_scope"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    node_id: Mapped[str] = mapped_column(ForeignKey("nodes.node_id"), index=True)
    category: Mapped[str] = mapped_column(String(64), index=True)
    period_start: Mapped[date] = mapped_column(Date)
    period_end: Mapped[date] = mapped_column(Date)
    allocated_usd: Mapped[float] = mapped_column(Float)
    # Reserved by open POs that have not yet been invoiced.
    committed_usd: Mapped[float] = mapped_column(Float, default=0.0)
    spent_usd: Mapped[float] = mapped_column(Float, default=0.0)

    @property
    def available_usd(self) -> float:
        return round(self.allocated_usd - self.committed_usd - self.spent_usd, 2)


class PromoCalendar(Base):
    """Planned promotions — the usual benign explanation for a demand spike."""

    __tablename__ = "promo_calendar"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    sku: Mapped[str] = mapped_column(ForeignKey("products.sku"), index=True)
    node_id: Mapped[Optional[str]] = mapped_column(String(32), nullable=True, index=True)
    start_date: Mapped[date] = mapped_column(Date)
    end_date: Mapped[date] = mapped_column(Date)
    expected_uplift_factor: Mapped[float] = mapped_column(Float, default=1.0)
    promo_type: Mapped[str] = mapped_column(String(48), default="discount")
    description: Mapped[str] = mapped_column(String(256), default="")


# ─────────────────────────────────────────────────────────────────────
# 2. Supplier communication (UNTRUSTED input)
# ─────────────────────────────────────────────────────────────────────


class SupplierMessage(Base):
    """An inbound supplier email / EDI note.

    The body is attacker-controlled from the agent's point of view: a
    supplier can write anything in it, including text shaped like an
    instruction. Tools that surface these always mark them untrusted, and
    policy requires any factual claim here to be corroborated against the
    supplier availability API before it drives an action.
    """

    __tablename__ = "supplier_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    supplier_id: Mapped[str] = mapped_column(ForeignKey("suppliers.supplier_id"), index=True)
    po_id: Mapped[Optional[str]] = mapped_column(String(32), nullable=True, index=True)
    channel: Mapped[str] = mapped_column(String(24), default="email")
    subject: Mapped[str] = mapped_column(String(256))
    body: Mapped[str] = mapped_column(Text)
    received_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    is_read: Mapped[bool] = mapped_column(Boolean, default=False)


# ─────────────────────────────────────────────────────────────────────
# 3. Agent observability
# ─────────────────────────────────────────────────────────────────────


class AgentRun(Base):
    __tablename__ = "agent_runs"

    run_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    scenario_id: Mapped[str] = mapped_column(String(64), index=True)
    case_type: Mapped[str] = mapped_column(String(48))
    sku: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    node_id: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="running", index=True)
    provider: Mapped[str] = mapped_column(String(32), default="deterministic")
    model: Mapped[str] = mapped_column(String(64), default="")
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    completed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    # Final structured decision + validation report, kept verbatim.
    result: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON, nullable=True)
    total_tokens: Mapped[int] = mapped_column(Integer, default=0)
    duration_ms: Mapped[int] = mapped_column(Integer, default=0)

    steps: Mapped[list["AgentStep"]] = relationship(
        back_populates="run", cascade="all, delete-orphan", lazy="selectin"
    )


class AgentStep(Base):
    """One observable event in a run: a graph node entry, a tool call, an
    LLM turn, or a validation verdict. This is the trace the UI renders."""

    __tablename__ = "agent_steps"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("agent_runs.run_id"), index=True)
    seq: Mapped[int] = mapped_column(Integer)
    graph_node: Mapped[str] = mapped_column(String(48))
    kind: Mapped[str] = mapped_column(String(32))  # node|tool_call|llm|validation|guardrail
    title: Mapped[str] = mapped_column(String(256), default="")
    tool_name: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    payload_in: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON, nullable=True)
    payload_out: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON, nullable=True)
    latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    tokens: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    run: Mapped[AgentRun] = relationship(back_populates="steps")


class EvidenceRecord(Base):
    """A fact the agent gathered, tagged with the slot it fills.

    Required slots per case type are declared in code; the graph will not
    let the agent decide until every required slot is populated. This is
    what turns "did it gather the right information?" into an assertion
    rather than an opinion.
    """

    __tablename__ = "evidence_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("agent_runs.run_id"), index=True)
    slot: Mapped[str] = mapped_column(String(48), index=True)
    tool_name: Mapped[str] = mapped_column(String(64))
    value: Mapped[dict[str, Any]] = mapped_column(JSON)
    trusted: Mapped[bool] = mapped_column(Boolean, default=True)
    collected_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Approval(Base):
    """A human-in-the-loop gate. Created when the autonomy policy says the
    agent may propose but not execute."""

    __tablename__ = "approvals"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("agent_runs.run_id"), index=True)
    status: Mapped[str] = mapped_column(String(24), default="pending", index=True)
    reason: Mapped[str] = mapped_column(Text)
    # The exact action that will execute on approval — no re-reasoning.
    proposed_action: Mapped[dict[str, Any]] = mapped_column(JSON)
    risk_level: Mapped[str] = mapped_column(String(16), default="medium")
    requested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    decided_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    decided_by: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    decision_note: Mapped[Optional[str]] = mapped_column(Text, nullable=True)


class DomainEvent(Base):
    """Append-only log of every state change made through the tool layer."""

    __tablename__ = "domain_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    entity_type: Mapped[str] = mapped_column(String(48), index=True)
    entity_id: Mapped[str] = mapped_column(String(64), index=True)
    event_type: Mapped[str] = mapped_column(String(48), index=True)
    actor: Mapped[str] = mapped_column(String(64), default="agent")
    run_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
