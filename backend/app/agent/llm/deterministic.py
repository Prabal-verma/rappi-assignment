"""A rule-based planner wearing the LLMClient interface.

This exists for three reasons, in order of importance:

1. **Reproducible evaluation.** A scored suite that cannot be re-run to the
   same answer is not a regression test. The deterministic planner gives
   every scenario a fixed, inspectable result to diff against.
2. **A baseline to beat.** It is the control arm: if the model does not
   decide better than a well-written rules engine on these cases, the model
   is not earning its cost.
3. **Graceful degradation.** When the provider is unreachable or returns
   garbage, the graph falls back here rather than failing the run. A
   purchasing system should not stop buying because inference is down.

It reads the same evidence the model reads — it reconstructs it from the
tool results already in the conversation — so the two arms are genuinely
comparable.
"""

from __future__ import annotations

import json
import re
from typing import Any

from app.agent.llm.base import LLMResponse, ToolCall, Turn

CASE_RE = re.compile(r"<case>(.*?)</case>", re.DOTALL)

# Ordered investigation playbooks. Each entry is (tool_name, extra_args).
PLAYBOOKS: dict[str, list[tuple[str, dict[str, Any]]]] = {
    "recommendation_review": [
        ("get_product", {}),
        ("get_inventory_position", {}),
        ("get_demand_forecast", {"horizon_days": 14}),
        ("list_open_purchase_orders", {}),
        ("get_supplier_terms", {}),
        ("get_budget_status", {}),
        ("get_storage_capacity", {}),
        ("analyze_replenishment", {}),
        ("search_policies", {"query": "open purchase orders maximum cover reject recommendation"}),
    ],
    "supplier_shortfall": [
        ("read_supplier_messages", {}),
        ("check_supplier_availability", {}),
        ("get_inventory_position", {}),
        ("list_open_purchase_orders", {}),
        ("get_demand_forecast", {"horizon_days": 14}),
        ("analyze_replenishment", {}),
        ("get_alternate_suppliers", {}),
        ("find_stock_at_other_nodes", {}),
        ("search_policies", {"query": "supplier confirmed less than ordered residual sourcing transfer"}),
    ],
    "demand_shift": [
        ("get_sales_history", {"lookback_days": 35}),
        ("detect_demand_anomaly", {}),
        ("get_promotions", {}),
        ("get_inventory_position", {}),
        ("get_demand_forecast", {"horizon_days": 14}),
        ("list_open_purchase_orders", {}),
        ("analyze_replenishment", {}),
        ("search_policies", {"query": "demand spike promotion extrapolate modify existing purchase order"}),
    ],
    "constrained_purchase": [
        ("get_product", {}),
        ("get_inventory_position", {}),
        ("get_demand_forecast", {"horizon_days": 14}),
        ("analyze_replenishment", {}),
        ("get_budget_status", {}),
        ("get_storage_capacity", {}),
        ("get_supplier_terms", {}),
        ("get_alternate_suppliers", {}),
        ("find_stock_at_other_nodes", {}),
        ("search_policies", {"query": "storage constraint budget exception escalate split delivery"}),
    ],
}


def _parse_case(turns: list[Turn]) -> dict[str, Any]:
    for turn in turns:
        if turn.role == "user" and turn.content:
            match = CASE_RE.search(turn.content)
            if match:
                try:
                    return json.loads(match.group(1))
                except json.JSONDecodeError:
                    continue
    return {}


def _called_tools(turns: list[Turn]) -> list[str]:
    names: list[str] = []
    for turn in turns:
        for call in turn.tool_calls:
            names.append(call.name)
    return names


def _evidence(turns: list[Turn]) -> dict[str, dict[str, Any]]:
    """Rebuild the evidence table from tool-result turns."""
    out: dict[str, dict[str, Any]] = {}
    for turn in turns:
        if turn.role != "tool" or not turn.tool_name:
            continue
        try:
            payload = json.loads(turn.content or "{}")
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            out[turn.tool_name] = payload.get("data", payload)
    return out


def _round_down_to_pack(units: int, pack: int) -> int:
    if pack <= 1:
        return max(0, units)
    return max(0, (units // pack) * pack)


class DeterministicClient:
    name = "deterministic"
    model = "rules-v1"

    # ---- investigation ------------------------------------------------

    def converse(
        self, system: str, turns: list[Turn], tools: list[dict[str, Any]] | None = None
    ) -> LLMResponse:
        case = _parse_case(turns)
        playbook = PLAYBOOKS.get(case.get("case_type", ""), PLAYBOOKS["recommendation_review"])
        already = _called_tools(turns)

        for tool_name, extra in playbook:
            if tool_name in already:
                continue
            args = dict(extra)
            if "sku" not in args and case.get("sku"):
                args["sku"] = case["sku"]
            if "node_id" not in args and case.get("node_id"):
                args["node_id"] = case["node_id"]
            if tool_name in {"get_supplier_terms", "check_supplier_availability"} and case.get("supplier_id"):
                args["supplier_id"] = case["supplier_id"]
            if tool_name == "check_supplier_availability":
                args["units"] = case.get("ordered_units") or case.get("recommended_units") or 0
            if tool_name == "get_alternate_suppliers" and case.get("supplier_id"):
                args["exclude_supplier_id"] = case["supplier_id"]
            if tool_name == "read_supplier_messages" and case.get("po_id"):
                args["po_id"] = case["po_id"]
                args.pop("sku", None)
                args.pop("node_id", None)

            return LLMResponse(
                text="",
                tool_calls=[ToolCall(id=f"det_{len(already)}", name=tool_name, arguments=args)],
                stop_reason="tool_use",
            )

        return LLMResponse(text="Investigation complete.", stop_reason="end_turn")

    # ---- decision -----------------------------------------------------

    def structured(self, system: str, turns: list[Turn], schema: dict[str, Any]) -> dict[str, Any]:
        case = _parse_case(turns)
        ev = _evidence(turns)
        case_type = case.get("case_type", "recommendation_review")

        analysis = ev.get("analyze_replenishment", {})
        sku = case.get("sku") or analysis.get("sku")
        node_id = case.get("node_id") or analysis.get("node_id")

        handler = {
            "recommendation_review": self._decide_recommendation,
            "supplier_shortfall": self._decide_shortfall,
            "demand_shift": self._decide_demand_shift,
            "constrained_purchase": self._decide_constrained,
        }.get(case_type, self._decide_recommendation)

        decision = handler(case, ev, analysis, sku, node_id)
        decision.setdefault("confidence", 0.75)
        decision.setdefault("risk_level", "medium")
        decision.setdefault("information_gaps", [])
        return decision

    # -- helpers --------------------------------------------------------

    @staticmethod
    def _expectation(
        analysis: dict,
        position_delta: int,
        unit_cost: float,
        secured_units: int | None = None,
    ) -> dict:
        """Predict the post-action world.

        `position_delta` is how much stock the action adds. `secured_units`
        is what validation will read off the purchase order — for a new PO
        they are the same, but amending an existing PO from 400 to 800 adds
        400 units of cover while the order line reads 800. Conflating the
        two makes every amendment look like drift.
        """
        position = analysis.get("inventory_position_units", 0) + position_delta
        daily = analysis.get("daily_demand_mean", 0) or 0
        cover = round(position / daily, 2) if daily > 0 else 0.0
        secured = position_delta if secured_units is None else secured_units
        return {
            "expected_units_secured": secured,
            "expected_spend_usd": round(secured * unit_cost, 2),
            "expected_inventory_position_units": position,
            "expected_days_of_cover": cover,
            "expected_stockout_risk": "low" if cover >= 7 else "medium",
            "expected_constraint_violations": 0,
            "notes": "Computed from the replenishment analysis at decision time.",
        }

    @staticmethod
    def _pick_supplier(ev: dict, stockout_days: float | None) -> dict | None:
        """First orderable supplier that lands before stock runs out."""
        suppliers = (ev.get("get_alternate_suppliers") or {}).get("suppliers") or []
        orderable = [s for s in suppliers if s.get("can_be_ordered_from")]
        if not orderable:
            return None
        if stockout_days is not None:
            in_time = [s for s in orderable if s.get("lead_time_days", 99) <= stockout_days]
            if in_time:
                return sorted(in_time, key=lambda s: s.get("unit_price_usd", 0))[0]
        return sorted(orderable, key=lambda s: (s.get("lead_time_days", 99), s.get("unit_price_usd", 0)))[0]

    # -- case handlers --------------------------------------------------

    def _decide_recommendation(self, case, ev, analysis, sku, node_id) -> dict:
        recommended = case.get("recommended_units") or 0
        supplier_id = case.get("supplier_id") or analysis.get("supplier_id")
        correct = analysis.get("recommended_order_rounded", 0)
        unit_cost = analysis.get("unit_cost_usd", 0.0)
        net_req = analysis.get("net_requirement_units", 0)
        factors = [
            f"Inventory position is {analysis.get('inventory_position_units')} units "
            f"({analysis.get('available_units')} available + {analysis.get('in_transit_units')} inbound).",
            f"Order-up-to level is {analysis.get('target_position_units')} units at "
            f"{analysis.get('daily_demand_mean')} units/day ({analysis.get('demand_source')}).",
            f"Net requirement is {net_req} units; computed order quantity {correct} "
            f"({analysis.get('rounding_note')}).",
        ]
        factors.extend(analysis.get("notes", [])[:3])

        if net_req <= 0:
            return {
                "decision_type": "reject",
                "recommended_units": 0,
                "rationale": (
                    f"The recommendation of {recommended} units should be rejected. Open purchase "
                    f"orders already bring the inventory position to "
                    f"{analysis.get('inventory_position_units')} units against an order-up-to level "
                    f"of {analysis.get('target_position_units')}, so the net requirement is zero. "
                    f"Buying more would duplicate cover already on the water."
                ),
                "key_factors": factors,
                "proposed_actions": [
                    {"action_type": "no_action", "sku": sku, "node_id": node_id,
                     "reason": "Existing open POs already cover the requirement."}
                ],
                "expectation": self._expectation(analysis, 0, unit_cost),
                "confidence": 0.85,
                "risk_level": "low",
            }

        deviation = abs(recommended - correct) / max(recommended, 1)
        action = [
            {
                "action_type": "create_purchase_order",
                "sku": sku,
                "node_id": node_id,
                "supplier_id": supplier_id,
                "units": correct,
                "reason": f"Replenishment to order-up-to level; net requirement {net_req} units.",
            }
        ]

        if deviation <= 0.1:
            return {
                "decision_type": "accept",
                "recommended_units": correct,
                "rationale": (
                    f"The recommendation of {recommended} units is within 10% of the computed "
                    f"requirement of {correct} units, and every constraint is satisfied. Accepting."
                ),
                "key_factors": factors,
                "proposed_actions": action,
                "expectation": self._expectation(analysis, correct, unit_cost),
                "confidence": 0.8,
                "risk_level": "low",
            }

        return {
            "decision_type": "modify",
            "recommended_units": correct,
            "rationale": (
                f"The recommendation of {recommended} units is {deviation:.0%} away from the "
                f"computed requirement of {correct}. The position already includes "
                f"{analysis.get('in_transit_units')} inbound units, and the shelf-life ceiling "
                f"caps useful stock at {analysis.get('max_useful_units')} units. Modifying to "
                f"{correct} units."
            ),
            "key_factors": factors,
            "proposed_actions": action,
            "expectation": self._expectation(analysis, correct, unit_cost),
            "confidence": 0.8,
            "risk_level": "medium",
        }

    def _decide_shortfall(self, case, ev, analysis, sku, node_id) -> dict:
        availability = ev.get("check_supplier_availability", {})
        net_req = analysis.get("net_requirement_units", 0)
        stockout = analysis.get("projected_stockout_in_days")
        factors = [
            f"Supplier availability API confirms {availability.get('available_units')} units "
            f"against a request for {availability.get('requested_units')} "
            f"(shortfall {availability.get('shortfall_units')}).",
            f"Counting only confirmed inbound units, the position is "
            f"{analysis.get('inventory_position_units')} against a target of "
            f"{analysis.get('target_position_units')}.",
            f"Residual requirement after the shortfall is {net_req} units; stock runs out in "
            f"{stockout} days.",
        ]

        if net_req <= 0:
            return {
                "decision_type": "accept",
                "recommended_units": 0,
                "rationale": (
                    f"The short shipment is absorbable. Even with only "
                    f"{availability.get('available_units')} units confirmed, the position of "
                    f"{analysis.get('inventory_position_units')} units covers "
                    f"{analysis.get('days_of_cover_now')} days against the target. No further "
                    f"sourcing is needed; the shortfall is recorded against the supplier."
                ),
                "key_factors": factors,
                "proposed_actions": [
                    {"action_type": "no_action", "sku": sku, "node_id": node_id,
                     "reason": "Confirmed quantity plus existing stock still covers demand."}
                ],
                "expectation": self._expectation(analysis, 0, analysis.get("unit_cost_usd", 0)),
                "confidence": 0.8,
                "risk_level": "low",
            }

        transfer = next(
            (
                c
                for c in (ev.get("find_stock_at_other_nodes") or {}).get("candidates", [])
                if c.get("transfer_possible")
            ),
            None,
        )

        # Policy POL-SUPPLIER prefers a transfer over a second purchase order
        # when it fully closes the gap: the stock is already paid for and it
        # usually arrives sooner. Only worth it if it covers the whole
        # residual — a partial transfer just relocates the problem.
        if transfer and transfer["transferable_surplus_units"] >= net_req:
            factors.append(
                f"{transfer['node_id']} holds {transfer['transferable_surplus_units']} units "
                f"above its own target, enough to cover the residual without buying."
            )
            return {
                "decision_type": "modify",
                "recommended_units": net_req,
                "rationale": (
                    f"The residual requirement of {net_req} units can be met without a purchase. "
                    f"{transfer['node_id']} holds {transfer['transferable_surplus_units']} units "
                    f"above its own target position, and policy prefers a transfer over a second "
                    f"purchase order when it closes the gap — the stock is already paid for and "
                    f"arrives in {transfer.get('estimated_transit_days', 1)} day(s), inside the "
                    f"{stockout}-day runway."
                ),
                "key_factors": factors,
                "proposed_actions": [
                    {
                        "action_type": "create_transfer_order",
                        "sku": sku,
                        "node_id": node_id,
                        "from_node_id": transfer["node_id"],
                        "units": net_req,
                        "reason": "Covering the supplier shortfall from same-country surplus.",
                    }
                ],
                "expectation": self._expectation(analysis, net_req, 0.0),
                "confidence": 0.8,
                "risk_level": "low",
            }

        alternate = self._pick_supplier(ev, stockout)
        if alternate is None:
            if transfer:
                units = min(net_req, transfer["transferable_surplus_units"])
                return {
                    "decision_type": "modify",
                    "recommended_units": units,
                    "rationale": (
                        f"No alternate supplier can deliver before stock runs out in {stockout} "
                        f"days, but {transfer['node_id']} holds {transfer['transferable_surplus_units']} "
                        f"units above its own target. Transferring {units} units closes the gap "
                        f"with stock already paid for."
                    ),
                    "key_factors": factors,
                    "proposed_actions": [
                        {
                            "action_type": "create_transfer_order",
                            "sku": sku,
                            "node_id": node_id,
                            "from_node_id": transfer["node_id"],
                            "units": units,
                            "reason": "Bridging the supplier shortfall from same-country surplus.",
                        }
                    ],
                    "expectation": self._expectation(analysis, units, 0.0),
                    "confidence": 0.7,
                    "risk_level": "medium",
                }
            return {
                "decision_type": "escalate",
                "recommended_units": net_req,
                "rationale": (
                    f"A residual requirement of {net_req} units remains after the short shipment, "
                    f"no active alternate supplier can deliver before the projected stockout in "
                    f"{stockout} days, and no node holds transferable surplus. This needs a human."
                ),
                "key_factors": factors,
                "proposed_actions": [
                    {"action_type": "escalate_to_human", "sku": sku, "node_id": node_id,
                     "units": net_req, "reason": "No viable source for the residual requirement."}
                ],
                "expectation": self._expectation(analysis, 0, 0.0),
                "confidence": 0.65,
                "risk_level": "high",
            }

        pack = alternate.get("case_pack", 1) or 1
        moq = alternate.get("moq_units", 0) or 0
        units = max(moq, -(-net_req // pack) * pack)
        return {
            "decision_type": "modify",
            "recommended_units": units,
            "rationale": (
                f"The supplier confirmed only {availability.get('available_units')} of "
                f"{availability.get('requested_units')} units, leaving a residual requirement of "
                f"{net_req}. {alternate['name']} is active, delivers in "
                f"{alternate['lead_time_days']} days — inside the {stockout}-day runway — at "
                f"${alternate['unit_price_usd']}/unit. Raising a second order for {units} units "
                f"(case pack {pack}, MOQ {moq})."
            ),
            "key_factors": factors,
            "proposed_actions": [
                {
                    "action_type": "create_purchase_order",
                    "sku": sku,
                    "node_id": node_id,
                    "supplier_id": alternate["supplier_id"],
                    "units": units,
                    "reason": "Covering the residual requirement left by the short shipment.",
                }
            ],
            "expectation": self._expectation(analysis, units, alternate.get("unit_price_usd", 0)),
            "confidence": 0.78,
            "risk_level": "medium",
        }

    def _decide_demand_shift(self, case, ev, analysis, sku, node_id) -> dict:
        anomaly = ev.get("detect_demand_anomaly", {})
        promos = (ev.get("get_promotions") or {}).get("promotions", [])
        live = [p for p in promos if p.get("is_live_today")]
        open_pos = (ev.get("list_open_purchase_orders") or {}).get("purchase_orders", [])
        net_req = analysis.get("net_requirement_units", 0)
        unit_cost = analysis.get("unit_cost_usd", 0.0)

        factors = [
            anomaly.get("explanation", ""),
            f"Demand basis used: {analysis.get('demand_source')} at "
            f"{analysis.get('daily_demand_mean')} units/day.",
            f"Position {analysis.get('inventory_position_units')} vs target "
            f"{analysis.get('target_position_units')}; stock out in "
            f"{analysis.get('projected_stockout_in_days')} days.",
        ]
        if live:
            p = live[0]
            factors.append(
                f"A promotion ({p.get('description') or p.get('promo_type')}) is live and ends in "
                f"{p.get('days_remaining')} day(s) — the uplift is partly temporary."
            )

        if not anomaly.get("detected"):
            return {
                "decision_type": "reject",
                "recommended_units": 0,
                "rationale": (
                    f"The apparent spike does not clear the bar for action: "
                    f"{anomaly.get('explanation')} Policy requires two baseline standard "
                    f"deviations on three consecutive days before re-planning. Leaving the "
                    f"existing plan unchanged."
                ),
                "key_factors": factors,
                "proposed_actions": [
                    {"action_type": "no_action", "sku": sku, "node_id": node_id,
                     "reason": "Uplift is within normal variation."}
                ],
                "expectation": self._expectation(analysis, 0, unit_cost),
                "confidence": 0.75,
                "risk_level": "low",
            }

        if net_req <= 0:
            return {
                "decision_type": "accept",
                "recommended_units": 0,
                "rationale": (
                    "Demand has genuinely shifted, but the existing purchase order still brings "
                    f"the position to {analysis.get('inventory_position_units')} units against a "
                    f"revised target of {analysis.get('target_position_units')}. No change needed."
                ),
                "key_factors": factors,
                "proposed_actions": [
                    {"action_type": "no_action", "sku": sku, "node_id": node_id,
                     "reason": "Existing plan still covers the revised demand."}
                ],
                "expectation": self._expectation(analysis, 0, unit_cost),
                "confidence": 0.75,
                "risk_level": "low",
            }

        amendable = next((p for p in open_pos if p.get("status") in {"submitted", "confirmed", "partially_confirmed"}), None)
        if amendable:
            # Add the *rounded* requirement, not the raw one: the amended
            # total still has to be a whole number of cases, and a supplier
            # cannot ship 749 units of a 20-pack.
            increment = analysis.get("recommended_order_rounded") or net_req
            new_units = amendable.get("ordered_units", 0) + increment
            promo_caveat = (
                " but partly promotional, so it is not extrapolated at full strength" if live else ""
            )
            return {
                "decision_type": "modify",
                "recommended_units": new_units,
                "rationale": (
                    f"The uplift is real{promo_caveat}. Against the revised demand of "
                    f"{analysis.get('daily_demand_mean')} units/day the position is short by "
                    f"{net_req} units. Policy prefers amending an open order to raising a second "
                    f"one, so {amendable['po_id']} is increased from "
                    f"{amendable.get('ordered_units')} to {new_units} units."
                ),
                "key_factors": factors,
                "proposed_actions": [
                    {
                        "action_type": "modify_purchase_order",
                        "po_id": amendable["po_id"],
                        "sku": sku,
                        "node_id": node_id,
                        "units": new_units,
                        "reason": "Covering confirmed demand uplift on the existing order.",
                    }
                ],
                "expectation": self._expectation(
                    analysis, increment, unit_cost, secured_units=new_units
                ),
                "confidence": 0.75,
                "risk_level": "medium",
            }

        units = analysis.get("recommended_order_rounded", net_req)
        return {
            "decision_type": "modify",
            "recommended_units": units,
            "rationale": (
                f"Confirmed demand uplift with no amendable open order. Raising a new purchase "
                f"order for {units} units to restore the order-up-to level."
            ),
            "key_factors": factors,
            "proposed_actions": [
                {
                    "action_type": "create_purchase_order",
                    "sku": sku,
                    "node_id": node_id,
                    "supplier_id": analysis.get("supplier_id"),
                    "units": units,
                    "reason": "Covering confirmed demand uplift.",
                }
            ],
            "expectation": self._expectation(analysis, units, unit_cost),
            "confidence": 0.72,
            "risk_level": "medium",
        }

    def _decide_constrained(self, case, ev, analysis, sku, node_id) -> dict:
        storage = ev.get("get_storage_capacity", {})
        budget = ev.get("get_budget_status", {})
        terms = ev.get("get_supplier_terms", {})
        net_req = analysis.get("net_requirement_units", 0)
        unit_cost = terms.get("unit_price_usd") or analysis.get("unit_cost_usd", 0.0)

        fits_storage = storage.get("max_additional_units_that_fit")
        affordable = int(budget.get("available_usd", 0) // unit_cost) if unit_cost else net_req
        feasible = min(x for x in [net_req, fits_storage, affordable] if x is not None)

        factors = [
            f"Requirement is {net_req} units.",
            f"Storage allows at most {fits_storage} more units ({storage.get('free_m3')} m³ free).",
            f"Budget of ${budget.get('available_usd')} allows at most {affordable} units at "
            f"${unit_cost}/unit.",
            f"Primary supplier MOQ is {terms.get('moq_units')} with case pack {terms.get('case_pack')}.",
        ]

        # Can the primary supplier work within the feasible envelope?
        pack = terms.get("case_pack", 1) or 1
        moq = terms.get("moq_units", 0) or 0
        primary_qty = _round_down_to_pack(feasible, pack)
        primary_viable = primary_qty >= moq and primary_qty > 0

        if primary_viable and primary_qty >= net_req:
            return {
                "decision_type": "accept",
                "recommended_units": primary_qty,
                "rationale": (
                    f"The full requirement of {net_req} units fits inside both constraints "
                    f"({fits_storage} units of space, {affordable} units of budget). Proceeding "
                    f"with {primary_qty} units."
                ),
                "key_factors": factors,
                "proposed_actions": [
                    {"action_type": "create_purchase_order", "sku": sku, "node_id": node_id,
                     "supplier_id": terms.get("supplier_id"), "units": primary_qty,
                     "reason": "Requirement fits within all constraints."}
                ],
                "expectation": self._expectation(analysis, primary_qty, unit_cost),
                "confidence": 0.8,
                "risk_level": "low",
            }

        # Primary blocked — look for a supplier whose MOQ fits the envelope.
        alternates = (ev.get("get_alternate_suppliers") or {}).get("suppliers") or []
        best = None
        for s in alternates:
            if not s.get("can_be_ordered_from"):
                continue
            s_pack = s.get("case_pack", 1) or 1
            s_afford = int(budget.get("available_usd", 0) // s.get("unit_price_usd", 1))
            s_feasible = _round_down_to_pack(
                min(net_req, fits_storage if fits_storage is not None else net_req, s_afford), s_pack
            )
            if s_feasible >= (s.get("moq_units", 0) or 0) and s_feasible > 0:
                if best is None or s_feasible > best[1]:
                    best = (s, s_feasible)

        cost = ev.get("estimate_stockout_cost", {})
        if best is not None:
            supplier, qty = best
            shortfall = max(0, net_req - qty)
            factors.append(
                f"{supplier['name']} has an MOQ of {supplier.get('moq_units')} which fits the "
                f"feasible envelope at ${supplier.get('unit_price_usd')}/unit."
            )
            return {
                "decision_type": "modify",
                "recommended_units": qty,
                "rationale": (
                    f"The full requirement of {net_req} units cannot be executed: storage allows "
                    f"{fits_storage} and budget allows {affordable}, and the primary supplier's "
                    f"MOQ of {moq} does not fit inside that envelope. {supplier['name']} does, so "
                    f"buying {qty} units there covers what can be covered. The remaining "
                    f"{shortfall} units of demand are exposed, worth roughly "
                    f"${cost.get('expected_lost_margin_usd', 0)} in lost margin, which is "
                    f"escalated separately rather than forced through."
                ),
                "key_factors": factors,
                "proposed_actions": [
                    {"action_type": "create_purchase_order", "sku": sku, "node_id": node_id,
                     "supplier_id": supplier["supplier_id"], "units": qty,
                     "reason": "Largest quantity that satisfies storage, budget and MOQ together."}
                ],
                "expectation": self._expectation(analysis, qty, supplier.get("unit_price_usd", 0)),
                "confidence": 0.7,
                "risk_level": "medium",
            }

        return {
            "decision_type": "escalate",
            "recommended_units": net_req,
            "rationale": (
                f"No executable purchase exists. The requirement of {net_req} units exceeds both "
                f"the {fits_storage} units of available space and the {affordable} units the "
                f"remaining budget allows, and no active supplier has a minimum order quantity "
                f"that fits inside that envelope. Escalating with the cost of inaction: roughly "
                f"${cost.get('expected_lost_margin_usd', 0)} of lost margin."
            ),
            "key_factors": factors,
            "proposed_actions": [
                {"action_type": "escalate_to_human", "sku": sku, "node_id": node_id,
                 "units": net_req, "reason": "Constraints cannot be satisfied simultaneously."}
            ],
            "expectation": self._expectation(analysis, 0, unit_cost),
            "confidence": 0.7,
            "risk_level": "high",
        }
