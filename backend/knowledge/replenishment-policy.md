# Replenishment Policy (POL-REPL)

Applies to all Turbo dark stores. Owned by Supply Planning. Reviewed quarterly.

## Ordering model

Nodes operate a periodic review (R, S) model. Each node has a review period (default 7 days)
and a service level target (default 95%). The order-up-to level S is
`daily demand x (lead time + review period) + safety stock`, and safety stock is
`z x sigma x sqrt(lead time + review period)`.

Never compute a requirement from on-hand stock alone. The correct basis is the **inventory
position**: on hand, minus reserved and damaged units, plus every unit genuinely inbound on an
open purchase order. Ignoring open POs is the single most common cause of over-ordering, and it
is the first thing to check when a recommendation looks too large.

## Treatment of open purchase orders

Count inbound units at their **confirmed** quantity, not the quantity originally ordered. Once a
supplier has confirmed a reduced amount, the difference is not inbound and must not be counted
as cover. A PO in draft or rejected status contributes nothing.

A delivery that arrives after the projected stockout date does not prevent the stockout. Always
compare the delivery date against the date stock reaches zero, not against the end of the period.

## Maximum cover

No single order may take the inventory position beyond **21 days of cover**, or beyond the
product shelf life if that is shorter. Cover beyond this ceiling ties up working capital and,
for perishables, guarantees write-off. Where a supplier minimum order quantity would breach the
ceiling, do not simply order the MOQ: look for a supplier with a lower MOQ, negotiate a split
delivery, or buy less and accept a shorter cycle.

## Rounding

Round a requirement **up** to the case pack, never down, so the order does not land short. Apply
MOQ after case-pack rounding. State the rounding in the rationale — a buyer reviewing an order
needs to see why the number is 264 rather than 260.

## When to reject a recommendation outright

Reject when the net requirement is zero because open POs already cover the target position;
when the product is being delisted within the cover horizon; or when the recommendation rests on
a forecast that observed sales have already contradicted. Rejecting is a legitimate outcome and
is preferred to ordering a token quantity to appear responsive.
