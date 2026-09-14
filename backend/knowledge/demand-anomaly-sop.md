# Demand Anomaly SOP (POL-DEMAND)

How to respond when actual sales diverge from forecast.

## Establish that the signal is real

Do not re-plan on a single day of sales. A demand shift is actionable only when the recent
selling rate exceeds the baseline by more than two baseline standard deviations, on at least
three consecutive days. Anything weaker is normal variation and must be left alone; reacting to
noise produces a whipsaw of over- and under-orders.

## Check for a promotional cause before extrapolating

Always consult the promotion calendar before treating a spike as a change in underlying demand.
A lift that coincides with a live promotion is temporary. Model it as elevated demand for the
remaining promotional days, then a residual halo of roughly 25% of the lift, and not as a
permanent step change. Extrapolating a full promotional uplift past the promotion end date is
the classic way to buy stock that will never sell.

Other benign explanations to rule out: a competitor stockout, a one-off bulk order from a single
customer, a pricing error, and cannibalisation from a sibling SKU that was itself out of stock.

## Censored demand

If the product was out of stock during the observation window, recorded sales understate true
demand. Note the censoring explicitly and treat the measured uplift as a lower bound.

## Acting on a confirmed shift

When a shift is confirmed, prefer **modifying an existing open purchase order** over raising a
new one: it avoids a second delivery, a second set of freight costs and a duplicate cover risk.
Only raise an additional order when the existing PO cannot be amended, or when its arrival date
is too late to matter.

Recompute safety stock as well as the cycle quantity. A higher mean demand with unchanged
variability still raises the order-up-to level, and forgetting this leaves the node exposed
precisely when it is selling well.

## Escalate the forecast, not just the order

A confirmed sustained shift means the published forecast is wrong. Record the divergence so
Demand Planning can retrain; fixing one purchase order without flagging the forecast leaves the
same error to recur every cycle.
