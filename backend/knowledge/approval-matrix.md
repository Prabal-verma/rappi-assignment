# Purchasing Approval Matrix (POL-AUTONOMY)

Defines what the automated buyer may execute unattended and what a human must see first.

## Autonomous execution

An action may execute without human approval only when **all** of the following hold:

- every hard constraint check passes, with no blocking violations;
- the total committed value is at or below the autonomous ceiling (USD 5,000 by default);
- the quantity is within 35% of the planning system recommendation;
- the supplier is active and its reliability score is at or above 0.85;
- the agent's own confidence in the decision is at or above 0.6.

## Mandatory human approval

Route to the buyer approval queue when any of these apply:

- committed value above the autonomous ceiling;
- deviation from the system recommendation beyond 35%, in either direction;
- the supplier reliability score is below 0.85 and the order exceeds USD 1,000;
- a perishable product would be bought close to its shelf-life limit;
- a budget or storage exception is being requested;
- the decision is self-assessed as high risk, or confidence is below 0.6.

An approval request must carry the exact action that will execute on approval. The approver is
agreeing to a specific purchase order, not to a plan that will be re-derived afterwards.

## Escalation

Escalate rather than act when constraints conflict irreconcilably — for example when the
quantity needed to avoid a stockout cannot fit in the node or cannot be funded from the
remaining budget. An escalation must quantify the cost of inaction (expected lost margin from
the projected stockout) so the receiving human can trade it against the constraint being
breached. "I could not proceed" without a number is not an escalation, it is an abdication.

## Auditability

Every executed action records the run that produced it, the evidence the decision rested on, the
constraint checks that were run, and the validation verdict. Any purchase order created by the
agent must be traceable to its run.
