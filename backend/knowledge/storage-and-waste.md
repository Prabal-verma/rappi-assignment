# Storage and Waste Policy (POL-STORAGE, POL-WASTE)

## Node capacity is a hard constraint

Dark stores are volume-constrained. Available space is the node capacity less the volume of stock
on hand and the volume of stock already inbound on open purchase orders. Stock that will arrive
occupies space that is not available now.

An order that does not fit cannot be received. There is no override at buyer level: reduce the
quantity, phase the delivery, or move the volume to a node with space.

## Options when storage binds

In order of preference:

1. **Reduce the quantity** to what fits, if the shorter cover is acceptable.
2. **Phase the delivery** across two drops where the supplier allows split delivery — this can
   satisfy a large MOQ without breaching capacity on any single day.
3. **Source from a supplier with a smaller MOQ**, even at a higher unit price, when the
   difference is less than the margin at risk from the stockout.
4. **Transfer from a node with excess stock** in the same country.
5. **Escalate** with the quantified cost of the shortfall if none of the above closes the gap.

## Perishables

For products with a shelf life shorter than the 21-day cover ceiling, the shelf life is the
binding limit. Never order a quantity that puts the position beyond what will sell within the
shelf life; the excess is a guaranteed write-off, not a buffer.

Perishable orders that approach the shelf-life limit require buyer approval even when they pass
every hard constraint, because the waste judgement depends on commercial context the planning
system does not hold.

## Quantifying the cost of not buying

When recommending a reduced quantity or an escalation, state the expected lost margin:
`units short x unit margin x (1 - substitution rate)`, with a default substitution rate of 30%
reflecting shoppers who buy an alternative instead. A constraint breach is a commercial
trade-off, and it cannot be made without both sides of the number.
