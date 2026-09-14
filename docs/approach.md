# Approach

How I broke the problem down, what I decided and why, and where the design is weak.

## Reading the brief

The line that set the direction was the last one: *"The goal is not to build a chatbot that
answers purchasing questions. The goal is to demonstrate a system capable of making, executing and
validating purchasing decisions."*

Two other lines shaped it as much:

- *"The purchasing recommendation provided to the agent should not necessarily be assumed to be
  correct."* — so the scenarios must be built such that agreeing with the input is the wrong
  answer. An agent that accepts recommendations looks competent on friendly data and is worthless
  on real data.
- *"If the outcome is different from what the agent expected, the system should be able to handle
  that situation appropriately."* — the word **expected** implies the agent must have had an
  expectation, explicitly, in a form the system can check. That became `ActionExpectation`, and it
  is the hinge the whole validation design turns on.

## Breaking it down

I split the problem into four questions and answered them in different places on purpose.

**1. What is true?** — the evidence problem.
An agent that decides on partial facts is confidently wrong. Rather than trusting the model to be
thorough, each case type declares the evidence slots it may not be decided without, tools declare
which slot they fill, and the graph refuses to proceed until they are populated — backfilling
anything the model skipped and recording that it had to. *Did it gather what it needed* became a
property that can be asserted rather than a judgement.

**2. What is the right number?** — the arithmetic problem.
Language models are unreliable at arithmetic and completely reliable at *sounding* right about
arithmetic, which is the dangerous combination. So every number is computed in Python — safety
stock, order-up-to level, net requirement, stockout date, shelf-life ceiling — and the model is
given the computed options. It chooses; it does not calculate.

**3. What is allowed?** — the constraint problem.
Business rules expressed in a prompt are requests. Expressed in code they are guarantees. Eleven
constraints and an autonomy policy sit between the model's proposal and the database, and the
model is never given a write tool at all. That last decision also closes the prompt-injection
surface as a side effect, which is why `SC2X` fails structurally rather than behaviourally.

**4. Did it work?** — the feedback problem.
The agent commits to a numeric prediction before acting. Afterwards, a validator that has never
seen the agent's reasoning re-derives the world from the database and diffs it. Drift feeds back
and the agent replans, bounded, then escalates.

## Things I got wrong on the first pass

Worth recording, because they were the interesting part and the evaluation suite is what caught
all of them.

**The replan loop decided on stale evidence.** After acting, the agent re-read state into the
evidence table — but not into the conversation transcript the reasoning layer actually reads. So
it kept solving a shortfall its own first order had already partly closed, and looped until it ran
out of attempts. Refreshed facts now land in both.

**The residual and the constraint engine disagreed about the same world.** The agent priced the
requirement against the incumbent supplier's seven-day lead time while the guard evaluated it
against the one-day replacement it had just switched to. Different exposure windows, so one said
"302 units short" and the other said "already covered", and the loop oscillated. Both now use the
sourcing supplier. This was the most interesting bug: not a typo, but two parts of the system
holding different models of the same situation.

**Transfer orders did not count as inbound stock.** A transfer looked like it had done nothing, so
the agent went back to buy stock it had already moved.

**Amendments looked like drift.** Raising a PO from 400 to 800 adds 400 units of cover but leaves
an order line reading 800. Conflating the two made every successful amendment fail validation.

**A Postgres-only seed bug.** Foreign-key inserts were unordered; SQLite tolerates that by default
and Postgres does not. It only surfaced when the compose stack came up, which is a decent argument
for running the same code against both.

## Where this design is weak

- **Five scenarios authored by the same person who wrote the agent.** They encode the failure
  modes I thought of. The failures that matter in production are the ones nobody thought of.
- **The rules planner matches the model on these cases.** `--compare` shows no delta. For cases
  this well-specified, a deterministic planner is sufficient — the model's value should appear on
  messier, less structured cases, and this suite does not contain any. I would rather state that
  plainly than imply the LLM is carrying the system.
- **Single-case, not batch.** Real replenishment runs thousands of SKU/node pairs nightly. The
  calculation layer would vectorise; the graph would not.
- **No durable checkpointing.** A run awaiting approval is reconstructed from the stored action
  rather than resumed from a persisted graph state. Fine here, wrong for production.
- **Demand modelling is deliberately simple.** Normal demand, a z-table, a two-sigma anomaly rule
  and a flat promotional halo. Real forecasting is a data-science problem; what mattered here was
  that the agent reasons correctly about whatever the forecast says, including when to distrust it.

## What I would build next

1. Persist the graph checkpointer so human-in-the-loop runs survive a restart.
2. Discount supplier promises by their observed fill rate — the data is already in the schema and
   used for autonomy decisions, but not yet to adjust what a supplier commits to.
3. Add an LLM judge for rationale quality alongside the deterministic checks, not instead of them.
4. Widen the suite until it can measure a rate rather than demonstrate a mechanism, and report
   variance across repeated runs.
