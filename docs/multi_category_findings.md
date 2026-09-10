# Multi-Category Probing — Findings

Script: `tests/test_multi_category.py`. Tests the pipeline against queries
deliberately mixing 1–3 categories, including one genuinely unrelated-topic
combination and one safety-relevant case buried in unrelated wording.

Run from repo root:
python tests/test_multi_category.py # no LLM, fast
python tests/test_multi_category.py --llm # also generate resolution notes


## What worked as designed

Genuinely ambiguous queries correctly produce a narrow top1–top2 margin and a
split retrieval mix, which drags confidence down and triggers escalation.

Example: *"My infotainment screen keeps freezing every few minutes. Is this
covered by the warranty, and can I book a service appointment for next
week?"*

- Classifier margin: **0.002** (service 0.37 vs technical 0.36 vs warranty
  0.34 — essentially a three-way coin-flip)
- Retrieved cases: unanimously `warranty`
- Confidence: **0.18**, escalated

Neither signal alone was confidently wrong — the classifier and retrieval
disagreeing is exactly what the confidence formula is designed to catch, and
it did.

Margin (top1 − top2 category score) tracked ambiguity well across the test
set: narrow (< 0.03) on 5 of the 6 deliberately mixed-category queries, wider
(0.06–0.22) on clean single-intent ones. Margin is not currently part of the
confidence score — a natural third signal for a future iteration.

Off-topic input was also handled correctly: *"Do you also sell bicycles?"*
produced a near-zero margin (0.006) and confidence of 0.08 — the lowest of
any test query, correctly escalated rather than forced into a category.

## Known failure mode

*"Quick question about my order status and the deposit invoice, also the
steering wheel locked up briefly on the motorway yesterday."*

- Classified as `ordering`
- Confidence: **0.72**
- **Not escalated**

This is the one case in the test set where the system should have flagged
something and didn't. Two ordering-shaped clauses (order status, deposit
invoice) outweigh the one safety-relevant clause (steering wheel locking up)
in a single-label, similarity-based classifier — the safety signal gets
diluted rather than surfaced.

No threshold adjustment fixes this: the query's *dominant* topic genuinely is
ordering-related by word count and retrieval similarity. The problem is
structural — the pipeline classifies the inquiry as a whole rather than
detecting that it contains multiple distinct intents, one of which is
safety-critical regardless of how small a fraction of the message it is.

**Documented in README as a known limitation, not fixed in this submission.**
Proposed direction: a lightweight safety-keyword pre-check ahead of
classification (independent of the main taxonomy), or intent splitting with
priority = max across all detected intents rather than a single dominant
category.