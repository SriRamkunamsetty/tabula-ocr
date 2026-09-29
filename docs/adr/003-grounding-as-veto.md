# ADR 3 — Grounding is a veto over the returned value, not one term in a weighted score

## Status
Accepted (supersedes an earlier weighted-sum design within the same
development cycle — see "History" below, kept because the failure it
documents is the exact failure mode this service exists to prevent)

## Context
Confidence for a field is built from three signals: cross-pass agreement,
grounding against the page's reference text, and format validity. The first
implementation combined them as a weighted sum —
`0.5·agreement + 0.35·grounding + 0.15·format` — and abstained below 0.62.

`tests/test_tabula.py::TestAntiHallucination::test_unanimous_hallucination_is_still_refused_by_grounding`
scripted two decoding passes that agreed with each other on a purchase-order
number that does not appear anywhere on the invoice. Under the weighted-sum
design this scored `0.5·1.0 + 0.35·0.0 + 0.15·1.0 = 0.65` — agreement and a
well-formed string carried it over the 0.62 threshold with **zero**
grounding, and the fabricated value was returned to the caller as if it were
a read value.

## Decision
Grounding is now a hard precondition, checked before confidence is computed:
a value that is not locatable in the reference text (`ground_value` below
`grounding_min_ratio`) is abstained on outright, regardless of how strongly
the passes agreed or how well-formed the string is. Among values that *do*
clear the grounding bar, agreement and format validity combine into the
confidence score as before, which is what a reviewer's triage queue sorts on.

The reasoning: two decoding passes agreeing with each other is not two
independent pieces of evidence about the document. Sampling temperature
changes which tokens a model reaches for; it does not change what the model
was trained to associate with a field named `purchase_order` on an invoice
that plausibly has one. Correlated hallucinations are exactly the case where
agreement is least trustworthy, which is the opposite of what the weighted
sum assumed.

When no reference text is available at all (a caller extracting from a
context where grounding cannot be checked), the veto is skipped rather than
forcing every field to abstain, and that fact is visible in the field's
`Provenance.grounded = False` rather than silently treated as "grounded."

## Consequences
- **Positive.** Closes the exact failure class this service is built to
  catch. `test_unanimous_hallucination_is_still_refused_by_grounding` pins
  the behaviour so a future refactor cannot silently reopen it.
- **Positive.** The confidence number stays interpretable: it is scaled down
  by the grounding score even on the abstained path
  (`consensus.py::score_field`), so a review queue sorted by confidence never
  ranks an unsupported value above a merely ambiguous one.
- **Negative.** A value that is genuinely correct but phrased differently
  from anything in the reference text (e.g. the model correctly computes a
  total that is never printed verbatim) is now abstained on rather than
  returned. This is an intentional trade: the brief is document *extraction*,
  not computation, so a value that cannot be pointed to on the page is out of
  scope for this service by design, not a bug to be tuned away.

## History
The weighted-sum version is worth recording rather than deleting from
history, because the bug it contained is the textbook version of the problem
this whole mini-challenge is about: a plausible-looking aggregate score that
lets a fabricated value through. Catching it in our own test suite before it
reached a demo is the result the anti-hallucination design is supposed to
produce.
