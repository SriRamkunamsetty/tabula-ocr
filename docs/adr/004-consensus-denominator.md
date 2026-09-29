# ADR 4 — Agreement is measured against every pass that ran, not against passes that answered

## Status
Accepted

## Context
`reach_consensus` groups each decoding pass's normalised value into buckets
and reports the winning bucket's share as the agreement score. The first
version divided by `len(observations)` — the number of passes that actually
produced a value for the field. A field that only one of two passes
attempted (the other reported it absent) therefore scored 1.0 agreement,
identical to a field both passes read identically.

That is backwards for exactly the fields most at risk of fabrication: a value
one pass invents and the other correctly omits should score as **weak**
agreement, not perfect agreement with a sample size of one.

## Decision
`reach_consensus` and `score_field` both take the true pass count
(`total_passes`, threaded from `OCRService.extract`, which knows exactly how
many passes completed successfully) and use it as the agreement denominator.
A pass that reports a field absent is counted as a dissenting vote, not
dropped from the denominator. The winning bucket's share of *all* completed
passes is what "agreement" means throughout the codebase.

## Consequences
- **Positive.** A field only one of N passes claims to have read now scores
  agreement `1/N`, correctly reading as weak evidence rather than perfect
  consensus — before the grounding veto in ADR 3 even applies.
- **Positive.** `usage.passes` (the count of passes that returned parseable
  output) is the single source of truth for the denominator; a pass that
  failed outright (timeout, malformed JSON) is excluded from both the
  numerator and denominator consistently, rather than one and not the other.
- **Neutral.** Raising `settings.passes` without a corresponding increase in
  how often the model is actually right will now visibly lower agreement
  scores across the board — which is the correct response to adding noisier
  votes, and is a config change worth re-running the evaluation ablation
  (`tabula evaluate`) after making.
