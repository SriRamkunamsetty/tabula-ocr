# ADR 2 — Structural validity is a decoding constraint, not a prompt request

## Status
Accepted

## Context
Early prototyping asked the model for JSON via instruction alone ("return
your answer as JSON"). It usually worked. "Usually" is not a property a
production pipeline can build on: a stray sentence of preamble, a trailing
comma, or a wrapped code fence turns "usually" into a 2 a.m. parsing failure,
and the failure mode is silent — the caller gets *a* string back, and only
discovers it wasn't valid JSON when something downstream breaks.

## Decision
Every extraction schema compiles to a JSON Schema
(`ExtractionSchema.to_json_schema`) and is passed to vLLM as `guided_json`.
The serving engine's grammar-constrained decoder (xgrammar) restricts the
token sampler to sequences that parse against that schema, so an
invalid-JSON response becomes structurally impossible rather than merely
unlikely. `VLMResponse.as_json()` still defends the boundary — a fenced code
block is stripped, and anything that still doesn't parse raises
`VLMProtocolError` with the first 200 characters attached — because a
serving engine misconfigured to ignore the constraint should fail loudly, not
be trusted silently.

The schema goes further than "give me JSON": every field is compiled as an
object requiring `value`, `quote`, and `bbox` as sibling required keys. This
is what forces the evidence contract structurally — a model literally cannot
return a bare value without the schema rejecting the completion, so quote and
bounding-box evidence is not something the model volunteers when it feels
like it.

## Consequences
- **Positive.** Parsing failures are eliminated as a source of pipeline
  errors, converting an entire class of production incident into "did not
  happen."
- **Positive.** The evidence contract (quote + bbox) is enforced by the
  decoder, which is what makes provenance in `Provenance` reliable enough to
  build the confidence score on.
- **Negative.** Grammar-constrained decoding has a real throughput cost
  versus unconstrained generation; `benchmarks/bench_pipeline.py --live`
  against the MI300X deployment is how that cost gets measured rather than
  assumed.
- **Negative.** Couples the serving choice to an engine that supports guided
  decoding (vLLM and SGLang both do; a bare `transformers` pipeline does
  not). Acceptable, since ADR 1 already commits to vLLM.

## Alternatives considered
- **Prompt-only JSON + post-hoc repair (e.g. `json_repair`).** Rejected:
  repair heuristics can silently "fix" a truncated or malformed response into
  something that parses but no longer reflects what the model actually said,
  which is worse than a loud failure.
- **Function/tool calling.** A reasonable alternative on engines that support
  it well; not chosen because guided JSON gives finer-grained control over
  the per-field evidence contract described above.
