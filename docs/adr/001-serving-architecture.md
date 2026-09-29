# ADR 1 — Serve the vision model ourselves on ROCm, behind an OpenAI-compatible endpoint

## Status
Accepted

## Context
Mini-challenge 2 asks for "prompting and serving a multi-modal model." Two
paths were available: call a hosted multimodal API (OpenAI, Anthropic,
Gemini), or serve open-weight OCR-specialised weights on the AMD Developer
Cloud MI300X allocation. Serving is one of the two words in the brief, so a
hosted-API-only submission does not actually answer it — and it forfeits the
$100 of MI300X credit the program provides specifically to be used.

## Decision
Serve a document-specialised vision-language model — PaddleOCR-VL or GLM-OCR
class weights, both ~0.9B parameters and both at or near the top of
OmniDocBench v1.7 as of this writing — with vLLM on ROCm, exposed as an
OpenAI-compatible `/v1/chat/completions` endpoint. The application layer
(`OpenAICompatibleVLM`) talks to that protocol and nothing more AMD-specific,
so the model behind it can be swapped by changing two environment variables.

## Consequences
- **Positive.** A 192 GB MI300X holds the OCR model, the transcription pass,
  and headroom for a reranker or a second model, on one device with no tensor
  parallelism. Guided decoding (`guided_json`) is available because we control
  the serving engine; a hosted API's structured-output support is narrower and
  not guaranteed to reject a non-conforming sample rather than silently
  degrade it.
- **Positive.** Cost is transparent: `$1.99/hr ÷ measured tokens/s`, not a
  per-token price list that changes without notice.
- **Negative.** We own operational concerns — health checks, retries, a
  circuit breaker, cold-start latency on model load — that a hosted API
  absorbs. `docker-compose.yml`'s `vlm` service and `deploy/provision_mi300x.sh`
  exist specifically to make this operationally boring.
- **Negative.** A ~0.9B specialised model is weaker at *general* visual
  reasoning than a frontier hosted model. This is the right trade for OCR
  specifically — the OmniDocBench numbers say the specialised small model
  wins on this exact task — but would not generalise to, say, visual
  question answering.

## Alternatives considered
- **Hosted multimodal API only.** Rejected: does not serve a model, forfeits
  the AMD compute allocation, and ties cost to an external price list.
  Kept as an interface-compatible fallback in principle — `VLMClient` is a
  `Protocol`, so a hosted-API implementation could be added without touching
  the pipeline — but not built, because it isn't what this challenge asks for.
- **Classical OCR (Tesseract/PaddleOCR text-only).** Rejected as the primary
  path: no free-form field understanding, no guided JSON output, and the
  brief specifically asks for a multimodal model. Remains the right fallback
  for clean machine-print where a VLM is overkill, and is a one-line swap
  behind the same `VLMClient` protocol.
