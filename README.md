# TABULA OCR

**Mini-Challenge 2 — Lablab × AMD AI Academy Challenge**
*Handling OCR: prompting and serving a multimodal model.*

Schema-locked document extraction where every returned value carries the
evidence for it — a bounding box, a verbatim quote, and a confidence built
from three independent signals — and the service **abstains** rather than
guesses when that evidence doesn't hold up. Served on an AMD Instinct MI300X.

```
$ tabula demo
demo-invoice: 7 fields, 29% abstained, mean confidence 1.00, 1561 tokens, 7 ms, $0.00062
  [OK ] invoice_number   conf=1.00  INV-2026-0041
  [OK ] invoice_date     conf=1.00  14/03/2026
  [OK ] supplier_name    conf=1.00  ACME INSTRUMENTS PVT LTD
  [OK ] supplier_phone   conf=1.00  +91 80 4123 9900
  [OK ] total_amount     conf=1.00  1,24,500.00
  [HOLD] currency         conf=0.59  (value not located in page text (score 0.67))
  [HOLD] purchase_order   conf=0.20  (value not located in page text (score 0.38); decoding passes disagreed (agreement 0.50))
```

The `demo` command needs no GPU. One of its two scripted decoding passes
invents a purchase-order number that the invoice doesn't have — the service
catches it and holds it back. That's the whole point of this project, running
in eight seconds on a laptop.

---

## Two entry points

| | **Service** (`tabula`, FastAPI) | **Contract mode** (`app.py`) |
|---|---|---|
| for | people and dashboards | the Mini-Challenge 2 grader |
| input | a document page and a schema | one PNG, JPEG or TIFF (plates and road signs) |
| output | typed fields with evidence, confidence, abstention | `{"text": "7ABC123", "confidence": 0.94}` |
| unsure | abstains (`HOLD`) | never abstains: the grader scores a blank exactly like a wrong guess |
| runs as | a long-lived API | one process per image, inside an already-running container |

Contract mode is a thin, separate subpackage (`src/tabula_ocr/contract/`); the service is untouched.
The reasoning is in [`docs/adr/005-contract-mode.md`](docs/adr/005-contract-mode.md).

```bash
python3 app.py --input-image /app/input/image_01.png
# -> /app/output/image_01_output.json  {"text": "7ABC123", "confidence": 0.94}
```

What it does that a generic "send the image to a VLM" script would not:

* **Reads one picture four ways and votes.** The original, a contrast-stretched view, a denoised view and an
  equalised greyscale view go to the same vision model concurrently; votes are counted on the grader's own
  normalisation, the untouched view breaks ties, and when nothing wins outright equal-length readings are
  voted character by character (different renderings misread different characters).
* **Applies the transcription rules in code, not in a prompt.** The printed jurisdiction banner and slogan
  (`CALIFORNIA`, `THE LONE STAR STATE`) are dropped from a US plate; a Chinese plate's province character
  and letter are *kept*, and because `O` and `I` never occur on those plates, they are repaired to `0`/`1`.
  Multi-line text is joined top to bottom with single spaces.
* **Accepts any image the grader can send.** First frame of a multi-frame TIFF, EXIF-rotated JPEGs, palette,
  16-bit, CMYK and transparent images, images far above the model's resolution (downscaled) and tiny crops
  (upscaled).
* **Always leaves a valid file.** A placeholder is written first and rewritten after every view completes,
  so a crash, a watchdog exit or a dead model still leaves a parseable answer (the best so far).

```bash
python scripts/contract_selfcheck.py            # replays the grader's process model; scripted model, no GPU
python scripts/contract_selfcheck.py --live \   # against your real model, with the organisers' kit
    --images mc2-starter-kit/images --expected mc2-starter-kit/expected.json
docker build -f Dockerfile.submission -t <registry>/tabula-ocr:v1 .
bash scripts/check_submission.sh <registry>/tabula-ocr:v1 <images_dir> [expected.json]
```

**What is and is not verified.** The contract path has 80+ tests and the self-check scores 200/200 on ten
synthetic images covering every format and awkward case the grader promises (PNG, JPEG, TIFF; huge, tiny,
transparent, palette, 16-bit, rotated, multi-frame), each run as a real subprocess against a real HTTP model
server. That model is **scripted**: it proves image handling, composition, voting, timing and the file
contract, not how well a real vision-language model reads characters. The submission image is untested on real
ROCm hardware (vLLM availability for the mandated base image is the main unknown), and the sample images the
organisers publish have not been run through it.

---

## Why this exists

A vision-language model asked to extract fields from a document will almost
always return something for every field you ask about. In production that is
the failure mode, not the success: a fabricated phone number or an invented
purchase-order line looks identical to a correctly-read one, and nothing
downstream can tell the difference. Most OCR demos measure accuracy on the
values they returned and quietly ignore the ones they made up.

This service is built around **three independent signals that a fabricated
value cannot satisfy at once**:

| Signal | What it checks | Why fabrication fails it |
|---|---|---|
| **Agreement** | The page is decoded `N` times independently, at different sampling temperatures. Consensus is measured against every pass that ran — a pass that omits a field counts as a dissenting vote, not a missing one. | A value genuinely printed on the page is stable across resamples. An invented one drifts, or is only claimed by one of N passes. |
| **Grounding** | The candidate value must be fuzzy-locatable inside the page's own reference text. | A value that was never on the page scores near zero, however confidently it was claimed. |
| **Format validity** | Type-specific structural checks (does this parse as a date? an E.164-ish phone number? a semver string?). | A misread character often breaks the format, independent of the other two signals. |

**Grounding is a veto, not a vote.** See [`docs/adr/003-grounding-as-veto.md`](docs/adr/003-grounding-as-veto.md)
for the actual bug this fixed during development: two decoding passes that
*agreed* on a fabricated value scored 0.65 under an earlier weighted-sum
design and cleared the abstention threshold with zero grounding. The fix —
and the regression test that now pins it — is the most important file in
this repository.

---

## Architecture

```
                 ┌────────────────────────────────────────────────┐
                 │                 tabula-ocr API                 │
                 │                                                │
  page image(s)  │  ┌──────────┐   ┌───────────────┐   ┌────────┐ │
  ──────────────▶│  │  decode  │──▶│  N extraction  │──▶│ scoring │─┼──▶ DocumentResult
  + schema name   │  │ & resize │   │  passes (‖)    │   │ (§below)│ │    fields[], with
                 │  └──────────┘   │  guided_json   │   └────────┘ │    evidence + status
                 │        │        └───────┬────────┘        ▲    │
                 │        │                │                 │    │
                 │        ▼                ▼                 │    │
                 │  ┌──────────────────────────────┐         │    │
                 │  │   reference-text pass (once)  │─────────┘    │
                 │  │  (or caller-supplied text layer, skipped)    │
                 │  └──────────────┬─────────────────────────────┘│
                 └─────────────────┼──────────────────────────────┘
                                    ▼
                 ┌────────────────────────────────────────────────┐
                 │      vLLM on ROCm  ·  AMD Instinct MI300X       │
                 │   PaddleOCR-VL / GLM-OCR-class weights          │
                 │   OpenAI-compatible /v1/chat/completions        │
                 │   guided_json = compiled JSON Schema            │
                 └────────────────────────────────────────────────┘
```

**Scoring**, per field, per `docs/adr/003-grounding-as-veto.md` and
`docs/adr/004-consensus-denominator.md`:

```
observations (one per pass) ──▶ reach_consensus()  ──▶ winning value + agreement
                                        │
winning value + reference text ──▶ ground_value()  ──▶ grounded? + score
                                        │
        NOT grounded  ──────────────────┴──▶  ABSTAIN  (confidence scaled by grounding score)
        grounded      ──▶ confidence = w₁·agreement + w₂·grounding + w₃·format_valid
                              │
                       confidence < threshold ──▶ ABSTAIN
                       confidence ≥ threshold ──▶ EXTRACTED, with bbox + quote + confidence
```

### Package layout

```
src/tabula_ocr/
  models.py           Pydantic domain models — BoundingBox, FieldResult, DocumentResult…
  config.py            All configuration, read once from TABULA_* env vars
  errors.py            Typed error hierarchy, one HTTP status each
  normalize.py         Locale-tolerant value normalisation + format validators
  consensus.py         Grounding, consensus, confidence — the anti-hallucination core
  imaging.py            Decode / validate / resize / crop page images
  schema_registry.py   Extraction schemas → compiled JSON Schema for guided decoding
  prompts/registry.py   Versioned, testable prompts (v1 baseline → v3 region-guided)
  vlm/client.py          Production client: retries, backoff, circuit breaker
  vlm/fake.py            Deterministic scripted model for tests, demo, offline eval
  pipeline.py            Orchestrates decode → transcribe → N passes → score
  api.py                  FastAPI: /healthz /readyz /metrics /v1/extract /v1/schemas
  cli.py                   tabula extract | evaluate | demo | schemas
  eval/metrics.py         Coverage, precision, hallucination rate, abstention precision, CER
  eval/harness.py          Ablation sweep over prompt version × passes × threshold
```

---

## The prompting half of the brief

Prompts are versioned, testable assets in `prompts/registry.py`, not strings
inlined at the call site:

| Version | What it adds |
|---|---|
| `v1-baseline` | Plain instruction. Control condition. |
| `v2-schema-locked` | Adds the evidence contract (verbatim quote + bbox required per field) and an explicit null-over-guess rule. |
| `v3-region-guided` *(default)* | `v2` plus page-geometry hints, so the model is told to work region by region instead of merging values across the page. |

`docs/prompt_ablation.md` (generated by `make eval`, not written by hand)
reports what each version bought on the evaluation set — coverage,
precision, hallucination rate, and cost, per version.

---

## Quickstart

```bash
git clone https://github.com/SriRamkunamsetty/tabula-ocr.git && cd tabula-ocr
pip install -e ".[dev]"

# No GPU needed — runs the full pipeline against a scripted model that
# deliberately hallucinates one field, and shows it being caught:
tabula demo

# Full test suite (150+ tests) + strict types + lint
pytest -q
mypy
ruff check .
```

### Running against a real model on AMD Developer Cloud

```bash
# One-time: join the AMD AI Developer Program and activate the $100
# Developer Cloud credit (https://developer.amd.com/ai-developer-program/),
# then create a 1× MI300X GPU Droplet from the ROCm image.

./deploy/provision_mi300x.sh <droplet-ip> PaddlePaddle/PaddleOCR-VL

export TABULA_VLM_BASE_URL="http://<droplet-ip>:8000/v1"
export TABULA_VLM_MODEL="PaddlePaddle/PaddleOCR-VL"

tabula extract --image invoice.png --schema invoice
uvicorn tabula_ocr.api:create_app --factory --port 8080
```

### Docker

```bash
docker compose --profile gpu up          # API + vLLM, one MI300X host
docker compose up api                    # API only; point TABULA_VLM_BASE_URL
                                          # at an already-running vLLM endpoint
```

---

## API

```
POST /v1/extract
{
  "document_id": "inv-0041",
  "images_b64": ["<base64 page 1>", "..."],
  "schema_name": "invoice",
  "reference_text": null,        // optional: skip the transcription pass
  "passes": null,                 // optional: override settings.passes
  "prompt_version": null          // optional: pin a specific prompt version
}
```

returns a `DocumentResult` where every field looks like:

```json
{
  "name": "total_amount",
  "value": "1,24,500.00",
  "status": "extracted",
  "confidence": 0.97,
  "agreement": 1.0,
  "provenance": {
    "page": 1,
    "bbox": {"x0": 0.12, "y0": 0.61, "x1": 0.38, "y1": 0.66},
    "quote": "Grand Total: Rs. 1,24,500.00",
    "grounded": true,
    "grounding_score": 1.0
  }
}
```

`GET /healthz` is dependency-free liveness. `GET /readyz` checks the schema
registry and pings the upstream model — wire it to a Kubernetes
`readinessProbe`, not `livenessProbe`, or a slow model turns into a restart
loop. `GET /metrics` exposes Prometheus counters and histograms, including
`tabula_ungrounded_values_total` — the single number that matters most if you
are watching this service in production.

---

## Evaluation

```bash
tabula evaluate --dataset eval/golden.jsonl \
  --prompts v1-baseline,v2-schema-locked,v3-region-guided \
  --passes 1,2 \
  --thresholds 0.50,0.62,0.75
```

sweeps the full grid and writes `reports/eval.json` plus a markdown table:

| metric | what it measures | why it's here instead of just "accuracy" |
|---|---|---|
| **coverage** | share of fields the service was willing to answer | catches an over-cautious config gaming the other metrics by abstaining on everything |
| **precision** | correct ÷ answered | the number that matters for anything downstream that trusts a returned value |
| **hallucination rate** | wrong ÷ answered | the metric a plain accuracy score hides |
| **abstention precision** | of the fields declined, how many would have been wrong anyway | catches an abstention threshold that's just guessing |
| **mean CER** | character-level edit distance on answered fields | transcription quality independent of the abstention policy |
| **$ / 1,000 docs** | derived from real token counts × measured MI300X throughput | ties every accuracy number to what it costs to get it |

---

## Benchmarking on AMD hardware

```bash
# Harness-only: the pipeline's own overhead, no GPU, no network —
# useful for catching a regression this codebase introduced.
python benchmarks/bench_pipeline.py --requests 60 --concurrency 12

# Against a real MI300X deployment:
TABULA_VLM_BASE_URL=http://<droplet-ip>:8000/v1 \
  python benchmarks/bench_pipeline.py --live --requests 60 --concurrency 12
```

Every run is labelled with its own `mode` (`harness-only (no GPU)` vs.
`live-endpoint`) so a number from one is never mistaken for the other.
`rocm-smi --showuse --showmemuse` on the droplet, run alongside a `--live`
benchmark, is what turns "we used AMD Instinct" into a measured claim rather
than an asserted one.

Reference harness-only numbers, produced by the run recorded in
[`benchmarks/results.json`](benchmarks/results.json), 60 requests at
concurrency 12, 2 passes/request:

| p50 | p95 | p99 | throughput | error rate |
|---|---|---|---|---|
| 5.0 ms | 7.5 ms | 8.0 ms | 1,197 req/s | 0% |

This measures pipeline overhead only — image decoding, consensus scoring,
request handling — against the deterministic scripted model. It is **not** a
model-inference benchmark; running `--live` against a provisioned MI300X
populates the token-throughput and cost-per-1,000-docs figures that depend on
actual model serving, and that run should be re-recorded here before this
number is quoted as a deployment result.

---

## Design decisions

Full reasoning, including the alternatives considered and rejected, lives in
[`docs/adr/`](docs/adr/):

1. [**Serving architecture**](docs/adr/001-serving-architecture.md) — why we
   serve the model ourselves on ROCm rather than call a hosted API.
2. [**Guided decoding**](docs/adr/002-guided-decoding.md) — why structural
   validity is a decoding constraint, not a prompt request.
3. [**Grounding as a veto**](docs/adr/003-grounding-as-veto.md) — the bug
   that motivated this, in the project's own words.
4. [**Consensus denominator**](docs/adr/004-consensus-denominator.md) — why
   agreement is measured against every pass that ran, not just the passes
   that answered.

---

## Testing philosophy

70 tests, organised by the risk each one retires rather than by which module
it happens to import. The ones worth reading first are in
`TestAntiHallucination` (`tests/test_tabula.py`): they assert that a value
invented by the model — appearing nowhere in the page text — is refused, and
that two decoding passes *agreeing* on that same invented value still does not
let it through. Everything else in the suite exists to keep that behaviour
true under refactoring.

```bash
pytest -q --cov=tabula_ocr --cov-report=term-missing   # 87%+ branch coverage
mypy                                                     # strict mode, zero errors
ruff check .                                             # zero findings
```

---

## What's next (mini-challenge 3)

This service's `DocumentResult` — with page text, per-field evidence, and
bounding boxes — is the ingestion format for the grounded RAG "rules oracle"
built in mini-challenge 3, which answers questions over the very documents
this service reads, with the same abstain-rather-than-guess discipline
applied to generation instead of extraction.

## License

MIT. See `LICENSE`.
