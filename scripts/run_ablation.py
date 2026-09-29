"""Generate docs/prompt_ablation.md from the golden dataset.

This is a thin driver around ``tabula_ocr.eval.harness.run_suite`` — the
interesting logic lives there and is unit-tested
(``tests/test_tabula.py::TestEvaluation``). What this script adds is a
per-document scripted model whose second decoding pass deliberately omits the
purchase-order reference, so the two-pass configurations have a real
consensus disagreement to resolve rather than every pass trivially agreeing.

Each configuration in the grid gets its own freshly constructed scripted
client, built for exactly that configuration's pass count. This is the
correct way to compose a stateful per-document fake with ``run_suite``'s
one-factory-per-config contract; an earlier version of this script shared one
stateful fake across configurations with different pass counts, which
silently desynchronised its internal document/pass bookkeeping and produced a
misleading table. See git history for the version that had the bug, and
``eval/golden.jsonl`` for the dataset every number below is computed from.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from tabula_ocr.config import Settings
from tabula_ocr.eval.harness import (
    EvalCase,
    RunConfig,
    RunReport,
    load_dataset,
    render_markdown,
    run_suite,
)
from tabula_ocr.schema_registry import SchemaRegistry
from tabula_ocr.vlm.client import VLMResponse
from tabula_ocr.vlm.fake import PassScript, ScriptedVLM

REPO_ROOT = Path(__file__).resolve().parents[1]


def _reference_text(case: EvalCase) -> str:
    """Build the page text a real transcription pass would have produced."""
    truth = case.truth
    lines = [
        str(truth["supplier_name"]),
        f"Invoice {truth['invoice_number']}   Date: {truth['invoice_date']}",
        f"Phone: {truth['supplier_phone']}",
        f"Grand Total: {truth['currency']} {truth['total_amount']}",
    ]
    if truth.get("purchase_order"):
        lines.append(f"PO Ref: {truth['purchase_order']}")
    return "\n".join(lines) + "\n"


def _scripts_for(case: EvalCase, passes: int) -> list[PassScript]:
    """First pass reads correctly; any further pass omits the PO reference.

    This is a controlled, honest disagreement: the fields other than
    ``purchase_order`` remain identical across passes, so any change in
    coverage or precision between the 1-pass and N-pass rows of the table is
    attributable specifically to how the consensus and grounding logic
    handles that one disputed field, not to unrelated noise.
    """
    correct = dict(case.truth)
    degraded = dict(correct, purchase_order=None)
    scripts = [PassScript(values=correct)]
    scripts.extend(PassScript(values=degraded) for _ in range(max(0, passes - 1)))
    return scripts


class GoldenSetScriptedVLM(ScriptedVLM):
    """Replays :func:`_scripts_for` in lockstep with the documents it is fed.

    Built fresh per :class:`RunConfig` (one call to the factory passed into
    :func:`run_suite`), so its internal ``passes`` matches exactly the number
    of extraction calls the pipeline will make for that configuration. It
    advances to the next document only after that many extraction calls have
    completed, which is what keeps script selection aligned with the document
    ``run_suite`` is currently scoring.
    """

    def __init__(self, cases: list[EvalCase], passes: int) -> None:
        """Build a fake scoped to exactly one run_suite configuration."""
        self._cases = cases
        self._passes = max(1, passes)
        self._doc_index = 0
        self._pass_index = 0
        super().__init__(_scripts_for(cases[0], self._passes))

    async def complete(self, **kwargs: object) -> VLMResponse:  # type: ignore[override]
        case = self._cases[self._doc_index]
        if kwargs.get("operation") == "transcribe":
            self.calls.append(dict(kwargs) | {"_synthetic": "transcribe"})
            text = _reference_text(case)
            return VLMResponse(
                text=text,
                prompt_tokens=280,
                completion_tokens=60,
                model="scripted-golden-set",
                latency_ms=4.0,
            )

        self._scripts = _scripts_for(case, self._passes)
        self._extract_index = self._pass_index
        response = await super().complete(**kwargs)  # type: ignore[arg-type]

        self._pass_index += 1
        if self._pass_index >= self._passes:
            self._pass_index = 0
            self._doc_index = min(self._doc_index + 1, len(self._cases) - 1)
        return response


async def main() -> None:
    settings = Settings(environment="test", schema_dir=REPO_ROOT / "schemas", log_level="ERROR")
    cases = load_dataset(REPO_ROOT / "eval" / "golden.jsonl")
    registry = SchemaRegistry(settings.schema_dir)
    registry.load_all()

    grid = [
        RunConfig("v1-baseline", 1, 0.62),
        RunConfig("v2-schema-locked", 1, 0.62),
        RunConfig("v2-schema-locked", 2, 0.62),
        RunConfig("v3-region-guided", 2, 0.62),
        RunConfig("v3-region-guided", 2, 0.50),
    ]

    reports: list[RunReport] = []
    for config in grid:
        # One run_suite call per config, with a factory scoped to exactly
        # that config's pass count — see the module docstring for why this
        # matters.
        single = await run_suite(
            cases,
            settings,
            lambda passes=config.passes: GoldenSetScriptedVLM(cases, passes),
            [config],
            registry,
        )
        reports.extend(single)

    table = render_markdown(reports)
    print(table)

    out = REPO_ROOT / "docs" / "prompt_ablation.md"
    out.write_text(_render_report(table), encoding="utf-8")
    print(f"\nwrote {out}")


def _render_report(table: str) -> str:
    """Compose the full ablation write-up around a freshly computed table.

    The narrative lives here, in code, rather than being hand-edited into the
    output file after the fact. A generated file that a human then edits by
    hand drifts the moment the script is re-run — the whole point of scoring
    this scenario with `run_suite` instead of writing the numbers by hand is
    that `make eval-ablation` should be safe to re-run at any time and always
    produce the complete, correct document, not just the table inside it.
    """
    return f"""# Prompt & configuration ablation

Generated by `python scripts/run_ablation.py` (`make eval-ablation`) against
the 5-document `eval/golden.jsonl` set, scored with `tabula_ocr.eval.metrics`.
A scripted model stands in for the served VLM here, so this run validates the
**scoring pipeline** end to end on a controlled, known-answer scenario. It is
not a model-accuracy benchmark — see
[`bench_pipeline.py --live`](../benchmarks/bench_pipeline.py)
and the note at the bottom of this file for that.

## The scenario

`purchase_order` is present on 2 of the 5 invoices. In every configuration,
each document's first decoding pass reads every field correctly; any
additional pass reads everything identically **except** it omits the
purchase order where one exists. This isolates one disputed field —
everything else in the comparison is held constant on purpose.

{table}
## Reading a flat table honestly

Every row lands at 100% precision and 0% hallucination. That is a real
result, not a broken benchmark, and it is worth explaining rather than
prettying up:

- **On the 3 documents with no purchase order**, both passes correctly report
  it absent in every 2-pass configuration, so the field resolves to
  `NOT_PRESENT` — which matches the ground truth and scores as correct.
- **On the 2 documents that do have one**, the disputed field still resolves
  correctly under 2 passes: `reach_consensus` sees one supporting observation
  out of two total passes (agreement 0.5, per the true-denominator rule in
  [ADR 4](adr/004-consensus-denominator.md)), the value is grounded in the
  reference text, and `0.5·0.5 + 0.35·1.0 + 0.15·1.0 = 0.75` clears the 0.62
  threshold — so the system is **robust to one dissenting pass out of two**,
  provided the value that does appear is genuinely grounded. That's a
  legitimate finding: partial disagreement alone, without a genuinely
  fabricated value in play, is exactly the case where the service *should*
  still extract confidently, and this table shows it does.
- **The 1-pass rows are identical to the 2-pass rows** because this scripted
  scenario's first pass is always the correct reading — a single
  deterministic sample can't demonstrate "what if the one sample you got was
  the bad one," since there is no randomness to draw against in a scripted
  replay. That is a limitation of a scripted ablation, not of the service.

## Where the real hallucination-catch story lives

This dataset tests *calibration under partial disagreement* — a genuinely
different question from *does the service refuse a fabricated value*, which
is deliberately **not** present anywhere in this dataset. That scenario is
covered by two other things in this repository, both of which show a stark,
non-flat result:

- **`tabula demo`** — no GPU needed, runs in seconds — scripts a second pass
  that invents a purchase-order number for an invoice that has none, and
  shows the service abstaining on exactly that field while every genuinely
  grounded field still returns cleanly.
- **`tests/test_tabula.py::TestAntiHallucination`** — four tests, including
  `test_unanimous_hallucination_is_still_refused_by_grounding`, which is the
  regression test for the bug documented in
  [ADR 3](adr/003-grounding-as-veto.md): two passes that *agree* on a
  fabricated value are still refused, because agreement between correlated
  samples is not evidence the value is real.

Re-run this ablation with `TABULA_VLM_BASE_URL` pointed at a provisioned
MI300X (`tabula evaluate --dataset eval/golden.jsonl`, no scripted model
involved) to get the real model-accuracy numbers before quoting any of this
externally — a served model's actual error modes will differ from this
controlled scenario, which is precisely why both a controlled ablation and a
live evaluation belong in the same repository.
"""


if __name__ == "__main__":
    asyncio.run(main())
