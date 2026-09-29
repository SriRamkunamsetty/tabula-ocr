# ADR 005: Contract mode, the shape the grader scores

## Status

Accepted. Added after the Mini-Challenge 2 specification was published; the service in the rest of this
repository was designed before it, from the one-line brief.

## Context

The published specification differs from the service's shape in ways that matter more than any accuracy
difference:

| | Service (ADR 1-4) | What is graded |
|---|---|---|
| input | a document page and a schema | one PNG, JPEG or TIFF (plates, road signs) |
| output | typed fields with evidence | `{"text", "confidence"}`; only `text` is graded |
| correctness | grounded and verified | equal to the expected text after normalisation |
| uncertainty | abstain | abstaining scores the same zero as a wrong answer |
| process model | long-running API | `python3 /app/app.py --input-image PATH` per image, in a running container |
| runtime | a separate vLLM server | one container on the mandated ROCm base, GPU required, 30 s per image |

Normalisation is uppercase, whitespace removed, and `- . _ ·` removed. Text printed *around* a plate number
(state banner, slogan) is not part of it; on a Chinese plate the leading province character and letter *are*.

## Decision

Keep the service; add `tabula_ocr.contract`, a second, thinner entry point:

1. **The model sees, code decides.** The model returns only `{"kind", "lines"}`: what the object is and the
   text lines it reads. Dropping banners, keeping province characters, joining lines, repairing characters a
   Chinese plate cannot contain (`O`, `I`) and voting are deterministic and unit-tested; none depends on
   prompt wording.
2. **Four views, one model, one vote.** Original, contrast-stretched, denoised and equalised renderings are
   read concurrently (about one model latency in wall-clock). Votes are counted on the normalised form; the
   original view weighs 1.5 and breaks ties; without a clear majority, equal-length readings are voted per
   character. This trades a few seconds of the 30 s budget for robustness to glare, blur and JPEG noise,
   which a single greedy read does not have.
3. **No abstention.** The service's HOLD status has no value here: a blank is a guaranteed loss and a guess
   is a chance. `confidence` (recorded, not scored) carries the doubt instead.
4. **Always a valid file.** A placeholder is written first, rewritten after every completed view, and a
   watchdog timer exits cleanly before the hard limit. A missing, empty or corrupt image yields the
   placeholder rather than a crash.
5. **Robust decoding.** Multi-frame TIFF (first frame), EXIF orientation, palette / 16-bit / CMYK /
   transparent modes, and out-of-range sizes (long edge forced into 640-1600 px) are handled in one module.
6. **One container, model loaded once.** `serve.py` starts vLLM with a GiB budget converted to a fraction of
   the card (a naive 0.9 would exceed the 48 GiB ceiling on a 192 GB accelerator), writes a ready marker and
   never exits, even if the model server dies.

## Alternatives considered

* **A dedicated OCR engine (Tesseract, PaddleOCR) as the primary reader.** Strong on clean text, but it
  brings a second runtime into the mandated image, needs its own plate/sign handling, and does not use the
  GPU, which the grader requires. Kept as a possible cross-check, not adopted.
* **A larger model or test-time sampling.** More accuracy per view, but the time budget is 30 s including
  process start and the peak-VRAM window is 1-48 GiB. Four cheap views of one 7B model fit both.
* **Fine-tuning on plates.** The hidden set is ten images and the samples are the only reference; a fine-tune
  would be measured on almost nothing. Revisit if the organisers release more labelled data.
* **Asking the model for the final string directly.** Cheaper, but the banner and province rules then live in
  a prompt, and the failure (a state name in the answer) is silent.

## Consequences

* The contract path is exercised by 80+ tests and by `scripts/contract_selfcheck.py`, which replays the
  grader's invocation model with real subprocesses and real HTTP against a **scripted** model. It validates
  decoding, composition, voting, timing and the file contract. It does **not** measure how well a real model
  reads characters; that needs `--live` on real hardware with the organisers' sample images.
* The composition rules encode the published examples (banner dropped, province kept). A jurisdiction banner
  not in the built-in list is still handled by choosing the most plate-like line, but is less certain.
* The kind of an object is decided by vote, with a deterministic override: text matching a mainland plate
  pattern is a Chinese plate whatever the model called it.
