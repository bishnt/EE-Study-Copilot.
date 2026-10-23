# EE-Study-Copilot.

A retrieval-augmented study assistant for Year III, Part I electrical engineering
(DSP, Digital Control, Switchgear & Protection, Machine Design, Thermodynamics, Power Electronics).
It answers **only** from approved, shareable class material, cites `[file, page]` for every claim,
and says `I could not find this in the documents.` when the material has no evidence.
Rules and targets live in [scope.md](scope.md).

## Pipeline

```
PDFs ─ ocr ─ extract ─ clean ─ chunk ─ embed ─► Chroma (dense) + chunks.jsonl (BM25)     [ingestion/]
                                                         │
question ─► scope gate ─► subject router ─► hybrid search (BM25 + dense, RRF) ─► evidence gate
                                                         │                         [app/, retrieval/]
              refusal ◄─ citation validator ◄─ LLM (context-only prompt) ◄────────┘  [generation/]
```

* **Scope gate** (`app/scope.py`): refuses ENEE 301, exam prediction, assignments, real equipment settings.
* **Hybrid retrieval**: BM25 (exact terms like "Nyquist") fused with bge embeddings by reciprocal rank fusion,
  optional subject filter, optional cross-encoder reranker (`retrieval/rerank.py`).
* **Evidence gate**: refuses before calling the model when neither retriever finds a close match
  (`min_dense_sim`, `min_bm25` in `app/config.py`).
* **Citation validator**: every claim needs a `[file, page]` that exists in the retrieved context.
  One repair attempt, then the answer is replaced by the refusal sentence. Nothing uncited is shown.
* **Shareable-only**: chunks carry `shareable`; anything not `yes` is dropped before the API call.
  Indexes built before this flag exist are rejected, rebuild them (below).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # add ANTHROPIC_API_KEY
```

Put PDFs in `corpus/<subject>/` with `corpus/manifest.csv`, then:

```bash
python -m ingestion.validate_corpus
python -m ingestion.build_class          # class index: shareable=yes only
python -m app.cli "state the ROC of a right-sided sequence" --sources
python -m app.cli                        # interactive
```

## Visual retrieval (ColPali, optional)

Text extraction misses what is *drawn*: circuit diagrams, waveforms, root-locus plots, tables that
come out as scrambled text, handwriting. [ColPali](https://github.com/illuin-tech/colpali) embeds
each PDF **page as an image** (~1000 patch vectors) and scores a question against it with
late-interaction MaxSim, so those pages become searchable. It is **off by default**; nothing
above changes until you turn it on.

```bash
pip install -r requirements-visual.txt            # adds colpali-engine (+ torch, transformers>=5.3)
python -m ingestion.build_visual --class          # embed shareable=yes pages (GPU strongly advised)
COPILOT_VISUAL=1 python -m app.cli "sketch the root locus of ..." --sources     # or: --visual
```

How it fits the pipeline:

```
page images ─ ColPali ─► vectordb_visual_class/ (emb.npy + pages.jsonl)            [ingestion/build_visual.py]
question ─► ... BM25 + dense + ColPali pages ─► RRF ─► shareable filter ─► evidence gate
                                       │
           hit on a page that also has a text chunk → boosts that chunk (no duplicate)
           hit on an image-only page               → its own hit, cited as [file, page]
                                       ▼
        pages that pass `min_visual` are re-rendered from the PDF and sent to Claude with the text
```

* **Same safety rules.** Page images obey `shareable`: non-shareable pages are filtered before
  anything is rendered or sent. Citations still need a `[file, page]` that is in the context, and a
  page whose image cannot be rendered and has no text is dropped (nothing to cite, so no evidence).
* **Calibrate `COPILOT_MIN_VISUAL`.** The score is the mean MaxSim per query vector. `0.50` is a
  starting guess, not a measured value. Too low and weak page matches pass the evidence gate (and
  get sent as images); too high and ColPali never counts as evidence. Run `python -m eval.run_eval`
  and pick a value between your answerable and unanswerable questions, as for `COPILOT_MIN_DENSE`.
* **Cost.** Embedding is a one-off per document (shards in `<index>/.shards`, resumable, skipped on
  re-run). ColPali-v1.3 is a ~3B model: use a GPU. `COPILOT_VISUAL_MODEL` also accepts the ColQwen2,
  ColQwen2.5 and ColIdefics3 (`colSmol`, smaller, for laptops) families; only `colpali-v1.3` is the
  default and the others are mapped but untested. Changing the model requires a rebuild (the app
  refuses to mix them).
  At answer time each matched page is one extra image on the API call (`COPILOT_MAX_IMAGES`, default 3).
* **Scope.** [scope.md](scope.md) lists "content that needs images, handwriting or scanned pages" as
  out of scope for Part I. This feature is what would lift that, but scope changes need a version bump and an
  eval re-run, so `scope.md` is left as is; update it when you adopt visual retrieval.
* Known limits: the optional cross-encoder reranker reads text only (leave it off with visual on),
  and the eval's faithfulness judge sees retrieved text, not images, so it can under-score answers
  that rely on a figure.

## Evaluate

Write questions to `eval/source/questions.jsonl` (git-ignored, format in `eval/questions.example.jsonl`;
list **all** relevant pages in `gold` so P@5 is meaningful), then:

```bash
python -m eval.run_eval --strict         # R1 citations, R2 refusal, R3 P@5, R4 faithfulness
```

The report (`eval/report.json`) includes the top similarity per question so you can calibrate
`COPILOT_MIN_DENSE` / `COPILOT_MIN_BM25` (answerable questions above, refusals below).

## Tests

```bash
python -m pytest -q        # offline: fake embeddings and fake LLM, no downloads, no API key
```
