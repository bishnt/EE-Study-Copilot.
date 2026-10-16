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
