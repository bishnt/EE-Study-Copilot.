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
  `python -m eval.run_retrieval_eval` (below) does this for you and prints a suggested value.
* **Cost.** Embedding is a one-off per document (shards in `<index>/.shards`, resumable, skipped on
  re-run). ColPali-v1.3 is a ~3B model: use a GPU. `COPILOT_VISUAL_MODEL` also accepts the ColQwen2,
  ColQwen2.5 and ColIdefics3 (`colSmol`, smaller, for laptops) families; only `colpali-v1.3` is the
  default and the others are mapped but untested. Changing the model requires a rebuild (the app
  refuses to mix them).
  At answer time each matched page is one extra image on the API call (`COPILOT_MAX_IMAGES`, default 3).
* **Eval and fine-tuning:** see the next two sections.
* **Scope.** [scope.md](scope.md) lists "content that needs images, handwriting or scanned pages" as
  out of scope for Part I. This feature is what would lift that, but scope changes need a version bump and an
  eval re-run, so `scope.md` is left as is; update it when you adopt visual retrieval.
* Known limits: the optional cross-encoder reranker reads text only (leave it off with visual on),
  and the eval's faithfulness judge sees retrieved text, not images, so it can under-score answers
  that rely on a figure.

## Retrieval eval (no LLM, no API key)

`eval.run_eval` grades whole answers. `eval.run_retrieval_eval` isolates the retrievers, so it is
cheap, deterministic, and the right tool for comparing BM25, dense, hybrid and ColPali, for
calibrating `COPILOT_MIN_VISUAL`, and for judging a fine-tuned model. It reads the same frozen
question file (`eval/source/questions.jsonl`).

```bash
python -m eval.run_retrieval_eval                       # bm25, dense, hybrid, visual, hybrid+visual
python -m eval.run_retrieval_eval --modes hybrid,hybrid+visual --out eval/retrieval_report.json
```

Metrics are **page-level** (chunks are collapsed to unique `[file, page]` in rank order, so text
and visual retrievers are comparable): `recall@k`, `p@k`, `hit@k`, `mrr`, `ndcg@k`, overall and per
subject. `chunk_p@5` is also reported with the same definition as R3, to relate it to `run_eval`.
`should_refuse` questions feed the calibration: it prints the smallest `COPILOT_MIN_VISUAL` that
keeps at least 90% of them below the gate (R2), and how many answerable questions still clear it.

## Fine-tuning ColPali on your corpus (optional)

ColPali is already strong. Fine-tuning only pays off if the base model misses *your* material
(EE jargon, your notes' layout, handwriting). Treat it as an experiment the frozen eval decides.

```bash
pip install -r requirements-visual.txt          # training needs nothing beyond this (no `datasets`)
python -m ingestion.build_visual --class        # page index; also the source of pages for step 1
python -m eval.run_retrieval_eval --out eval/retrieval_report.json          # baseline, BEFORE training

python -m training.make_pairs --per-page 3 --out train/pairs.jsonl --dry-run   # see the plan/cost
python -m training.make_pairs --per-page 3 --out train/pairs.jsonl             # Claude writes questions
python -m training.train_colpali --pairs train/pairs.jsonl --dry-run           # validate, show plan
python -m training.train_colpali --pairs train/pairs.jsonl --out adapters/ee-colpali --epochs 2

python -m ingestion.build_visual --class --adapter adapters/ee-colpali --out vectordb_visual_ft
python -m eval.run_retrieval_eval --adapter adapters/ee-colpali --visual-dir vectordb_visual_ft \
    --out eval/retrieval_report_ft.json
python -m eval.run_retrieval_eval --compare eval/retrieval_report.json eval/retrieval_report_ft.json --strict
```

Adopt it by setting `COPILOT_VISUAL_ADAPTER=adapters/ee-colpali` and `COPILOT_VISUAL_DIR=vectordb_visual_ft`.
The index records the model *and* adapter that built it and the app refuses a mismatch, because
base and fine-tuned embeddings are not comparable.

What the training does: LoRA (r=32) on the language model's attention/MLP layers and the output
projection, vision tower frozen; in-batch contrastive loss (the library's `ColbertLoss`), where each
query's page is the positive and every other page in the batch a negative. Batches never contain
the same page twice. Each epoch it measures recall on held-out pages and keeps the best epoch.

Data rules, enforced in code (`training/data.py`, with tests):

* **Shareable only.** `make_pairs` sends nothing but `shareable=yes` pages to the API (no override),
  and training drops pairs for non-shareable documents unless you pass `--include-non-shareable`
  (the weights would then encode that content, so don't share the adapter).
* **No eval leakage.** Training queries that equal or nearly equal an eval question are dropped, and
  by default so is every pair on an eval question's gold page (a model that memorised those pages
  would flatter the eval). Training refuses to start without `eval/source/questions.jsonl` unless
  you pass `--no-eval-guard`. When you have decided to ship, you may retrain once with
  `--allow-eval-pages` for production, but then that adapter can no longer be evaluated cleanly.
* **Held-out validation is split by page**, never by query.

Things to know before you spend GPU time:

* **Batch size is your number of negatives.** `--grad-accum` does not add any. Use the largest batch
  that fits; 8 is a floor, not a recommendation. The default `--lr 2e-5` is a starting point.
* **Synthetic questions are only as good as their prompts.** Read a sample of `train/pairs.jsonl`
  before training. Generated questions are phrased like the page; real student questions are not.
* **Validation gain is not a verdict.** A small held-out set from the same documents is optimistic.
  `run_retrieval_eval --compare --strict` (fails on a >0.02 drop in recall@5 or MRR for
  `hybrid+visual`, tunable with `--tolerance`) against the frozen questions is the real test.
* The trainer warns if the best epoch does not beat the base model on validation.

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

`tests/test_training_loop.py` trains a tiny stand-in model on CPU and needs torch + peft +
colpali-engine; it is skipped automatically where they are not installed (the light CI job).
