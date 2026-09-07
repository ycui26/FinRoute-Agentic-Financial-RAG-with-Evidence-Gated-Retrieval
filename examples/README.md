# End-to-end example: synthetic filings, no downloads needed

This example lets you run the full FinRoute workflow — index build, retrieval, evidence
grading, evaluation, and (optionally) grounded generation — without downloading the
FinanceBench dataset. `make_example_data.py` generates four fictional 10-K filings
(two companies × two fiscal years, seven pages each) with internally consistent
financials, plus `documents.csv` metadata and `evaluation.jsonl` gold labels.

The filings are plain text PDFs with income statements, balance sheets, and cash flow
statements, so document routing, table-block retrieval, and the evidence grader all
exercise realistically.

## Run it natively

```bash
# from the repository root, with the package installed (pip install -e .)
python examples/run_example.py
```

That single command:

1. generates the synthetic dataset into `data/` if `data/pdfs/` is empty,
2. builds the index into `artifacts/` (skipped when `artifacts/manifest.json` exists),
3. asks three demo questions through the Python API (`FinRoutePipeline.ask`) and prints
   the route, routed documents, selected pages, and evidence grade,
4. runs the offline evaluation (`retrieval_metrics`) against `data/evaluation.jsonl`
   and prints per-question and mean metrics.

Useful flags:

```bash
python examples/run_example.py --skip-index   # reuse an existing artifacts/ build
python examples/run_example.py --generate     # also run grounded Qwen generation
python examples/run_example.py --artifacts /tmp/artifacts   # keep builds separate
```

## Run it with Docker

```bash
docker build -t finroute:local .    # or pull from ghcr.io/<owner>/finroute

# The image entrypoint is the finroute CLI, so run the example via --entrypoint python:
docker run --rm --entrypoint python \
  -v "$PWD/data:/app/data" -v "$PWD/artifacts:/app/artifacts" \
  finroute:local examples/run_example.py
```

The `examples/` directory is baked into the image, so only `data/` and `artifacts/`
need to be mounted (plus the HF cache volume if you want the model weights to persist
across containers — recommended). With compose, which already wires those volumes:

```bash
docker compose run --rm --entrypoint python finroute examples/run_example.py
```

On Windows PowerShell, use `${PWD}` instead of `$PWD` for the bind paths.

## What to expect on the first run

- **One-time model downloads** into the Hugging Face cache (`HF_HOME`, the
  `finroute_hf_cache` compose volume): the BGE embedder (~130 MB) and the BGE reranker
  (~2.2 GB). Later runs reuse them. Add `--generate` only if you also want to download
  Qwen3-4B (~8 GB) — generation is slow on CPU.
- **A printed trace per question**, for example:

  ```text
  Q: What was the operating margin of Meridian Technologies in 2023?
     route=calculation metric=operating_margin operands=['operating income', 'revenue'] years=['2023']
     routed documents: ['meridian_2023_10k', 'cobalt_2023_10k', 'meridian_2022_10k']
     page meridian_2023_10k p4 [Consolidated Statements of Operations] score=0.985
     grade: complete=True coverage=1.00 abstain=False (evidence_complete)
  ```

- **Evaluation output** with router recall and page-level hit/recall/MRR/nDCG@5.
  Note that a 4-document corpus makes retrieval easy — these numbers demonstrate the
  pipeline plumbing, not FinanceBench-scale quality.

## About `config.example.yaml`

`run_example.py` uses `examples/config.example.yaml`, which matches
`configs/default.yaml` except for `grading.minimum_candidate_pages: 12`. The stock
value of 20 expects FinanceBench-scale retrieval, where the candidate pool always
saturates the 20-page shortlist; a 4-filing corpus tops out at 19 candidate pages for
calculation-routed questions, so with the stock config the grader would correctly
abstain with `only 19 candidate pages`. See both behaviors:

```bash
python examples/run_example.py --skip-index                          # example config
python examples/run_example.py --skip-index --config configs/default.yaml  # stock config: abstains
```

Abstention on a corpus this small is the system working as designed, not a bug —
real deployments tune this threshold to corpus size.

## Files

| File | Purpose |
| --- | --- |
| `make_example_data.py` | Writes `data/pdfs/*.pdf`, `data/documents.csv`, `data/evaluation.jsonl` |
| `run_example.py` | Index build + `ask()` demo + offline evaluation in one command |
| `config.example.yaml` | Stock config with a toy-corpus-sized grading threshold |

The generator is covered by `tests/test_example_data.py` (it runs in the Docker CI job
where `pymupdf` is installed; the fast torch-free CI job skips it).
