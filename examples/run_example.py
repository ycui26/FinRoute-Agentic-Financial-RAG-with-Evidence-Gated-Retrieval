"""End-to-end FinRoute example over the synthetic dataset from make_example_data.py.

Run from the repository root (or from /app inside the container):

    python examples/run_example.py               # build index, ask, evaluate
    python examples/run_example.py --skip-index  # reuse an existing artifacts/ build
    python examples/run_example.py --generate    # also run grounded Qwen generation

The first run downloads the embedding model (~130 MB) and the reranker (~2.2 GB)
into the Hugging Face cache; later runs reuse them.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from finroute.build import build_artifacts
from finroute.config import FinRouteConfig
from finroute.evaluation import retrieval_metrics
from finroute.pipeline import FinRoutePipeline

EXAMPLES_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXAMPLES_DIR))

import make_example_data  # noqa: E402  (local module, path set above)

# The toy corpus cannot saturate the stock 20-candidate-page grading threshold,
# so the example ships its own config (see the header of config.example.yaml).
DEFAULT_CONFIG = EXAMPLES_DIR / "config.example.yaml"

DEMO_QUESTIONS = [
    "What was the operating margin of Meridian Technologies in 2023?",
    "What was the current ratio of Cobalt Energy Corp in 2023?",
    "Why did Meridian Technologies increase its revolving credit facility in 2023?",
]


def print_result(question: str, result: dict) -> None:
    plan = result["plan"]
    print(f"\nQ: {question}")
    print(
        f"   route={plan['route']} metric={plan['metric_name']} "
        f"operands={plan['operands']} years={plan['years']}"
    )
    print(f"   routed documents: {result['allowed_documents']}")
    for page in result["pages"][:3]:
        print(
            f"   page {page['doc_name']} p{page['page_num']} "
            f"[{page.get('section', '')[:60]}] "
            f"score={page.get('combined_score', 0):.3f}"
        )
    grade = result["grade"]
    print(
        f"   grade: complete={grade['complete']} coverage={grade['term_coverage']:.2f} "
        f"abstain={result['abstain']} ({result['stop_reason']})"
    )
    if not grade["complete"]:
        print(f"   grade detail: {grade['reason']}")
    generation = result.get("generation")
    if generation:
        print(f"   answer: {generation.get('answer', '')}")
        print(f"   citations: {generation.get('citations', [])}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the FinRoute example end to end")
    parser.add_argument("--data-dir", default="data", help="dataset directory (default: data)")
    parser.add_argument("--artifacts", default="artifacts", help="index directory (default: artifacts)")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="YAML config path")
    parser.add_argument("--generate", action="store_true", help="also run grounded Qwen generation")
    parser.add_argument("--skip-index", action="store_true", help="reuse the existing index build")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    artifacts = Path(args.artifacts)
    pdf_dir = data_dir / "pdfs"

    if not any(pdf_dir.glob("*.pdf")):
        print(f"No PDFs under {pdf_dir}/ - generating the synthetic dataset first.")
        make_example_data.generate(data_dir)

    config = FinRouteConfig.from_yaml(args.config)
    if not args.skip_index or not (artifacts / "manifest.json").exists():
        print(f"Building index in {artifacts}/ (first run downloads the embedding model)...")
        manifest = build_artifacts(pdf_dir, data_dir / "documents.csv", artifacts, config)
        print(f"Index built: {json.dumps(manifest)}")
    else:
        print(f"Reusing existing index in {artifacts}/")

    pipeline = FinRoutePipeline.from_artifacts(artifacts, config)

    print("\n" + "=" * 78)
    print("PART 1 - ask(): retrieval, evidence grading, and optional generation")
    print("=" * 78)
    for question in DEMO_QUESTIONS:
        print_result(question, pipeline.ask(question, generate=args.generate))

    questions_path = data_dir / "evaluation.jsonl"
    if questions_path.exists():
        print("\n" + "=" * 78)
        print("PART 2 - offline evaluation against gold page labels")
        print("=" * 78)
        rows = pd.read_json(questions_path, lines=True).to_dict(orient="records")
        metric_keys = None
        totals: dict[str, float] = {}
        for row in rows:
            result = pipeline.ask(str(row["question"]))
            retrieved = [(p["doc_name"], int(p["page_idx0"])) for p in result["pages"]]
            gold = {
                (e.get("doc_name", row["doc_name"]), int(e["evidence_page_num"]))
                for e in row["evidence"]
            }
            metrics = retrieval_metrics(retrieved, gold, k=5)
            metrics["router_recall@1"] = int(row["doc_name"] in result["allowed_documents"][:1])
            metric_keys = metric_keys or list(metrics)
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + value
            gold_text = ", ".join(f"{doc} p{page + 1}" for doc, page in sorted(gold))
            top_text = ", ".join(f"{doc} p{page + 1}" for doc, page in retrieved[:2])
            print(f"\n{row['question_id']}: {row['question']}")
            print(f"   gold: {gold_text}")
            print(f"   top-2 retrieved: {top_text}")
            print(
                "   " + "  ".join(
                    f"{key}={metrics[key]:.2f}" if isinstance(metrics[key], float) else f"{key}={metrics[key]}"
                    for key in metric_keys if key in metrics
                )
            )
        print("\nSummary over", len(rows), "questions:")
        for key in metric_keys or []:
            print(f"   mean {key} = {totals[key] / len(rows):.3f}")
        print(
            "\nNote: a tiny 4-document corpus makes these numbers easy to read but not\n"
            "comparable to FinanceBench-scale results."
        )


if __name__ == "__main__":
    main()
