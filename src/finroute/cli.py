from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from finroute.build import build_artifacts
from finroute.config import FinRouteConfig
from finroute.evaluation import retrieval_metrics
from finroute.pipeline import FinRoutePipeline


def load_config(path: str) -> FinRouteConfig:
    return FinRouteConfig.from_yaml(path)


def build_command(args: argparse.Namespace) -> None:
    manifest = build_artifacts(
        args.pdf_dir,
        args.metadata,
        args.artifacts,
        load_config(args.config),
    )
    print(json.dumps(manifest, indent=2))


def ask_command(args: argparse.Namespace) -> None:
    pipeline = FinRoutePipeline.from_artifacts(
        args.artifacts, load_config(args.config)
    )
    print(json.dumps(pipeline.ask(args.question, generate=args.generate), indent=2, default=str))


def gold_pairs(row: pd.Series) -> set[tuple[str, int]]:
    pairs = set()
    for evidence in row["evidence"]:
        pairs.add(
            (
                str(evidence.get("doc_name", evidence.get("evidence_doc_name", row["doc_name"]))),
                int(evidence["evidence_page_num"]),
            )
        )
    return pairs


def evaluate_command(args: argparse.Namespace) -> None:
    pipeline = FinRoutePipeline.from_artifacts(
        args.artifacts, load_config(args.config)
    )
    questions = pd.read_json(args.questions, lines=True)
    records = []
    for row in questions.to_dict(orient="records"):
        started = time.perf_counter()
        result = pipeline.ask(str(row["question"]), generate=False)
        retrieved = [
            (str(page["doc_name"]), int(page["page_idx0"]))
            for page in result["pages"]
        ]
        gold = gold_pairs(pd.Series(row))
        routed = result["allowed_documents"]
        records.append(
            {
                "question_id": row.get("financebench_id", row.get("question_id")),
                "question": row["question"],
                "router_recall@1": int(row["doc_name"] in routed[:1]),
                "router_recall@3": int(row["doc_name"] in routed[:3]),
                **retrieval_metrics(retrieved, gold, k=5),
                "candidate_count": result["candidate_count"],
                "evidence_complete": result["grade"]["complete"],
                "abstain": result["abstain"],
                "latency_seconds": time.perf_counter() - started,
                "retrieved_pages": result["pages"],
                "trace": result["trace"],
            }
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_json(output, orient="records", lines=True)
    print(f"Wrote {len(records)} evaluation rows to {output}")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="FinRoute financial filing RAG")
    subparsers = root.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build-index", help="parse PDFs and build persistent indexes")
    build.add_argument("--pdf-dir", required=True)
    build.add_argument("--metadata", required=True)
    build.add_argument("--artifacts", required=True)
    build.add_argument("--config", default="configs/default.yaml")
    build.set_defaults(handler=build_command)

    ask = subparsers.add_parser("ask", help="retrieve evidence and optionally generate an answer")
    ask.add_argument("--artifacts", required=True)
    ask.add_argument("--question", required=True)
    ask.add_argument("--config", default="configs/default.yaml")
    ask.add_argument("--generate", action="store_true")
    ask.set_defaults(handler=ask_command)

    evaluate = subparsers.add_parser("evaluate", help="run page-level retrieval evaluation")
    evaluate.add_argument("--artifacts", required=True)
    evaluate.add_argument("--questions", required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--config", default="configs/default.yaml")
    evaluate.set_defaults(handler=evaluate_command)
    return root


def main() -> None:
    args = parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
