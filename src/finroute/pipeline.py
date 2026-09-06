from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from finroute.config import FinRouteConfig
from finroute.generation import GroundedGenerator
from finroute.grading import EvidenceGrader
from finroute.indexing import EmbeddingEncoder, HybridIndex
from finroute.planning import RulePlanner
from finroute.reranking import QueryCenteredReranker
from finroute.retrieval import PageRetriever
from finroute.routing import DocumentRouter
from finroute.workflow import FinRouteWorkflow


class FinRoutePipeline:
    def __init__(
        self,
        workflow: FinRouteWorkflow,
        generator: GroundedGenerator,
    ):
        self.workflow = workflow
        self.generator = generator

    @classmethod
    def from_artifacts(
        cls, artifacts_directory: str | Path, config: FinRouteConfig
    ) -> "FinRoutePipeline":
        artifacts = Path(artifacts_directory)
        encoder = EmbeddingEncoder(
            config.models.embedding,
            query_prefix=config.models.query_prefix,
            local_files_only=config.models.local_files_only,
        )
        chroma = artifacts / "chroma"
        document_index = HybridIndex.load(
            artifacts / "documents", encoder, config, chroma
        )
        child_index = HybridIndex.load(
            artifacts / "narrative", encoder, config, chroma
        )
        table_index = HybridIndex.load(
            artifacts / "tables", encoder, config, chroma
        )
        pages = pd.read_parquet(artifacts / "pages.parquet")
        planner = RulePlanner()
        router = DocumentRouter(document_index)
        retriever = PageRetriever(child_index, table_index, pages, config)
        reranker = QueryCenteredReranker(config.models.reranker, config)
        grader = EvidenceGrader(config)
        workflow = FinRouteWorkflow(
            planner, router, retriever, reranker, grader, config
        )
        return cls(workflow, GroundedGenerator(config))

    def ask(self, question: str, generate: bool = False) -> dict[str, Any]:
        retrieval = self.workflow.run(question)
        response = retrieval.to_dict()
        if generate:
            response["generation"] = self.generator.generate(retrieval)
        return response
