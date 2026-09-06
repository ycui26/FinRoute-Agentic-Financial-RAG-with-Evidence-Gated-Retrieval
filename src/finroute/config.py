from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    embedding: str = "BAAI/bge-small-en-v1.5"
    reranker: str = "BAAI/bge-reranker-v2-m3"
    generator: str = "Qwen/Qwen3-4B-Instruct-2507"
    query_prefix: str = "Represent this sentence for searching relevant passages: "
    local_files_only: bool = False


@dataclass
class ChunkingConfig:
    parent_size: int = 1800
    parent_overlap: int = 250
    child_size: int = 650
    child_overlap: int = 100
    minimum_chars: int = 80


@dataclass
class RetrievalConfig:
    dense_candidates: int = 100
    bm25_candidates: int = 100
    rrf_k: int = 60
    union_candidates: int = 120
    page_shortlist: int = 20
    final_top_k: int = 5
    routed_document_top_k: int = 3
    table_blocks_per_query: int = 20
    child_pages_per_query: int = 5
    minimum_document_candidates: int = 2
    minimum_table_candidates: int = 12
    maximum_child_only_candidates: int = 8


@dataclass
class RerankingConfig:
    reranker_weight: float = 0.70
    fusion_weight: float = 0.20
    structure_weight: float = 0.10
    excerpt_max_chars: int = 2000
    excerpt_line_window: int = 1
    operand_gain_weight: float = 0.22
    statement_gain_weight: float = 0.08
    year_gain_weight: float = 0.05


@dataclass
class GradingConfig:
    minimum_term_coverage: float = 0.45
    minimum_candidate_pages: int = 20
    require_numeric_anchor: bool = True
    numeric_anchor_window: int = 180
    maximum_retries: int = 1


@dataclass
class RuntimeConfig:
    seed: int = 42
    use_chroma: bool = True
    generator_backend: str = "local_transformers"
    generator_base_url: str = "http://127.0.0.1:8000/v1"
    generator_api_key: str = "EMPTY"


@dataclass
class FinRouteConfig:
    models: ModelConfig = field(default_factory=ModelConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    reranking: RerankingConfig = field(default_factory=RerankingConfig)
    grading: GradingConfig = field(default_factory=GradingConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "FinRouteConfig":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        config = cls(
            models=ModelConfig(**raw.get("models", {})),
            chunking=ChunkingConfig(**raw.get("chunking", {})),
            retrieval=RetrievalConfig(**raw.get("retrieval", {})),
            reranking=RerankingConfig(**raw.get("reranking", {})),
            grading=GradingConfig(**raw.get("grading", {})),
            runtime=RuntimeConfig(**raw.get("runtime", {})),
        )
        config.apply_environment()
        return config

    def apply_environment(self) -> None:
        self.models.embedding = os.getenv("FINROUTE_EMBEDDING_MODEL", self.models.embedding)
        self.models.reranker = os.getenv("FINROUTE_RERANKER_MODEL", self.models.reranker)
        self.models.generator = os.getenv("FINROUTE_GENERATOR_MODEL", self.models.generator)
        self.models.local_files_only = os.getenv(
            "FINROUTE_HF_LOCAL_ONLY", str(int(self.models.local_files_only))
        ).lower() in {"1", "true", "yes"}
        self.runtime.generator_backend = os.getenv(
            "FINROUTE_GENERATOR_BACKEND", self.runtime.generator_backend
        )
        self.runtime.generator_base_url = os.getenv(
            "FINROUTE_GENERATOR_BASE_URL", self.runtime.generator_base_url
        )
        self.runtime.generator_api_key = os.getenv(
            "FINROUTE_GENERATOR_API_KEY", self.runtime.generator_api_key
        )

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        return asdict(self)
