from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

from finroute.config import FinRouteConfig


TOKEN_RE = re.compile(r"[a-z0-9]+(?:[./%-][a-z0-9]+)*", re.I)


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(str(text).lower())


def top_positions(scores: np.ndarray, valid: np.ndarray, top_n: int) -> np.ndarray:
    if len(valid) == 0:
        return np.array([], dtype=int)
    top_n = min(top_n, len(valid))
    values = scores[valid]
    if top_n == len(valid):
        local = np.argsort(-values)
    else:
        local = np.argpartition(values, -top_n)[-top_n:]
        local = local[np.argsort(-values[local])]
    return valid[local]


class EmbeddingEncoder:
    def __init__(self, model_name: str, query_prefix: str = "", local_files_only: bool = False):
        self.model_name = model_name
        self.query_prefix = query_prefix
        self.local_files_only = local_files_only
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                self.model_name,
                local_files_only=self.local_files_only,
            )
        return self._model

    def encode_documents(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        values = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        return np.asarray(values, dtype="float32")

    def encode_query(self, query: str) -> np.ndarray:
        value = self.model.encode(
            [f"{self.query_prefix}{query}"],
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        return np.asarray(value, dtype="float32")


@dataclass
class HybridIndex:
    frame: pd.DataFrame
    embeddings: np.ndarray
    text_column: str
    encoder: EmbeddingEncoder
    config: FinRouteConfig
    collection_name: str
    chroma_directory: Path | None = None

    def __post_init__(self) -> None:
        self.frame = self.frame.reset_index(drop=True)
        if len(self.frame) != len(self.embeddings):
            raise ValueError("frame and embeddings must have the same number of rows")
        self.bm25 = BM25Okapi([tokenize(text) for text in self.frame[self.text_column].astype(str)])
        self._chroma_collection = None

    def build_chroma(self, recreate: bool = False, batch_size: int = 5000) -> None:
        if self.chroma_directory is None:
            return
        import chromadb

        self.chroma_directory.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(self.chroma_directory))
        if recreate:
            try:
                client.delete_collection(self.collection_name)
            except Exception:
                pass
        collection = client.get_or_create_collection(
            self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        if collection.count() != len(self.frame):
            if collection.count():
                client.delete_collection(self.collection_name)
                collection = client.get_or_create_collection(
                    self.collection_name,
                    metadata={"hnsw:space": "cosine"},
                )
            for start in range(0, len(self.frame), batch_size):
                stop = min(start + batch_size, len(self.frame))
                records = self.frame.iloc[start:stop]
                collection.add(
                    ids=[f"{self.collection_name}:{i}" for i in range(start, stop)],
                    embeddings=self.embeddings[start:stop].tolist(),
                    metadatas=[
                        {
                            "corpus_pos": int(i),
                            "doc_name": str(row.doc_name),
                        }
                        for i, row in zip(range(start, stop), records.itertuples(index=False))
                    ],
                )
        self._chroma_collection = collection

    def _chroma_dense(
        self, query_vector: np.ndarray, allowed_docs: list[str] | None, top_n: int
    ) -> tuple[np.ndarray, dict[int, float]] | None:
        if not self.config.runtime.use_chroma or self.chroma_directory is None:
            return None
        try:
            if self._chroma_collection is None:
                self.build_chroma()
            where = {"doc_name": {"$in": allowed_docs}} if allowed_docs else None
            result = self._chroma_collection.query(
                query_embeddings=[query_vector.tolist()],
                n_results=top_n,
                where=where,
                include=["metadatas", "distances"],
            )
            metadata = result["metadatas"][0]
            distances = result["distances"][0]
            positions = np.asarray([int(item["corpus_pos"]) for item in metadata], dtype=int)
            scores = {
                int(item["corpus_pos"]): 1.0 - float(distance)
                for item, distance in zip(metadata, distances)
            }
            return positions, scores
        except Exception:
            return None

    def search(
        self,
        query: str,
        allowed_docs: list[str] | None = None,
        top_n: int | None = None,
    ) -> pd.DataFrame:
        rc = self.config.retrieval
        top_n = top_n or rc.union_candidates
        valid = (
            np.arange(len(self.frame))
            if not allowed_docs
            else np.flatnonzero(self.frame["doc_name"].isin(allowed_docs).to_numpy())
        )
        if not len(valid):
            return pd.DataFrame()

        query_vector = self.encoder.encode_query(query)
        sparse_scores = np.asarray(self.bm25.get_scores(tokenize(query)), dtype="float32")
        chroma = self._chroma_dense(
            query_vector, allowed_docs, min(rc.dense_candidates, len(valid))
        )
        if chroma is None:
            dense_scores = self.embeddings @ query_vector
            dense_positions = top_positions(dense_scores, valid, rc.dense_candidates)
            dense_lookup = {int(pos): float(dense_scores[pos]) for pos in dense_positions}
            dense_backend = "numpy"
        else:
            dense_positions, dense_lookup = chroma
            dense_backend = "chroma"
        sparse_positions = top_positions(sparse_scores, valid, rc.bm25_candidates)
        dense_ranks = {int(pos): rank for rank, pos in enumerate(dense_positions, 1)}
        sparse_ranks = {int(pos): rank for rank, pos in enumerate(sparse_positions, 1)}

        rows = []
        for pos in set(dense_ranks) | set(sparse_ranks):
            dense_rank, sparse_rank = dense_ranks.get(pos), sparse_ranks.get(pos)
            rrf = 0.0
            if dense_rank is not None:
                rrf += 1.0 / (rc.rrf_k + dense_rank)
            if sparse_rank is not None:
                rrf += 1.0 / (rc.rrf_k + sparse_rank)
            row = self.frame.iloc[pos].to_dict()
            row.update(
                {
                    "dense_rank": dense_rank,
                    "bm25_rank": sparse_rank,
                    "dense_score": dense_lookup.get(pos, math.nan),
                    "bm25_score": float(sparse_scores[pos]),
                    "rrf_score": rrf,
                    "score": rrf,
                    "dense_backend": dense_backend,
                }
            )
            rows.append(row)
        return (
            pd.DataFrame(rows)
            .sort_values(["rrf_score", "dense_rank", "bm25_rank"], ascending=[False, True, True])
            .head(top_n)
            .reset_index(drop=True)
        )

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.frame.to_parquet(directory / "corpus.parquet", index=False)
        np.save(directory / "embeddings.npy", self.embeddings)
        (directory / "index.json").write_text(
            json.dumps(
                {
                    "text_column": self.text_column,
                    "collection_name": self.collection_name,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(
        cls,
        directory: str | Path,
        encoder: EmbeddingEncoder,
        config: FinRouteConfig,
        chroma_directory: str | Path | None = None,
    ) -> "HybridIndex":
        directory = Path(directory)
        metadata = json.loads((directory / "index.json").read_text(encoding="utf-8"))
        return cls(
            frame=pd.read_parquet(directory / "corpus.parquet"),
            embeddings=np.load(directory / "embeddings.npy"),
            text_column=metadata["text_column"],
            encoder=encoder,
            config=config,
            collection_name=metadata["collection_name"],
            chroma_directory=Path(chroma_directory) if chroma_directory else None,
        )
