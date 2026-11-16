"""Hybrid schema search over ``world/dictionary.yaml`` (BM25 + optional sentence embeddings).

Documents are tables (name, description, notes, synonyms, column names) and columns (table.column +
description). Scores are rank-fused per table. Only objects visible to the role are indexed. If
``sentence-transformers`` is not installed (optional extra ``embeddings``), search runs BM25-only and
:attr:`SchemaRetriever.mode` says so.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import numpy as np
import yaml
from rank_bm25 import BM25Okapi

from .paths import WORLD

log = logging.getLogger(__name__)

_TOKEN = re.compile(r"[a-z0-9]+")
STOP = {
    "the",
    "a",
    "an",
    "of",
    "in",
    "by",
    "for",
    "and",
    "or",
    "to",
    "what",
    "which",
    "how",
    "many",
    "is",
    "are",
    "was",
    "were",
    "per",
    "with",
    "on",
    "at",
    "from",
    "show",
    "me",
    "give",
    "list",
    "each",
    "our",
    "do",
    "does",
    "did",
    "we",
    "last",
    "this",
    "that",
    "all",
    "top",
    "most",
}


def tokenize(text: str) -> list[str]:
    toks = []
    for t in _TOKEN.findall(text.lower().replace("_", " ")):
        if t in STOP:
            continue
        if len(t) > 4 and t.endswith("ies"):
            t = t[:-3] + "y"
        elif len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
            t = t[:-1]
        toks.append(t)
    return toks


@lru_cache(maxsize=1)
def load_dictionary() -> dict[str, Any]:
    with open(WORLD / "dictionary.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass
class Hit:
    table: str
    score: float
    description: str
    columns: list[dict[str, str]] = field(default_factory=list)
    joins: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "table": self.table,
            "description": self.description,
            "columns": self.columns,
            "joins": self.joins,
            "notes": self.notes,
        }


class SchemaRetriever:
    def __init__(
        self,
        role: str,
        bm25_weight: float = 0.5,
        embedding_model: str | None = None,
        use_embeddings: bool = True,
    ) -> None:
        self.role = role
        self.bm25_weight = bm25_weight
        d = load_dictionary()["tables"]
        self.tables: dict[str, dict[str, Any]] = {}
        for t, body in d.items():
            if role != "admin" and role not in body.get("views", []):
                continue
            cols = {c: v for c, v in body["columns"].items() if role == "admin" or role in v.get("views", [])}
            self.tables[t] = {**body, "columns": cols}
        self.docs: list[tuple[str, str | None, str]] = []  # (table, column|None, text)
        for t, body in self.tables.items():
            text = " ".join(
                [
                    t,
                    t.replace("_", " "),
                    body["description"],
                    " ".join(body.get("notes", [])),
                    " ".join(body.get("synonyms", [])),
                    " ".join(body["columns"]),
                ]
            )
            self.docs.append((t, None, text))
            for c, cb in body["columns"].items():
                self.docs.append((t, c, f"{t} {c} {c.replace('_', ' ')} {cb['description']}"))
        self.bm25 = BM25Okapi([tokenize(x[2]) for x in self.docs])
        self.mode = "bm25"
        self._emb = None
        self._doc_vecs: np.ndarray | None = None
        if use_embeddings and embedding_model:
            self._init_embeddings(embedding_model)

    def _init_embeddings(self, model_name: str) -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]
        except ImportError:
            log.info("sentence-transformers not installed: schema search runs BM25-only")
            return
        try:
            self._emb = SentenceTransformer(model_name)
            self._doc_vecs = np.asarray(
                self._emb.encode([x[2] for x in self.docs], normalize_embeddings=True)
            )
            self.mode = "hybrid"
        except Exception as e:  # noqa: BLE001 - model download can fail offline; BM25 still works
            log.warning("embedding model unavailable (%s): BM25-only", e)
            self._emb = None

    def search(self, query: str, k: int = 10) -> list[Hit]:
        toks = tokenize(query)
        bm = np.asarray(self.bm25.get_scores(toks)) if toks else np.zeros(len(self.docs))
        score = self._rank_norm(bm)
        if self._emb is not None and self._doc_vecs is not None:
            q = np.asarray(self._emb.encode([query], normalize_embeddings=True))[0]
            score = self.bm25_weight * score + (1 - self.bm25_weight) * self._rank_norm(self._doc_vecs @ q)
        table_score: dict[str, float] = {}
        col_score: dict[str, list[tuple[float, str]]] = {}
        for (t, c, _), s in zip(self.docs, score, strict=True):
            if c is None:
                table_score[t] = max(table_score.get(t, 0.0), float(s))
            else:
                table_score[t] = max(table_score.get(t, 0.0), 0.9 * float(s))
                col_score.setdefault(t, []).append((float(s), c))
        ranked = sorted(table_score.items(), key=lambda x: (-x[1], x[0]))[:k]
        hits = []
        for t, s in ranked:
            body = self.tables[t]
            cols = sorted(col_score.get(t, []), reverse=True)[:4]
            hits.append(
                Hit(
                    table=t,
                    score=round(s, 4),
                    description=body["description"],
                    columns=[{"name": c, "description": body["columns"][c]["description"]} for _, c in cols],
                    joins=body.get("joins", [])[:3],
                    notes=body.get("notes", [])[:2],
                )
            )
        return hits

    @staticmethod
    def _rank_norm(x: np.ndarray) -> np.ndarray:
        """Reciprocal-rank style normalisation so BM25 and cosine scores are comparable."""
        if len(x) == 0:
            return x
        order = np.argsort(-x, kind="stable")
        ranks = np.empty(len(x))
        ranks[order] = np.arange(1, len(x) + 1)
        out = 1.0 / (ranks + 10.0)
        out[x <= 0] = 0.0 if x.max() > 0 else out[x <= 0]
        return out / out.max() if out.max() > 0 else out


def recall_at_k(retrieved: list[str], gold: list[str], k: int) -> float:
    g = {x.lower() for x in gold}
    if not g:
        return 1.0
    return len(g & {x.lower() for x in retrieved[:k]}) / len(g)
