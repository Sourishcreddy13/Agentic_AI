"""Persistent Chroma store with Sentence-Transformers embeddings."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import chromadb

from src.config import CONFIG, PROJECT_ROOT
from src.memory.eviction import (
    MAX_FACTS_PER_USER,
    importance_for_fact_type,
    select_evictions,
    retention_score,
)
from src.state.schema import MemoryFact


COLLECTION_NAME = "applicant_facts"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=4)
def _embedding_model(model_name: str):
    """Load a local Sentence-Transformers model once per process per name.

    ChromaMemoryStore is constructed fresh on essentially every node
    invocation that touches long-term memory (intake's memory lookup,
    memory_consolidation's post-decision write) — previously each
    construction loaded its own SentenceTransformer instance from disk.
    Caching by model name (the same pattern src/rag/policy_store.py already
    uses for the RAG embedding model) means the model is loaded once and
    reused for the life of the process.
    """
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


class ChromaMemoryStore:
    """Small explicit persistence adapter around a Chroma collection."""

    def __init__(self, persist_directory: str | Path | None = None):
        configured = CONFIG.get("memory", {}).get(
            "chroma_persist_directory", "data/chroma_memory"
        )
        path = Path(persist_directory) if persist_directory is not None else Path(configured)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        path.mkdir(parents=True, exist_ok=True)

        self.persist_directory = path
        self.client = chromadb.PersistentClient(path=str(path))
        self.collection = self.client.get_or_create_collection(name=COLLECTION_NAME)

        model_name = CONFIG.get("memory", {}).get(
            "embedding_model", DEFAULT_EMBEDDING_MODEL
        )
        self.embedder = _embedding_model(model_name)

    def _embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self.embedder.encode(texts, normalize_embeddings=True)
        return vectors.tolist()

    def upsert_facts(self, facts: list[MemoryFact]) -> None:
        if not facts:
            return

        now = datetime.now(timezone.utc)
        for fact in facts:
            fact.importance = importance_for_fact_type(fact.fact_type)
            metadata = {
                "user_id": fact.user_id,
                "fact_type": fact.fact_type,
                "importance": float(fact.importance),
                "session_ts": fact.session_ts,
                "thread_id": fact.thread_id,
                "usage_count": int(fact.usage_count),
                "created_at": fact.session_ts,
                "last_access_ts": fact.last_access_ts or fact.session_ts,
            }
            self.collection.upsert(
                ids=[fact.fact_id],
                documents=[fact.value],
                embeddings=self._embed([fact.value]),
                metadatas=[metadata],
            )

        self.evict_user(facts[0].user_id, now=now)

    def search(self, user_id: str, query: str, k: int = 5) -> list[MemoryFact]:
        if self.collection.count() == 0:
            return []

        results = self.collection.query(
            query_embeddings=self._embed([query]),
            n_results=k,
            where={"user_id": user_id},
        )

        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]

        facts: list[MemoryFact] = []
        for fact_id, document, metadata in zip(ids, documents, metadatas):
            metadata = metadata or {}
            facts.append(
                MemoryFact(
                    fact_id=fact_id,
                    user_id=user_id,
                    fact_type=str(metadata.get("fact_type", "employment")),
                    value=str(document),
                    importance=float(metadata.get("importance", 0.40)),
                    session_ts=str(metadata.get("session_ts")),
                    thread_id=str(metadata.get("thread_id")),
                    usage_count=int(metadata.get("usage_count", 0)) + 1,
                    last_access_ts=str(metadata.get("last_access_ts", metadata.get("session_ts"))),
                )
            )

        # Update usage metadata for retrieved memories.
        for fact in facts:
            self.collection.update(
                ids=[fact.fact_id],
                metadatas=[{
                    "user_id": fact.user_id,
                    "fact_type": fact.fact_type,
                    "importance": float(fact.importance),
                    "session_ts": fact.session_ts,
                    "thread_id": fact.thread_id,
                    "usage_count": fact.usage_count,
                    "created_at": fact.session_ts,
                    "last_access_ts": datetime.now(timezone.utc).isoformat(),
                }],
            )

        return facts

    def list_user_facts(self, user_id: str) -> list[MemoryFact]:
        results = self.collection.get(where={"user_id": user_id})
        ids = results.get("ids", [])
        documents = results.get("documents", [])
        metadatas = results.get("metadatas", [])

        facts: list[MemoryFact] = []
        for fact_id, document, metadata in zip(ids, documents, metadatas):
            metadata = metadata or {}
            facts.append(
                MemoryFact(
                    fact_id=fact_id,
                    user_id=user_id,
                    fact_type=str(metadata.get("fact_type", "employment")),
                    value=str(document),
                    importance=float(metadata.get("importance", 0.40)),
                    session_ts=str(metadata.get("session_ts")),
                    thread_id=str(metadata.get("thread_id")),
                    usage_count=int(metadata.get("usage_count", 0)),
                    last_access_ts=str(metadata.get("last_access_ts", metadata.get("session_ts"))),
                )
            )
        return facts

    def delete_facts(self, fact_ids: list[str]) -> None:
        if fact_ids:
            self.collection.delete(ids=fact_ids)

    def evict_user(self, user_id: str, *, now: datetime | None = None) -> None:
        facts = self.list_user_facts(user_id)
        if not facts:
            return

        access_counts = {fact.fact_id: fact.usage_count for fact in facts}
        evictions = select_evictions(facts, access_counts=access_counts, now=now)
        if len(facts) > MAX_FACTS_PER_USER:
            # Safety fallback in case future policy changes alter selection.
            ordered = sorted(
                facts,
                key=lambda fact: retention_score(
                    fact,
                    access_count=access_counts.get(fact.fact_id, 0),
                    now=now,
                ),
            )
            evictions.extend(ordered[: len(facts) - MAX_FACTS_PER_USER])

        self.delete_facts(list(dict.fromkeys(f.fact_id for f in evictions)))
