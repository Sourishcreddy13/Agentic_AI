"""Chroma-backed clause-level semantic retrieval for synthetic lending policy."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any
import re


DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_COLLECTION = "lending_policy"
INDEX_VERSION = "phase5-clause-v2"
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _slug(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_") or "section"


def _split_policy_documents(corpus_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    for doc_path in sorted(corpus_dir.glob("*.md")):
        text = doc_path.read_text(encoding="utf-8").strip()
        if not text:
            continue

        lines = text.splitlines()
        policy_id = next(
            (
                match.group(1).strip()
                for line in lines
                if (
                    match := re.match(
                        r"^\*\*Policy ID:\*\*\s*(\S+)",
                        line.strip(),
                    )
                )
            ),
            doc_path.stem,
        )
        title = next(
            (line.lstrip("#").strip() for line in lines if line.startswith("# ")),
            doc_path.stem,
        )
        header_lines: list[str] = []
        sections: list[tuple[str, list[str]]] = []
        current_heading: str | None = None
        current_lines: list[str] = []

        for line in lines:
            if line.startswith("## "):
                if current_heading is not None:
                    sections.append((current_heading, current_lines))
                current_heading = line[3:].strip()
                current_lines = []
            elif current_heading is None:
                header_lines.append(line)
            else:
                current_lines.append(line)

        if current_heading is not None:
            sections.append((current_heading, current_lines))

        if not sections:
            sections = [(title, lines)]
            header_lines = []

        header = "\n".join(line for line in header_lines if line.strip())
        for heading, section_lines in sections:
            body = "\n".join(section_lines).strip()
            if not body:
                continue
            chunk = f"{header}\n\n## {heading}\n{body}".strip()
            clause_id = f"{doc_path.stem}::{_slug(heading)}"
            records.append(
                {
                    "id": clause_id,
                    "document": chunk,
                    "metadata": {
                        "policy_id": policy_id,
                        "clause_id": clause_id,
                        "heading": heading,
                        "source_file": doc_path.name,
                        "index_version": INDEX_VERSION,
                    },
                }
            )

    return records


@lru_cache(maxsize=1)
def _embedding_model(model_name: str = DEFAULT_MODEL):
    """Load the local Sentence-Transformers model once per process."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


class PolicyVectorStore:
    """Persistent Chroma vector store for clause-level synthetic policy retrieval."""

    def __init__(
        self,
        persist_directory: str | Path,
        *,
        collection_name: str = DEFAULT_COLLECTION,
        embedding_model: str = DEFAULT_MODEL,
    ) -> None:
        import chromadb

        self.persist_directory = Path(persist_directory)
        if not self.persist_directory.is_absolute():
            self.persist_directory = PROJECT_ROOT / self.persist_directory
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model

        client = chromadb.PersistentClient(path=str(self.persist_directory))
        existing = None
        try:
            existing = client.get_collection(name=collection_name)
        except Exception:
            pass

        if existing is not None:
            metadata = existing.metadata or {}
            if metadata.get("index_version") != INDEX_VERSION:
                client.delete_collection(name=collection_name)

        self.collection = client.get_or_create_collection(
            name=collection_name,
            metadata={
                "hnsw:space": "cosine",
                "index_version": INDEX_VERSION,
            },
        )

    def build_index(self, corpus_dir: str | Path) -> int:
        """Rebuild the clause-level synthetic policy index."""
        records = _split_policy_documents(Path(corpus_dir))
        if not records:
            raise RuntimeError(f"No policy documents found in {corpus_dir}")

        existing_ids = self.collection.get().get("ids", [])
        if existing_ids:
            self.collection.delete(ids=existing_ids)

        model = _embedding_model(self.embedding_model_name)
        documents = [record["document"] for record in records]
        embeddings = model.encode(
            documents,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        self.collection.upsert(
            ids=[record["id"] for record in records],
            documents=documents,
            metadatas=[record["metadata"] for record in records],
            embeddings=embeddings.tolist(),
        )
        return len(records)

    def ensure_indexed(self, corpus_dir: str | Path) -> None:
        if self.collection.count() == 0:
            self.build_index(corpus_dir)

    def search(
        self,
        query: str,
        *,
        k: int = 3,
        corpus_dir: str | Path,
    ) -> list[dict[str, Any]]:
        self.ensure_indexed(corpus_dir)

        k = max(1, min(int(k), 10))
        model = _embedding_model(self.embedding_model_name)
        query_embedding = model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )[0]

        result = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        output: list[dict[str, Any]] = []
        for document, metadata, distance in zip(documents, metadatas, distances):
            metadata = metadata or {}
            score = max(0.0, min(1.0, 1.0 - float(distance)))
            output.append(
                {
                    "clause_id": str(metadata.get("clause_id", "unknown")),
                    "policy_id": str(metadata.get("policy_id", "unknown")),
                    "heading": str(metadata.get("heading", "")),
                    "source_file": str(metadata.get("source_file", "")),
                    "snippet": str(document)[:800].replace("\n", " ").strip(),
                    "score": round(score, 4),
                }
            )

        output.sort(key=lambda item: item["score"], reverse=True)
        return output
