"""Build the persistent Chroma index for the synthetic lending-policy corpus."""
from __future__ import annotations

import json
from pathlib import Path

from src.config import get_rag_config
from src.rag.policy_store import PolicyVectorStore


ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = ROOT / "rag" / "corpus"


def main() -> None:
    config = get_rag_config()
    store = PolicyVectorStore(
        config["persist_directory"],
        collection_name=config["collection_name"],
        embedding_model=config["embedding_model"],
    )
    count = store.build_index(CORPUS_DIR)
    print(json.dumps({"collection": config["collection_name"], "documents_indexed": count}))


if __name__ == "__main__":
    main()
