"""
Pure tool logic — no FastMCP dependency here. This module is imported by
both mcp_server/server.py (for registration) and by tests (for unit testing
the logic directly, without spawning a subprocess).
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
APPLICANTS_FILE = REPO_ROOT / "synthetic_data" / "applicants.json"
BUREAU_FILE = REPO_ROOT / "synthetic_data" / "bureau_records.json"
CORPUS_DIR = REPO_ROOT / "rag" / "corpus"


# --------------------------------------------------------------------------
# Internal helpers
# --------------------------------------------------------------------------

def _load_json(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


# --------------------------------------------------------------------------
# Tool logic
# --------------------------------------------------------------------------

def applicant_lookup(applicant_id: str) -> dict[str, Any]:
    """
    Look up synthetic applicant history by applicant_id.
    Returns found=False for unknown IDs rather than raising.
    No real PII is accessed — all data is from synthetic_data/applicants.json.
    """
    records = _load_json(APPLICANTS_FILE)
    match = next((r for r in records if r["applicant_id"] == applicant_id), None)
    if match is None:
        return {
            "applicant_id": applicant_id,
            "found": False,
            "prior_applications": 0,
            "synthetic_watchlist_hit": False,
        }
    return {
        "applicant_id": applicant_id,
        "found": True,
        "prior_applications": match["prior_applications"],
        "synthetic_watchlist_hit": match["watchlist_hit"],
    }


def bureau_check(applicant_id: str, declared_income: float) -> dict[str, Any]:
    """
    Simulated bureau pull — reads from synthetic_data/bureau_records.json
    and applies a deterministic synthetic DTI calculation.
    No real credit data; scores are synthetic and deterministic.
    """
    records = _load_json(BUREAU_FILE)
    match = next((r for r in records if r["applicant_id"] == applicant_id), None)

    if match is None:
        # Unknown applicant → generate a thin-file synthetic record
        synthetic_score = max(300, min(500, int(300 + declared_income / 500)))
        return {
            "applicant_id": applicant_id,
            "found_in_bureau": False,
            "synthetic_score": synthetic_score,
            "delinquencies_24m": 0,
            "thin_file": True,
            "dti_estimate": round(1500 / max(declared_income / 12, 1), 3),
        }

    monthly_income = max(declared_income / 12, 1)
    monthly_obligations = match["total_outstanding_synthetic"] / 24  # assume 24m amortization
    dti = round(monthly_obligations / monthly_income, 3)

    return {
        "applicant_id": applicant_id,
        "found_in_bureau": True,
        "synthetic_score": match["synthetic_score"],
        "delinquencies_24m": match["delinquencies_24m"],
        "thin_file": match["thin_file"],
        "dti_estimate": dti,
    }


@lru_cache(maxsize=4)
def _get_policy_store(persist_directory: str, collection_name: str, embedding_model: str):
    """Cached PolicyVectorStore construction.

    Previously every `lending_policy_search` call rebuilt a fresh
    PolicyVectorStore (a chromadb PersistentClient + collection handle) from
    scratch. The heaviest part — the Sentence-Transformers model load — was
    already cached one layer down (`policy_store.py::_embedding_model`), but
    the store wrapper itself was not. Caching by the three config values that
    fully determine store identity (all plain strings, hashable) removes the
    remaining redundant client/collection construction on every call, the
    same pattern this project already uses for the embedding model itself.
    Call `reset_policy_store_cache()` to force a fresh instance (mirrors
    `src/mcp_client.py::reset_mcp_tools_cache`), e.g. in tests that rebuild
    the index under a fresh path.
    """
    from src.rag.policy_store import PolicyVectorStore

    return PolicyVectorStore(
        persist_directory,
        collection_name=collection_name,
        embedding_model=embedding_model,
    )


def reset_policy_store_cache() -> None:
    """Clear the cached PolicyVectorStore. Mainly useful for test isolation."""
    _get_policy_store.cache_clear()


def lending_policy_search(query: str, k: int = 3) -> list[dict[str, Any]]:
    """Semantically search the synthetic lending-policy corpus with Chroma.

    Sentence-Transformers creates local embeddings and Chroma performs cosine
    vector retrieval. The MCP tool interface remains unchanged for the agentic
    loop: query + k in, ranked policy clauses out.
    """
    from src.config import get_rag_config

    config = get_rag_config()
    store = _get_policy_store(
        config["persist_directory"],
        config["collection_name"],
        config["embedding_model"],
    )
    return store.search(
        query,
        k=k or config.get("default_top_k", 3),
        corpus_dir=CORPUS_DIR,
    )


def credit_policy_manual_content() -> str:
    """Returns the full synthetic credit policy manual text (used by the MCP resource)."""
    manual_path = REPO_ROOT / "mcp_server" / "resources" / "credit_policy_manual.md"
    return manual_path.read_text()
