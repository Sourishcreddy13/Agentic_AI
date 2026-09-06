"""
AC-09: Custom MCP server exposes ≥ 2 tools and 1 resource.

These tests import from mcp_server.tools directly — no subprocess or
transport involved — so they run fast and offline.
"""
from src import config as app_config
import pytest

from mcp_server.tools import (
    applicant_lookup,
    bureau_check,
    lending_policy_search,
    credit_policy_manual_content,
)


@pytest.fixture
def isolated_rag(monkeypatch, tmp_path):
    """Use a per-test Chroma index for policy-retrieval tests."""
    config = dict(app_config.get_rag_config())
    config["persist_directory"] = str(tmp_path / "chroma_rag")
    monkeypatch.setattr(app_config, "get_rag_config", lambda: config)
    return config


# ---------- applicant_lookup ----------

def test_applicant_lookup_known_applicant():
    result = applicant_lookup("SYN-0001")
    assert result["found"] is True
    assert result["applicant_id"] == "SYN-0001"
    assert isinstance(result["prior_applications"], int)
    assert isinstance(result["synthetic_watchlist_hit"], bool)


def test_applicant_lookup_watchlist_hit():
    result = applicant_lookup("SYN-0003")
    assert result["found"] is True
    assert result["synthetic_watchlist_hit"] is True


def test_applicant_lookup_unknown_applicant_returns_not_found():
    result = applicant_lookup("SYN-9999")
    assert result["found"] is False
    assert result["prior_applications"] == 0


# ---------- bureau_check ----------

def test_bureau_check_strong_applicant():
    result = bureau_check("SYN-0001", 85000)
    assert result["found_in_bureau"] is True
    assert 300 <= result["synthetic_score"] <= 900
    assert result["thin_file"] is False
    assert result["delinquencies_24m"] == 0


def test_bureau_check_thin_file_applicant():
    result = bureau_check("SYN-0002", 22000)
    assert result["thin_file"] is True
    assert result["synthetic_score"] < 500


def test_bureau_check_unknown_applicant_returns_thin_file():
    result = bureau_check("SYN-UNKNOWN", 30000)
    assert result["found_in_bureau"] is False
    assert result["thin_file"] is True


# ---------- lending_policy_search (AC-09 + AC-11 unit-level) ----------

def test_lending_policy_search_returns_results(isolated_rag):
    results = lending_policy_search("thin file manual underwriting", k=3)
    assert len(results) >= 1
    assert all("clause_id" in r for r in results)
    assert all("score" in r for r in results)


def test_lending_policy_search_scores_are_sorted_descending(isolated_rag):
    results = lending_policy_search("income eligibility minimum", k=5)
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_lending_policy_search_dti_query_finds_dti_doc(isolated_rag):
    results = lending_policy_search("debt to income ratio DTI limit", k=3)
    clause_ids = [r["clause_id"] for r in results]
    assert any("dti" in cid.lower() for cid in clause_ids)


def test_lending_policy_search_kyc_query_finds_kyc_doc(isolated_rag):
    results = lending_policy_search("KYC watchlist fail decline", k=3)
    clause_ids = [r["clause_id"] for r in results]
    assert any("kyc" in cid.lower() for cid in clause_ids)




def test_lending_policy_search_uses_semantic_similarity(isolated_rag):
    """Phase 5: semantically related wording still retrieves the DTI policy."""
    results = lending_policy_search(
        "monthly debt obligations relative to gross monthly earnings",
        k=3,
    )
    assert results
    assert results[0]["policy_id"] == "DTI-001"
    assert results[0]["clause_id"].startswith("policy_02_dti_ratios::")
    assert 0.0 <= results[0]["score"] <= 1.0

def test_lending_policy_search_returns_clause_level_metadata(isolated_rag):
    results = lending_policy_search("watchlist screening and KYC fail", k=3)
    assert results
    assert all("::" in r["clause_id"] for r in results)
    assert all(r["policy_id"] == "KYC-001" for r in results[:1])


def test_lending_policy_search_respects_k_limit(isolated_rag):
    results = lending_policy_search("loan approval", k=2)
    assert len(results) <= 2


# ---------- resource ----------

def test_credit_policy_manual_resource_returns_content():
    content = credit_policy_manual_content()
    assert len(content) > 100
    assert "eligib" in content.lower() or "kyc" in content.lower()
