"""
Custom MCP server for the loan-origination copilot (AC-09).

Exposes:
  Tools:    applicant_lookup, bureau_check, lending_policy_search
  Resource: policy://credit_policy_manual

Runs over stdio transport (langchain-mcp-adapters spawns this as a subprocess).
Tool logic lives in mcp_server/tools.py so it can be unit-tested independently.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make the repo root importable when the server is spawned as a subprocess.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mcp.server.fastmcp import FastMCP

from mcp_server.tools import (
    applicant_lookup as _applicant_lookup,
    bureau_check as _bureau_check,
    lending_policy_search as _lending_policy_search,
    credit_policy_manual_content,
)

mcp = FastMCP(
    "loan-origination-mcp",
    instructions=(
        "This server provides tools for loan origination: applicant history lookup, "
        "synthetic bureau checks, and lending policy search. All data is synthetic."
    ),
)


# --------------------------------------------------------------------------
# Tools
# --------------------------------------------------------------------------

@mcp.tool()
def applicant_lookup(applicant_id: str) -> dict:
    """
    Look up synthetic applicant history by applicant_id.
    Returns prior application count and synthetic watchlist flag.
    No real PII — all data is synthetic.
    """
    return _applicant_lookup(applicant_id)


@mcp.tool()
def bureau_check(applicant_id: str, declared_income: float) -> dict:
    """
    Perform a simulated bureau check.
    Returns synthetic credit score, delinquency count, thin-file flag, and DTI estimate.
    No real credit bureau is called — all scores are deterministic synthetic values.
    """
    return _bureau_check(applicant_id, declared_income)


@mcp.tool()
def lending_policy_search(query: str, k: int = 3) -> list:
    """
    Search the lending policy corpus for clauses relevant to the query.
    Use this tool to look up eligibility rules, DTI limits, KYC requirements,
    thin-file policy, or pricing tiers before making a credit decision.
    Returns the top-k matching policy clauses with relevance scores.
    """
    return _lending_policy_search(query, k)


# --------------------------------------------------------------------------
# Resource
# --------------------------------------------------------------------------

@mcp.resource("policy://credit_policy_manual")
def credit_policy_manual() -> str:
    """Full text of the synthetic credit policy manual."""
    return credit_policy_manual_content()


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run()   # stdio transport by default
