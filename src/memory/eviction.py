"""Importance-weighted + TTL memory eviction policy."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable

from src.state.schema import MemoryFact

WEIGHT_IMPORTANCE = 0.45
WEIGHT_RECENCY = 0.30
WEIGHT_USAGE = 0.25
HARD_TTL_DAYS = 365
EVICTION_THRESHOLD = 0.25
MAX_FACTS_PER_USER = 50

IMPORTANCE_BY_TYPE = {
    "kyc_outcome": 0.95,
    "credit_outcome": 0.90,
    "prior_application_count": 0.70,
    "declared_income_band": 0.60,
    "employment": 0.55,
    "preferred_term": 0.40,
}


def importance_for_fact_type(fact_type: str) -> float:
    return IMPORTANCE_BY_TYPE.get(fact_type, 0.40)


def recency_score(session_ts: str, now: datetime | None = None) -> float:
    now = now or datetime.now(timezone.utc)
    created = datetime.fromisoformat(session_ts.replace("Z", "+00:00"))
    age_days = max(0.0, (now - created).total_seconds() / 86400)
    if age_days < 30:
        return 1.0
    if age_days >= 180:
        return 0.0
    return (180.0 - age_days) / 150.0


def usage_score(access_count: int) -> float:
    return min(max(access_count, 0) / 10.0, 1.0)


def retention_score(
    fact: MemoryFact,
    *,
    access_count: int = 0,
    now: datetime | None = None,
) -> float:
    return (
        WEIGHT_IMPORTANCE * fact.importance
        + WEIGHT_RECENCY * recency_score(fact.session_ts, now)
        + WEIGHT_USAGE * usage_score(access_count)
    )


def is_expired(fact: MemoryFact, now: datetime | None = None) -> bool:
    now = now or datetime.now(timezone.utc)
    created = datetime.fromisoformat(fact.session_ts.replace("Z", "+00:00"))
    age_days = max(0.0, (now - created).total_seconds() / 86400)
    return age_days > HARD_TTL_DAYS


def rank_for_eviction(
    facts: Iterable[MemoryFact],
    *,
    access_counts: dict[str, int] | None = None,
    now: datetime | None = None,
) -> list[MemoryFact]:
    """Return facts from least worth retaining to most worth retaining."""
    access_counts = access_counts or {}
    return sorted(
        facts,
        key=lambda fact: (
            retention_score(
                fact,
                access_count=access_counts.get(fact.fact_id, 0),
                now=now,
            ),
            fact.last_access_ts or fact.session_ts,
        ),
    )


def select_evictions(
    facts: list[MemoryFact],
    *,
    access_counts: dict[str, int] | None = None,
    now: datetime | None = None,
) -> list[MemoryFact]:
    """Select facts for expiry, low retention, or capacity pressure."""
    access_counts = access_counts or {}
    now = now or datetime.now(timezone.utc)

    evict_ids = {
        fact.fact_id
        for fact in facts
        if is_expired(fact, now)
        or retention_score(
            fact,
            access_count=access_counts.get(fact.fact_id, 0),
            now=now,
        ) < EVICTION_THRESHOLD
    }

    remaining = [fact for fact in facts if fact.fact_id not in evict_ids]
    if len(remaining) > MAX_FACTS_PER_USER:
        overflow = len(remaining) - MAX_FACTS_PER_USER
        capacity_order = sorted(
            remaining,
            key=lambda fact: (
                retention_score(
                    fact,
                    access_count=access_counts.get(fact.fact_id, 0),
                    now=now,
                ),
                fact.last_access_ts or fact.session_ts,
            ),
        )
        evict_ids.update(fact.fact_id for fact in capacity_order[:overflow])

    return [fact for fact in facts if fact.fact_id in evict_ids]
