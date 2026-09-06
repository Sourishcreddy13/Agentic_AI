from datetime import datetime, timedelta, timezone

from src.memory.eviction import (
    EVICTION_THRESHOLD,
    HARD_TTL_DAYS,
    importance_for_fact_type,
    is_expired,
    retention_score,
    select_evictions,
)
from src.state.schema import MemoryFact


def make_fact(fact_id: str, fact_type: str, days_old: int) -> MemoryFact:
    ts = (datetime.now(timezone.utc) - timedelta(days=days_old)).isoformat()
    return MemoryFact(
        fact_id=fact_id,
        user_id="EV-USER",
        fact_type=fact_type,
        value=f"value-{fact_id}",
        importance=importance_for_fact_type(fact_type),
        session_ts=ts,
        thread_id="EV-THREAD",
        usage_count=0,
    )


def test_importance_scores_match_policy_table():
    assert importance_for_fact_type("kyc_outcome") == 0.95
    assert importance_for_fact_type("credit_outcome") == 0.90
    assert importance_for_fact_type("preferred_term") == 0.40


def test_old_fact_hits_hard_ttl():
    fact = make_fact("old", "employment", HARD_TTL_DAYS + 1)
    assert is_expired(fact) is True


def test_low_retention_fact_is_evicted():
    fact = make_fact("weak", "preferred_term", 180)
    assert retention_score(fact, access_count=0) < EVICTION_THRESHOLD
    assert fact in select_evictions([fact], access_counts={fact.fact_id: 0})


def test_high_importance_recently_used_fact_is_retained():
    fact = make_fact("strong", "kyc_outcome", 1)
    assert retention_score(fact, access_count=10) > EVICTION_THRESHOLD
    assert fact not in select_evictions([fact], access_counts={fact.fact_id: 10})
