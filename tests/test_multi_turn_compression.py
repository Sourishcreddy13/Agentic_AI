"""AC/NFR-08 reachability proof: compression fires via the real graph.

`tests/test_context_engineering.py` proves `summarize_if_long()` and
`prepare_worker_context()` behave correctly in isolation, against a
hand-built message list constructed directly in the test. That is a
correctness test, not a reachability proof: it does not show that 20+
messages can ever actually accumulate on a thread during real use of the
shipped graph, checkpointer, and CLI entry point.

This test drives the same production call path `cli.py` uses --
`build_graph()` + `get_checkpointer()` + `cli._build_initial_state()` --
repeatedly against the *same* thread_id, exactly as a returning applicant
reusing a conversation thread across multiple sessions/applications would.
Only the LLM and RAG calls are stubbed (the same doubles the rest of the
hermetic suite uses via the autouse `deterministic_phase_models` fixture);
the graph, routing, checkpointing, and `src/context/compress.py` itself all
run for real.

Why messages accumulate across runs while decisions don't carry over:
`messages` is an `Annotated[..., add]` (accumulating) channel, so each new
invocation's initial HumanMessage is appended to -- not replacing -- the
thread's prior history, and each worker's own `AIMessage` (see
intake/kyc/credit/offer agents) adds to it too. `applicant`, `kyc_result`,
`credit_assessment`, `offer`, `retry_count`, and `reflection_log` are plain
(last-value) channels, so a fresh `new_state()`-shaped input genuinely
starts a new application on the same thread rather than being blocked by
the prior one's completed decision -- which is exactly the resume-aware
supervisor's job (see src/graph/supervisor.py, src/graph/routing.py).
"""
from __future__ import annotations

import json
from pathlib import Path

import cli
from src.graph.build_graph import build_graph
from src.memory.checkpointer import get_checkpointer

SAMPLE_INPUT = Path(__file__).resolve().parent.parent / "sample_inputs" / "applicant_strong.json"


def test_compression_is_reached_by_repeated_real_graph_runs(tmp_path):
    db_path = tmp_path / "multi_turn_checkpoints.sqlite"
    thread_id = "multi-turn-reachability-thread"
    payload = json.loads(SAMPLE_INPUT.read_text(encoding="utf-8"))

    message_counts: list[int] = []
    compressed_after_run: list[bool] = []

    with get_checkpointer(db_path) as checkpointer:
        graph = build_graph(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": thread_id, "memory_enabled": False}}

        # Five separate "sessions" on the same thread -- e.g. the same
        # applicant returning to reapply, or a resubmission after an
        # earlier decision -- each driven through cli._build_initial_state,
        # the exact function cli.main() uses per invocation.
        for _ in range(5):
            state = cli._build_initial_state(payload, thread_id=thread_id, user_id="reachability-user")
            for _update in graph.stream(state, config=config, stream_mode="updates"):
                pass

            snapshot = graph.get_state(config)
            values = snapshot.values
            message_counts.append(len(values.get("messages", [])))
            compressed_after_run.append(values.get("compressed_summary") is not None)

        # Captured while the checkpointer connection is still open.
        final_values = graph.get_state(config).values

    # Messages genuinely accumulate run over run (the accumulating channel).
    assert message_counts == sorted(message_counts)
    assert message_counts[-1] > message_counts[0]

    # Below NFR-08's threshold (20), compression has not fired.
    assert all(
        not compressed
        for count, compressed in zip(message_counts, compressed_after_run)
        if count <= 20
    )

    # Once real accumulated history crosses the threshold, the real
    # summarize_if_long() path (not a monkeypatched stand-in) has fired and
    # left a real summary on checkpointed state.
    assert any(compressed_after_run), (
        f"compression never fired across 5 runs; message_counts={message_counts}"
    )
    crossing_index = next(
        i for i, count in enumerate(message_counts) if count > 20
    )
    assert compressed_after_run[crossing_index] is True

    assert isinstance(final_values.get("compressed_summary"), str)
    assert len(final_values["compressed_summary"]) > 0
    # The applicant's most recent (5th) application still completed on the
    # same thread -- compression optimizes context, it never blocks or
    # alters the decision pipeline.
    assert final_values.get("offer") is not None

    evidence_dir = Path("evidence")
    evidence_dir.mkdir(parents=True, exist_ok=True)
    (evidence_dir / "multi_turn_compression.json").write_text(
        json.dumps(
            {
                "ac_reference": "NFR-08",
                "scenario": (
                    "Real graph + real SqliteSaver checkpointer + real "
                    "src.context.compress.summarize_if_long, driven through "
                    "the same thread_id across 5 separate cli-style runs, "
                    "showing the 20-message compression threshold is "
                    "actually reachable in normal multi-session use, not "
                    "only in an isolated unit test."
                ),
                "message_counts_per_run": message_counts,
                "compressed_after_run": compressed_after_run,
                "final_offer_present": final_values.get("offer") is not None,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
