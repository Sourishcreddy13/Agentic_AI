import json
from pathlib import Path

from langchain_core.messages import HumanMessage

from src.state.schema import new_state

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "sample_inputs"


def load_initial_state(
    sample_filename: str,
    thread_id: str = "test-thread",
    user_id: str = "test-user",
):
    raw = (SAMPLE_DIR / sample_filename).read_text()
    state = new_state(thread_id=thread_id, user_id=user_id)
    state["messages"] = [HumanMessage(content=raw)]
    return state
