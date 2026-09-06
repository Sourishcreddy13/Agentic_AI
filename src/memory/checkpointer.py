"""SQLite checkpointer factory for AC-05."""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from langgraph.checkpoint.sqlite import SqliteSaver

from src.config import CONFIG, PROJECT_ROOT


def default_checkpoint_path() -> Path:
    configured = CONFIG.get("memory", {}).get("checkpoint_db", "data/checkpoints.sqlite")
    path = Path(configured)
    return path if path.is_absolute() else PROJECT_ROOT / path


@contextmanager
def get_checkpointer(db_path: str | Path | None = None) -> Iterator[SqliteSaver]:
    """Open, initialize, and close a synchronous SQLite checkpointer."""
    path = Path(db_path) if db_path is not None else default_checkpoint_path()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)

    with SqliteSaver.from_conn_string(str(path)) as saver:
        saver.setup()
        yield saver
