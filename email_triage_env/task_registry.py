import json
from pathlib import Path

_TASKS_DIR = Path(__file__).parent.parent / "tasks"


def load_task(task_id: str) -> dict:
    fixture_path = _TASKS_DIR / f"{task_id}.json"
    if not fixture_path.exists():
        raise ValueError(f"Unknown task: {task_id}")
    return json.loads(fixture_path.read_text())
