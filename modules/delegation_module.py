class DelegationModule:
    def __init__(self, tasks: list[dict]):
        self._tasks = tasks
        self._index = 0
        self._last_ground_truth: dict | None = None

    def current_task(self) -> dict:
        t = self._tasks[self._index]
        return {
            "task_id": t["task_id"],
            "title": t["title"],
            "source_email_id": t["source_email_id"],
            "urgency": t["urgency"],
            "requires_technical": t["requires_technical"],
            "task_number": self._index + 1,
            "total_tasks": len(self._tasks),
        }

    def advance(self) -> None:
        self._last_ground_truth = self._tasks[self._index]["ground_truth"]
        self._index += 1

    def done(self) -> bool:
        return self._index >= len(self._tasks)

    def ground_truth(self) -> dict:
        if self._last_ground_truth is None:
            raise RuntimeError("No task has been advanced past yet.")
        return self._last_ground_truth
