class CalendarModule:
    def __init__(self, conflicts: list[dict]):
        self._conflicts = conflicts
        self._index = 0
        self._last_ground_truth: dict | None = None

    def current_conflict(self) -> dict:
        c = self._conflicts[self._index]
        return {
            "conflict_id": c["conflict_id"],
            "event_a": c["event_a"],
            "event_b": c["event_b"],
            "conflict_reason": c["conflict_reason"],
            "resolution_options": c["resolution_options"],
            "conflict_number": self._index + 1,
            "total_conflicts": len(self._conflicts),
        }

    def advance(self) -> None:
        self._last_ground_truth = self._conflicts[self._index]["ground_truth"]
        self._index += 1

    def done(self) -> bool:
        return self._index >= len(self._conflicts)

    def ground_truth(self) -> dict:
        if self._last_ground_truth is None:
            raise RuntimeError("No conflict has been advanced past yet.")
        return self._last_ground_truth
