class CalendarGrader:
    def score(self, action: dict, ground_truth: dict, conflict: dict | None = None) -> tuple[float, dict]:
        # --- resolution_score ---
        if action.get("resolution") == ground_truth["correct_resolution"]:
            resolution_score = 1.0
        elif action.get("resolution") in ground_truth.get("acceptable_resolutions", []):
            resolution_score = 0.5
        else:
            resolution_score = 0.0

        # --- vip_protection_score ---
        vip_protection_score = 1.0
        if conflict is not None:
            resolution = action.get("resolution", "")
            event_a = conflict.get("event_a", {})
            event_b = conflict.get("event_b", {})

            cancelled_event = None
            other_event = None
            if resolution in ("cancel_a",):
                cancelled_event, other_event = event_a, event_b
            elif resolution in ("cancel_b",):
                cancelled_event, other_event = event_b, event_a

            if cancelled_event is not None and cancelled_event.get("vip", False):
                # Check if a non-VIP alternative existed
                if other_event is not None and not other_event.get("vip", False):
                    vip_protection_score = max(0.0, vip_protection_score - 1.0)
                else:
                    vip_protection_score = max(0.0, vip_protection_score - 0.5)

        # --- rationale_score ---
        rationale = (action.get("rationale") or "").strip()
        word_count = len(rationale.split()) if rationale else 0
        if word_count >= 5:
            rationale_score = 1.0
        elif word_count >= 1:
            rationale_score = 0.5
        else:
            rationale_score = 0.0

        # --- final score ---
        final = 0.5 * resolution_score + 0.3 * vip_protection_score + 0.2 * rationale_score
        final = max(0.0, min(1.0, final))

        return final, {
            "resolution_score": resolution_score,
            "vip_protection": vip_protection_score,
            "rationale_score": rationale_score,
        }
