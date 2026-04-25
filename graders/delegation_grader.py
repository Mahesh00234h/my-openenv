class DelegationGrader:
    def score(self, action: dict, ground_truth: dict) -> tuple[float, dict]:
        assignee = action.get("assignee", "")

        # --- assignee_score ---
        if assignee == ground_truth["correct_assignee"]:
            assignee_score = 1.0
        elif assignee in ground_truth.get("acceptable_assignees", []):
            assignee_score = 0.5
        else:
            assignee_score = 0.0

        # Special penalty: drop on urgent task
        if assignee == "drop" and ground_truth.get("urgency") == "urgent":
            assignee_score = max(0.0, assignee_score - 0.3)

        # --- message_quality_score ---
        required_keywords = ground_truth.get("required_message_keywords", [])
        message = (action.get("delegation_message") or "").lower()
        if not required_keywords:
            message_quality_score = 1.0
        elif not message:
            message_quality_score = 0.0
        else:
            matched = sum(1 for kw in required_keywords if kw.lower() in message)
            message_quality_score = matched / len(required_keywords)

        # --- escalation_appropriateness ---
        requires_technical = ground_truth.get("requires_technical", False)
        if requires_technical and assignee in ("manager", "external"):
            escalation_score = 1.0
        elif not requires_technical and assignee in ("self", "junior"):
            escalation_score = 1.0
        else:
            escalation_score = 0.5

        # --- final score ---
        final = 0.5 * assignee_score + 0.3 * message_quality_score + 0.2 * escalation_score
        final = max(0.0, min(1.0, final))

        return final, {
            "assignee_score": assignee_score,
            "message_score": message_quality_score,
            "escalation_score": escalation_score,
        }
