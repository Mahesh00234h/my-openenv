from email_triage_env.models import Action, Reward

PRIORITY_SCALE = ["low", "medium", "high", "urgent"]


def grade(action: Action, ground_truth: dict) -> Reward:
    """Deterministic grader: scores an Action against ground_truth labels."""

    # --- category_score ---
    gt_category = ground_truth["category"]
    near_miss = ground_truth.get("near_miss_categories", [])
    if action.category == gt_category:
        category_score = 1.0
    elif action.category in near_miss:
        category_score = 0.5
    else:
        category_score = 0.0

    # --- priority_score ---
    gt_priority = ground_truth["priority"]
    try:
        agent_idx = PRIORITY_SCALE.index(action.priority)
        gt_idx = PRIORITY_SCALE.index(gt_priority)
        diff = abs(agent_idx - gt_idx)
    except ValueError:
        diff = 2  # unknown priority treated as 2+ levels off
    if diff == 0:
        priority_score = 1.0
    elif diff == 1:
        priority_score = 0.5
    else:
        priority_score = 0.0

    # --- response_score ---
    required_keywords = ground_truth.get("required_response_keywords", [])
    if not required_keywords:
        response_score = 1.0
    elif not action.suggested_response:
        response_score = 0.0
    else:
        response_lower = action.suggested_response.lower()
        matched = sum(1 for kw in required_keywords if kw.lower() in response_lower)
        response_score = matched / len(required_keywords)

    # --- base weighted score ---
    base_score = 0.4 * category_score + 0.3 * priority_score + 0.3 * response_score

    # --- bonus / penalty ---
    bonus_applied = False
    penalty_applied = False

    if action.priority == "urgent" and gt_priority == "urgent":
        base_score = min(1.0, base_score + 0.05)
        bonus_applied = True
    elif action.priority == "urgent" and gt_priority in ("low", "medium"):
        base_score = max(0.0, base_score - 0.1)
        penalty_applied = True

    final_score = base_score

    # --- explanation ---
    parts = [
        f"category_score={category_score:.2f}",
        f"priority_score={priority_score:.2f}",
        f"response_score={response_score:.2f}",
        f"base=0.4*{category_score:.2f}+0.3*{priority_score:.2f}+0.3*{response_score:.2f}",
    ]
    if bonus_applied:
        parts.append("urgent bonus +0.05 applied (capped at 1.0)")
    if penalty_applied:
        parts.append("false-urgent penalty -0.1 applied (floored at 0.0)")
    explanation = "; ".join(parts)

    return Reward(
        score=final_score,
        category_score=category_score,
        priority_score=priority_score,
        response_score=response_score,
        explanation=explanation,
    )
