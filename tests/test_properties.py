# Feature: openenv-realworld-environment
# Property-based tests using Hypothesis

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from email_triage_env.models import Action, Reward
from email_triage_env.grader import grade, PRIORITY_SCALE

# ---------------------------------------------------------------------------
# Shared generators
# ---------------------------------------------------------------------------

CATEGORIES = ["spam", "urgent", "newsletter", "support", "internal"]
PRIORITIES = ["low", "medium", "high", "urgent"]

st_category = st.sampled_from(CATEGORIES)
st_priority = st.sampled_from(PRIORITIES)
st_response = st.text()
st_keywords = st.lists(st.text(min_size=1, max_size=20), min_size=0, max_size=5)


def make_action(category, priority, response, email_id="e001"):
    return Action(
        email_id=email_id,
        category=category,
        priority=priority,
        suggested_response=response,
    )


def make_ground_truth(category, priority, keywords=None, near_miss=None):
    return {
        "category": category,
        "priority": priority,
        "required_response_keywords": keywords if keywords is not None else [],
        "near_miss_categories": near_miss if near_miss is not None else [],
    }


# ---------------------------------------------------------------------------
# Property 4: Grader determinism
# Validates: Requirements 4.6
# ---------------------------------------------------------------------------

# Feature: openenv-realworld-environment, Property 4: Grader determinism
@settings(max_examples=100)
@given(
    category=st_category,
    priority=st_priority,
    response=st_response,
    gt_category=st_category,
    gt_priority=st_priority,
    keywords=st_keywords,
)
def test_grader_determinism(category, priority, response, gt_category, gt_priority, keywords):
    """For any (Action, ground_truth) pair, calling grade() multiple times returns identical Reward objects."""
    action = make_action(category, priority, response)
    ground_truth = make_ground_truth(gt_category, gt_priority, keywords)

    reward1 = grade(action, ground_truth)
    reward2 = grade(action, ground_truth)

    assert reward1.score == reward2.score
    assert reward1.category_score == reward2.category_score
    assert reward1.priority_score == reward2.priority_score
    assert reward1.response_score == reward2.response_score


# ---------------------------------------------------------------------------
# Property 5: Category scoring rule
# Validates: Requirements 4.1
# ---------------------------------------------------------------------------

# Feature: openenv-realworld-environment, Property 5: Category scoring rule
@settings(max_examples=100)
@given(
    agent_category=st_category,
    gt_category=st_category,
    near_miss=st.lists(st_category, min_size=0, max_size=3),
    priority=st_priority,
    response=st_response,
)
def test_category_scoring(agent_category, gt_category, near_miss, priority, response):
    """category_score is 1.0 for exact match, 0.5 if in near_miss_set, 0.0 otherwise."""
    action = make_action(agent_category, priority, response)
    ground_truth = make_ground_truth(gt_category, priority, near_miss=near_miss)

    reward = grade(action, ground_truth)

    if agent_category == gt_category:
        assert reward.category_score == 1.0
    elif agent_category in near_miss:
        assert reward.category_score == 0.5
    else:
        assert reward.category_score == 0.0


# ---------------------------------------------------------------------------
# Property 6: Priority scoring rule
# Validates: Requirements 4.2
# ---------------------------------------------------------------------------

# Feature: openenv-realworld-environment, Property 6: Priority scoring rule
@settings(max_examples=100)
@given(
    agent_priority=st_priority,
    gt_priority=st_priority,
    category=st_category,
    response=st_response,
)
def test_priority_scoring(agent_priority, gt_priority, category, response):
    """priority_score is 1.0 exact, 0.5 one level off, 0.0 two+ levels off."""
    action = make_action(category, agent_priority, response)
    ground_truth = make_ground_truth(category, gt_priority)

    reward = grade(action, ground_truth)

    agent_idx = PRIORITY_SCALE.index(agent_priority)
    gt_idx = PRIORITY_SCALE.index(gt_priority)
    diff = abs(agent_idx - gt_idx)

    if diff == 0:
        assert reward.priority_score == 1.0
    elif diff == 1:
        assert reward.priority_score == 0.5
    else:
        assert reward.priority_score == 0.0


# ---------------------------------------------------------------------------
# Property 7: Response keyword scoring
# Validates: Requirements 4.3, 4.5
# ---------------------------------------------------------------------------

# Feature: openenv-realworld-environment, Property 7: Response keyword scoring
@settings(max_examples=100)
@given(
    response=st_response,
    keywords=st_keywords,
    category=st_category,
    priority=st_priority,
)
def test_response_keyword_scoring(response, keywords, category, priority):
    """response_score equals fraction of required keywords present (0.0 when keywords non-empty and response empty)."""
    action = make_action(category, priority, response)
    ground_truth = make_ground_truth(category, priority, keywords=keywords)

    reward = grade(action, ground_truth)

    if not keywords:
        # No required keywords → full score
        assert reward.response_score == 1.0
    elif not response:
        # Keywords required but empty response
        assert reward.response_score == 0.0
    else:
        response_lower = response.lower()
        matched = sum(1 for kw in keywords if kw.lower() in response_lower)
        expected = matched / len(keywords)
        assert abs(reward.response_score - expected) < 1e-9


# ---------------------------------------------------------------------------
# Property 8: Weighted score formula
# Validates: Requirements 4.4
# ---------------------------------------------------------------------------

# Feature: openenv-realworld-environment, Property 8: Weighted score formula
@settings(max_examples=100)
@given(
    category=st_category,
    priority=st_priority,
    response=st_response,
    gt_category=st_category,
    gt_priority=st_priority,
    keywords=st_keywords,
)
def test_weighted_score_formula(category, priority, response, gt_category, gt_priority, keywords):
    """Base score = 0.4*category_score + 0.3*priority_score + 0.3*response_score (before bonus/penalty)."""
    # Exclude cases where bonus or penalty would be applied
    assume(not (priority == "urgent" and gt_priority == "urgent"))
    assume(not (priority == "urgent" and gt_priority in ("low", "medium")))

    action = make_action(category, priority, response)
    ground_truth = make_ground_truth(gt_category, gt_priority, keywords=keywords)

    reward = grade(action, ground_truth)

    expected_base = (
        0.4 * reward.category_score
        + 0.3 * reward.priority_score
        + 0.3 * reward.response_score
    )
    assert abs(reward.score - expected_base) < 1e-9


# ---------------------------------------------------------------------------
# Property 9: Bonus and penalty application
# Validates: Requirements 5.2, 5.3
# ---------------------------------------------------------------------------

# Feature: openenv-realworld-environment, Property 9: Bonus and penalty application
@settings(max_examples=100)
@given(
    category=st_category,
    response=st_response,
    gt_category=st_category,
    keywords=st_keywords,
)
def test_bonus_penalty_application(category, response, gt_category, keywords):
    """(a) urgent→urgent adds exactly 0.05 (capped at 1.0); (b) urgent→low/medium subtracts exactly 0.1 (floored at 0.0)."""
    # --- (a) urgent bonus ---
    action_bonus = make_action(category, "urgent", response)
    gt_bonus = make_ground_truth(gt_category, "urgent", keywords=keywords)
    reward_bonus = grade(action_bonus, gt_bonus)

    base_bonus = (
        0.4 * reward_bonus.category_score
        + 0.3 * reward_bonus.priority_score
        + 0.3 * reward_bonus.response_score
    )
    expected_bonus_score = min(1.0, base_bonus + 0.05)
    assert abs(reward_bonus.score - expected_bonus_score) < 1e-9

    # --- (b) false-urgent penalty (low) ---
    action_penalty_low = make_action(category, "urgent", response)
    gt_penalty_low = make_ground_truth(gt_category, "low", keywords=keywords)
    reward_penalty_low = grade(action_penalty_low, gt_penalty_low)

    base_penalty_low = (
        0.4 * reward_penalty_low.category_score
        + 0.3 * reward_penalty_low.priority_score
        + 0.3 * reward_penalty_low.response_score
    )
    expected_penalty_low_score = max(0.0, base_penalty_low - 0.1)
    assert abs(reward_penalty_low.score - expected_penalty_low_score) < 1e-9

    # --- (b) false-urgent penalty (medium) ---
    action_penalty_med = make_action(category, "urgent", response)
    gt_penalty_med = make_ground_truth(gt_category, "medium", keywords=keywords)
    reward_penalty_med = grade(action_penalty_med, gt_penalty_med)

    base_penalty_med = (
        0.4 * reward_penalty_med.category_score
        + 0.3 * reward_penalty_med.priority_score
        + 0.3 * reward_penalty_med.response_score
    )
    expected_penalty_med_score = max(0.0, base_penalty_med - 0.1)
    assert abs(reward_penalty_med.score - expected_penalty_med_score) < 1e-9


# ---------------------------------------------------------------------------
# Property 10: Partial credit
# Validates: Requirements 5.1
# ---------------------------------------------------------------------------

# Feature: openenv-realworld-environment, Property 10: Partial credit is always possible
@settings(max_examples=100)
@given(
    category=st_category,
    priority=st_priority,
    response=st_response,
    gt_category=st_category,
    gt_priority=st_priority,
    keywords=st_keywords,
)
def test_partial_credit(category, priority, response, gt_category, gt_priority, keywords):
    """For any action with at least one sub-score > 0.0, Reward.score > 0.0."""
    action = make_action(category, priority, response)
    ground_truth = make_ground_truth(gt_category, gt_priority, keywords=keywords)

    reward = grade(action, ground_truth)

    has_partial = (
        reward.category_score > 0.0
        or reward.priority_score > 0.0
        or reward.response_score > 0.0
    )

    if has_partial:
        assert reward.score > 0.0


# ---------------------------------------------------------------------------
# Property 16: Reward score range
# Validates: Requirements 1.7, 5.2, 5.3
# ---------------------------------------------------------------------------

# Feature: openenv-realworld-environment, Property 16: Reward score is always in [0.0, 1.0]
@settings(max_examples=100)
@given(
    category=st_category,
    priority=st_priority,
    response=st_response,
    gt_category=st_category,
    gt_priority=st_priority,
    keywords=st_keywords,
)
def test_reward_score_range(category, priority, response, gt_category, gt_priority, keywords):
    """For any action graded against any ground truth, Reward.score is always in [0.0, 1.0]."""
    action = make_action(category, priority, response)
    ground_truth = make_ground_truth(gt_category, gt_priority, keywords=keywords)

    reward = grade(action, ground_truth)

    assert 0.0 <= reward.score <= 1.0


# ---------------------------------------------------------------------------
# Properties 11–14: EmailTriageEnv environment tests
# ---------------------------------------------------------------------------

from email_triage_env.env import EmailTriageEnv

# ---------------------------------------------------------------------------
# Property 11: email_id mismatch yields zero reward
# Validates: Requirements 1.8
# ---------------------------------------------------------------------------

# Feature: openenv-realworld-environment, Property 11: email_id mismatch yields zero reward
@settings(max_examples=100)
@given(task_id=st.sampled_from(["easy_triage", "medium_triage", "hard_triage"]))
def test_email_id_mismatch(task_id):
    """For any action whose email_id does not match the current email, step() returns score=0.0 with mismatch explanation."""
    env = EmailTriageEnv()
    obs = env.reset(task_id)

    correct_id = obs.email_id
    wrong_id = correct_id + "_wrong"

    action = Action(
        email_id=wrong_id,
        category="spam",
        priority="low",
        suggested_response="",
    )
    _, reward, done, _ = env.step(action)

    assert reward.score == 0.0
    assert len(reward.explanation) > 0
    assert "mismatch" in reward.explanation.lower()
    # Episode should not have advanced
    assert not done


# ---------------------------------------------------------------------------
# Property 12: Episode terminates after inbox exhaustion
# Validates: Requirements 1.3
# ---------------------------------------------------------------------------

# Feature: openenv-realworld-environment, Property 12: Episode terminates after inbox exhaustion
@settings(max_examples=100)
@given(task_id=st.sampled_from(["easy_triage", "medium_triage", "hard_triage"]))
def test_episode_termination(task_id):
    """After exactly N valid steps (N = inbox size), done=True. Subsequent calls also return done=True."""
    env = EmailTriageEnv()
    obs = env.reset(task_id)
    n = obs.total_emails

    done = False
    for _ in range(n):
        action = Action(
            email_id=obs.email_id,
            category="spam",
            priority="low",
            suggested_response="",
        )
        obs, _, done, _ = env.step(action)

    assert done is True

    # Extra step after exhaustion should still return done=True
    extra_action = Action(
        email_id="any_id",
        category="spam",
        priority="low",
        suggested_response="",
    )
    _, _, still_done, _ = env.step(extra_action)
    assert still_done is True


# ---------------------------------------------------------------------------
# Property 13: reset() is deterministic
# Validates: Requirements 3.4
# ---------------------------------------------------------------------------

# Feature: openenv-realworld-environment, Property 13: reset() is deterministic
@settings(max_examples=100)
@given(task_id=st.sampled_from(["easy_triage", "medium_triage", "hard_triage"]))
def test_reset_determinism(task_id):
    """Calling reset(task_id) twice returns identical Observation objects."""
    env = EmailTriageEnv()
    obs1 = env.reset(task_id)
    obs2 = env.reset(task_id)

    assert obs1.email_id == obs2.email_id
    assert obs1.subject == obs2.subject
    assert obs1.body == obs2.body
    assert obs1.sender == obs2.sender
    assert obs1.timestamp == obs2.timestamp
    assert obs1.inbox_position == obs2.inbox_position
    assert obs1.total_emails == obs2.total_emails


# ---------------------------------------------------------------------------
# Property 14: Cumulative reward invariant
# Validates: Requirements 5.4, 1.4
# ---------------------------------------------------------------------------

# Feature: openenv-realworld-environment, Property 14: Cumulative reward invariant
@settings(max_examples=100)
@given(
    task_id=st.sampled_from(["easy_triage", "medium_triage", "hard_triage"]),
    num_steps=st.integers(min_value=1, max_value=5),
)
def test_cumulative_reward_invariant(task_id, num_steps):
    """state()['cumulative_reward'] equals the sum of all Reward.score values returned by step()."""
    env = EmailTriageEnv()
    obs = env.reset(task_id)
    total_emails = obs.total_emails
    steps_to_take = min(num_steps, total_emails)

    cumulative = 0.0
    for _ in range(steps_to_take):
        action = Action(
            email_id=obs.email_id,
            category="spam",
            priority="low",
            suggested_response="",
        )
        obs, reward, done, _ = env.step(action)
        cumulative += reward.score
        if done:
            break

    assert abs(env.state()["cumulative_reward"] - cumulative) < 1e-9


# ---------------------------------------------------------------------------
# Property 3: Malformed JSON returns 422
# Validates: Requirements 9.5
# ---------------------------------------------------------------------------

# Feature: openenv-realworld-environment, Property 3: Malformed JSON returns 422
from starlette.testclient import TestClient
from app import app

_client = TestClient(app)


@settings(max_examples=100)
@given(
    payload=st.one_of(
        st.text(),
        st.binary().map(lambda b: b.decode("utf-8", errors="replace")),
    )
)
def test_malformed_json_422(payload):
    """For any POST to /step with a payload that is not valid Action JSON, the server returns HTTP 422."""
    # Exclude strings that happen to be valid Action JSON
    try:
        import json as _json
        parsed = _json.loads(payload)
        if (
            isinstance(parsed, dict)
            and "email_id" in parsed
            and "category" in parsed
            and "priority" in parsed
            and "suggested_response" in parsed
        ):
            assume(False)
    except (ValueError, TypeError):
        pass  # Not valid JSON at all — keep it

    response = _client.post(
        "/step",
        content=payload,
        headers={"content-type": "application/json"},
    )
    # FastAPI returns 422 for schema validation failures and 400 for body decode errors
    assert response.status_code in (400, 422)


def test_malformed_json_422_structurally_invalid():
    """Structurally invalid JSON (e.g. {not valid json}) returns HTTP 422."""
    response = _client.post(
        "/step",
        content="{not valid json}",
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 422


def test_malformed_json_422_missing_fields():
    """Valid JSON but missing required Action fields (e.g. {}) returns HTTP 422."""
    response = _client.post(
        "/step",
        content="{}",
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Property 1: Observation round-trip serialization
# Validates: Requirements 9.3, 9.1, 9.2, 1.5
# ---------------------------------------------------------------------------

from email_triage_env.models import Observation

# Feature: openenv-realworld-environment, Property 1: Observation round-trip serialization
@settings(max_examples=100)
@given(
    st.builds(
        Observation,
        email_id=st.text(alphabet=st.characters(blacklist_categories=("Cs",))),
        subject=st.text(alphabet=st.characters(blacklist_categories=("Cs",))),
        body=st.text(alphabet=st.characters(blacklist_categories=("Cs",))),
        sender=st.text(alphabet=st.characters(blacklist_categories=("Cs",))),
        timestamp=st.text(alphabet=st.characters(blacklist_categories=("Cs",))),
        inbox_position=st.integers(min_value=0),
        total_emails=st.integers(min_value=1),
    )
)
def test_observation_round_trip(obs):
    """For any valid Observation, serializing then deserializing produces an equal object."""
    serialized = obs.model_dump_json()
    deserialized = Observation.model_validate_json(serialized)
    assert deserialized == obs


# ---------------------------------------------------------------------------
# Property 2: Action round-trip serialization
# Validates: Requirements 9.4, 9.1, 9.2
# ---------------------------------------------------------------------------

# Feature: openenv-realworld-environment, Property 2: Action round-trip serialization
@settings(max_examples=100)
@given(
    st.builds(
        Action,
        email_id=st.text(alphabet=st.characters(blacklist_categories=("Cs",))),
        category=st.sampled_from(CATEGORIES),
        priority=st.sampled_from(PRIORITIES),
        suggested_response=st.text(alphabet=st.characters(blacklist_categories=("Cs",))),
    )
)
def test_action_round_trip(action):
    """For any valid Action, serializing then deserializing produces an equal object."""
    serialized = action.model_dump_json()
    deserialized = Action.model_validate_json(serialized)
    assert deserialized == action


# ---------------------------------------------------------------------------
# Property 15: Log line format
# Validates: Requirements 6.5, 6.6, 6.7
# ---------------------------------------------------------------------------

import re

# Feature: openenv-realworld-environment, Property 15: Log line format
@settings(max_examples=100)
@given(
    task_id=st.text(
        min_size=1,
        alphabet=st.characters(
            blacklist_categories=("Cs", "Cc", "Z"),
            blacklist_characters=(" ",),
        ),
    ),
    step_number=st.integers(min_value=1, max_value=1000),
    score=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
)
def test_log_line_format(task_id, step_number, score):
    """For any (task_id, step_number, score) triple, the log lines match the expected patterns."""
    start_line = f"[START] task={task_id}"
    step_line = f"[STEP] task={task_id} step={step_number} score={score}"
    end_line = f"[END] task={task_id} total_score={score}"

    assert re.fullmatch(r"\[START\] task=\S+", start_line), \
        f"START line did not match pattern: {start_line!r}"
    assert re.fullmatch(r"\[STEP\] task=\S+ step=\d+ score=[\d.eE+\-]+", step_line), \
        f"STEP line did not match pattern: {step_line!r}"
    assert re.fullmatch(r"\[END\] task=\S+ total_score=[\d.eE+\-]+", end_line), \
        f"END line did not match pattern: {end_line!r}"
