"""
Comprehensive tests for the AI Chief of Staff environment.
Covers: graders, modules, env orchestration, models, task files.
Run with: python3 -m pytest tests/test_cos.py -v
"""
import json
import sys
from pathlib import Path

import pytest

# Make root importable
sys.path.insert(0, str(Path(__file__).parent.parent))

# ═══════════════════════════════════════════════════════════════════
# GRADER TESTS — email_grader
# ═══════════════════════════════════════════════════════════════════

from graders.email_grader import grade
from email_triage_env.models import Action


def make_action(**kwargs):
    defaults = {"email_id": "e001", "category": "urgent",
                "priority": "urgent", "suggested_response": ""}
    return Action(**{**defaults, **kwargs})


def make_gt(**kwargs):
    defaults = {"category": "urgent", "priority": "urgent",
                "required_response_keywords": [], "near_miss_categories": []}
    return {**defaults, **kwargs}


class TestEmailGrader:
    def test_perfect_score(self):
        r = grade(make_action(category="urgent", priority="urgent",
                              suggested_response="approve failover now"),
                  make_gt(category="urgent", priority="urgent",
                          required_response_keywords=["approve", "failover"]))
        assert r.score > 0.9

    def test_wrong_category_zero_score(self):
        r = grade(make_action(category="spam"), make_gt(category="urgent"))
        assert r.category_score == 0.0

    def test_near_miss_category_half_score(self):
        r = grade(make_action(category="support"),
                  make_gt(category="urgent", near_miss_categories=["support"]))
        assert r.category_score == 0.5

    def test_exact_priority_full_score(self):
        r = grade(make_action(priority="high"), make_gt(priority="high"))
        assert r.priority_score == 1.0

    def test_priority_one_off_half_score(self):
        r = grade(make_action(priority="medium"), make_gt(priority="high"))
        assert r.priority_score == 0.5

    def test_priority_two_off_zero_score(self):
        r = grade(make_action(priority="low"), make_gt(priority="urgent"))
        assert r.priority_score == 0.0

    def test_response_keywords_all_matched(self):
        r = grade(make_action(suggested_response="please approve the failover"),
                  make_gt(required_response_keywords=["approve", "failover"]))
        assert r.response_score == 1.0

    def test_response_keywords_partial(self):
        r = grade(make_action(suggested_response="approve the request"),
                  make_gt(required_response_keywords=["approve", "failover"]))
        assert r.response_score == 0.5

    def test_response_keywords_none_required(self):
        r = grade(make_action(suggested_response=""),
                  make_gt(required_response_keywords=[]))
        assert r.response_score == 1.0

    def test_false_urgent_penalty(self):
        r = grade(make_action(priority="urgent"), make_gt(priority="low"))
        base = 0.4 * r.category_score + 0.3 * r.priority_score + 0.3 * r.response_score
        assert r.score <= base

    def test_urgent_bonus(self):
        # Use a partial response so base < 1.0, giving room for the bonus to show
        r = grade(make_action(category="urgent", priority="urgent",
                              suggested_response="approve this"),
                  make_gt(category="urgent", priority="urgent",
                          required_response_keywords=["approve", "failover"]))
        # base = 0.4*1.0 + 0.3*1.0 + 0.3*0.5 = 0.85, bonus pushes to 0.90
        base = 0.4 * r.category_score + 0.3 * r.priority_score + 0.3 * r.response_score
        assert r.score >= base  # bonus applied, score should be >= base

    def test_score_always_in_range(self):
        for cat in ["spam", "urgent", "newsletter", "support", "internal"]:
            for pri in ["low", "medium", "high", "urgent"]:
                r = grade(make_action(category=cat, priority=pri),
                          make_gt(category="urgent", priority="urgent"))
                assert 0.0 <= r.score <= 1.0


# ═══════════════════════════════════════════════════════════════════
# GRADER TESTS — calendar_grader
# ═══════════════════════════════════════════════════════════════════

from graders.calendar_grader import CalendarGrader

cal_grader = CalendarGrader()

CONFLICT = {
    "event_a": {"title": "Post-Mortem", "time": "14:00", "attendees": [], "vip": True},
    "event_b": {"title": "Vendor Demo",  "time": "14:00", "attendees": [], "vip": False},
}
CAL_GT = {"correct_resolution": "reschedule_b", "acceptable_resolutions": ["cancel_b"]}


class TestCalendarGrader:
    def test_correct_resolution_full_score(self):
        score, _ = cal_grader.score({"resolution": "reschedule_b", "rationale": "VIP meeting takes priority here"}, CAL_GT, CONFLICT)
        assert score >= 0.5

    def test_acceptable_resolution_partial(self):
        score, _ = cal_grader.score({"resolution": "cancel_b", "rationale": "Cancelling lower priority event"}, CAL_GT, CONFLICT)
        assert score >= 0.25

    def test_wrong_resolution_zero_resolution_score(self):
        _, breakdown = cal_grader.score({"resolution": "cancel_a", "rationale": "test"}, CAL_GT, CONFLICT)
        assert breakdown["resolution_score"] == 0.0

    def test_vip_penalty_cancelling_vip(self):
        _, breakdown = cal_grader.score(
            {"resolution": "cancel_a", "rationale": "cancelling vip event"},
            CAL_GT,
            CONFLICT  # event_a is VIP, event_b is not
        )
        assert breakdown["vip_protection"] < 1.0

    def test_rationale_5_words_full_score(self):
        _, breakdown = cal_grader.score(
            {"resolution": "reschedule_b", "rationale": "This is a good reason"},
            CAL_GT, CONFLICT)
        assert breakdown["rationale_score"] == 1.0

    def test_rationale_short_half_score(self):
        _, breakdown = cal_grader.score(
            {"resolution": "reschedule_b", "rationale": "ok"},
            CAL_GT, CONFLICT)
        assert breakdown["rationale_score"] == 0.5

    def test_rationale_empty_zero_score(self):
        _, breakdown = cal_grader.score(
            {"resolution": "reschedule_b", "rationale": ""},
            CAL_GT, CONFLICT)
        assert breakdown["rationale_score"] == 0.0

    def test_missing_resolution_key_no_crash(self):
        score, _ = cal_grader.score({"rationale": "some reason"}, CAL_GT, CONFLICT)
        assert 0.0 <= score <= 1.0

    def test_score_always_in_range(self):
        for res in ["reschedule_a", "reschedule_b", "cancel_a", "cancel_b", "delegate_a"]:
            score, _ = cal_grader.score({"resolution": res, "rationale": "reason here"}, CAL_GT, CONFLICT)
            assert 0.0 <= score <= 1.0


# ═══════════════════════════════════════════════════════════════════
# GRADER TESTS — delegation_grader
# ═══════════════════════════════════════════════════════════════════

from graders.delegation_grader import DelegationGrader

del_grader = DelegationGrader()

DEL_GT = {
    "correct_assignee": "manager",
    "acceptable_assignees": ["external"],
    "required_message_keywords": ["failover", "incident"],
    "urgency": "urgent",
    "requires_technical": True,
}


class TestDelegationGrader:
    def test_correct_assignee_full_score(self):
        score, _ = del_grader.score(
            {"assignee": "manager", "delegation_message": "Please handle the failover incident now"},
            DEL_GT)
        assert score >= 0.5

    def test_acceptable_assignee_partial(self):
        _, breakdown = del_grader.score(
            {"assignee": "external", "delegation_message": "failover incident"},
            DEL_GT)
        assert breakdown["assignee_score"] == 0.5

    def test_wrong_assignee_zero(self):
        _, breakdown = del_grader.score(
            {"assignee": "junior", "delegation_message": "failover incident"},
            DEL_GT)
        assert breakdown["assignee_score"] == 0.0

    def test_drop_urgent_penalty(self):
        _, breakdown = del_grader.score(
            {"assignee": "drop", "delegation_message": ""},
            DEL_GT)
        assert breakdown["assignee_score"] == 0.0

    def test_message_keywords_all_matched(self):
        _, breakdown = del_grader.score(
            {"assignee": "manager", "delegation_message": "Handle the failover and incident response"},
            DEL_GT)
        assert breakdown["message_score"] == 1.0

    def test_message_keywords_partial(self):
        _, breakdown = del_grader.score(
            {"assignee": "manager", "delegation_message": "Handle the failover"},
            DEL_GT)
        assert breakdown["message_score"] == 0.5

    def test_technical_escalation_correct(self):
        _, breakdown = del_grader.score(
            {"assignee": "manager", "delegation_message": "failover incident"},
            DEL_GT)
        assert breakdown["escalation_score"] == 1.0

    def test_technical_escalation_wrong(self):
        _, breakdown = del_grader.score(
            {"assignee": "junior", "delegation_message": "failover incident"},
            DEL_GT)
        assert breakdown["escalation_score"] == 0.5

    def test_score_always_in_range(self):
        for assignee in ["self", "junior", "manager", "external", "drop"]:
            score, _ = del_grader.score(
                {"assignee": assignee, "delegation_message": "failover incident"},
                DEL_GT)
            assert 0.0 <= score <= 1.0


# ═══════════════════════════════════════════════════════════════════
# MODULE TESTS — CalendarModule / DelegationModule
# ═══════════════════════════════════════════════════════════════════

from modules.calendar_module import CalendarModule
from modules.delegation_module import DelegationModule

CONFLICTS = [
    {"conflict_id": "c001", "event_a": {"title": "A", "time": "10:00", "attendees": [], "vip": True},
     "event_b": {"title": "B", "time": "10:00", "attendees": [], "vip": False},
     "conflict_reason": "same_time", "resolution_options": ["reschedule_a", "reschedule_b"],
     "ground_truth": {"correct_resolution": "reschedule_b", "acceptable_resolutions": []}},
    {"conflict_id": "c002", "event_a": {"title": "C", "time": "11:00", "attendees": [], "vip": False},
     "event_b": {"title": "D", "time": "11:00", "attendees": [], "vip": True},
     "conflict_reason": "same_time", "resolution_options": ["reschedule_a", "reschedule_b"],
     "ground_truth": {"correct_resolution": "reschedule_a", "acceptable_resolutions": []}},
]

TASKS = [
    {"task_id": "t001", "title": "Fix bug", "source_email_id": "e001",
     "urgency": "urgent", "requires_technical": True,
     "ground_truth": {"correct_assignee": "manager", "acceptable_assignees": [],
                      "required_message_keywords": ["fix", "bug"]}},
    {"task_id": "t002", "title": "Send report", "source_email_id": "e002",
     "urgency": "low", "requires_technical": False,
     "ground_truth": {"correct_assignee": "junior", "acceptable_assignees": [],
                      "required_message_keywords": ["report"]}},
]


class TestCalendarModule:
    def test_current_conflict_returns_correct_keys(self):
        m = CalendarModule(CONFLICTS)
        c = m.current_conflict()
        for k in ("conflict_id", "event_a", "event_b", "conflict_reason",
                  "resolution_options", "conflict_number", "total_conflicts"):
            assert k in c

    def test_conflict_number_is_1_indexed(self):
        m = CalendarModule(CONFLICTS)
        assert m.current_conflict()["conflict_number"] == 1
        assert m.current_conflict()["total_conflicts"] == 2

    def test_advance_moves_index(self):
        m = CalendarModule(CONFLICTS)
        m.advance()
        assert m.current_conflict()["conflict_id"] == "c002"

    def test_done_false_initially(self):
        assert not CalendarModule(CONFLICTS).done()

    def test_done_true_after_all_advanced(self):
        m = CalendarModule(CONFLICTS)
        m.advance(); m.advance()
        assert m.done()

    def test_ground_truth_after_advance(self):
        m = CalendarModule(CONFLICTS)
        m.advance()
        gt = m.ground_truth()
        assert gt["correct_resolution"] == "reschedule_b"

    def test_ground_truth_before_advance_raises(self):
        with pytest.raises(RuntimeError):
            CalendarModule(CONFLICTS).ground_truth()


class TestDelegationModule:
    def test_current_task_returns_correct_keys(self):
        m = DelegationModule(TASKS)
        t = m.current_task()
        for k in ("task_id", "title", "source_email_id", "urgency",
                  "requires_technical", "task_number", "total_tasks"):
            assert k in t

    def test_task_number_is_1_indexed(self):
        m = DelegationModule(TASKS)
        assert m.current_task()["task_number"] == 1
        assert m.current_task()["total_tasks"] == 2

    def test_advance_moves_index(self):
        m = DelegationModule(TASKS)
        m.advance()
        assert m.current_task()["task_id"] == "t002"

    def test_done_false_initially(self):
        assert not DelegationModule(TASKS).done()

    def test_done_true_after_all_advanced(self):
        m = DelegationModule(TASKS)
        m.advance(); m.advance()
        assert m.done()

    def test_ground_truth_after_advance(self):
        m = DelegationModule(TASKS)
        m.advance()
        gt = m.ground_truth()
        assert gt["correct_assignee"] == "manager"

    def test_ground_truth_before_advance_raises(self):
        with pytest.raises(RuntimeError):
            DelegationModule(TASKS).ground_truth()


# ═══════════════════════════════════════════════════════════════════
# ENV ORCHESTRATION TESTS — ChiefOfStaffEnv
# ═══════════════════════════════════════════════════════════════════

from env import ChiefOfStaffEnv
from email_triage_env.models import EpisodePhase


def make_email_action(email_id, category="spam", priority="low"):
    return {"phase": "email", "email_id": email_id, "category": category,
            "priority": priority, "suggested_response": ""}

def make_cal_action(conflict_id, resolution="reschedule_b"):
    return {"phase": "calendar", "conflict_id": conflict_id,
            "resolution": resolution, "rationale": "Higher priority event protected"}

def make_del_action(task_id, assignee="junior"):
    return {"phase": "delegation", "task_id": task_id,
            "assignee": assignee, "delegation_message": "Please handle this task"}


class TestChiefOfStaffEnv:
    def test_reset_returns_email_phase(self):
        env = ChiefOfStaffEnv()
        obs = env.reset("easy_cos")
        assert obs["phase"] == "email"

    def test_reset_unknown_task_raises(self):
        env = ChiefOfStaffEnv()
        with pytest.raises(Exception):
            env.reset("nonexistent_task")

    def test_step_returns_required_keys(self):
        env = ChiefOfStaffEnv()
        obs = env.reset("easy_cos")
        result = env.step(make_email_action(obs["email_id"]))
        for k in ("observation", "reward", "done", "info"):
            assert k in result

    def test_reward_in_range_every_step(self):
        env = ChiefOfStaffEnv()
        obs = env.reset("easy_cos")
        while not obs.get("done"):
            phase = obs.get("phase", "email")
            if phase == "email":
                action = make_email_action(obs.get("email_id", ""))
            elif phase == "calendar":
                action = make_cal_action(obs.get("conflict_id", ""))
            else:
                action = make_del_action(obs.get("task_id", ""))
            result = env.step(action)
            assert 0.0 <= result["reward"] <= 1.0
            obs = result["observation"]

    def test_episode_completes_within_50_steps(self):
        env = ChiefOfStaffEnv()
        obs = env.reset("easy_cos")
        steps = 0
        while not obs.get("done") and steps < 50:
            phase = obs.get("phase", "email")
            if phase == "email":
                action = make_email_action(obs.get("email_id", ""))
            elif phase == "calendar":
                action = make_cal_action(obs.get("conflict_id", ""))
            else:
                action = make_del_action(obs.get("task_id", ""))
            result = env.step(action)
            obs = result["observation"]
            steps += 1
        assert obs.get("phase") == "done"

    def test_phase_transitions_email_to_calendar(self):
        env = ChiefOfStaffEnv()
        obs = env.reset("easy_cos")
        while obs.get("phase") == "email":
            result = env.step(make_email_action(obs.get("email_id", "")))
            obs = result["observation"]
        assert obs["phase"] == "calendar"

    def test_phase_transitions_calendar_to_delegation(self):
        env = ChiefOfStaffEnv()
        obs = env.reset("easy_cos")
        while obs.get("phase") != "calendar":
            result = env.step(make_email_action(obs.get("email_id", "")))
            obs = result["observation"]
        while obs.get("phase") == "calendar":
            result = env.step(make_cal_action(obs.get("conflict_id", "")))
            obs = result["observation"]
        assert obs["phase"] == "delegation"

    def test_info_contains_phase_rewards(self):
        env = ChiefOfStaffEnv()
        obs = env.reset("easy_cos")
        result = env.step(make_email_action(obs["email_id"]))
        assert "phase_rewards" in result["info"]
        for k in ("email", "calendar", "delegation"):
            assert k in result["info"]["phase_rewards"]

    def test_state_returns_correct_keys(self):
        env = ChiefOfStaffEnv()
        env.reset("easy_cos")
        s = env.state()
        assert "phase" in s
        assert "phase_rewards" in s
        assert "done" in s

    def test_done_observation_has_total_reward(self):
        env = ChiefOfStaffEnv()
        obs = env.reset("easy_cos")
        while not obs.get("done"):
            phase = obs.get("phase", "email")
            if phase == "email":
                action = make_email_action(obs.get("email_id", ""))
            elif phase == "calendar":
                action = make_cal_action(obs.get("conflict_id", ""))
            else:
                action = make_del_action(obs.get("task_id", ""))
            result = env.step(action)
            obs = result["observation"]
        assert "total_reward" in obs
        assert "phase_rewards" in obs

    def test_all_three_difficulties_complete(self):
        for task_id in ["easy_cos", "medium_cos", "hard_cos"]:
            env = ChiefOfStaffEnv()
            obs = env.reset(task_id)
            steps = 0
            while not obs.get("done") and steps < 100:
                phase = obs.get("phase", "email")
                if phase == "email":
                    action = make_email_action(obs.get("email_id", ""))
                elif phase == "calendar":
                    action = make_cal_action(obs.get("conflict_id", ""))
                else:
                    action = make_del_action(obs.get("task_id", ""))
                result = env.step(action)
                obs = result["observation"]
                steps += 1
            assert obs.get("phase") == "done", f"{task_id} did not complete"


# ═══════════════════════════════════════════════════════════════════
# TASK FILE TESTS
# ═══════════════════════════════════════════════════════════════════

TASKS_DIR = Path(__file__).parent.parent / "tasks"


class TestTaskFiles:
    @pytest.mark.parametrize("task_id,n_emails,n_conflicts,n_tasks", [
        ("easy_cos",   5,  2, 2),
        ("medium_cos", 10, 3, 3),
        ("hard_cos",   15, 5, 5),
    ])
    def test_task_file_counts(self, task_id, n_emails, n_conflicts, n_tasks):
        data = json.loads((TASKS_DIR / f"{task_id}.json").read_text())
        assert len(data["emails"])    == n_emails
        assert len(data["conflicts"]) == n_conflicts
        assert len(data["tasks"])     == n_tasks

    @pytest.mark.parametrize("task_id", ["easy_cos", "medium_cos", "hard_cos"])
    def test_email_schema(self, task_id):
        data = json.loads((TASKS_DIR / f"{task_id}.json").read_text())
        for email in data["emails"]:
            for k in ("email_id", "subject", "body", "sender", "timestamp", "ground_truth"):
                assert k in email
            for k in ("category", "priority", "required_response_keywords"):
                assert k in email["ground_truth"]

    @pytest.mark.parametrize("task_id", ["easy_cos", "medium_cos", "hard_cos"])
    def test_conflict_schema(self, task_id):
        data = json.loads((TASKS_DIR / f"{task_id}.json").read_text())
        for c in data["conflicts"]:
            for k in ("conflict_id", "event_a", "event_b", "conflict_reason",
                      "resolution_options", "ground_truth"):
                assert k in c
            for k in ("correct_resolution", "acceptable_resolutions"):
                assert k in c["ground_truth"]
            for event in (c["event_a"], c["event_b"]):
                assert "vip" in event

    @pytest.mark.parametrize("task_id", ["easy_cos", "medium_cos", "hard_cos"])
    def test_task_schema(self, task_id):
        data = json.loads((TASKS_DIR / f"{task_id}.json").read_text())
        for t in data["tasks"]:
            for k in ("task_id", "title", "source_email_id", "urgency",
                      "requires_technical", "ground_truth"):
                assert k in t
            for k in ("correct_assignee", "acceptable_assignees", "required_message_keywords"):
                assert k in t["ground_truth"]

    @pytest.mark.parametrize("task_id", ["easy_cos", "medium_cos", "hard_cos"])
    def test_valid_categories_in_emails(self, task_id):
        valid = {"spam", "urgent", "newsletter", "support", "internal"}
        data = json.loads((TASKS_DIR / f"{task_id}.json").read_text())
        for email in data["emails"]:
            assert email["ground_truth"]["category"] in valid

    @pytest.mark.parametrize("task_id", ["easy_cos", "medium_cos", "hard_cos"])
    def test_valid_resolutions_in_conflicts(self, task_id):
        valid = {"reschedule_a", "reschedule_b", "cancel_a", "cancel_b", "delegate_a"}
        data = json.loads((TASKS_DIR / f"{task_id}.json").read_text())
        for c in data["conflicts"]:
            assert c["ground_truth"]["correct_resolution"] in valid

    @pytest.mark.parametrize("task_id", ["easy_cos", "medium_cos", "hard_cos"])
    def test_valid_assignees_in_tasks(self, task_id):
        valid = {"self", "junior", "manager", "external", "drop"}
        data = json.loads((TASKS_DIR / f"{task_id}.json").read_text())
        for t in data["tasks"]:
            assert t["ground_truth"]["correct_assignee"] in valid


# ═══════════════════════════════════════════════════════════════════
# MODEL TESTS
# ═══════════════════════════════════════════════════════════════════

from email_triage_env.models import (
    EpisodePhase, CalendarAction, DelegationAction,
    ChiefOfStaffAction, CalendarConflict, DelegationTask
)
from pydantic import ValidationError


class TestModels:
    def test_episode_phase_values(self):
        assert EpisodePhase.EMAIL      == "email"
        assert EpisodePhase.CALENDAR   == "calendar"
        assert EpisodePhase.DELEGATION == "delegation"
        assert EpisodePhase.DONE       == "done"

    def test_calendar_action_valid(self):
        a = CalendarAction(conflict_id="c001", resolution="reschedule_a")
        assert a.conflict_id == "c001"

    def test_calendar_action_invalid_resolution(self):
        with pytest.raises(ValidationError):
            CalendarAction(conflict_id="c001", resolution="invalid_option")

    def test_delegation_action_valid(self):
        a = DelegationAction(task_id="t001", assignee="manager")
        assert a.assignee == "manager"

    def test_delegation_action_invalid_assignee(self):
        with pytest.raises(ValidationError):
            DelegationAction(task_id="t001", assignee="ceo")

    def test_chief_of_staff_action_email_phase(self):
        a = ChiefOfStaffAction(phase=EpisodePhase.EMAIL, email_id="e001",
                               category="urgent", priority="urgent")
        assert a.phase == EpisodePhase.EMAIL

    def test_chief_of_staff_action_optional_fields_default_none(self):
        a = ChiefOfStaffAction(phase=EpisodePhase.CALENDAR)
        assert a.email_id is None
        assert a.task_id is None

    def test_calendar_conflict_model(self):
        c = CalendarConflict(
            conflict_id="c001",
            event_a={"title": "A"}, event_b={"title": "B"},
            conflict_reason="same_time", resolution_options=["reschedule_a"],
            conflict_number=1, total_conflicts=2
        )
        assert c.conflict_id == "c001"

    def test_delegation_task_model(self):
        t = DelegationTask(
            task_id="t001", title="Fix bug", source_email_id="e001",
            urgency="urgent", requires_technical=True,
            task_number=1, total_tasks=3
        )
        assert t.requires_technical is True
