import json
from pathlib import Path

from modules.calendar_module import CalendarModule
from modules.delegation_module import DelegationModule
from graders.email_grader import grade as email_grade
from graders.calendar_grader import CalendarGrader
from graders.delegation_grader import DelegationGrader
from email_triage_env.models import Action, EpisodePhase


class ChiefOfStaffEnv:
    def __init__(self):
        self._calendar_grader = CalendarGrader()
        self._delegation_grader = DelegationGrader()
        self.scenario = None
        self._emails: list[dict] = []
        self._email_index: int = 0
        self._email_step_count: int = 0
        self._calendar: CalendarModule | None = None
        self._delegation: DelegationModule | None = None
        self.phase: EpisodePhase = EpisodePhase.DONE
        self.phase_rewards: dict = {"email": 0.0, "calendar": 0.0, "delegation": 0.0}
        self.urgent_email_ids: list[str] = []

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def reset(self, task_id: str) -> dict:
        task_path = Path(__file__).parent / "tasks" / f"{task_id}.json"
        with open(task_path) as f:
            self.scenario = json.load(f)

        self._emails = self.scenario["emails"]
        self._email_index = 0
        self._email_step_count = 0
        self._calendar = CalendarModule(self.scenario["conflicts"])
        self._delegation = DelegationModule(self.scenario["tasks"])

        self.phase = EpisodePhase.EMAIL
        self.phase_rewards = {"email": 0.0, "calendar": 0.0, "delegation": 0.0}
        self.urgent_email_ids = []

        return self._get_observation()

    def step(self, action: dict) -> dict:
        if self.phase == EpisodePhase.DONE:
            return {
                "observation": self._get_observation(),
                "reward": 0.0,
                "done": True,
                "info": {"phase": EpisodePhase.DONE, "phase_rewards": self.phase_rewards},
            }

        current_phase = self.phase

        if self.phase == EpisodePhase.EMAIL:
            reward, breakdown = self._step_email(action)
        elif self.phase == EpisodePhase.CALENDAR:
            reward, breakdown = self._step_calendar(action)
        else:  # DELEGATION
            reward, breakdown = self._step_delegation(action)

        self.phase_rewards[current_phase.value] += reward

        return {
            "observation": self._get_observation(),
            "reward": reward,
            "done": self.phase == EpisodePhase.DONE,
            "info": {
                "phase": current_phase,
                "phase_rewards": self.phase_rewards,
                **breakdown,
            },
        }

    def state(self) -> dict:
        return {
            "phase": self.phase,
            "phase_rewards": self.phase_rewards,
            "done": self.phase == EpisodePhase.DONE,
        }

    # ------------------------------------------------------------------ #
    #  Phase handlers                                                      #
    # ------------------------------------------------------------------ #

    def _step_email(self, action: dict) -> tuple[float, dict]:
        email = self._emails[self._email_index]
        gt = email["ground_truth"]

        # Build the Pydantic Action the email grader expects
        pydantic_action = Action(
            email_id=action.get("email_id", ""),
            category=action.get("category", ""),
            priority=action.get("priority", ""),
            suggested_response=action.get("suggested_response", ""),
        )

        reward_obj = email_grade(
            pydantic_action,
            gt,
            inbox_position=self._email_index,
            total_emails=len(self._emails),
        )
        score = reward_obj.score

        # Track urgent emails that were correctly categorised
        if gt.get("priority") == "urgent" and action.get("category") == gt.get("category"):
            self.urgent_email_ids.append(email["email_id"])

        self._email_index += 1
        if self._email_index >= len(self._emails):
            self.phase = EpisodePhase.CALENDAR

        return score, {
            "category_score": reward_obj.category_score,
            "priority_score": reward_obj.priority_score,
            "response_score": reward_obj.response_score,
            "explanation": reward_obj.explanation,
        }

    def _step_calendar(self, action: dict) -> tuple[float, dict]:
        conflict = self._calendar.current_conflict()
        # Retrieve the full conflict dict (with vip flags) for the grader
        raw_conflict = self.scenario["conflicts"][self._calendar._index]
        self._calendar.advance()
        gt = self._calendar.ground_truth()

        score, breakdown = self._calendar_grader.score(action, gt, raw_conflict)

        if self._calendar.done():
            self.phase = EpisodePhase.DELEGATION

        return score, breakdown

    def _step_delegation(self, action: dict) -> tuple[float, dict]:
        self._delegation.advance()
        gt = self._delegation.ground_truth()

        score, breakdown = self._delegation_grader.score(action, gt)

        if self._delegation.done():
            self.phase = EpisodePhase.DONE

        return score, breakdown

    # ------------------------------------------------------------------ #
    #  Observation builder                                                 #
    # ------------------------------------------------------------------ #

    def _get_observation(self) -> dict:
        if self.phase == EpisodePhase.EMAIL:
            email = self._emails[self._email_index]
            return {
                "phase": "email",
                "email_id": email["email_id"],
                "subject": email["subject"],
                "body": email["body"],
                "sender": email["sender"],
                "timestamp": email["timestamp"],
                "inbox_position": self._email_index,
                "total_emails": len(self._emails),
            }
        if self.phase == EpisodePhase.CALENDAR:
            return {"phase": "calendar", **self._calendar.current_conflict()}
        if self.phase == EpisodePhase.DELEGATION:
            return {"phase": "delegation", **self._delegation.current_task()}
        # DONE
        return {
            "phase": "done",
            "phase_rewards": self.phase_rewards,
            "total_reward": sum(self.phase_rewards.values()),
        }

    # ------------------------------------------------------------------ #
    #  Episode-level combined reward (logging only)                        #
    # ------------------------------------------------------------------ #

    def episode_reward(self) -> float:
        """0.40 * mean_email + 0.35 * mean_calendar + 0.25 * mean_delegation"""
        n_emails = max(len(self._emails), 1)
        n_conflicts = max(len(self.scenario["conflicts"]), 1) if self.scenario else 1
        n_tasks = max(len(self.scenario["tasks"]), 1) if self.scenario else 1

        mean_email = self.phase_rewards["email"] / n_emails
        mean_calendar = self.phase_rewards["calendar"] / n_conflicts
        mean_delegation = self.phase_rewards["delegation"] / n_tasks

        return 0.40 * mean_email + 0.35 * mean_calendar + 0.25 * mean_delegation
