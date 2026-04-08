from email_triage_env.models import Action, Observation, Reward
from email_triage_env.grader import grade
from email_triage_env.task_registry import load_task


class EmailTriageEnv:
    def __init__(self):
        self._task_id: str | None = None
        self._inbox: list[dict] | None = None
        self._current_index: int = 0
        self._cumulative_reward: float = 0.0
        self._done: bool = False

    def _make_observation(self, email_dict: dict, index: int, total: int) -> Observation:
        return Observation(
            email_id=email_dict["email_id"],
            subject=email_dict["subject"],
            body=email_dict["body"],
            sender=email_dict["sender"],
            timestamp=email_dict["timestamp"],
            inbox_position=index,
            total_emails=total,
        )

    def _terminal_observation(self) -> Observation:
        total = len(self._inbox)
        return Observation(
            email_id="",
            subject="",
            body="",
            sender="",
            timestamp="",
            inbox_position=total,
            total_emails=total,
        )

    def reset(self, task_id: str = "easy_triage") -> Observation:
        task = load_task(task_id)  # raises ValueError for unknown task_id
        self._task_id = task_id
        self._inbox = task["emails"]
        self._current_index = 0
        self._cumulative_reward = 0.0
        self._done = False
        return self._make_observation(self._inbox[0], 0, len(self._inbox))

    def step(self, action: Action) -> tuple[Observation, Reward, bool, dict]:
        if self._inbox is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        total = len(self._inbox)
        info = {"task_id": self._task_id, "step": self._current_index}

        # Already done
        if self._done:
            terminal_obs = self._terminal_observation()
            reward = Reward(
                score=0.001,
                category_score=0.0,
                priority_score=0.0,
                response_score=0.0,
                explanation="Episode already terminated.",
            )
            return terminal_obs, reward, True, info

        current_email = self._inbox[self._current_index]

        # email_id mismatch — do NOT advance index
        if action.email_id != current_email["email_id"]:
            obs = self._make_observation(current_email, self._current_index, total)
            reward = Reward(
                score=0.001,
                category_score=0.0,
                priority_score=0.0,
                response_score=0.0,
                explanation=(
                    f"email_id mismatch: expected {current_email['email_id']}, got {action.email_id}"
                ),
            )
            return obs, reward, self._done, info

        # Grade the action
        reward = grade(action, current_email["ground_truth"])
        self._cumulative_reward += reward.score
        self._current_index += 1

        # Check if inbox exhausted
        if self._current_index >= total:
            self._done = True
            obs = self._terminal_observation()
        else:
            obs = self._make_observation(self._inbox[self._current_index], self._current_index, total)

        info = {"task_id": self._task_id, "step": self._current_index}
        return obs, reward, self._done, info

    def state(self) -> dict:
        remaining = 0 if self._inbox is None else max(0, len(self._inbox) - self._current_index)
        total = 0 if self._inbox is None else len(self._inbox)
        steps_done = self._current_index
        # Normalize cumulative reward to a per-step mean, clamped strictly to (0, 1)
        if steps_done > 0:
            mean_score = self._cumulative_reward / steps_done
        else:
            mean_score = 0.5  # neutral default before any steps
        mean_score = max(0.001, min(0.999, mean_score))
        return {
            "task_id": self._task_id,
            "current_index": self._current_index,
            "cumulative_reward": mean_score,
            "remaining_emails": remaining,
        }
