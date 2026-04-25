"""Smoke test for ChiefOfStaffEnv — plain Python, no pytest."""
import json
import random
import sys
from pathlib import Path

from env import ChiefOfStaffEnv

CATEGORIES = ["spam", "urgent", "newsletter", "support", "internal"]
PRIORITIES  = ["low", "medium", "high", "urgent"]
ASSIGNEES   = ["self", "junior", "manager", "external", "drop"]
TASKS       = ["easy_cos", "medium_cos", "hard_cos"]
MAX_STEPS   = 100


def random_action(observation: dict) -> dict:
    phase = observation.get("phase", "email")
    if phase == "email":
        return {
            "phase": "email",
            "email_id": observation.get("email_id", ""),
            "category": random.choice(CATEGORIES),
            "priority": random.choice(PRIORITIES),
            "suggested_response": "Acknowledged, will follow up shortly.",
        }
    if phase == "calendar":
        options = observation.get("resolution_options", ["reschedule_a"])
        return {
            "phase": "calendar",
            "conflict_id": observation.get("conflict_id", ""),
            "resolution": random.choice(options),
            "rationale": "Prioritising the higher impact event for the team.",
        }
    if phase == "delegation":
        return {
            "phase": "delegation",
            "task_id": observation.get("task_id", ""),
            "assignee": random.choice(ASSIGNEES),
            "delegation_message": "Please handle this task and report back with an update.",
        }
    return {"phase": phase}


def mean(lst: list) -> float:
    return sum(lst) / len(lst) if lst else 0.0


def run_task(task_id: str) -> dict:
    env = ChiefOfStaffEnv()
    obs_raw = env.reset(task_id)

    phase_rewards: dict[str, list] = {"email": [], "calendar": [], "delegation": []}
    steps = 0
    done = False

    while not done:
        assert steps < MAX_STEPS, f"Episode did not finish within {MAX_STEPS} steps"

        phase = obs_raw.get("phase", "email")
        action = random_action(obs_raw)
        result = env.step(action)

        for key in ("observation", "reward", "done", "info"):
            assert key in result, f"Step result missing key: '{key}'"

        reward = result["reward"]
        assert 0.0 <= reward <= 1.0, f"Reward out of range: {reward}"

        info = result["info"]
        assert "phase_rewards" in info, "info missing 'phase_rewards'"
        for k in ("email", "calendar", "delegation"):
            assert k in info["phase_rewards"], f"phase_rewards missing key: '{k}'"

        if phase in phase_rewards:
            phase_rewards[phase].append(reward)

        done = result["done"]
        obs_raw = result["observation"]
        steps += 1

    assert done, "Episode never reached done=True"

    return {
        "email":      round(mean(phase_rewards["email"]), 3),
        "calendar":   round(mean(phase_rewards["calendar"]), 3),
        "delegation": round(mean(phase_rewards["delegation"]), 3),
        "steps":      steps,
    }


def print_table(results: dict) -> None:
    h_task  = 12
    h_steps =  6
    h_score =  10

    top    = f"╔{'═'*(h_task+2)}╦{'═'*(h_steps+2)}╦{'═'*(h_score+2)}╦{'═'*(h_score+2)}╦{'═'*(h_score+2)}╗"
    header = f"║ {'Task':<{h_task}} ║ {'Steps':>{h_steps}} ║ {'Email':^{h_score}} ║ {'Calendar':^{h_score}} ║ {'Delegation':^{h_score}} ║"
    sep    = f"╠{'═'*(h_task+2)}╬{'═'*(h_steps+2)}╬{'═'*(h_score+2)}╬{'═'*(h_score+2)}╬{'═'*(h_score+2)}╣"
    bot    = f"╚{'═'*(h_task+2)}╩{'═'*(h_steps+2)}╩{'═'*(h_score+2)}╩{'═'*(h_score+2)}╩{'═'*(h_score+2)}╝"

    print(top)
    print(header)
    print(sep)
    for task_id, r in results.items():
        row = (f"║ {task_id:<{h_task}} ║ {r['steps']:>{h_steps}} ║ "
               f"{r['email']:^{h_score}.3f} ║ {r['calendar']:^{h_score}.3f} ║ "
               f"{r['delegation']:^{h_score}.3f} ║")
        print(row)
    print(bot)


def save_results(results: dict) -> None:
    out = Path(__file__).parent / "results" / "smoke_test_results.json"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[SAVED] {out}")


if __name__ == "__main__":
    all_results = {}
    failed = False

    for task_id in TASKS:
        try:
            all_results[task_id] = run_task(task_id)
        except AssertionError as e:
            print(f"FAILED: {task_id} — {e}")
            failed = True
        except Exception as e:
            print(f"FAILED: {task_id} — Unexpected error: {e}")
            failed = True

    if failed:
        sys.exit(1)

    print_table(all_results)
    save_results(all_results)
    print(f"\nSMOKE TEST PASSED — all {len(TASKS)} difficulties")
