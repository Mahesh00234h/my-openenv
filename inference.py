"""
Baseline inference script for the Email Triage Environment.

Follows the OpenEnv sample inference.py format exactly.
"""

import json
import os
import sys
import time
from typing import List, Optional

from openai import OpenAI

from email_triage_env.env import EmailTriageEnv
from email_triage_env.models import Action

# Environment variables
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY", "")
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

TASKS = ["easy_triage", "medium_triage", "hard_triage"]
BENCHMARK = "email-triage"
MAX_RETRIES = 3
BACKOFF_DELAYS = [1, 2, 4]
SUCCESS_SCORE_THRESHOLD = 0.1


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def build_prompt(obs) -> str:
    return f"""You are an email triage assistant. Analyze the following email and respond with a JSON object.

Email details:
- email_id: {obs.email_id}
- subject: {obs.subject}
- sender: {obs.sender}
- timestamp: {obs.timestamp}
- body: {obs.body}

Respond with ONLY a valid JSON object matching this schema:
{{
  "email_id": "<same email_id as above>",
  "category": "<one of: spam, urgent, newsletter, support, internal>",
  "priority": "<one of: low, medium, high, urgent>",
  "suggested_response": "<brief suggested response text>"
}}

Do not include any explanation or markdown — only the raw JSON object."""


def call_llm_with_retry(client: OpenAI, prompt: str, email_id: str) -> Action:
    last_exc = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            content = response.choices[0].message.content.strip()
            data = json.loads(content)
            return Action(**data)
        except Exception as exc:
            last_exc = exc
            if attempt < MAX_RETRIES:
                delay = BACKOFF_DELAYS[attempt]
                print(
                    f"[DEBUG] LLM call failed (attempt {attempt + 1}): {exc}. Retrying in {delay}s...",
                    file=sys.stderr, flush=True,
                )
                time.sleep(delay)

    print(f"[DEBUG] LLM failed after {MAX_RETRIES} retries: {last_exc}", file=sys.stderr, flush=True)
    return Action(email_id=email_id, category="spam", priority="low", suggested_response="")


def run_task(env: EmailTriageEnv, client: OpenAI, task_id: str) -> float:
    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    obs = env.reset(task_id)
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    try:
        step = 0
        while True:
            if obs.email_id == "" and obs.body == "":
                break

            step += 1
            prompt = build_prompt(obs)
            action = call_llm_with_retry(client, prompt, obs.email_id)
            action_str = f"triage({obs.email_id!r})"

            obs, reward_obj, done, _info = env.step(action)
            reward = float(reward_obj.score)
            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action_str, reward=reward, done=done, error=None)

            if done:
                break

        # Compute score as mean reward, clamped to (0, 1) exclusive
        if rewards:
            score = sum(rewards) / len(rewards)
        else:
            score = 0.5
        score = max(0.001, min(0.999, score))
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


def main():
    if not API_KEY:
        print("[DEBUG] Warning: no API key set", file=sys.stderr, flush=True)

    client = OpenAI(api_key=API_KEY or "missing", base_url=API_BASE_URL)
    env = EmailTriageEnv()

    task_scores = {}
    for task_id in TASKS:
        score = run_task(env, client, task_id)
        task_scores[task_id] = score

    print("\n--- Results ---", flush=True)
    for task_id, score in task_scores.items():
        print(f"  {task_id}: {score:.4f}", flush=True)
    mean_score = sum(task_scores.values()) / len(task_scores)
    print(f"  overall_mean: {mean_score:.4f}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[DEBUG] Error: {e}", file=sys.stderr, flush=True)
        sys.exit(1)
