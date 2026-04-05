"""
Baseline inference script for the Email Triage Environment.

Reads OPENAI_API_KEY, API_BASE_URL, MODEL_NAME from environment variables,
runs all three tasks sequentially using the OpenAI Python client, and emits
structured log lines for each task/step.
"""

import json
import os
import sys
import time

from openai import OpenAI

from email_triage_env.env import EmailTriageEnv
from email_triage_env.models import Action

TASKS = ["easy_triage", "medium_triage", "hard_triage"]

MAX_RETRIES = 3
BACKOFF_DELAYS = [1, 2, 4]  # seconds

# Environment variables — API_BASE_URL and MODEL_NAME have defaults; HF_TOKEN does not
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")
# Optional — used when loading the environment from a local Docker image
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")


def get_env_vars() -> tuple[str, str, str]:
    """Read required environment variables, exit with code 1 if OPENAI_API_KEY is missing."""
    api_key = os.environ.get("OPENAI_API_KEY", "")
    base_url = API_BASE_URL
    model_name = MODEL_NAME

    if not api_key:
        print("Error: missing required environment variable: OPENAI_API_KEY", file=sys.stderr)
        sys.exit(1)

    return api_key, base_url, model_name


def build_prompt(obs) -> str:
    """Build a prompt asking the LLM to triage the given email."""
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


def call_llm_with_retry(client: OpenAI, model_name: str, prompt: str, email_id: str) -> Action:
    """
    Call the LLM to produce an Action, retrying up to MAX_RETRIES times with
    exponential backoff (1s, 2s, 4s). Returns a zero-score dummy Action on
    final failure.
    """
    last_exc = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=model_name,
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
                    f"  [WARN] LLM call failed (attempt {attempt + 1}/{MAX_RETRIES + 1}): {exc}. "
                    f"Retrying in {delay}s...",
                    file=sys.stderr,
                )
                time.sleep(delay)

    # Final failure — return a dummy Action with score=0.0
    print(
        f"  [ERROR] LLM call failed after {MAX_RETRIES} retries: {last_exc}. Recording score=0.0.",
        file=sys.stderr,
    )
    return Action(
        email_id=email_id,
        category="spam",
        priority="low",
        suggested_response="",
    )


def run_task(env: EmailTriageEnv, client: OpenAI, model_name: str, task_id: str) -> float:
    """Run a single task and return the total score."""
    print(f"[START] task={task_id}")

    obs = env.reset(task_id)
    total_score = 0.0
    step_num = 0

    while True:
        # Terminal observation — episode done
        if obs.email_id == "" and obs.body == "":
            break

        step_num += 1
        prompt = build_prompt(obs)
        action = call_llm_with_retry(client, model_name, prompt, obs.email_id)

        obs, reward, done, _info = env.step(action)
        total_score += reward.score

        print(f"[STEP] task={task_id} step={step_num} score={reward.score:.4f}")

        if done:
            break

    print(f"[END] task={task_id} total_score={total_score:.4f}")
    return total_score


def main():
    api_key, base_url, model_name = get_env_vars()

    client = OpenAI(api_key=api_key, base_url=base_url)
    env = EmailTriageEnv()

    task_scores: dict[str, float] = {}

    for task_id in TASKS:
        score = run_task(env, client, model_name, task_id)
        task_scores[task_id] = score

    print("\n--- Results ---")
    for task_id, score in task_scores.items():
        print(f"  {task_id}: {score:.4f}")

    mean_score = sum(task_scores.values()) / len(task_scores)
    print(f"  overall_mean: {mean_score:.4f}")


if __name__ == "__main__":
    main()
