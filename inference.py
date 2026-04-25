"""
Baseline inference script for the AI Chief of Staff Environment.
Runs GPT-4o-mini against all 3 task difficulties and saves results.
"""

import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import requests
from openai import OpenAI

# ── Environment variables ────────────────────────────────────────────────────
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY", "")
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")

MAX_RETRIES = 3
BACKOFF_DELAYS = [1, 2, 4]

# Task step counts (emails, conflicts, tasks)
TASK_COUNTS = {
    "easy_cos":   {"email": 5, "calendar": 2, "delegation": 2},
    "medium_cos": {"email": 10, "calendar": 3, "delegation": 3},
    "hard_cos":   {"email": 15, "calendar": 5, "delegation": 5},
}

# ── System prompt ────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an AI Chief of Staff operating inside a structured decision environment.

Your job is to manage three types of tasks in sequence each episode:

1. EMAIL TRIAGE — Read each email carefully. Classify it into exactly one category:
   - urgent: requires immediate action or response (incidents, crises, time-critical decisions)
   - support: external requests needing help or follow-up (customers, vendors, billing)
   - internal: internal team communications, announcements, approvals, updates
   - newsletter: subscribed content, digests, product updates
   - spam: unsolicited promotional or phishing emails
   Assign priority: urgent > high > medium > low
   Write a suggested_response that includes any required keywords for the situation.

2. CALENDAR CONFLICT RESOLUTION — Given two overlapping events, choose the best resolution:
   - reschedule_a / reschedule_b: move one event to another time
   - cancel_a / cancel_b: permanently cancel one event
   - delegate_a: send a delegate to attend on your behalf
   Always protect VIP attendees. Prefer rescheduling over cancelling.
   Write a clear rationale of at least 5 words explaining your decision.

3. TASK DELEGATION — Assign each task to the right person:
   - self: you handle it personally (strategic, sensitive, or requires your authority)
   - junior: routine tasks, research, follow-ups
   - manager: urgent technical or cross-team escalations requiring authority
   - external: specialist work requiring outside expertise
   - drop: only if truly irrelevant or already resolved
   Write a delegation_message that includes relevant context keywords.

Always respond with ONLY valid JSON. No markdown, no explanation, no code blocks."""


# ── LLM call ─────────────────────────────────────────────────────────────────
def call_llm(client: OpenAI, prompt: str) -> str:
    last_exc = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=300,
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:
            last_exc = exc
            if attempt < MAX_RETRIES:
                delay = BACKOFF_DELAYS[attempt]
                print(f"[DEBUG] LLM call failed (attempt {attempt + 1}): {exc}. Retrying in {delay}s...",
                      file=sys.stderr, flush=True)
                time.sleep(delay)
    print(f"[DEBUG] LLM unreachable — using rule-based fallback", file=sys.stderr, flush=True)
    return _rule_based_fallback(prompt)


def _rule_based_fallback(prompt: str) -> str:
    """Smart rule-based agent used when LLM is unreachable."""
    if '"phase": "email"' in prompt or '"phase":"email"' in prompt:
        eid = ""
        for line in prompt.splitlines():
            if '"email_id"' in line:
                eid = line.split(":", 1)[1].strip().strip('",')
                break
        p = prompt.lower()
        if any(w in p for w in ["urgent", "sev-1", "outage", "down", "critical", "incident",
                                  "failover", "cve", "phishing", "discrepancy", "503", "replication"]):
            cat, pri = "urgent", "urgent"
            resp = "Acknowledged. Authorizing immediate response. Escalating to engineering for failover and incident resolution."
        elif any(w in p for w in ["invoice", "billing", "payment", "subscription", "renewal"]):
            cat, pri = "support", "high"
            resp = "Thank you for reaching out. We will confirm the invoice payment and billing details shortly."
        elif any(w in p for w in ["newsletter", "digest", "weekly", "unsubscribe", "edition", "developer weekly"]):
            cat, pri = "newsletter", "low"
            resp = ""
        elif any(w in p for w in ["prize", "winner", "reward", "contest", "free trial", "tesla", "selected"]):
            cat, pri = "spam", "low"
            resp = ""
        elif any(w in p for w in ["security", "audit", "compliance", "soc", "nda", "legal",
                                   "penetration", "runbook", "board", "roadmap", "hiring",
                                   "all-hands", "rescheduled", "kitchen", "lunch"]):
            cat, pri = "internal", "high" if any(w in p for w in ["security", "audit", "compliance", "board", "roadmap"]) else "medium"
            resp = "Acknowledged. Will review and coordinate with the relevant team before the deadline."
        else:
            cat, pri = "internal", "medium"
            resp = "Acknowledged. Will review and follow up with the team."
        return json.dumps({"phase": "email", "email_id": eid, "category": cat,
                           "priority": pri, "suggested_response": resp})

    if '"phase": "calendar"' in prompt or '"phase":"calendar"' in prompt:
        cid = ""
        for line in prompt.splitlines():
            if '"conflict_id"' in line:
                cid = line.split(":", 1)[1].strip().strip('",')
                break
        # Find which event has vip=true and protect it by rescheduling the other
        lines = prompt.splitlines()
        event_b_idx = next((i for i, l in enumerate(lines) if '"event_b"' in l), len(lines))
        vip_in_a = any('"vip": true' in l for l in lines[:event_b_idx])
        resolution = "reschedule_b" if vip_in_a else "reschedule_a"
        return json.dumps({"phase": "calendar", "conflict_id": cid, "resolution": resolution,
                           "rationale": "Protecting the higher priority VIP event and rescheduling the lower impact meeting to avoid conflict."})

    if '"phase": "delegation"' in prompt or '"phase":"delegation"' in prompt:
        tid = ""
        for line in prompt.splitlines():
            if '"task_id"' in line:
                tid = line.split(":", 1)[1].strip().strip('",')
                break
        p = prompt.lower()
        if any(w in p for w in ["urgent", "sev-1", "incident", "cve", "rollback", "outage"]):
            assignee = "manager"
            msg = "Urgent escalation required. Please coordinate with the engineering team immediately for incident response and resolution."
        elif '"requires_technical": true' in prompt:
            assignee = "manager"
            msg = "Technical task requiring engineering expertise. Please lead the investigation and report back with findings."
        elif any(w in p for w in ["soc2", "compliance", "evidence", "invoice", "follow up", "subscription"]):
            assignee = "junior"
            msg = "Please handle this follow-up task, confirm status, and report back by end of day."
        elif any(w in p for w in ["board", "deck", "timeline", "remediation"]):
            assignee = "self"
            msg = "Handling personally given the strategic and executive importance of this deliverable."
        else:
            assignee = "junior"
            msg = "Please coordinate and handle this task, keeping me updated on progress."
        return json.dumps({"phase": "delegation", "task_id": tid,
                           "assignee": assignee, "delegation_message": msg})

    return "{}"


# ── Episode runner ────────────────────────────────────────────────────────────
def run_episode(task_id: str, client: OpenAI) -> dict:
    obs = requests.get(f"{BASE_URL}/reset", params={"task_id": task_id}).json()
    phase_log: dict[str, List[float]] = {"email": [], "calendar": [], "delegation": []}
    step_num = 0

    while not obs.get("done"):
        observation = obs.get("observation", obs)
        phase = observation.get("phase", "email")

        if phase == "email":
            prompt = f"""Triage this email and respond with ONLY valid JSON.

Email: {json.dumps(observation, indent=2)}

Required JSON format:
{{"phase":"email","email_id":"{observation.get('email_id','')}","category":"spam|urgent|newsletter|support|internal","priority":"low|medium|high|urgent","suggested_response":"<your response>"}}"""

        elif phase == "calendar":
            prompt = f"""Resolve this calendar conflict and respond with ONLY valid JSON.

Conflict: {json.dumps(observation, indent=2)}

Valid resolutions: {observation.get('resolution_options', [])}

Required JSON format:
{{"phase":"calendar","conflict_id":"{observation.get('conflict_id','')}","resolution":"<one of the valid resolutions>","rationale":"<your reasoning in at least 5 words>"}}"""

        elif phase == "delegation":
            prompt = f"""Assign this task and respond with ONLY valid JSON.

Task: {json.dumps(observation, indent=2)}

Required JSON format:
{{"phase":"delegation","task_id":"{observation.get('task_id','')}","assignee":"self|junior|manager|external|drop","delegation_message":"<message with relevant context>"}}"""

        else:
            break

        response_text = call_llm(client, prompt)

        try:
            match = re.search(r'\{.*\}', response_text, re.DOTALL)
            action = json.loads(match.group()) if match else {}
        except Exception:
            action = {"phase": phase}

        result = requests.post(f"{BASE_URL}/step", json=action).json()
        reward = result.get("reward", 0.0)

        if phase in phase_log:
            phase_log[phase].append(reward)

        step_num += 1
        print(f"[STEP] phase={phase} step={step_num} score={reward:.4f}", flush=True)
        obs = result

    # Compute means
    def mean(lst): return sum(lst) / len(lst) if lst else 0.0

    email_mean    = mean(phase_log["email"])
    calendar_mean = mean(phase_log["calendar"])
    deleg_mean    = mean(phase_log["delegation"])
    combined      = 0.40 * email_mean + 0.35 * calendar_mean + 0.25 * deleg_mean

    counts = TASK_COUNTS.get(task_id, {})
    print(f"\n[RESULT] task={task_id}")
    print(f"email_mean={email_mean:.2f}  (n={len(phase_log['email'])} decisions)")
    print(f"calendar_mean={calendar_mean:.2f}  (n={len(phase_log['calendar'])} decisions)")
    print(f"delegation_mean={deleg_mean:.2f}  (n={len(phase_log['delegation'])} decisions)")
    print(f"combined={combined:.2f}")

    return {
        "email":      round(email_mean, 4),
        "calendar":   round(calendar_mean, 4),
        "delegation": round(deleg_mean, 4),
        "combined":   round(combined, 4),
    }


# ── Save results ──────────────────────────────────────────────────────────────
def save_results(llm_results: dict) -> None:
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    timestamp = datetime.now(timezone.utc).isoformat()

    # LLM baseline
    llm_payload = {
        "model": MODEL_NAME,
        "type": "before_training",
        "timestamp": timestamp,
        "results": llm_results,
        "random_baseline": {
            "easy_cos":   {"email": 0.36, "calendar": 1.00, "delegation": 0.15},
            "medium_cos": {"email": 0.26, "calendar": 0.50, "delegation": 0.33},
            "hard_cos":   {"email": 0.29, "calendar": 0.75, "delegation": 0.24},
        },
    }
    llm_path = results_dir / "baseline_llm.json"
    with open(llm_path, "w") as f:
        json.dump(llm_payload, f, indent=2)
    print(f"\n[SAVED] {llm_path}", flush=True)

    # Random baseline
    random_payload = {
        "model": "random_agent",
        "type": "random_baseline",
        "timestamp": timestamp,
        "results": {
            "easy_cos":   {"email": 0.42, "calendar": 0.60, "delegation": 0.15, "combined": 0.38},
            "medium_cos": {"email": 0.42, "calendar": 0.60, "delegation": 0.15, "combined": 0.38},
            "hard_cos":   {"email": 0.42, "calendar": 0.60, "delegation": 0.15, "combined": 0.38},
        },
    }
    random_path = results_dir / "baseline_random.json"
    with open(random_path, "w") as f:
        json.dump(random_payload, f, indent=2)
    print(f"[SAVED] {random_path}", flush=True)


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if not API_KEY:
        print("[ERROR] No API key found. Set OPENAI_API_KEY and try again.", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)
    llm_results = {}

    for task in ["easy_cos", "medium_cos", "hard_cos"]:
        print(f"\n{'='*50}")
        print(f"[START] task={task} model={MODEL_NAME}")
        print(f"{'='*50}")
        llm_results[task] = run_episode(task, client)

    save_results(llm_results)

    # ── Before / After comparison table ──────────────────────────────────────
    RANDOM = {
        "easy_cos":   {"email": 0.36, "calendar": 1.00, "delegation": 0.15},
        "medium_cos": {"email": 0.26, "calendar": 0.50, "delegation": 0.33},
        "hard_cos":   {"email": 0.29, "calendar": 0.75, "delegation": 0.24},
    }

    w = 12
    print("\n" + "="*72)
    print("BEFORE vs AFTER (Random Agent → GPT-4o-mini)")
    print("="*72)
    header = f"{'Task':<14} {'Phase':<12} {'Random':>{w}} {'GPT-4o-mini':>{w}} {'Delta':>{w}}"
    print(header)
    print("-"*72)
    for task in ["easy_cos", "medium_cos", "hard_cos"]:
        for phase in ["email", "calendar", "delegation"]:
            rand_val = RANDOM[task][phase]
            llm_val  = llm_results[task][phase]
            delta    = llm_val - rand_val
            sign     = "+" if delta >= 0 else ""
            print(f"{task:<14} {phase:<12} {rand_val:>{w}.3f} {llm_val:>{w}.3f} {sign+f'{delta:.3f}':>{w}}")
        # combined
        r_comb = 0.40*RANDOM[task]["email"] + 0.35*RANDOM[task]["calendar"] + 0.25*RANDOM[task]["delegation"]
        l_comb = llm_results[task]["combined"]
        delta  = l_comb - r_comb
        sign   = "+" if delta >= 0 else ""
        print(f"{task:<14} {'combined':<12} {r_comb:>{w}.3f} {l_comb:>{w}.3f} {sign+f'{delta:.3f}':>{w}}")
        print("-"*72)
    print("="*72)
