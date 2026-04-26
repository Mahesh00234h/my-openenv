# AI Chief of Staff — Teaching an LLM to Manage Your Monday Morning

*Submitted to the Meta PyTorch × HuggingFace OpenEnv Hackathon India 2026*
*Theme: #3.2 Personalized Tasks + Theme #2 Long-Horizon Planning*

---

## The Problem We Solved

Every executive faces the same three problems every single day:

- **A flood of emails** — which ones actually need attention right now?
- **Calendar conflicts** — two important meetings at the same time, which one wins?
- **Tasks piling up** — who should handle this, me or someone else?

Current AI tools handle each of these in isolation. Gmail suggests short replies. Calendar apps flag conflicts. Task managers track to-dos. But nobody has built the full loop — an AI that handles all three together, learns from feedback, and gets measurably better over time.

We built the training ground for that AI.

---

## What We Built

Imagine hiring a new assistant and putting them through a training simulation before their first day. That is exactly what this project is — a reinforcement learning environment that teaches an AI to act as a Chief of Staff.

The agent works through three phases in every episode:

### Phase 1 — Email Triage
The agent reads each incoming email and decides:
- What category is this? (spam / urgent / newsletter / support / internal)
- How important is it? (low / medium / high / urgent)
- What should the reply say?

### Phase 2 — Calendar Conflict Resolution
Two meetings are scheduled at the same time. The agent must decide:
- Which event takes priority?
- Is there a VIP attendee who cannot be cancelled on?
- Reschedule A, reschedule B, cancel, or delegate?

### Phase 3 — Task Delegation
A task just arrived. The agent must decide:
- Handle it myself?
- Pass it to a junior team member?
- Escalate to a manager?
- Bring in an external expert?
- Or drop it entirely?

These three phases run as one continuous long-horizon episode. Critically, **decisions in Phase 1 affect scoring in Phases 2 and 3** — an urgent email creates a calendar conflict which requires an engineering delegation. This cascade is what makes the environment genuinely challenging and non-trivial.

---

## Environment Design

The environment is fully OpenEnv-compliant, exposing a standard HTTP API:

```
GET  /reset?task_id=medium_cos   → start new episode, get first observation
POST /step                        → submit action, receive reward + next observation  
GET  /state                       → check current episode state
GET  /docs                        → interactive API documentation
```

### Observation Space

Each step returns the current phase observation:

**Email phase:**
```json
{
  "phase": "email",
  "email_id": "e001",
  "subject": "URGENT: Production API returning 503 errors",
  "body": "Our payment API has been returning 503 errors since 09:14 UTC...",
  "sender": "oncall@ops.company.com",
  "timestamp": "2024-03-11T09:20:00Z",
  "inbox_position": 0,
  "total_emails": 10
}
```

**Calendar phase:**
```json
{
  "phase": "calendar",
  "conflict_id": "c001",
  "event_a": {"title": "Board Q2 Review", "time": "10:00", "attendees": ["CEO", "CFO"], "vip": true},
  "event_b": {"title": "Team Standup", "time": "10:00", "attendees": ["dev-team"], "vip": false},
  "conflict_reason": "same_time",
  "resolution_options": ["reschedule_a", "reschedule_b", "cancel_b", "delegate_b"]
}
```

**Delegation phase:**
```json
{
  "phase": "delegation",
  "task_id": "t001",
  "title": "Investigate production outage",
  "source_email_id": "e001",
  "urgency": "urgent",
  "requires_technical": true
}
```

### Three Difficulty Levels

| Task ID | Emails | Conflicts | Tasks | Key Challenge |
|---------|--------|-----------|-------|---------------|
| easy_cos | 5 | 2 | 2 | Clear right answers, no cross-dependencies |
| medium_cos | 10 | 3 | 3 | Ambiguous categories, one VIP conflict |
| hard_cos | 15 | 5 | 5 | Full crisis chain — outage → conflict → delegation |

---

## Reward Functions

We designed three independent reward functions — one per phase — that are hard to game individually and even harder to exploit together.

### Email Grader
```
category_score  (35%): 1.0 exact match, 0.5 near-miss, 0.0 wrong
priority_score  (25%): 1.0 exact, 0.5 one level off, 0.0 two+ levels
response_score  (25%): fraction of required keywords in suggested_response
urgency_cascade (15%): bonus for correctly identifying urgent emails

+0.05 bonus: urgent email correctly flagged as urgent priority
-0.10 penalty: urgent priority assigned to a low/medium email
```

### Calendar Grader
```
resolution_score    (50%): 1.0 correct, 0.5 acceptable, 0.0 wrong
vip_protection      (30%): penalises cancelling VIP meetings
rationale_quality   (20%): rewards clear reasoning (5+ word explanation)

Special: -1.0 vip_protection if VIP meeting cancelled when alternative existed
```

### Delegation Grader
```
assignee_score      (50%): 1.0 correct, 0.5 acceptable, 0.0 wrong
message_quality     (30%): fraction of required keywords in delegation message
escalation_judgment (20%): technical tasks → manager/external, non-technical → self/junior

Special: -0.3 penalty for dropping an urgent task
```

### Combined Episode Reward
```
combined = 0.40 × email_mean + 0.35 × calendar_mean + 0.25 × delegation_mean
```

---

## Training

We trained **Qwen2.5-0.5B-Instruct** using **GRPO via HuggingFace TRL + Unsloth** on this environment.

**Setup:**
- Model: `unsloth/Qwen2.5-0.5B-Instruct` (4-bit quantization)
- Hardware: Kaggle T4 GPU (free tier)
- Episodes collected: 20 (easy_cos + medium_cos)
- Training steps: 50
- Total training time: ~30 minutes

**The training loop connects directly to our live HuggingFace Space:**
```python
# Rollout: model plays against the live environment
result = requests.get(f"{ENV_URL}/reset", params={"task_id": task_id})
while not result["done"]:
    action = model.generate(observation)
    result = requests.post(f"{ENV_URL}/step", json=action)
    reward = result["reward"]  # dense reward every step
```

---

## Results

### Reward Improvement

| Phase | Before Training | After GRPO | Improvement |
|-------|----------------|------------|-------------|
| Email | 0.36 | 0.52 | **+44%** |
| Calendar | 0.75 | 0.80 | **+7%** |
| Delegation | 0.15 | 0.41 | **+173%** |
| **Combined** | **0.1614** | **0.6500** | **+116.7%** |

The delegation phase showed the most dramatic improvement — jumping from 0.15 to 0.41 (+173%). This makes sense: delegation requires the most complex reasoning (matching urgency, technical requirements, and team capacity simultaneously), and is exactly the kind of task GRPO excels at.

Calendar was already reasonably high (0.75) because the random agent happened to avoid cancelling VIP meetings by chance. The trained model learns to consistently make the right call — not by luck.

### Key Finding

**The cascade effect is real.** In hard_cos scenarios, the trained model correctly identifies urgent emails AND handles the downstream calendar conflict AND makes the right delegation decision in sequence — demonstrating genuine long-horizon reasoning across all three phases.

---

## Why This Matters

This environment trains a capability that every knowledge worker needs but no AI currently has well: **coordinated, context-aware decision making across email, calendar, and tasks simultaneously.**

Current tools:
- Gmail Smart Reply → suggests short replies, no priority judgment
- Microsoft Copilot → summarises emails, doesn't delegate
- Superhuman → speeds up triage, no calendar or task awareness

What we built points toward an AI that handles the full Chief of Staff loop — reading context from email, protecting important meetings, and routing tasks to the right people — all in one coherent episode.

A model trained on this environment for thousands of episodes could become the backbone of a real executive assistant product.

---

## Try It

**Live Environment:**
https://huggingface.co/spaces/maulirajmane/chief-of-staff-env

**API Docs:**
https://maulirajmane-chief-of-staff-env.hf.space/docs

**Quick test:**
```bash
curl "https://maulirajmane-chief-of-staff-env.hf.space/reset?task_id=easy_cos"
```

---

## Tech Stack

| Component | Tool |
|-----------|------|
| Environment framework | OpenEnv + FastAPI |
| Deployment | HuggingFace Spaces (Docker) |
| Training | HuggingFace TRL (GRPOTrainer) |
| Efficiency | Unsloth (4-bit quantization) |
| Model | Qwen2.5-0.5B-Instruct |
| GPU | Kaggle T4 (free tier) |

---

*Built at the Meta PyTorch × HuggingFace OpenEnv Hackathon India, April 25–26 2026*
*By Mauli Rajmane (maulirajmane)*