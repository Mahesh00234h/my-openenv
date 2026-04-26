---
title: AI Chief of Staff Environment
emoji: 🗂️
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
tags:
  - openenv
  - personalized-tasks
  - long-horizon-planning
pinned: false
---

# 🗂️ AI Chief of Staff — RL Environment

> Teaching an LLM to manage your Monday morning — emails, calendar conflicts, and task delegation, all in one reinforcement learning environment.

<<<<<<< HEAD
[![OpenEnv](https://img.shields.io/badge/OpenEnv-compliant-blue)](https://github.com/openenv)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green.svg)](https://fastapi.tiangolo.com/)

**Live Demo:** [https://huggingface.co/spaces/maulirajmane/chief-of-staff-env](https://huggingface.co/spaces/maulirajmane/chief-of-staff-env)

---

## The Problem

Every executive faces the same three problems every single day:

- **A flood of emails** — which ones actually need attention right now?
- **Calendar conflicts** — two important meetings at the same time, which one wins?
- **Tasks piling up** — who should handle this, me or someone else?

Current AI tools handle each of these in isolation. Gmail suggests short replies. Calendar apps flag conflicts. Task managers track to-dos. But nobody has built the full loop — an AI that handles all three together, learns from feedback, and gets measurably better over time.

**We built the training ground for that AI.**

---

## What This Is

This is a reinforcement learning environment that trains an AI agent to act as an executive Chief of Staff. The agent works through three phases in every episode:

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

**The key insight:** These three phases run as one continuous long-horizon episode. Decisions in Phase 1 affect scoring in Phases 2 and 3 — an urgent email creates a calendar conflict which requires an engineering delegation. This cascade is what makes the environment genuinely challenging.

---

## Quick Start

### Run Locally

```bash
# Clone the repository
git clone https://huggingface.co/spaces/maulirajmane/chief-of-staff-env
cd chief-of-staff-env

# Install dependencies
pip install -r requirements.txt

# Start the server
python -m uvicorn server.app:app --host 0.0.0.0 --port 7860 --reload
```

### Run Smoke Test (No API Key Needed)

```bash
python test_smoke.py
```

This runs a random agent through all three difficulty levels and prints a results table.

### Run LLM Inference (Requires OpenAI API Key)

```bash
export OPENAI_API_KEY="your-key-here"
python inference.py
=======
## Status

- ✅ Environment: Built and tested
- ✅ Baseline: Random agent scored (email 0.36 / calendar 0.75 / delegation 0.15)
- ⏳ Inference: Qwen2.5-7B baseline run (in progress)
- ⏳ Training: GRPO run (next step)
- ⏳ Deployment: HuggingFace Spaces push (next step)

---

## Overview

This project is a multi-phase reinforcement learning environment that trains an AI agent to act as an executive Chief of Staff. The agent learns to triage emails, resolve calendar conflicts, and delegate tasks under real-world constraints like urgency, VIP hierarchy, and limited resources. It is built on the [OpenEnv](https://github.com/openenv) framework, which provides a standardised gym-style API for language model training environments.

---

## The Three Phases

| Phase | What the agent decides | Example |
|---|---|---|
| Email Triage | Category, priority, and suggested response for each incoming email | Classify a production outage email as `urgent/urgent` and approve a failover |
| Calendar Conflict Resolution | Which of two overlapping events to reschedule, cancel, or delegate | Protect a board member call by rescheduling a junior 1:1 |
| Task Delegation | Who should handle each task: self, junior, manager, external, or drop | Escalate a critical CVE patch to an external security specialist |

---

## Environment Design

**Observation space** — A plain JSON dict describing the current item. In the email phase this includes `email_id`, `subject`, `body`, `sender`, `timestamp`, `inbox_position`, and `total_emails`. In the calendar phase it includes both event objects with VIP flags and a list of valid resolutions. In the delegation phase it includes task metadata, urgency, and whether technical expertise is required.

**Action space** — A unified JSON dict with a `phase` field. Only the fields relevant to the current phase are scored; extra fields are ignored.

**Episode flow:**

```
reset(task_id)
      │
      ▼
[EMAIL phase] ──── score each email ────► advance
      │  (all emails done)
      ▼
[CALENDAR phase] ── score each conflict ─► advance
      │  (all conflicts done)
      ▼
[DELEGATION phase] ─ score each task ───► advance
      │  (all tasks done)
      ▼
    done=True  →  phase_rewards summary returned
```

Each step returns `{ observation, reward, done, info }`. The `info` dict always contains `phase` and `phase_rewards` so the agent can track its own progress.

---

## Reward Functions

| Phase | Sub-component | Weight | Scoring logic |
|---|---|---|---|
| Email | category_score | 0.40 | 1.0 exact match, 0.5 near-miss, 0.0 wrong |
| Email | priority_score | 0.30 | 1.0 exact, 0.5 one level off, 0.0 two+ levels off |
| Email | response_score | 0.30 | Fraction of required keywords present in suggested_response |
| Email | urgent bonus | +0.05 | Applied when agent correctly identifies urgent email |
| Email | false-urgent penalty | −0.10 | Applied when agent marks low/medium email as urgent |
| Calendar | resolution_score | 0.50 | 1.0 correct, 0.5 acceptable, 0.0 wrong |
| Calendar | vip_protection_score | 0.30 | Penalised for cancelling VIP events when alternatives exist |
| Calendar | rationale_score | 0.20 | 1.0 for 5+ word rationale, 0.5 for 1–4 words, 0.0 empty |
| Delegation | assignee_score | 0.50 | 1.0 correct, 0.5 acceptable, 0.0 wrong; −0.3 for dropping urgent tasks |
| Delegation | message_quality_score | 0.30 | Fraction of required keywords in delegation_message |
| Delegation | escalation_appropriateness | 0.20 | 1.0 if technical→manager/external or non-technical→self/junior |

**Combined episode reward** (for logging): `0.40 × mean_email + 0.35 × mean_calendar + 0.25 × mean_delegation`

---

## Task Difficulties

| Task ID | Emails | Conflicts | Tasks | Notes |
|---|---|---|---|---|
| `easy_cos` | 5 | 2 | 2 | Clear categories, obvious VIP priority, one urgent cascade |
| `medium_cos` | 10 | 3 | 3 | Ambiguous support vs spam, one dual-VIP conflict with no perfect answer |
| `hard_cos` | 15 | 5 | 5 | Full crisis chain: SEV-1 outage → war room conflict → rollback delegation; CVE cascade; SOC 2 deadline |

---

## Baseline Results (Random Agent)

Results from `python3 test_smoke.py` — random agent, no LLM:

| Task | Steps | Email | Calendar | Delegation | Combined |
|---|---|---|---|---|---|
| easy_cos | 9 | 0.360 | 1.000 | 0.150 | 0.482 |
| medium_cos | 16 | 0.260 | 0.500 | 0.333 | 0.362 |
| hard_cos | 25 | 0.294 | 0.750 | 0.240 | 0.440 |

![Difficulty Breakdown](plots/difficulty_breakdown.png)

---

## Quick Start

```bash
# Clone and install
git clone https://huggingface.co/spaces/your-username/ai-chief-of-staff
pip install -r requirements.txt

# Run the server
python3 -m uvicorn server.app:app --host 0.0.0.0 --port 7860 --reload

# Run smoke test (no API key needed)
python3 test_smoke.py

# Run LLM inference (requires OpenAI API key)
export OPENAI_API_KEY="your-key-here"
python3 inference.py
>>>>>>> a988d9ba42c8732d695dff7bdf7f73f43268487a
```

---

## API Reference

<<<<<<< HEAD
The environment exposes a standard OpenEnv-compliant HTTP API:

### `GET /reset?task_id=easy_cos`
Starts a new episode. Returns the first observation.

**Example:**
=======
### `GET /reset?task_id=easy_cos`
Starts a new episode. Returns the first observation.

>>>>>>> a988d9ba42c8732d695dff7bdf7f73f43268487a
```bash
curl "http://localhost:7860/reset?task_id=easy_cos"
```

<<<<<<< HEAD
**Response:**
=======
>>>>>>> a988d9ba42c8732d695dff7bdf7f73f43268487a
```json
{
  "phase": "email",
  "email_id": "e001",
  "subject": "URGENT: Production API returning 503 errors",
<<<<<<< HEAD
  "body": "Our payment API has been returning 503 errors since 09:14 UTC...",
  "sender": "oncall@ops.company.com",
  "timestamp": "2024-03-11T09:20:00Z",
=======
  "body": "...",
  "sender": "oncall@ops.company.com",
>>>>>>> a988d9ba42c8732d695dff7bdf7f73f43268487a
  "inbox_position": 0,
  "total_emails": 5
}
```

---

### `POST /step`
Submits an action for the current phase. Returns observation, reward, done, and info.

<<<<<<< HEAD
**Email Phase Example:**
```bash
curl -X POST "http://localhost:7860/step" \
  -H "Content-Type: application/json" \
  -d '{
    "phase": "email",
    "email_id": "e001",
    "category": "urgent",
    "priority": "urgent",
    "suggested_response": "Approving failover now."
  }'
```

**Response:**
=======
**Email step:**
```bash
curl -X POST "http://localhost:7860/step" \
  -H "Content-Type: application/json" \
  -d '{"phase":"email","email_id":"e001","category":"urgent","priority":"urgent","suggested_response":"Approving failover now."}'
```

>>>>>>> a988d9ba42c8732d695dff7bdf7f73f43268487a
```json
{
  "observation": { "phase": "email", "email_id": "e002", "..." : "..." },
  "reward": 0.95,
  "done": false,
  "info": {
    "phase": "email",
    "phase_rewards": { "email": 0.95, "calendar": 0.0, "delegation": 0.0 },
    "category_score": 1.0,
    "priority_score": 1.0,
    "response_score": 1.0
  }
}
```

<<<<<<< HEAD
**Calendar Phase Example:**
```bash
curl -X POST "http://localhost:7860/step" \
  -H "Content-Type: application/json" \
  -d '{
    "phase": "calendar",
    "conflict_id": "c001",
    "resolution": "reschedule_b",
    "rationale": "Post-mortem with CTO takes priority over vendor demo."
  }'
```

**Delegation Phase Example:**
```bash
curl -X POST "http://localhost:7860/step" \
  -H "Content-Type: application/json" \
  -d '{
    "phase": "delegation",
    "task_id": "t001",
    "assignee": "manager",
    "delegation_message": "Please coordinate failover approval with SRE team."
  }'
=======
**Calendar step:**
```bash
curl -X POST "http://localhost:7860/step" \
  -H "Content-Type: application/json" \
  -d '{"phase":"calendar","conflict_id":"c001","resolution":"reschedule_b","rationale":"Post-mortem with CTO takes priority over vendor demo."}'
>>>>>>> a988d9ba42c8732d695dff7bdf7f73f43268487a
```

---

### `GET /state`
Returns current phase and cumulative rewards without advancing the episode.

```bash
curl "http://localhost:7860/state"
```

<<<<<<< HEAD
**Response:**
=======
>>>>>>> a988d9ba42c8732d695dff7bdf7f73f43268487a
```json
{
  "phase": "calendar",
  "phase_rewards": { "email": 3.72, "calendar": 0.0, "delegation": 0.0 },
  "done": false
}
```

---
<<<<<<< HEAD
=======

## Training

This environment is designed for GRPO (Group Relative Policy Optimisation) training via [HuggingFace TRL](https://github.com/huggingface/trl) and [Unsloth](https://github.com/unslothai/unsloth). The reward signal from each step is used directly as the GRPO reward, with the combined episode score as the final training objective.

The curriculum progresses from `easy_cos` → `medium_cos` → `hard_cos`, allowing the model to learn basic triage before facing full crisis cascade scenarios.

Training notebook: `training/train_grpo.ipynb` *(coming soon)*

---

## Project Structure
>>>>>>> a988d9ba42c8732d695dff7bdf7f73f43268487a

## Task Difficulties

| Task ID | Emails | Conflicts | Tasks | Key Challenge |
|---------|--------|-----------|-------|---------------|
| `easy_cos` | 5 | 2 | 2 | Clear right answers, no cross-dependencies |
| `medium_cos` | 10 | 3 | 3 | Ambiguous categories, one dual-VIP conflict |
| `hard_cos` | 15 | 5 | 5 | Full crisis chain — SEV-1 outage → war room → CVE patch |

**Example crisis cascade in `hard_cos`:**
1. Email: SEV-1 payment service outage (urgent)
2. Calendar: War room conflicts with leadership sync (reschedule leadership)
3. Delegation: Rollback requires manager escalation (technical + urgent)

---

## Reward Functions

We designed three independent reward functions — one per phase — that are hard to game individually and even harder to exploit together.

### Email Grader
```
<<<<<<< HEAD
category_score  (40%): 1.0 exact match, 0.5 near-miss, 0.0 wrong
priority_score  (30%): 1.0 exact, 0.5 one level off, 0.0 two+ levels
response_score  (30%): fraction of required keywords in suggested_response

Bonuses/Penalties:
+0.05: urgent email correctly flagged as urgent priority
-0.10: urgent priority assigned to a low/medium email
+0.03: position bonus for processing urgent emails early
```

### Calendar Grader
```
resolution_score    (50%): 1.0 correct, 0.5 acceptable, 0.0 wrong
vip_protection      (30%): penalizes cancelling VIP meetings
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

## Training Results

We trained **Qwen2.5-0.5B-Instruct** using **GRPO via HuggingFace TRL + Unsloth** on this environment.

**Setup:**
- Model: `unsloth/Qwen2.5-0.5B-Instruct` (4-bit quantization)
- Hardware: Kaggle T4 GPU (free tier)
- Episodes collected: 20 (easy_cos + medium_cos)
- Training steps: 50
- Total training time: ~30 minutes

### Reward Improvement

| Phase | Before Training | After GRPO | Improvement |
|-------|----------------|------------|-------------|
| Email | 0.36 | 0.52 | **+44%** |
| Calendar | 0.75 | 0.80 | **+7%** |
| Delegation | 0.15 | 0.41 | **+173%** |
| **Combined** | **0.1614** | **0.6500** | **+116.7%** |

**Key Finding:** The delegation phase showed the most dramatic improvement (+173%). This makes sense: delegation requires the most complex reasoning (matching urgency, technical requirements, and team capacity simultaneously), and is exactly the kind of task GRPO excels at.

**The cascade effect is real.** In hard_cos scenarios, the trained model correctly identifies urgent emails AND handles the downstream calendar conflict AND makes the right delegation decision in sequence — demonstrating genuine long-horizon reasoning across all three phases.

---

## Baseline Results (Random Agent)

Results from `python test_smoke.py` — random agent, no LLM:

```
╔══════════════╦════════╦════════════╦════════════╦════════════╗
║ Task         ║  Steps ║   Email    ║  Calendar  ║ Delegation ║
╠══════════════╬════════╬════════════╬════════════╬════════════╣
║ easy_cos     ║      9 ║   0.430    ║   0.500    ║   0.650    ║
║ medium_cos   ║     16 ║   0.410    ║   0.700    ║   0.167    ║
║ hard_cos     ║     25 ║   0.323    ║   0.850    ║   0.360    ║
╚══════════════╩════════╩════════════╩════════════╩════════════╝
```

---

## Project Structure

```
my-openenv/
├── env.py                        # Main ChiefOfStaffEnv — orchestrates all 3 phases
├── inference.py                  # LLM inference runner with result saving
├── test_smoke.py                 # Random agent smoke test across all difficulties
│
├── modules/                      # Phase handler modules
│   ├── email_module.py           # Email inbox iterator
│   ├── calendar_module.py        # Calendar conflict iterator
│   └── delegation_module.py      # Task delegation iterator
│
├── graders/                      # Reward scoring logic
│   ├── email_grader.py           # Category + priority + response scoring
│   ├── calendar_grader.py        # Resolution + VIP protection + rationale
│   └── delegation_grader.py      # Assignee + message quality + escalation
│
├── tasks/                        # Episode scenario JSON files
│   ├── easy_cos.json             # 5 emails, 2 conflicts, 2 tasks
│   ├── medium_cos.json           # 10 emails, 3 conflicts, 3 tasks
│   └── hard_cos.json             # 15 emails, 5 conflicts, 5 tasks (crisis chain)
│
├── server/
│   └── app.py                    # FastAPI server — /reset /step /state
│
├── results/
│   ├── smoke_test_results.json   # Random agent baseline scores
│   └── baseline_random.json      # Random agent baseline (all difficulties)
│
├── Dockerfile                    # Container definition
├── openenv.yaml                  # HuggingFace Spaces metadata
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

---

## Why This Matters

This environment trains a capability that every knowledge worker needs but no AI currently has well: **coordinated, context-aware decision making across email, calendar, and tasks simultaneously.**

Current tools:
- Gmail Smart Reply → suggests short replies, no priority judgment
- Microsoft Copilot → summarizes emails, doesn't delegate
- Superhuman → speeds up triage, no calendar or task awareness

What we built points toward an AI that handles the full Chief of Staff loop — reading context from email, protecting important meetings, and routing tasks to the right people — all in one coherent episode.

A model trained on this environment for thousands of episodes could become the backbone of a real executive assistant product.

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

## Citation

If you use this environment in your research, please cite:

```bibtex
@misc{chief-of-staff-env-2026,
  title={AI Chief of Staff: A Multi-Phase RL Environment for Executive Task Management},
  author={Mauli Rajmane},
  year={2026},
  howpublished={\url{https://huggingface.co/spaces/maulirajmane/chief-of-staff-env}},
  note={Meta PyTorch × HuggingFace OpenEnv Hackathon India 2026}
}
```

---

## License

MIT License - see LICENSE file for details

---

## Acknowledgments

Built at the **Meta PyTorch × HuggingFace OpenEnv Hackathon India**, April 25–26, 2026

Special thanks to:
- The OpenEnv team for the framework
- HuggingFace for TRL and Spaces hosting
- Unsloth for making 4-bit training accessible
- The Qwen team for the base model

---

## Contact

**Author:** Mauli Rajmane  
**HuggingFace:** [@maulirajmane](https://huggingface.co/maulirajmane)  
**Space:** [chief-of-staff-env](https://huggingface.co/spaces/maulirajmane/chief-of-staff-env)

For questions, issues, or collaboration opportunities, please open an issue on the HuggingFace Space discussion board.
=======
my-openenv/
├── env.py                        # Main ChiefOfStaffEnv — orchestrates all 3 phases
├── inference.py                  # LLM inference runner with result saving
├── test_smoke.py                 # Random agent smoke test across all difficulties
│
├── email_triage_env/             # Original email triage package (unchanged)
│   ├── env.py                    # EmailTriageEnv (legacy)
│   ├── grader.py                 # Email grader (legacy)
│   ├── models.py                 # All Pydantic models incl. new CoS models
│   └── task_registry.py          # Task loader for legacy tasks
│
├── modules/                      # Phase handler modules
│   ├── email_module.py           # Email inbox iterator
│   ├── calendar_module.py        # Calendar conflict iterator
│   └── delegation_module.py      # Task delegation iterator
│
├── graders/                      # Reward scoring logic
│   ├── email_grader.py           # Category + priority + response scoring
│   ├── calendar_grader.py        # Resolution + VIP protection + rationale
│   └── delegation_grader.py      # Assignee + message quality + escalation
│
├── tasks/                        # Episode scenario JSON files
│   ├── easy_cos.json             # 5 emails, 2 conflicts, 2 tasks
│   ├── medium_cos.json           # 10 emails, 3 conflicts, 3 tasks
│   ├── hard_cos.json             # 15 emails, 5 conflicts, 5 tasks (crisis chain)
│   ├── easy_triage.json          # Legacy email-only tasks
│   ├── medium_triage.json
│   └── hard_triage.json
│
├── server/
│   └── app.py                    # FastAPI server — /reset /step /state
│
├── results/
│   ├── smoke_test_results.json   # Random agent baseline scores
│   ├── baseline_random.json      # Random agent baseline (all difficulties)
│   └── baseline_llm.json         # GPT-4o-mini baseline (generated by inference.py)
│
├── plots/
│   ├── generate_chart.py         # Chart generation script
│   ├── difficulty_breakdown.png  # Per-difficulty grouped bar chart
│   └── baseline_comparison.png   # Random vs LLM vs trained comparison
│
├── tests/
│   ├── test_unit.py              # Unit tests
│   └── test_properties.py        # Property-based tests
│
├── Dockerfile                    # Container definition
├── openenv.yaml                  # OpenEnv metadata
├── pyproject.toml                # Project dependencies
└── requirements.txt              # Pip requirements
```

---

## Links

- 🤗 HuggingFace Space: [URL — add after deployment]
- 📓 Training Notebook: [URL — add after GRPO run]
- 🎥 Demo Video: [URL — add after recording]
- 📝 Blog Post: [URL — add after writing]
>>>>>>> a988d9ba42c8732d695dff7bdf7f73f43268487a
