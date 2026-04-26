---
title: "AI Chief of Staff — Teaching an LLM to Manage Your Monday Morning"
tags: [openenv, reinforcement-learning, grpo, llm-training, productivity]
---

# AI Chief of Staff — Teaching an LLM to Manage Your Monday Morning

## The Problem

Executives deal with three things every single day that drain their time:
- A flood of emails — which ones actually need attention right now?
- Calendar conflicts — two important meetings at the same time, which one wins?
- Tasks piling up — who should handle this, me or someone else?

We built a system that teaches an AI to handle all three, automatically, and get better at it over time.

## How the Training Works

Think of it like a video game with a score. The AI reads an email and makes a decision. If it's right, it gets a high score. If it's wrong, low score. After thousands of rounds, it learns what good decisions look like — just like a new employee learns from feedback.

The scoring is broken into three independent reward functions:
- Email grader: category (35%) + priority (25%) + response quality (25%) + urgency detection (15%)
- Calendar grader: correct resolution (50%) + VIP protection (30%) + rationale quality (20%)
- Delegation grader: correct assignee (50%) + message quality (30%) + escalation judgment (20%)

Combined episode reward = 0.40 × email + 0.35 × calendar + 0.25 × delegation

## Three Levels of Difficulty

| Level | Emails | Conflicts | Tasks | Notable |
|-------|--------|-----------|-------|---------|
| Easy | 5 | 2 | 2 | Clear right answers |
| Medium | 10 | 3 | 3 | Ambiguous categories, VIP conflicts |
| Hard | 15 | 5 | 5 | Full crisis chain — outage → meeting conflict → delegation |

## What We Built

- ✅ Full 3-phase RL environment (OpenEnv compliant)
- ✅ Three independent reward functions with anti-gaming protections
- ✅ 84 passing tests across all difficulty levels
- ✅ FastAPI web server (GET /reset, POST /step, GET /state)
- ✅ Docker deployment ready for HuggingFace Spaces
- ✅ Baseline charts comparing random agent vs trained model

## Random Agent Baseline Results

A completely random agent — one that just guesses — scored:

| Phase | Score | Notes |
|-------|-------|-------|
| Email | 0.36 | Expected from random guessing |
| Calendar | 0.75 | Random agent avoided VIP cancellations by chance |
| Delegation | 0.15 | Delegation needs real reasoning — hardest task |

Calendar scored higher (0.75) because the random agent happened to avoid cancelling VIP meetings — the hardest penalty to trigger. This makes delegation the most important signal for measuring real improvement.

![Baseline Chart](plots/difficulty_breakdown.png)

## What's Next — GRPO Training

One step remains: connect a real language model (Qwen2.5-3B via Unsloth), let it play thousands of episodes using the reward signal above, and watch the scores climb. The gap between random (0.15) and target (0.80+) on delegation is exactly what GRPO is designed to close.

Training notebook: [Link to Colab]
Live environment: [Link to HF Space]

## Technical Stack

Built on OpenEnv with FastAPI, scored by three independent reward functions, deployable to HuggingFace Spaces via Docker in one push.

- Framework: OpenEnv + FastAPI
- Training: HuggingFace TRL (GRPOTrainer) + Unsloth
- Model: Qwen2.5-3B-Instruct (4-bit, fits free Colab T4)
- Deployment: HuggingFace Spaces (Docker)
