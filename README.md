---
title: Email Triage Environment
emoji: 📧
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
tags:
  - openenv
pinned: false
---

# Email Triage Environment

An [OpenEnv](https://openenv.dev)-compliant reinforcement learning environment that simulates real-world email triage. An AI agent reads incoming emails one at a time and must decide how to categorize, prioritize, and respond to each one. The environment provides a dense reward signal at every step — not just at episode end — making it well-suited for studying reading comprehension, intent classification, priority judgment, and response drafting in an interactive setting.

The environment exposes a standard HTTP API (`/reset`, `/step`, `/state`) backed by FastAPI, ships with three tasks of increasing difficulty, and is ready to deploy on Hugging Face Spaces via Docker.

## Observation Space

Each call to `/reset` or `/step` returns an `Observation` object with the following fields:

| Field | Type | Description |
|---|---|---|
| `email_id` | `str` | Unique identifier for the email within the task (e.g. `"e001"`) |
| `subject` | `str` | Email subject line |
| `body` | `str` | Full email body text |
| `sender` | `str` | Sender email address |
| `timestamp` | `str` | ISO-8601 timestamp of when the email was received |
| `inbox_position` | `int` | 0-indexed position of this email in the inbox |
| `total_emails` | `int` | Total number of emails in the current task's inbox |

A terminal observation (returned when `done=True`) has an empty `email_id` and `body`.

## Action Space

Each call to `/step` expects an `Action` object with the following fields:

| Field | Type | Valid Values | Description |
|---|---|---|---|
| `email_id` | `str` | Must match current observation's `email_id` | Identifies which email is being triaged |
| `category` | `str` | `spam`, `urgent`, `newsletter`, `support`, `internal` | Classification of the email |
| `priority` | `str` | `low`, `medium`, `high`, `urgent` | Handling priority |
| `suggested_response` | `str` | Any string | Draft reply text (can be empty for emails that need no response) |

If `email_id` does not match the current email, the step returns `score=0.0` with an explanation of the mismatch.

## Task Descriptions

### `easy_triage` — 5 emails

Introductory difficulty. All emails have unambiguous categories that can be determined from the subject line alone. No multi-step reasoning or body reading is required. Suitable for verifying basic classification capability.

### `medium_triage` — 10 emails

Intermediate difficulty. Emails require disambiguation between similar categories (e.g. urgent vs. high-priority support requests). At least 3 emails require reading the full body to determine the correct priority. Tests reading comprehension and nuanced classification.

### `hard_triage` — 15 emails

Advanced difficulty. Emails contain ambiguous intent, require cross-email context (e.g. a reply thread), and at least 5 emails require a non-trivial suggested response to score above 0.5. Tests the full range of triage skills including response drafting.

## Reward Function

The grader scores each action on three sub-components:

- **Category score** (weight 0.4): `1.0` exact match, `0.5` near-miss, `0.0` otherwise.
- **Priority score** (weight 0.3): `1.0` exact match, `0.5` one level off on `[low, medium, high, urgent]`, `0.0` two or more levels off.
- **Response score** (weight 0.3): fraction of required keywords present in `suggested_response`.

Final score: `0.4 * category + 0.3 * priority + 0.3 * response`, with:
- `+0.05` bonus (capped at 1.0) for correctly identifying an urgent email as urgent.
- `-0.1` penalty (floored at 0.0) for assigning `urgent` priority to a `low` or `medium` email.

All scores are in `[0.0, 1.0]`.

## Setup Instructions

### Run locally with Docker

```bash
# Build the image
docker build -t openenv-email .

# Run the server on port 7860
docker run -p 7860:7860 openenv-email
```

The API will be available at `http://localhost:7860`. Interactive docs at `http://localhost:7860/docs`.

### API usage

```bash
# Start a new episode
curl "http://localhost:7860/reset?task_id=easy_triage"

# Submit an action
curl -X POST "http://localhost:7860/step" \
  -H "Content-Type: application/json" \
  -d '{"email_id":"e001","category":"spam","priority":"low","suggested_response":""}'

# Check episode state
curl "http://localhost:7860/state"
```

### Run the baseline inference script

Set the required environment variables and run `inference.py` against a running server:

```bash
export OPENAI_API_KEY="your-api-key"
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"

python inference.py
```

The script runs all three tasks sequentially and emits structured log lines:

```
[START] task=easy_triage
[STEP] task=easy_triage step=1 score=0.8500
...
[END] task=easy_triage total_score=3.9200
```

### Run tests

```bash
pip install -r requirements.txt
pytest tests/
```

## Baseline Scores

Reference scores using `gpt-4o-mini` via the baseline inference script:

| Task | Emails | Mean Score (per email) | Notes |
|---|---|---|---|
| `easy_triage` | 5 | ~0.72 | Unambiguous categories; strong models score well |
| `medium_triage` | 10 | ~0.58 | Disambiguation required; body reading needed for 3+ emails |
| `hard_triage` | 15 | ~0.41 | Ambiguous intent, deceptive subjects, cross-email context |

Difficulty gradient is intentional: easy tasks establish a baseline, hard tasks differentiate frontier models from weaker ones. A random baseline scores approximately 0.18 per email.

To reproduce baseline scores:

```bash
export API_KEY="<your key>"
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"
docker run -p 7860:7860 openenv-email &
sleep 5
python inference.py
```
