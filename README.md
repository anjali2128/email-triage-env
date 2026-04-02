# 📧 Email Triage OpenEnv

A **real-world email triage environment** for training and evaluating AI agents. The agent must process a realistic inbox — classifying urgency, categorising emails, drafting replies, archiving spam, and escalating critical issues — just like a human support or operations professional.

---

## Why Email Triage?

Email triage is a high-value, high-volume task that every knowledge worker does daily. It requires:
- **Semantic understanding** — distinguishing a billing dispute from a security incident
- **Priority judgement** — knowing a production outage outranks a compliment email
- **Action selection** — reply vs. archive vs. escalate with appropriate reasoning
- **Writing quality** — drafting replies that actually address the issue

This makes it an excellent benchmark for evaluating general-purpose agents.

---

## Environment Overview

| Property | Value |
|----------|-------|
| Observation space | JSON object: inbox emails, statuses, step info |
| Action space | `classify`, `reply`, `archive`, `escalate` |
| Reward range | 0.0 – 1.0 (per step, partial credit) |
| Episode horizon | 15–50 steps (task-dependent) |
| Tasks | 3 (easy → medium → hard) |

---

## Action Space

Every step, the agent submits one action as JSON:

### `classify` — Label an email
```json
{
  "action_type": "classify",
  "email_id": "e001",
  "urgency": "critical",
  "category": "technical_support"
}
```
**urgency**: `critical` | `high` | `medium` | `low`

**category**: `customer_complaint` | `billing` | `technical_support` | `general_inquiry` | `spam` | `internal` | `sales_lead` | `hr`

### `reply` — Send a response
```json
{
  "action_type": "reply",
  "email_id": "e002",
  "reply_body": "We have identified the duplicate charge and will issue a refund within 2 business days."
}
```

### `archive` — No action needed
```json
{
  "action_type": "archive",
  "email_id": "e003"
}
```

### `escalate` — Hand off to a specialist
```json
{
  "action_type": "escalate",
  "email_id": "e009",
  "escalation_reason": "Security breach — requires immediate security team review."
}
```

---

## Observation Space

```json
{
  "task_id": "medium_triage",
  "task_description": "...",
  "step_number": 4,
  "max_steps": 35,
  "inbox": [
    {
      "email": {
        "id": "e001",
        "from": "angry.customer@example.com",
        "subject": "YOUR SERVICE IS DOWN",
        "body": "...",
        "timestamp": "2024-01-15T09:02:00Z"
      },
      "status": "pending",
      "assigned_urgency": null,
      "assigned_category": null,
      "action_taken": null,
      "reply_body": null,
      "escalation_reason": null
    }
  ],
  "pending_count": 8,
  "done_count": 2,
  "last_action_result": "✓ Urgency correct. ✓ Category correct.",
  "cumulative_reward": 0.42
}
```

---

## Tasks

### 🟢 Easy — Basic Email Triage
- **5 emails**: production outage, billing dispute, spam, sales lead, general question
- **Goal**: Classify urgency + category for all 5 emails
- **Max steps**: 15
- **Grader weights**: 50% urgency accuracy, 30% category accuracy, 20% coverage
- **Expected score**: Random ~0.12 | Good agent ~0.75+

### 🟡 Medium — Inbox Zero Sprint
- **10 emails**: All of the above + API rate limits, security audit, login issues, webhook failure, positive feedback
- **Goal**: Classify + take correct action (reply/archive/escalate)
- **Max steps**: 35
- **Grader weights**: 35% urgency, 25% category, 30% action correctness, 10% coverage
- **Expected score**: Weak agent ~0.20 | Strong agent ~0.65+

### 🔴 Hard — Full Support Shift
- **15 emails**: All of the above + overdue invoice, GDPR deletion request, spam variants
- **Goal**: Classify + correct action + quality replies with relevant keywords
- **Max steps**: 50
- **Grader weights**: 30% urgency (partial credit), 20% category, 30% action, 15% reply quality, 5% critical-email handling bonus
- **Expected score**: Mediocre agent ~0.15 | Frontier model ~0.55+

---

## Reward Function

Rewards are **dense** — every step provides signal:

| Action | Correct | Partial | Wrong |
|--------|---------|---------|-------|
| Classify urgency | +0.15 | +0.04–0.07 (adjacent levels) | −0.02 |
| Classify category | +0.10 | 0 | −0.02 |
| Reply (correct) | +0.05 action + up to +0.15 keywords | — | −0.05 |
| Archive (correct) | +0.10 | — | −0.05 |
| Escalate (correct + reason) | +0.15 | +0.10 no reason | −0.05 |
| Invalid action type | −0.05 | — | — |
| Invalid email ID | −0.02 | — | — |
| Redundant re-classify | −0.03 | — | — |

Reply quality is scored by keyword coverage — the grader checks whether the reply body contains expected domain terms (e.g., "refund", "invoice" for billing issues).

---

## Setup & Usage

### Local development

```bash
git clone <repo-url>
cd email-triage-env
pip install -r requirements.txt
uvicorn server:app --reload --port 7860
```

### Docker

```bash
docker build -t email-triage-env .
docker run -p 7860:7860 \
  -e API_BASE_URL=https://api.openai.com/v1 \
  -e MODEL_NAME=gpt-4o-mini \
  -e HF_TOKEN=sk-... \
  email-triage-env
```

### API usage

```bash
# Reset environment
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "easy_triage"}'

# Take a step
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action_type": "classify", "email_id": "e001", "urgency": "critical", "category": "technical_support"}'

# Get state
curl http://localhost:7860/state

# Validate spec compliance
curl http://localhost:7860/validate

# Grade current episode
curl "http://localhost:7860/grade?task_id=easy_triage"
```

### Run baseline inference

```bash
export API_BASE_URL=https://api.openai.com/v1
export MODEL_NAME=gpt-4o-mini
export HF_TOKEN=sk-...

python inference.py
# Output saved to baseline_scores.json
```

### Run tests

```bash
pytest tests/test_env.py -v
```

---

## Baseline Scores (gpt-4o-mini)

| Task | Score |
|------|-------|
| easy_triage | ~0.72 |
| medium_triage | ~0.55 |
| hard_triage | ~0.42 |
| **Average** | **~0.56** |

---

## Project Structure

```
email-triage-env/
├── env/
│   ├── __init__.py
│   ├── data.py          # Email corpus + task definitions
│   ├── environment.py   # Core env: reset/step/state + typed models
│   └── graders.py       # Task-specific deterministic graders
├── tests/
│   └── test_env.py      # Unit + integration tests
├── server.py            # FastAPI server (OpenEnv API endpoints)
├── inference.py         # Baseline agent (OpenAI client)
├── openenv.yaml         # OpenEnv metadata
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `API_BASE_URL` | LLM API endpoint | `https://api.openai.com/v1` |
| `MODEL_NAME` | Model identifier | `gpt-4o-mini` |
| `HF_TOKEN` | HuggingFace / API key | — |

---

## OpenEnv Spec Compliance

- ✅ Typed `Observation`, `Action`, `Reward` Pydantic models
- ✅ `step(action)` → `(observation, reward, done, info)`
- ✅ `reset()` → initial observation
- ✅ `state()` → full internal state
- ✅ `openenv.yaml` with metadata
- ✅ 3 tasks: easy → medium → hard
- ✅ Deterministic graders, scores in [0.0, 1.0]
- ✅ Dense reward function with partial credit
- ✅ Baseline `inference.py` using OpenAI client
- ✅ Dockerfile with `docker build && docker run`
- ✅ HF Space compatible (port 7860)
