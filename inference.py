"""
inference.py — Baseline agent for Email Triage OpenEnv.

Prints structured [START]/[STEP]/[END] blocks required by the validator.

Environment variables:
  API_BASE_URL  : LLM API base URL
  MODEL_NAME    : Model identifier  
  HF_TOKEN      : API key (required, no default)
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from openai import OpenAI
except ImportError:
    print("ERROR: openai package not installed.", flush=True)
    sys.exit(1)

# ── Load .env if present ──────────────────────────────────────────────────────
def _load_dotenv(path: str = ".env") -> None:
    env_file = Path(path)
    if not env_file.exists():
        return
    for raw in env_file.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key not in os.environ:
            os.environ[key] = val

_load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN")  # No default per validator requirements

# ── Import env ───────────────────────────────────────────────────────────────
try:
    from env.environment import Action, EmailTriageEnv
    from env.graders import GRADERS
    LOCAL_ENV_AVAILABLE = True
except ImportError:
    LOCAL_ENV_AVAILABLE = False

TASKS = ["easy_triage", "medium_triage", "hard_triage"]

SYSTEM_PROMPT = """You are an expert email triage agent.

For each email, output ONE JSON action:
{
  "action_type": "classify" | "reply" | "archive" | "escalate",
  "email_id": "<id>",
  "urgency": "critical" | "high" | "medium" | "low",
  "category": "customer_complaint" | "billing" | "technical_support" | "general_inquiry" | "spam" | "internal" | "sales_lead" | "hr",
  "reply_body": "<text if replying>",
  "escalation_reason": "<text if escalating>"
}

First classify all emails, then take actions."""

_FATAL = "__FATAL__"


def call_llm(client: OpenAI, messages: List[Dict]) -> Optional[str]:
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME, messages=messages, temperature=0.2, max_tokens=512,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        err = str(e)
        if any(x in err for x in ["402", "401", "insufficient_quota", "depleted"]):
            return _FATAL
        return None


def parse_action(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    text = text.replace("```json", "").replace("```", "").strip()
    s, e = text.find("{"), text.rfind("}") + 1
    if s == -1 or e == 0:
        return None
    try:
        return json.loads(text[s:e])
    except json.JSONDecodeError:
        return None


def build_prompt(obs: Dict[str, Any]) -> str:
    pending = [e for e in obs["inbox"] if e["status"] == "pending"]
    classified = [e for e in obs["inbox"] if e["status"] == "classified"]
    lines = [f"Step {obs['step_number']}/{obs['max_steps']} | Pending:{obs['pending_count']} Done:{obs['done_count']}", ""]
    emails = pending[:3] if pending else classified[:3]
    label = "CLASSIFY THESE" if pending else "ACT ON THESE"
    lines.append(f"=== {label} ===")
    for entry in emails:
        e = entry["email"]
        lines.append(f"ID:{e['id']} Subject:{e['subject']}")
        lines.append(f"Body:{e['body'][:200]}")
        if not pending:
            lines.append(f"Urgency:{entry.get('assigned_urgency')} Category:{entry.get('assigned_category')}")
    lines.append("\nOutput ONE JSON action:")
    return "\n".join(lines)


def run_task(task_id: str, client: OpenAI) -> Dict[str, Any]:
    print(f"[START] task={task_id}", flush=True)

    env = EmailTriageEnv()
    obs = env.reset(task_id=task_id)
    obs_dict = obs.model_dump()
    conversation = [{"role": "system", "content": SYSTEM_PROMPT}]
    step = 0

    while not env._is_done() and step < obs.max_steps:
        conversation.append({"role": "user", "content": build_prompt(obs_dict)})
        if len(conversation) > 12:
            conversation = [conversation[0]] + conversation[-10:]

        llm_response = call_llm(client, conversation)

        if llm_response == _FATAL:
            print(f"[STEP] step={step+1} reward=0.0 action=quota_exhausted", flush=True)
            break

        if not llm_response:
            step += 1
            continue

        conversation.append({"role": "assistant", "content": llm_response})
        action_dict = parse_action(llm_response)

        if not action_dict:
            step += 1
            continue

        try:
            action = Action(
                action_type=action_dict.get("action_type", ""),
                email_id=action_dict.get("email_id", ""),
                urgency=action_dict.get("urgency"),
                category=action_dict.get("category"),
                reply_body=action_dict.get("reply_body"),
                escalation_reason=action_dict.get("escalation_reason"),
            )
            result = env.step(action)
            obs_dict = result.observation.model_dump()

            print(
                f"[STEP] step={step+1} reward={result.reward.step_reward:.4f} "
                f"action={action.action_type} email={action.email_id}",
                flush=True,
            )

            if result.done:
                break

        except Exception:
            print(f"[STEP] step={step+1} reward=0.0 action=error", flush=True)

        step += 1
        time.sleep(0.3)

    grade = GRADERS[task_id](env)
    print(f"[END] task={task_id} score={grade['score']:.4f} steps={step}", flush=True)
    return grade


def main():
    if not HF_TOKEN:
        print("ERROR: HF_TOKEN not set.", flush=True)
        sys.exit(1)

    if not LOCAL_ENV_AVAILABLE:
        print("ERROR: Could not import env module.", flush=True)
        sys.exit(1)

    client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)
    print(f"Model: {MODEL_NAME}", flush=True)
    print(f"API:   {API_BASE_URL}", flush=True)

    results = {}
    start = time.time()

    for task_id in TASKS:
        results[task_id] = run_task(task_id, client)

    elapsed = time.time() - start
    print("=" * 60, flush=True)
    print("BASELINE RESULTS SUMMARY", flush=True)
    print("=" * 60, flush=True)
    for task_id, grade in results.items():
        print(f"  {task_id:20s}: {grade['score']:.4f}", flush=True)
    avg = sum(g["score"] for g in results.values()) / len(results)
    print(f"  {'AVERAGE':20s}: {avg:.4f}", flush=True)
    print(f"  Total time: {elapsed:.1f}s", flush=True)

    with open("baseline_scores.json", "w") as f:
        json.dump({
            "model": MODEL_NAME, "api_base": API_BASE_URL,
            "scores": {t: g["score"] for t, g in results.items()},
            "details": results, "elapsed_seconds": round(elapsed, 2),
        }, f, indent=2)
    print("Results saved to baseline_scores.json", flush=True)


if __name__ == "__main__":
    main()