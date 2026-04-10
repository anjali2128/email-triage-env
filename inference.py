"""
inference.py — Baseline agent for Email Triage OpenEnv.

Uses the OpenAI client (compatible API) to run an LLM agent against all 3 tasks.
Reads credentials from environment variables (or .env file):
  - API_BASE_URL   : LLM API base URL
  - MODEL_NAME     : Model identifier
  - OPENAI_API_KEY : OpenAI API key  (preferred when using api.openai.com)
  - HF_TOKEN       : HuggingFace token (only for HF Inference endpoints)

Usage:
  python inference.py
  python inference.py --task easy_triage
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from openai import OpenAI
except ImportError:
    print("ERROR: openai package not installed. Run: pip install openai")
    sys.exit(1)


# ── Minimal .env loader (runs before anything else) ──────────────────────────
def _load_dotenv(path: str = ".env") -> None:
    """Load .env file. Shell-exported vars always take priority."""
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
        if key not in os.environ:          # never overwrite shell exports
            os.environ[key] = val

_load_dotenv()


# ── Try importing env directly ───────────────────────────────────────────────
try:
    from env.environment import Action, EmailTriageEnv
    from env.graders import GRADERS
    from env.data import TASK_DESCRIPTIONS
    LOCAL_ENV_AVAILABLE = True
except ImportError:
    LOCAL_ENV_AVAILABLE = False


# ── Config ────────────────────────────────────────────────────────────────────
# Key selection logic:
#   - OPENAI_API_KEY (sk-...) must be used against api.openai.com
#   - HF_TOKEN      (hf_...) must be used against HF Inference API
#   - They are NOT interchangeable — using hf_... against OpenAI returns 401.
#
# Rule: prefer OPENAI_API_KEY; fall back to HF_TOKEN only if OPENAI_API_KEY absent.

# HuggingFace Inference API is the default — free, no credit card, works with hf_ tokens.
# To use OpenAI instead: set API_BASE_URL=https://api.openai.com/v1 and OPENAI_API_KEY=sk-...
_HF_BASE    = "https://router.huggingface.co/v1"
_HF_MODEL   = "Qwen/Qwen2.5-72B-Instruct"   # free on HF Inference

API_BASE_URL = os.environ.get("API_BASE_URL", _HF_BASE)
MODEL_NAME   = os.environ.get("MODEL_NAME",   _HF_MODEL)

_openai_key = os.environ.get("OPENAI_API_KEY", "").strip()
_hf_token   = os.environ.get("HF_TOKEN", "").strip()

# Key selection: for HF endpoints use HF_TOKEN; for OpenAI use OPENAI_API_KEY
_using_hf_endpoint = "huggingface.co" in API_BASE_URL or "hf.co" in API_BASE_URL or "router.huggingface" in API_BASE_URL
if _using_hf_endpoint:
    HF_TOKEN = _hf_token if _hf_token else _openai_key   # HF token preferred for HF
else:
    HF_TOKEN = _openai_key if _openai_key else _hf_token  # OpenAI key preferred for OpenAI


TASKS = ["easy_triage", "medium_triage", "hard_triage"]

SYSTEM_PROMPT = """You are an expert email triage agent. Your job is to process an inbox
of emails efficiently and accurately.

For each email you MUST:
1. CLASSIFY it with urgency (critical/high/medium/low) and category
   (customer_complaint, billing, technical_support, general_inquiry, spam, internal, sales_lead, hr)
2. Take ONE of these actions:
   - REPLY with a helpful response (for emails needing human attention)
   - ARCHIVE (for spam, low-priority, or no-action-needed emails)
   - ESCALATE with a reason (for critical issues, legal matters, security incidents)

Rules:
- critical urgency = production outages, security breaches, legal deadlines
- high urgency = significant business impact, paying customer issues
- medium urgency = general support, onboarding issues
- low urgency = spam, compliments, general questions

Output ONE action at a time as valid JSON with this exact structure:
{
  "action_type": "classify" | "reply" | "archive" | "escalate",
  "email_id": "<id>",
  "urgency": "<level>",          // only for classify
  "category": "<category>",      // only for classify
  "reply_body": "<text>",        // only for reply
  "escalation_reason": "<text>"  // only for escalate
}

Process emails in two steps: first classify, then action. Be concise but thorough."""


def build_user_message(obs: Dict[str, Any]) -> str:
    """Build a clear prompt from the current observation."""
    pending = [e for e in obs["inbox"] if e["status"] == "pending"]
    classified = [e for e in obs["inbox"] if e["status"] == "classified"]

    lines = [
        f"Task: {obs['task_description']}",
        f"Step {obs['step_number']}/{obs['max_steps']} | "
        f"Pending: {obs['pending_count']} | Done: {obs['done_count']}",
        f"Last result: {obs.get('last_action_result', 'N/A')}",
        "",
    ]

    if pending:
        lines.append("=== PENDING EMAILS (need classification) ===")
        for entry in pending[:3]:  # Show max 3 at a time
            e = entry["email"]
            lines.append(f"\nID: {e['id']}")
            lines.append(f"From: {e.get('from', e.get('from_address', ''))}")
            lines.append(f"Subject: {e['subject']}")
            lines.append(f"Body: {e['body'][:300]}")
    elif classified:
        lines.append("=== CLASSIFIED EMAILS (need action) ===")
        for entry in classified[:3]:
            e = entry["email"]
            lines.append(f"\nID: {e['id']}")
            lines.append(f"From: {e.get('from', e.get('from_address', ''))}")
            lines.append(f"Subject: {e['subject']}")
            lines.append(f"Body: {e['body'][:300]}")
            lines.append(f"Urgency: {entry.get('assigned_urgency')} | Category: {entry.get('assigned_category')}")

    lines.append("\nOutput ONE JSON action now.")
    return "\n".join(lines)


# Sentinel returned when the API key is exhausted / unauthorized — stop immediately
_FATAL_ERROR = "__FATAL__"

def call_llm(client: OpenAI, messages: List[Dict]) -> Optional[str]:
    """Call the LLM. Returns None on transient error, _FATAL_ERROR on quota/auth failure."""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.2,
            max_tokens=512,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        err_str = str(e)
        print(f"  [LLM error] {e}")
        # 402 = out of credits, 401 = bad key — both are fatal, stop looping
        if "402" in err_str or "401" in err_str or "insufficient_quota" in err_str or "depleted" in err_str:
            return _FATAL_ERROR
        return None


def parse_action(text: str) -> Optional[Dict[str, Any]]:
    """Parse JSON action from LLM output, handling code blocks."""
    if not text:
        return None
    # Strip markdown code blocks
    text = text.replace("```json", "").replace("```", "").strip()
    # Find JSON object
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        return None
    try:
        return json.loads(text[start:end])
    except json.JSONDecodeError:
        return None


def run_local_task(task_id: str, client: OpenAI) -> Dict[str, Any]:
    """Run one task using the local environment (no HTTP)."""
    print(f"[START] task={task_id}", flush=True)

    env = EmailTriageEnv()
    obs = env.reset(task_id=task_id)
    obs_dict = obs.model_dump()

    conversation = [{"role": "system", "content": SYSTEM_PROMPT}]
    step = 0
    max_steps = obs.max_steps

    while not env._is_done() and step < max_steps:
        user_msg = build_user_message(obs_dict)
        conversation.append({"role": "user", "content": user_msg})

        # Keep conversation manageable
        if len(conversation) > 12:
            conversation = [conversation[0]] + conversation[-10:]

        llm_response = call_llm(client, conversation)
        if llm_response == _FATAL_ERROR:
            print(f"  [FATAL] API quota exhausted — stopping task early. Partial score will be saved.", flush=True)
            break
        if not llm_response:
            step += 1
            continue

        conversation.append({"role": "assistant", "content": llm_response})
        action_dict = parse_action(llm_response)

        if not action_dict:
            print(f"  Step {step+1}: Could not parse action from LLM output", flush=True)
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
                f"[STEP] step={step+1} reward={result.reward.step_reward:.3f} "
                f"action={action.action_type} email={action.email_id}",
                flush=True
            )

            if result.done:
                break

        except Exception as e:
            print(f"  Step {step+1}: Error applying action: {e}", flush=True)

        step += 1
        time.sleep(0.3)  # Rate limit friendliness

    # Final grade
    grader = GRADERS[task_id]
    grade = grader(env)
    print(f"[END] task={task_id} score={grade['score']:.4f} steps={step}", flush=True)
    return grade


def main():
    parser = argparse.ArgumentParser(description="Email Triage OpenEnv baseline agent")
    parser.add_argument("--task", default=None, help="Run only this task (default: all)")
    parser.add_argument("--local", action="store_true", help="Use local env (no HTTP)")
    args = parser.parse_args()

    if not HF_TOKEN:
        print("ERROR: No API key found.")
        print("  Set OPENAI_API_KEY=sk-... in your .env or shell for OpenAI.")
        print("  Set HF_TOKEN=hf_...   in your .env or shell for HF Inference.")
        sys.exit(1)

    # Guard: hf_ token against openai.com always returns 401
    if HF_TOKEN.startswith("hf_") and "openai.com" in API_BASE_URL:
        print("ERROR: Your HF_TOKEN (hf_...) cannot be used with api.openai.com.")
        print("  Fix: remove API_BASE_URL from .env (defaults to HF Inference),")
        print("  or set OPENAI_API_KEY=sk-... and point API_BASE_URL to OpenAI.")
        sys.exit(1)

    key_hint = HF_TOKEN[:12] + "..." if len(HF_TOKEN) > 12 else HF_TOKEN
    key_src  = "HF_TOKEN" if (os.environ.get("HF_TOKEN") and HF_TOKEN == os.environ.get("HF_TOKEN","").strip()) else "OPENAI_API_KEY"
    client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)
    print(f"Model:  {MODEL_NAME}")
    print(f"API:    {API_BASE_URL}")
    print(f"Key:    {key_hint} (from {key_src})")

    tasks_to_run = [args.task] if args.task else TASKS
    results = {}
    start = time.time()

    for task_id in tasks_to_run:
        if task_id not in TASKS:
            print(f"Unknown task: {task_id}. Valid: {TASKS}")
            continue
        result = run_local_task(task_id, client)
        results[task_id] = result

    elapsed = time.time() - start
    print(f"\n{'='*60}")
    print("BASELINE RESULTS SUMMARY")
    print('='*60)
    for task_id, grade in results.items():
        print(f"  {task_id:20s}: {grade['score']:.4f}")
    if results:
        avg = sum(g["score"] for g in results.values()) / len(results)
        print(f"  {'AVERAGE':20s}: {avg:.4f}")
    print(f"\n  Total time: {elapsed:.1f}s")
    print('='*60)

    # Write results to file for reproducibility
    with open("baseline_scores.json", "w") as f:
        json.dump(
            {
                "model": MODEL_NAME,
                "api_base": API_BASE_URL,
                "scores": {t: g["score"] for t, g in results.items()},
                "details": results,
                "elapsed_seconds": round(elapsed, 2),
            },
            f,
            indent=2,
        )
    print("Results saved to baseline_scores.json")


if __name__ == "__main__":
    main()