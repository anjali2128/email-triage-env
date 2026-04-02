"""
FastAPI server for the Email Triage OpenEnv environment.
Exposes: POST /reset, POST /step, GET /state, GET /tasks, GET /validate
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from env.environment import Action, EmailTriageEnv
from env.graders import GRADERS
from env.data import TASK_DESCRIPTIONS

app = FastAPI(
    title="Email Triage OpenEnv",
    description=(
        "A real-world email triage environment. "
        "An AI agent classifies, replies to, archives, and escalates emails."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global env instance (single-session; for multi-session use session tokens)
_env = EmailTriageEnv()


# ── request bodies ───────────────────────────────


class ResetRequest(BaseModel):
    task_id: str = "easy_triage"


class StepRequest(BaseModel):
    action_type: str
    email_id: str
    urgency: Optional[str] = None
    category: Optional[str] = None
    reply_body: Optional[str] = None
    escalation_reason: Optional[str] = None


# ── endpoints ────────────────────────────────────


@app.get("/")
def root():
    return {
        "name": "email-triage-env",
        "version": "1.0.0",
        "tasks": list(TASK_DESCRIPTIONS.keys()),
        "endpoints": ["/reset", "/step", "/state", "/tasks", "/validate", "/grade"],
    }


@app.post("/reset")
def reset(req: Optional[ResetRequest] = None):
    """Reset environment for a new episode on the given task."""
    try:
        task_id = req.task_id if req else "easy_triage"
        obs = _env.reset(task_id=task_id)
        return obs.model_dump()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step")
def step(req: StepRequest):
    """Execute one agent action."""
    action = Action(
        action_type=req.action_type,
        email_id=req.email_id,
        urgency=req.urgency,
        category=req.category,
        reply_body=req.reply_body,
        escalation_reason=req.escalation_reason,
    )
    try:
        result = _env.step(action)
        return result.model_dump()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/state")
def state():
    """Return full internal environment state."""
    return _env.state()


@app.get("/tasks")
def tasks():
    """List all available tasks with metadata."""
    return {
        task_id: {
            "name": meta["name"],
            "difficulty": meta["difficulty"],
            "description": meta["description"],
            "goal": meta["goal"],
            "max_steps": meta["max_steps"],
            "email_count": len(meta["emails"]),
        }
        for task_id, meta in TASK_DESCRIPTIONS.items()
    }


@app.get("/validate")
def validate():
    """OpenEnv spec validation — checks all required endpoints respond correctly."""
    checks: Dict[str, Any] = {}

    # Check reset
    try:
        obs = _env.reset(task_id="easy_triage")
        checks["reset"] = "pass" if obs.task_id == "easy_triage" else "fail"
    except Exception as e:
        checks["reset"] = f"fail: {e}"

    # Check step
    try:
        from env.data import EMAIL_CORPUS
        action = Action(
            action_type="classify",
            email_id=EMAIL_CORPUS[0]["id"],
            urgency="critical",
            category="technical_support",
        )
        result = _env.step(action)
        checks["step"] = "pass" if result.reward.step_reward is not None else "fail"
    except Exception as e:
        checks["step"] = f"fail: {e}"

    # Check state
    try:
        st = _env.state()
        checks["state"] = "pass" if "task_id" in st else "fail"
    except Exception as e:
        checks["state"] = f"fail: {e}"

    # Check graders
    grader_results = {}
    for task_id, grader in GRADERS.items():
        try:
            _env.reset(task_id=task_id)
            result = grader(_env)
            ok = 0.0 <= result["score"] <= 1.0
            grader_results[task_id] = "pass" if ok else f"fail: score={result['score']}"
        except Exception as e:
            grader_results[task_id] = f"fail: {e}"
    checks["graders"] = grader_results

    all_pass = all(
        v == "pass" for v in checks.values() if isinstance(v, str)
    ) and all(v == "pass" for v in checks["graders"].values())

    return {"status": "pass" if all_pass else "fail", "checks": checks}


@app.get("/grade")
def grade(task_id: str = "easy_triage"):
    """Run the grader on the current episode state."""
    if task_id not in GRADERS:
        raise HTTPException(status_code=400, detail=f"Unknown task_id: {task_id}")
    result = GRADERS[task_id](_env)
    return result