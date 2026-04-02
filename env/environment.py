"""
Email Triage OpenEnv — core environment implementation.
Implements step() / reset() / state() with full typed Pydantic models.
"""
from __future__ import annotations

import copy
import time
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from env.data import (
    CATEGORIES,
    TASK_DESCRIPTIONS,
    URGENCY_LEVELS,
    VALID_ACTIONS,
    get_task_emails,
)


# ─────────────────────────────────────────────
# Typed models (OpenEnv spec)
# ─────────────────────────────────────────────


class Email(BaseModel):
    id: str
    from_address: str = Field(default="", alias="from")
    subject: str
    body: str
    timestamp: str

    class Config:
        populate_by_name = True


class EmailWithStatus(BaseModel):
    email: Email
    status: str = "pending"          # pending | classified | actioned | done
    assigned_urgency: Optional[str] = None
    assigned_category: Optional[str] = None
    action_taken: Optional[str] = None
    reply_body: Optional[str] = None
    escalation_reason: Optional[str] = None


class Observation(BaseModel):
    task_id: str
    task_description: str
    step_number: int
    max_steps: int
    inbox: List[Dict[str, Any]]      # serialized EmailWithStatus list
    pending_count: int
    done_count: int
    last_action_result: Optional[str] = None
    cumulative_reward: float = 0.0


class Action(BaseModel):
    action_type: str = Field(
        description="One of: classify, reply, archive, escalate"
    )
    email_id: str = Field(description="ID of the email to act on")
    # classify fields
    urgency: Optional[str] = Field(
        default=None,
        description="One of: critical, high, medium, low"
    )
    category: Optional[str] = Field(
        default=None,
        description=(
            "One of: customer_complaint, billing, technical_support, "
            "general_inquiry, spam, internal, sales_lead, hr"
        ),
    )
    # reply field
    reply_body: Optional[str] = Field(
        default=None,
        description="Body of the reply to send"
    )
    # escalate field
    escalation_reason: Optional[str] = Field(
        default=None,
        description="Reason for escalation"
    )


class Reward(BaseModel):
    step_reward: float = Field(description="Reward earned this step (0.0–1.0 scaled)")
    cumulative_reward: float
    breakdown: Dict[str, float] = Field(
        description="Component-wise reward breakdown"
    )


class StepResult(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: Dict[str, Any]


# ─────────────────────────────────────────────
# Environment
# ─────────────────────────────────────────────


class EmailTriageEnv:
    """
    OpenEnv-compliant Email Triage environment.

    An agent triages a realistic inbox by:
      1. classify(email_id, urgency, category)  — label the email
      2. reply(email_id, reply_body)            — write a response
         OR archive(email_id)                  — mark as no action needed
         OR escalate(email_id, reason)         — hand off to a specialist

    Grader scores urgency accuracy, category accuracy, action correctness,
    and reply quality (keyword coverage). Partial credit at every step.
    """

    def __init__(self) -> None:
        self.task_id: Optional[str] = None
        self._emails: List[EmailWithStatus] = []
        self._step_num: int = 0
        self._max_steps: int = 50
        self._cumulative_reward: float = 0.0
        self._last_result: Optional[str] = None
        self._action_log: List[Dict[str, Any]] = []
        self._start_time: float = time.time()

    # ── public OpenEnv API ──────────────────────

    def reset(self, task_id: str = "easy_triage") -> Observation:
        """Reset environment for a new episode on the specified task."""
        if task_id not in TASK_DESCRIPTIONS:
            raise ValueError(
                f"Unknown task_id '{task_id}'. "
                f"Valid: {list(TASK_DESCRIPTIONS.keys())}"
            )

        self.task_id = task_id
        task = TASK_DESCRIPTIONS[task_id]
        raw_emails = get_task_emails(task_id)

        self._emails = [
            EmailWithStatus(
                email=Email(
                    id=e["id"],
                    **{k: v for k, v in e.items() if k != "id"},
                )
            )
            for e in raw_emails
        ]
        self._step_num = 0
        self._max_steps = task["max_steps"]
        self._cumulative_reward = 0.0
        self._last_result = "Episode started. Triage your inbox."
        self._action_log = []
        self._start_time = time.time()

        return self._build_observation()

    def step(self, action: Action) -> StepResult:
        """Execute one agent action and return (observation, reward, done, info)."""
        if self.task_id is None:
            raise RuntimeError("Call reset() before step().")

        self._step_num += 1
        step_reward, breakdown, message = self._apply_action(action)
        self._cumulative_reward = min(1.0, self._cumulative_reward + step_reward)
        self._last_result = message

        self._action_log.append(
            {
                "step": self._step_num,
                "action": action.model_dump(),
                "reward": step_reward,
                "message": message,
            }
        )

        done = self._is_done()
        obs = self._build_observation()
        reward = Reward(
            step_reward=round(step_reward, 4),
            cumulative_reward=round(self._cumulative_reward, 4),
            breakdown=breakdown,
        )

        # Final-episode info
        info: Dict[str, Any] = {
            "step": self._step_num,
            "max_steps": self._max_steps,
            "task_id": self.task_id,
        }
        if done:
            final_score = self._compute_episode_score()
            info["final_score"] = final_score
            info["episode_summary"] = self._episode_summary()

        return StepResult(observation=obs, reward=reward, done=done, info=info)

    def state(self) -> Dict[str, Any]:
        """Return full internal state (for debugging / logging)."""
        return {
            "task_id": self.task_id,
            "step_number": self._step_num,
            "max_steps": self._max_steps,
            "cumulative_reward": self._cumulative_reward,
            "emails": [e.model_dump() for e in self._emails],
            "action_log": self._action_log,
            "episode_score": self._compute_episode_score() if self.task_id else 0.0,
        }

    # ── action handling ─────────────────────────

    def _apply_action(
        self, action: Action
    ) -> tuple[float, Dict[str, float], str]:
        """Validate and apply an action; return (step_reward, breakdown, message)."""
        breakdown: Dict[str, float] = {}

        # Validate action_type
        if action.action_type not in VALID_ACTIONS:
            breakdown["invalid_action"] = -0.05
            return -0.05, breakdown, f"Invalid action_type '{action.action_type}'."

        # Validate email_id
        email_entry = self._find_email(action.email_id)
        if email_entry is None:
            breakdown["invalid_email"] = -0.02
            return -0.02, breakdown, f"Email '{action.email_id}' not found."

        gt = self._get_ground_truth(action.email_id)

        if action.action_type == "classify":
            return self._handle_classify(action, email_entry, gt, breakdown)
        elif action.action_type == "reply":
            return self._handle_reply(action, email_entry, gt, breakdown)
        elif action.action_type == "archive":
            return self._handle_archive(action, email_entry, gt, breakdown)
        elif action.action_type == "escalate":
            return self._handle_escalate(action, email_entry, gt, breakdown)

        return 0.0, breakdown, "No-op."

    def _handle_classify(self, action, entry, gt, breakdown):
        reward = 0.0
        msgs = []

        # Urgency scoring (0.0–0.15 per email)
        if action.urgency not in URGENCY_LEVELS:
            breakdown["urgency_invalid"] = -0.02
            reward -= 0.02
            msgs.append(f"Invalid urgency '{action.urgency}'.")
        else:
            entry.assigned_urgency = action.urgency
            if action.urgency == gt["urgency"]:
                breakdown["urgency_correct"] = 0.15
                reward += 0.15
                msgs.append("✓ Urgency correct.")
            else:
                # Partial credit for adjacent levels
                idx_pred = URGENCY_LEVELS.index(action.urgency)
                idx_true = URGENCY_LEVELS.index(gt["urgency"])
                distance = abs(idx_pred - idx_true)
                partial = max(0.0, 0.07 - distance * 0.03)
                breakdown["urgency_partial"] = partial
                reward += partial
                msgs.append(
                    f"✗ Urgency: got '{action.urgency}', expected '{gt['urgency']}'. "
                    f"Partial credit: {partial:.2f}."
                )

        # Category scoring (0.0–0.10 per email)
        if action.category not in CATEGORIES:
            breakdown["category_invalid"] = -0.02
            reward -= 0.02
            msgs.append(f"Invalid category '{action.category}'.")
        else:
            entry.assigned_category = action.category
            if action.category == gt["category"]:
                breakdown["category_correct"] = 0.10
                reward += 0.10
                msgs.append("✓ Category correct.")
            else:
                breakdown["category_wrong"] = 0.0
                msgs.append(
                    f"✗ Category: got '{action.category}', expected '{gt['category']}'."
                )

        # Penalize re-classification (agent already classified this)
        if entry.status not in ("pending",):
            breakdown["redundant"] = -0.03
            reward -= 0.03
            msgs.append("⚠ Email already classified — small penalty.")
        else:
            entry.status = "classified"

        total = max(-0.1, reward)
        return total, breakdown, " ".join(msgs)

    def _handle_reply(self, action, entry, gt, breakdown):
        reward = 0.0
        msgs = []

        if not action.reply_body or len(action.reply_body.strip()) < 10:
            breakdown["empty_reply"] = -0.05
            return -0.05, breakdown, "Reply body too short — must be at least 10 chars."

        # Check action is appropriate
        if gt["action"] != "reply":
            breakdown["wrong_action"] = -0.05
            reward -= 0.05
            msgs.append(
                f"✗ Action 'reply' not ideal here (expected '{gt['action']}')."
            )
        else:
            breakdown["action_correct"] = 0.05
            reward += 0.05
            msgs.append("✓ Reply is the correct action.")

        # Keyword coverage in reply body
        keywords = gt.get("reply_keywords", [])
        if keywords:
            body_lower = action.reply_body.lower()
            matched = sum(1 for kw in keywords if kw.lower() in body_lower)
            coverage = matched / len(keywords)
            kw_reward = round(coverage * 0.15, 4)
            breakdown["reply_keywords"] = kw_reward
            reward += kw_reward
            msgs.append(
                f"Reply keyword coverage: {matched}/{len(keywords)} ({coverage:.0%}) → +{kw_reward:.2f}"
            )
        else:
            # No reply expected but agent replied anyway — small penalty
            breakdown["unnecessary_reply"] = -0.03
            reward -= 0.03
            msgs.append("⚠ Reply not needed for this email.")

        # Reply length quality bonus
        length = len(action.reply_body.split())
        if 15 <= length <= 150:
            breakdown["reply_length_ok"] = 0.02
            reward += 0.02

        entry.reply_body = action.reply_body
        entry.action_taken = "reply"
        entry.status = "done"

        return round(max(-0.1, reward), 4), breakdown, " ".join(msgs)

    def _handle_archive(self, action, entry, gt, breakdown):
        reward = 0.0
        if gt["action"] == "archive":
            breakdown["action_correct"] = 0.10
            reward += 0.10
            msg = "✓ Correctly archived."
        else:
            breakdown["wrong_action"] = -0.05
            reward -= 0.05
            msg = f"✗ Archive not ideal (expected '{gt['action']}')."

        entry.action_taken = "archive"
        entry.status = "done"
        return round(reward, 4), breakdown, msg

    def _handle_escalate(self, action, entry, gt, breakdown):
        reward = 0.0
        msgs = []

        if gt["action"] == "escalate":
            breakdown["action_correct"] = 0.10
            reward += 0.10
            msgs.append("✓ Escalation is correct.")
        else:
            breakdown["wrong_action"] = -0.05
            reward -= 0.05
            msgs.append(f"✗ Escalation not ideal (expected '{gt['action']}').")

        # Check escalation reason quality
        if action.escalation_reason and len(action.escalation_reason.strip()) >= 10:
            breakdown["escalation_reason"] = 0.05
            reward += 0.05
            msgs.append("✓ Reason provided.")
        else:
            msgs.append("⚠ No escalation reason given — include a reason.")

        entry.escalation_reason = action.escalation_reason
        entry.action_taken = "escalate"
        entry.status = "done"

        return round(max(-0.1, reward), 4), breakdown, " ".join(msgs)

    # ── helpers ────────────────────────────────

    def _find_email(self, email_id: str) -> Optional[EmailWithStatus]:
        for e in self._emails:
            if e.email.id == email_id:
                return e
        return None

    def _get_ground_truth(self, email_id: str) -> Dict[str, Any]:
        from env.data import EMAIL_CORPUS
        for e in EMAIL_CORPUS:
            if e["id"] == email_id:
                return e["ground_truth"]
        return {}

    def _is_done(self) -> bool:
        if self._step_num >= self._max_steps:
            return True
        all_done = all(e.status == "done" for e in self._emails)
        return all_done

    def _build_observation(self) -> Observation:
        inbox = []
        for e in self._emails:
            d = e.model_dump()
            # Handle 'from' alias: model_dump may use 'from_address' as the key
            email_dict = d.get("email", {})
            if isinstance(email_dict, dict):
                if "from_address" in email_dict:
                    email_dict["from"] = email_dict.pop("from_address")
                d["email"] = email_dict
            inbox.append(d)

        return Observation(
            task_id=self.task_id or "",
            task_description=TASK_DESCRIPTIONS.get(self.task_id, {}).get(
                "description", ""
            ),
            step_number=self._step_num,
            max_steps=self._max_steps,
            inbox=inbox,
            pending_count=sum(1 for e in self._emails if e.status == "pending"),
            done_count=sum(1 for e in self._emails if e.status == "done"),
            last_action_result=self._last_result,
            cumulative_reward=round(self._cumulative_reward, 4),
        )

    def _compute_episode_score(self) -> float:
        """Compute a 0.0–1.0 grader score for the full episode."""
        if not self._emails:
            return 0.0

        total = 0.0
        max_per_email = 0.45  # 0.15 urgency + 0.10 category + 0.10 action + 0.10 reply

        for entry in self._emails:
            gt = self._get_ground_truth(entry.email.id)
            score = 0.0

            # Urgency
            if entry.assigned_urgency == gt.get("urgency"):
                score += 0.15
            elif entry.assigned_urgency:
                idx_pred = URGENCY_LEVELS.index(entry.assigned_urgency) if entry.assigned_urgency in URGENCY_LEVELS else 99
                idx_true = URGENCY_LEVELS.index(gt.get("urgency", "low")) if gt.get("urgency") in URGENCY_LEVELS else 99
                distance = abs(idx_pred - idx_true)
                score += max(0.0, 0.07 - distance * 0.03)

            # Category
            if entry.assigned_category == gt.get("category"):
                score += 0.10

            # Action
            if entry.action_taken == gt.get("action"):
                score += 0.10

            # Reply quality
            if entry.action_taken == "reply" and entry.reply_body:
                keywords = gt.get("reply_keywords", [])
                if keywords:
                    body_lower = entry.reply_body.lower()
                    matched = sum(1 for kw in keywords if kw.lower() in body_lower)
                    score += (matched / len(keywords)) * 0.10

            total += score

        raw = total / (len(self._emails) * max_per_email)
        return round(min(1.0, raw), 4)

    def _episode_summary(self) -> Dict[str, Any]:
        correct_urgency = sum(
            1
            for e in self._emails
            if e.assigned_urgency == self._get_ground_truth(e.email.id).get("urgency")
        )
        correct_action = sum(
            1
            for e in self._emails
            if e.action_taken == self._get_ground_truth(e.email.id).get("action")
        )
        total = len(self._emails)
        return {
            "total_emails": total,
            "correct_urgency": correct_urgency,
            "correct_action": correct_action,
            "urgency_accuracy": round(correct_urgency / total, 3) if total else 0,
            "action_accuracy": round(correct_action / total, 3) if total else 0,
            "final_score": self._compute_episode_score(),
            "steps_used": self._step_num,
        }
