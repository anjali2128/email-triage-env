"""
Agent graders for each task. Deterministic, reproducible 0.0–1.0 scores.
"""
from __future__ import annotations
from typing import Dict, Any

from env.environment import EmailTriageEnv
from env.data import URGENCY_LEVELS


def grade_easy_triage(env: EmailTriageEnv) -> Dict[str, Any]:
    """
    Easy task grader: 5 emails, score purely on urgency + category accuracy.
    Expected baseline: random agent ~0.15, good agent ~0.75+
    """
    state = env.state()
    emails = state["emails"]
    total = len(emails)
    if total == 0:
        return {"score": 0.0, "details": "No emails found"}

    urgency_correct = 0
    category_correct = 0
    actioned = 0

    for entry in emails:
        email_id = entry["email"]["id"]
        gt = env._get_ground_truth(email_id)

        if entry.get("assigned_urgency") == gt.get("urgency"):
            urgency_correct += 1
        if entry.get("assigned_category") == gt.get("category"):
            category_correct += 1
        if entry.get("action_taken") is not None:
            actioned += 1

    # Weighted: 50% urgency, 30% category, 20% coverage
    score = (
        0.50 * (urgency_correct / total)
        + 0.30 * (category_correct / total)
        + 0.20 * (actioned / total)
    )

    return {
        "score": round(score, 4),
        "urgency_accuracy": round(urgency_correct / total, 3),
        "category_accuracy": round(category_correct / total, 3),
        "coverage": round(actioned / total, 3),
        "total_emails": total,
    }


def grade_medium_triage(env: EmailTriageEnv) -> Dict[str, Any]:
    """
    Medium task grader: 10 emails, score on urgency + category + correct action.
    Expected baseline: weak agent ~0.2, strong agent ~0.65+
    """
    state = env.state()
    emails = state["emails"]
    total = len(emails)
    if total == 0:
        return {"score": 0.0, "details": "No emails found"}

    urgency_correct = 0
    category_correct = 0
    action_correct = 0
    covered = 0

    for entry in emails:
        email_id = entry["email"]["id"]
        gt = env._get_ground_truth(email_id)

        if entry.get("assigned_urgency") == gt.get("urgency"):
            urgency_correct += 1
        if entry.get("assigned_category") == gt.get("category"):
            category_correct += 1
        if entry.get("action_taken") == gt.get("action"):
            action_correct += 1
        if entry.get("action_taken") is not None:
            covered += 1

    score = (
        0.35 * (urgency_correct / total)
        + 0.25 * (category_correct / total)
        + 0.30 * (action_correct / total)
        + 0.10 * (covered / total)
    )

    return {
        "score": round(score, 4),
        "urgency_accuracy": round(urgency_correct / total, 3),
        "category_accuracy": round(category_correct / total, 3),
        "action_accuracy": round(action_correct / total, 3),
        "coverage": round(covered / total, 3),
        "total_emails": total,
    }


def grade_hard_triage(env: EmailTriageEnv) -> Dict[str, Any]:
    """
    Hard task grader: 15 emails, full scoring including reply quality.
    Expected baseline: mediocre agent ~0.15, frontier model ~0.55+
    """
    state = env.state()
    emails = state["emails"]
    total = len(emails)
    if total == 0:
        return {"score": 0.0, "details": "No emails found"}

    urgency_score = 0.0
    category_correct = 0
    action_correct = 0
    reply_score = 0.0
    critical_handled = 0
    critical_total = 0

    for entry in emails:
        email_id = entry["email"]["id"]
        gt = env._get_ground_truth(email_id)

        # Urgency with partial credit
        pred_urg = entry.get("assigned_urgency")
        true_urg = gt.get("urgency", "low")
        if pred_urg == true_urg:
            urgency_score += 1.0
        elif pred_urg in URGENCY_LEVELS and true_urg in URGENCY_LEVELS:
            dist = abs(URGENCY_LEVELS.index(pred_urg) - URGENCY_LEVELS.index(true_urg))
            urgency_score += max(0.0, 1.0 - dist * 0.4)

        if entry.get("assigned_category") == gt.get("category"):
            category_correct += 1
        if entry.get("action_taken") == gt.get("action"):
            action_correct += 1

        # Reply quality
        if gt.get("action") == "reply" and entry.get("reply_body"):
            keywords = gt.get("reply_keywords", [])
            if keywords:
                body_lower = entry["reply_body"].lower()
                matched = sum(1 for kw in keywords if kw.lower() in body_lower)
                reply_score += matched / len(keywords)
            else:
                reply_score += 0.5

        # Critical email handling bonus/malus
        if true_urg == "critical":
            critical_total += 1
            if entry.get("action_taken") in ("escalate", "reply"):
                critical_handled += 1

    reply_emails = sum(1 for e in emails if env._get_ground_truth(e["email"]["id"]).get("action") == "reply")
    reply_norm = reply_score / max(1, reply_emails)
    critical_bonus = (critical_handled / max(1, critical_total)) * 0.05 if critical_total else 0.0

    score = (
        0.30 * (urgency_score / total)
        + 0.20 * (category_correct / total)
        + 0.30 * (action_correct / total)
        + 0.15 * reply_norm
        + critical_bonus
    )

    return {
        "score": round(min(1.0, score), 4),
        "urgency_score": round(urgency_score / total, 3),
        "category_accuracy": round(category_correct / total, 3),
        "action_accuracy": round(action_correct / total, 3),
        "reply_quality": round(reply_norm, 3),
        "critical_handling": round(critical_handled / max(1, critical_total), 3),
        "total_emails": total,
    }


GRADERS = {
    "easy_triage": grade_easy_triage,
    "medium_triage": grade_medium_triage,
    "hard_triage": grade_hard_triage,
}
