"""
Tests for the Email Triage OpenEnv environment.
Run with: python -m pytest tests/test_env.py -v
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from env.environment import Action, EmailTriageEnv
from env.graders import GRADERS


@pytest.fixture
def env():
    return EmailTriageEnv()


# ── reset() tests ────────────────────────────────

def test_reset_easy(env):
    obs = env.reset("easy_triage")
    assert obs.task_id == "easy_triage"
    assert obs.step_number == 0
    assert obs.pending_count == 5
    assert obs.done_count == 0


def test_reset_medium(env):
    obs = env.reset("medium_triage")
    assert obs.pending_count == 10


def test_reset_hard(env):
    obs = env.reset("hard_triage")
    assert obs.pending_count == 15


def test_reset_invalid_task(env):
    with pytest.raises(ValueError):
        env.reset("nonexistent_task")


def test_reset_clears_state(env):
    env.reset("easy_triage")
    # Take an action
    action = Action(action_type="classify", email_id="e001", urgency="critical", category="technical_support")
    env.step(action)
    assert env._step_num == 1
    # Reset
    obs = env.reset("easy_triage")
    assert env._step_num == 0
    assert obs.pending_count == 5


# ── step() tests ─────────────────────────────────

def test_step_before_reset(env):
    with pytest.raises(RuntimeError):
        action = Action(action_type="classify", email_id="e001", urgency="low", category="spam")
        env.step(action)


def test_classify_correct(env):
    env.reset("easy_triage")
    action = Action(
        action_type="classify",
        email_id="e001",
        urgency="critical",
        category="technical_support",
    )
    result = env.step(action)
    assert result.reward.step_reward > 0
    assert result.reward.breakdown.get("urgency_correct", 0) == 0.15
    assert result.reward.breakdown.get("category_correct", 0) == 0.10


def test_classify_wrong_urgency(env):
    env.reset("easy_triage")
    action = Action(
        action_type="classify",
        email_id="e001",
        urgency="low",  # Wrong — should be critical
        category="technical_support",
    )
    result = env.step(action)
    assert result.reward.breakdown.get("urgency_correct", 0) == 0
    assert result.reward.step_reward < 0.15  # Less than full credit


def test_archive_spam_correctly(env):
    env.reset("easy_triage")
    action = Action(action_type="archive", email_id="e003")
    result = env.step(action)
    assert result.reward.breakdown.get("action_correct", 0) == 0.10


def test_archive_wrong_email(env):
    env.reset("easy_triage")
    action = Action(action_type="archive", email_id="e001")  # Should escalate
    result = env.step(action)
    assert result.reward.step_reward < 0


def test_reply_with_keywords(env):
    env.reset("easy_triage")
    action = Action(
        action_type="reply",
        email_id="e002",  # Billing issue
        reply_body="Thank you for letting us know. We will review the duplicate invoice and issue a refund to your billing account.",
    )
    result = env.step(action)
    assert result.reward.step_reward > 0.05


def test_reply_too_short(env):
    env.reset("easy_triage")
    action = Action(action_type="reply", email_id="e002", reply_body="Ok")
    result = env.step(action)
    assert result.reward.step_reward < 0


def test_escalate_critical_with_reason(env):
    env.reset("easy_triage")
    action = Action(
        action_type="escalate",
        email_id="e001",
        escalation_reason="Production outage — requires immediate senior engineer response.",
    )
    result = env.step(action)
    assert result.reward.breakdown.get("action_correct", 0) == 0.10
    assert result.reward.breakdown.get("escalation_reason", 0) == 0.05


def test_invalid_action_type(env):
    env.reset("easy_triage")
    action = Action(action_type="delete", email_id="e001")
    result = env.step(action)
    assert result.reward.step_reward < 0


def test_invalid_email_id(env):
    env.reset("easy_triage")
    action = Action(action_type="classify", email_id="e999", urgency="low", category="spam")
    result = env.step(action)
    assert result.reward.step_reward < 0


def test_done_when_all_actioned(env):
    env.reset("easy_triage")
    email_ids = ["e001", "e002", "e003", "e004", "e005"]
    actions = [
        Action(action_type="escalate", email_id="e001", escalation_reason="Critical outage"),
        Action(action_type="reply", email_id="e002", reply_body="We will review the billing and issue a refund for the duplicate invoice."),
        Action(action_type="archive", email_id="e003"),
        Action(action_type="reply", email_id="e004", reply_body="Happy to schedule a demo for your enterprise team. Please contact our sales team."),
        Action(action_type="reply", email_id="e005", reply_body="You can export your data as CSV from Settings > Data Export."),
    ]
    result = None
    for action in actions:
        result = env.step(action)
    assert result.done is True


def test_done_on_max_steps(env):
    env.reset("easy_triage")
    env._max_steps = 2
    action = Action(action_type="classify", email_id="e001", urgency="low", category="spam")
    env.step(action)
    result = env.step(action)
    assert result.done is True


# ── state() tests ────────────────────────────────

def test_state_structure(env):
    env.reset("easy_triage")
    st = env.state()
    assert "task_id" in st
    assert "emails" in st
    assert "action_log" in st
    assert "episode_score" in st


# ── grader tests ─────────────────────────────────

def test_graders_return_valid_scores(env):
    for task_id, grader in GRADERS.items():
        env.reset(task_id)
        result = grader(env)
        assert "score" in result
        assert 0.0 <= result["score"] <= 1.0, f"{task_id}: score {result['score']} out of range"


def test_perfect_easy_triage():
    """Simulate a near-perfect run of easy_triage and check score."""
    env = EmailTriageEnv()
    env.reset("easy_triage")

    perfect_actions = [
        Action(action_type="escalate", email_id="e001", escalation_reason="Production outage — critical, needs senior engineer now."),
        Action(action_type="reply", email_id="e002", reply_body="We sincerely apologize. We've identified the duplicate billing on invoice #4521 and will issue a refund within 2 business days."),
        Action(action_type="archive", email_id="e003"),
        Action(action_type="reply", email_id="e004", reply_body="We'd love to schedule a demo! Our enterprise plan fits your needs. I'll have someone from our sales contact you this week."),
        Action(action_type="reply", email_id="e005", reply_body="Great question! You can export your data as a CSV file in Settings > Data Export."),
    ]
    for act in perfect_actions:
        env.step(act)

    # Also classify them for full score
    env2 = EmailTriageEnv()
    env2.reset("easy_triage")
    classify_actions = [
        Action(action_type="classify", email_id="e001", urgency="critical", category="technical_support"),
        Action(action_type="classify", email_id="e002", urgency="high", category="billing"),
        Action(action_type="classify", email_id="e003", urgency="low", category="spam"),
        Action(action_type="classify", email_id="e004", urgency="high", category="sales_lead"),
        Action(action_type="classify", email_id="e005", urgency="low", category="general_inquiry"),
    ]
    for act in classify_actions:
        env2.step(act)

    grade = GRADERS["easy_triage"](env2)
    assert grade["score"] > 0.5, f"Expected good score, got {grade['score']}"


def test_grader_deterministic():
    """Same episode should always produce same grader score."""
    def run_episode():
        e = EmailTriageEnv()
        e.reset("easy_triage")
        e.step(Action(action_type="classify", email_id="e001", urgency="critical", category="technical_support"))
        e.step(Action(action_type="archive", email_id="e003"))
        return GRADERS["easy_triage"](e)["score"]

    assert run_episode() == run_episode()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
