"""
Email corpus and task definitions for the Email Triage environment.
Each email has ground-truth labels for grading.
"""
from typing import List, Dict, Any

# Urgency levels
URGENCY_LEVELS = ["critical", "high", "medium", "low"]

# Email categories
CATEGORIES = [
    "customer_complaint",
    "billing",
    "technical_support",
    "general_inquiry",
    "spam",
    "internal",
    "sales_lead",
    "hr",
]

# Valid actions
VALID_ACTIONS = ["classify", "reply", "archive", "escalate"]

# Full email corpus (25 emails, used across tasks)
EMAIL_CORPUS: List[Dict[str, Any]] = [
    {
        "id": "e001",
        "from": "angry.customer@example.com",
        "subject": "YOUR SERVICE IS DOWN AND I'M LOSING MONEY",
        "body": "Our entire production environment has been down for 2 hours. This is unacceptable. We are losing thousands of dollars per minute. I need a senior engineer on this NOW. This is a P0 incident.",
        "timestamp": "2024-01-15T09:02:00Z",
        "ground_truth": {
            "urgency": "critical",
            "category": "technical_support",
            "action": "escalate",
            "reply_keywords": ["apologize", "engineer", "priority", "escalat"],
        },
    },
    {
        "id": "e002",
        "from": "billing@client-corp.com",
        "subject": "Invoice #4521 - Duplicate Charge",
        "body": "Hi, we noticed we were charged twice for our subscription this month. Invoice #4521 and #4522 appear to be duplicates. Please review and issue a refund for one of them. Thank you.",
        "timestamp": "2024-01-15T09:15:00Z",
        "ground_truth": {
            "urgency": "high",
            "category": "billing",
            "action": "reply",
            "reply_keywords": ["refund", "invoice", "review", "billing"],
        },
    },
    {
        "id": "e003",
        "from": "newsletter@deals.example.net",
        "subject": "🔥 AMAZING DEALS THIS WEEK ONLY 🔥",
        "body": "Don't miss out! Click here for 90% off everything! Limited time offer! Unsubscribe below.",
        "timestamp": "2024-01-15T09:20:00Z",
        "ground_truth": {
            "urgency": "low",
            "category": "spam",
            "action": "archive",
            "reply_keywords": [],
        },
    },
    {
        "id": "e004",
        "from": "john.smith@potential-client.com",
        "subject": "Interested in Enterprise Plan",
        "body": "Hello, we're a 500-person company looking to migrate from our current solution. I'd love to schedule a demo and discuss pricing for an enterprise contract. Our budget is around $50k/year.",
        "timestamp": "2024-01-15T09:45:00Z",
        "ground_truth": {
            "urgency": "high",
            "category": "sales_lead",
            "action": "reply",
            "reply_keywords": ["demo", "schedule", "enterprise", "contact"],
        },
    },
    {
        "id": "e005",
        "from": "user123@gmail.com",
        "subject": "How do I export my data?",
        "body": "Hi there, I've been using your service for a few months and I'd like to know how I can export all my data as a CSV file. I looked in settings but couldn't find it. Thanks!",
        "timestamp": "2024-01-15T10:00:00Z",
        "ground_truth": {
            "urgency": "low",
            "category": "general_inquiry",
            "action": "reply",
            "reply_keywords": ["export", "settings", "csv", "data"],
        },
    },
    {
        "id": "e006",
        "from": "hr@company-internal.com",
        "subject": "Q1 All-Hands Meeting - Please RSVP",
        "body": "Hi team, our Q1 All-Hands is scheduled for Jan 20th at 2pm. Please RSVP using the link below so we can get a headcount. Light refreshments will be provided.",
        "timestamp": "2024-01-15T10:15:00Z",
        "ground_truth": {
            "urgency": "medium",
            "category": "hr",
            "action": "archive",
            "reply_keywords": [],
        },
    },
    {
        "id": "e007",
        "from": "security-alert@bank.example.com",
        "subject": "URGENT: Suspicious login attempt on your account",
        "body": "We detected a suspicious login attempt from IP 192.168.1.1 in Russia. If this was not you, please click here immediately to secure your account: http://totally-legit-bank.phishing.com/secure",
        "timestamp": "2024-01-15T10:30:00Z",
        "ground_truth": {
            "urgency": "low",
            "category": "spam",
            "action": "archive",
            "reply_keywords": [],
        },
    },
    {
        "id": "e008",
        "from": "dev@startup-client.io",
        "subject": "API rate limiting - affecting our production app",
        "body": "Hey, we're hitting rate limits on your API during peak hours (6-8pm UTC). Our users are seeing errors. We're on the Pro plan. Can we get a temporary increase or discuss Enterprise limits? This is impacting our users.",
        "timestamp": "2024-01-15T11:00:00Z",
        "ground_truth": {
            "urgency": "high",
            "category": "technical_support",
            "action": "reply",
            "reply_keywords": ["rate limit", "api", "increase", "enterprise", "plan"],
        },
    },
    {
        "id": "e009",
        "from": "cto@enterprise-client.com",
        "subject": "Security audit findings - requires immediate attention",
        "body": "Our security team completed an audit and found a potential data exposure vulnerability in your API endpoints. I'm attaching the full report. We need your security team to review this within 24 hours per our SLA. This could be a breach.",
        "timestamp": "2024-01-15T11:30:00Z",
        "ground_truth": {
            "urgency": "critical",
            "category": "technical_support",
            "action": "escalate",
            "reply_keywords": ["security", "team", "review", "24 hours", "urgent"],
        },
    },
    {
        "id": "e010",
        "from": "happy.user@example.com",
        "subject": "Love the new dashboard update!",
        "body": "Just wanted to say the new dashboard is fantastic. The performance improvements are night and day. Keep up the great work! Your team is doing amazing.",
        "timestamp": "2024-01-15T12:00:00Z",
        "ground_truth": {
            "urgency": "low",
            "category": "general_inquiry",
            "action": "archive",
            "reply_keywords": [],
        },
    },
    {
        "id": "e011",
        "from": "accounts@vendor-supplies.com",
        "subject": "Payment overdue - Invoice #8834",
        "body": "This is a reminder that Invoice #8834 for $12,450 was due on January 1st and remains unpaid. Please process this payment within 5 business days to avoid late fees and service interruption.",
        "timestamp": "2024-01-15T12:30:00Z",
        "ground_truth": {
            "urgency": "high",
            "category": "billing",
            "action": "escalate",
            "reply_keywords": ["payment", "invoice", "finance", "overdue"],
        },
    },
    {
        "id": "e012",
        "from": "new.signup@trial-user.com",
        "subject": "Can't log in after signing up",
        "body": "I just signed up 10 minutes ago and I can't log in. I keep getting 'Invalid credentials' even though I'm sure my password is correct. I haven't received a verification email either.",
        "timestamp": "2024-01-15T13:00:00Z",
        "ground_truth": {
            "urgency": "medium",
            "category": "technical_support",
            "action": "reply",
            "reply_keywords": ["verification", "email", "password", "login", "account"],
        },
    },
    {
        "id": "e013",
        "from": "partner@integration-co.com",
        "subject": "Webhook integration stopped working",
        "body": "Our webhook integration that's been running for 6 months suddenly stopped delivering events yesterday around 3pm. We haven't changed anything on our end. This is breaking our automated workflow.",
        "timestamp": "2024-01-15T13:15:00Z",
        "ground_truth": {
            "urgency": "high",
            "category": "technical_support",
            "action": "reply",
            "reply_keywords": ["webhook", "investigate", "engineer", "fix"],
        },
    },
    {
        "id": "e014",
        "from": "compliance@regulated-co.com",
        "subject": "GDPR Data Deletion Request - Legal Deadline",
        "body": "Per GDPR Article 17 (Right to Erasure), we formally request deletion of all personal data for user ID 98234 within the legally required 30-day period, which expires January 30th. Please confirm receipt and provide a deletion timeline.",
        "timestamp": "2024-01-15T14:00:00Z",
        "ground_truth": {
            "urgency": "critical",
            "category": "billing",
            "action": "escalate",
            "reply_keywords": ["gdpr", "deletion", "legal", "confirm", "compliance"],
        },
    },
    {
        "id": "e015",
        "from": "freelancer@upwork-stuff.net",
        "subject": "I can grow your Instagram to 100k followers!!!",
        "body": "Hello Business Owner! I am professional social media expert. I grow your Instagram 100k followers guaranteed! Very cheap price. Reply for special offer!",
        "timestamp": "2024-01-15T14:30:00Z",
        "ground_truth": {
            "urgency": "low",
            "category": "spam",
            "action": "archive",
            "reply_keywords": [],
        },
    },
]


def get_task_emails(task_id: str) -> List[Dict[str, Any]]:
    """Return the subset of emails for a given task."""
    if task_id == "easy_triage":
        return EMAIL_CORPUS[:5]
    elif task_id == "medium_triage":
        return EMAIL_CORPUS[:10]
    elif task_id == "hard_triage":
        return EMAIL_CORPUS[:15]
    else:
        raise ValueError(f"Unknown task_id: {task_id}")


TASK_DESCRIPTIONS = {
    "easy_triage": {
        "name": "Basic Email Triage",
        "difficulty": "easy",
        "description": (
            "You have 5 emails in your inbox. For each email, classify its urgency "
            "(critical/high/medium/low) and category, then decide to reply, archive, or escalate. "
            "You must classify all 5 emails to complete the task."
        ),
        "goal": "Classify all emails with correct urgency and category. Take the right action.",
        "max_steps": 15,
        "emails": EMAIL_CORPUS[:5],
    },
    "medium_triage": {
        "name": "Inbox Zero Sprint",
        "difficulty": "medium",
        "description": (
            "You have 10 emails to triage. Classify each email's urgency and category, "
            "then take the appropriate action: reply (with a helpful response), archive, or escalate. "
            "Emails requiring replies need substantive responses with relevant keywords."
        ),
        "goal": "Triage all 10 emails correctly. Draft helpful replies for action items.",
        "max_steps": 35,
        "emails": EMAIL_CORPUS[:10],
    },
    "hard_triage": {
        "name": "Full Support Shift",
        "difficulty": "hard",
        "description": (
            "Handle a full inbox of 15 emails of varying urgency. You must: correctly classify "
            "urgency and category for all emails, draft appropriate replies for emails needing responses, "
            "escalate critical/legal issues with clear reasons, and archive low-priority items. "
            "Partial credit is awarded for correct classifications even if actions are suboptimal."
        ),
        "goal": "Handle all 15 emails with correct classification, appropriate actions, and quality replies.",
        "max_steps": 50,
        "emails": EMAIL_CORPUS[:15],
    },
}
