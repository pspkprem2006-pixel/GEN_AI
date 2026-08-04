# Project 1: Email Automation

## Overview
An automated workflow that personalizes and "sends" an email to everyone in a list. This demonstrates the classic **trigger → process → action** automation pattern.

## How It Works
1. **Trigger**: Loop over a list of recipients
2. **Process**: Build a personalized prompt and generate an email
3. **Action**: "Send" the email (saves to `outbox/` in mock mode)

## Files
- `email_automation.py` - Main automation script
- `outbox/` - Generated emails are saved here (created automatically)

## How to Run
```bash
python email_automation.py
```

## Configuration
- `USE_REAL_API = False` - Set to `True` to use a real LLM API (requires API key)
- `RECIPIENTS` - List of recipients with name, email, and role

## Sample Output
```
============================================================
PROJECT 1: EMAIL AUTOMATION
============================================================
Processing 3 recipients...
Mode: Mock (saves to outbox/)

[1/3] Processing Aarav Sharma <aarav@example.com>
        Subject: A quick update from the AI Program team
        [SENT -> saved to outbox/aarav_sharma.txt]

[2/3] Processing Priya Patel <priya@example.com>
        Subject: A quick update from the AI Program team
        [SENT -> saved to outbox/priya_patel.txt]

[3/3] Processing Rahul Kumar <rahul@example.com>
        Subject: A quick update from the AI Program team
        [SENT -> saved to outbox/rahul_kumar.txt]

============================================================
Emails generated and 'sent': 3
Check the 'outbox/' folder to see the generated emails.
============================================================
```

## Challenges
1. Add more recipients to the list
2. Customize the email template for different roles
3. Add a feature to schedule emails
4. Load recipients from a CSV file
5. Add email tracking (sent/delivered status)

## Real-World Usage
In production, you would:
1. Use a real email service (SMTP, SendGrid, etc.)
2. Connect to a database for recipient lists
3. Add error handling and retry logic
4. Implement email templates with HTML formatting