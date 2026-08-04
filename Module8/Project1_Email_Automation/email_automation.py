"""
Project 1: Email Automation
Module 8 - AI Agents & Automation

An automated workflow that personalizes and "sends" an email to everyone in a list.
Runs in mock mode by default (saves to outbox/). Set USE_REAL_API = True for real LLM.
"""

import os
import json
from datetime import datetime

# Configuration
USE_REAL_API = False  # Set True + provide API key for real LLM emails
OUTBOX_DIR = "outbox"

# Sample recipient list
RECIPIENTS = [
    {"name": "Aarav Sharma", "email": "aarav@example.com", "role": "Data Analyst"},
    {"name": "Priya Patel", "email": "priya@example.com", "role": "Project Manager"},
    {"name": "Rahul Kumar", "email": "rahul@example.com", "role": "Software Engineer"},
]

# Email template
EMAIL_TEMPLATE = """Subject: {subject}

Dear {name},

This is a personalized update from the AI Program team.

We wanted to reach out regarding your role as {role}. The AI Powered Engineering
Upskilling Program is making great progress, and we're excited to share some updates
with you.

Key highlights:
- Module 8 covers AI Agents & Automation
- You'll learn to build automated workflows
- Hands-on projects include email automation, AI agents, and multi-agent systems

Best regards,
The AI Program Team

---
Generated on: {timestamp}
"""


def build_prompt(recipient):
    """Build a personalized prompt for email generation."""
    return f"Write a professional email to {recipient['name']} ({recipient['role']}) about the AI Program update."


def generate_email(recipient):
    """Generate a personalized email for the recipient."""
    if USE_REAL_API:
        # In real mode, you would call an LLM API here
        # For now, we'll use the template
        pass
    
    # Use template for mock mode
    subject = f"A quick update from the AI Program team"
    body = EMAIL_TEMPLATE.format(
        name=recipient["name"],
        role=recipient["role"],
        subject=subject,
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )
    return subject, body


def send_email(recipient, subject, body):
    """Send email (mock: saves to outbox/ directory)."""
    # Create outbox directory if it doesn't exist
    os.makedirs(OUTBOX_DIR, exist_ok=True)
    
    # Create filename from recipient name
    filename = recipient["name"].lower().replace(" ", "_") + ".txt"
    filepath = os.path.join(OUTBOX_DIR, filename)
    
    # Save email to file
    with open(filepath, "w") as f:
        f.write(f"To: {recipient['email']}\n")
        f.write(f"Subject: {subject}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 50 + "\n\n")
        f.write(body)
    
    return filepath


def main():
    """Main email automation workflow."""
    print("=" * 60)
    print("PROJECT 1: EMAIL AUTOMATION")
    print("=" * 60)
    print(f"Processing {len(RECIPIENTS)} recipients...")
    print(f"Mode: {'Real API' if USE_REAL_API else 'Mock (saves to outbox/)'}")
    print()
    
    sent_count = 0
    
    for i, recipient in enumerate(RECIPIENTS, 1):
        print(f"[{i}/{len(RECIPIENTS)}] Processing {recipient['name']} <{recipient['email']}>")
        
        # Step 1: Build prompt (personalize)
        prompt = build_prompt(recipient)
        
        # Step 2: Generate email
        subject, body = generate_email(recipient)
        print(f"        Subject: {subject}")
        
        # Step 3: Send email (mock)
        filepath = send_email(recipient, subject, body)
        print(f"        [SENT -> saved to {filepath}]")
        
        sent_count += 1
        print()
    
    print("=" * 60)
    print(f"Emails generated and 'sent': {sent_count}")
    print(f"Check the '{OUTBOX_DIR}/' folder to see the generated emails.")
    print("=" * 60)


if __name__ == "__main__":
    main()