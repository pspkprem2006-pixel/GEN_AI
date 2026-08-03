"""
Project 1 - AI Resume Generator
Module 7: Generative AI & Prompt Engineering

Turns a person's facts into a polished, achievement-focused resume using a
2-part engineered prompt:
  - system prompt: sets the ROLE and RULES
  - user prompt:  carries the TASK, the DATA, the FORMAT and the RULES

Runs OFFLINE in mock mode by default (no API key, no installs). Set
USE_REAL_API = True and add an ANTHROPIC_API_KEY to have Claude write the
resume for real. Saves the result to resume.md.
"""

USE_REAL_API = False
MODEL = "claude-sonnet-5"   # pick the smallest model that does the job well
MAX_TOKENS = 1024

# ---------------------------------------------------------------------------
# Sample candidate profile - edit with your own details
# ---------------------------------------------------------------------------
PROFILE = {
    "name": "Alex Rivera",
    "target_role": "Junior AI/ML Engineer",
    "summary_facts": (
        "3rd-year computer science student. Built a customer-churn prediction "
        "model (87% accuracy) with scikit-learn. Interned at a fintech startup, "
        "building a Python API with Flask. Comfortable with Python, Pandas, "
        "scikit-learn, OpenCV, Git."
    ),
    "skills": ["Python", "Pandas", "scikit-learn", "OpenCV",
               "Prompt Engineering", "Git"],
    "projects": [
        ("Customer churn prediction", "Built a churn model with 87% accuracy "
         "using scikit-learn; deployed it as a Flask API."),
        ("Face detection app", "Built a real-time face detection app with "
         "OpenCV and Haar cascades."),
    ],
}


# ---------------------------------------------------------------------------
# The 2-part engineered prompt
# ---------------------------------------------------------------------------
def build_system_prompt():
    """ROLE + RULES: sets who the model is and how it must behave."""
    return ("You are an expert technical resume writer and career coach. "
            "You write concise, achievement-focused resumes. "
            "You never invent facts; you only use what you are given.")


def build_user_prompt(profile):
    """TASK + DATA + FORMAT + RULES: the specific request for this resume."""
    facts = profile["summary_facts"]
    skills = " | ".join(profile["skills"])
    projects = "\n".join(
        f"- {title}: {desc}" for title, desc in profile["projects"]
    )
    return f"""Write a resume for {profile['name']} targeting {profile['target_role']}.

Candidate details (use ONLY these facts, never invent anything):
Summary facts: {facts}
Skills: {skills}
Projects:
{projects}

FORMAT: use these exact section headings:
## Professional Summary
## Key Skills
## Experience & Projects

RULES:
- Professional Summary must be 2-3 sentences.
- Use action-verb bullets for projects.
- Keep the whole resume under 250 words.
- Never invent facts, job titles, dates, or companies."""


# ---------------------------------------------------------------------------
# Calling the LLM API (Anthropic Claude)
# ---------------------------------------------------------------------------
def call_claude(system_prompt, user_prompt, model=MODEL, max_tokens=MAX_TOKENS):
    """Send the prompt to Claude and return the text reply."""
    import anthropic

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY env var
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    # The reply is a list of content blocks; collect the text:
    return "".join(block.text for block in response.content if block.type == "text")


# ---------------------------------------------------------------------------
# Mock output - representative of what the real model produces
# ---------------------------------------------------------------------------
def mock_resume(profile):
    name = profile["name"]
    skills = " | ".join(profile["skills"])
    project_bullets = [
        "Built a customer-churn prediction model (87% accuracy) with scikit-learn.",
        "Deployed the model as a Flask API used by the startup's data team.",
        "Built a real-time face detection app with OpenCV.",
    ]
    lines = [
        f"# {name}",
        "## Professional Summary",
        "Aspiring Junior AI/ML Engineer with hands-on experience building "
        "end-to-end AI applications, from data cleaning to model deployment. "
        "Strong foundation in Python and machine learning, with an internship "
        "building production APIs. Looking for a role where I can keep learning "
        "and shipping real products.",
        "## Key Skills",
        skills,
        "## Experience & Projects",
        *[f"- {b}" for b in project_bullets],
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("AI RESUME GENERATOR  (Module 7, Project 1)")
    print(f"Target role : {PROFILE['target_role']}")
    print(f"Model       : {MODEL}")
    print(f"Real API    : {USE_REAL_API}")
    print("=" * 60)

    system_prompt = build_system_prompt()
    user_prompt = build_user_prompt(PROFILE)

    print("\n--- SYSTEM PROMPT (role + rules) ----------------------------")
    print(system_prompt)
    print("\n--- USER PROMPT (task + data + format + rules) --------------")
    print(user_prompt)

    if USE_REAL_API:
        print("\nCalling Claude...")
        resume_text = call_claude(system_prompt, user_prompt)
    else:
        print("\nMOCK MODE: representative output shown (no API call).")
        resume_text = mock_resume(PROFILE)

    print("\n--- RESUME ---------------------------------------------------")
    print(resume_text)
    print("---")

    with open("resume.md", "w", encoding="utf-8") as f:
        f.write(resume_text + "\n")
    print("\nSaved to resume.md")


if __name__ == "__main__":
    main()
