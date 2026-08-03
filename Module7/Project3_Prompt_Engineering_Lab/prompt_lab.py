"""
Project 3 - Prompt Engineering Lab
Module 7: Generative AI & Prompt Engineering

A guided tour of the five core prompting techniques. For each of
zero-shot, few-shot, role, chain-of-thought, and structured-output, the lab
prints the prompt, a representative response, and when to use the technique.

Runs OFFLINE in mock mode by default (no API key, no installs). Set
USE_REAL_API = True and add an ANTHROPIC_API_KEY to watch each technique
change Claude's actual answer - the fastest way to feel why prompt
engineering matters.
"""

USE_REAL_API = False
MODEL = "claude-sonnet-5"   # pick the smallest model that does the job well
MAX_TOKENS = 1024

# ---------------------------------------------------------------------------
# The five techniques: (name, prompt, mock response, when to use)
# ---------------------------------------------------------------------------
LABS = [
    {
        "name": "ZERO-SHOT",
        "prompt": (
            'Classify the sentiment of this review as Positive or Negative:\n'
            '"The battery dies within an hour. Very disappointing."'
        ),
        "mock_response": "Negative",
        "when": "simple/common tasks the model has seen many times",
    },
    {
        "name": "FEW-SHOT",
        "prompt": (
            'Review: "I love it!"            -> Positive\n'
            'Review: "Total waste of money." -> Negative\n'
            'Review: "Fast delivery, great quality." ->'
        ),
        "mock_response": "Positive",
        "when": "you need a consistent format or a task the model might "
                "interpret loosely",
    },
    {
        "name": "ROLE",
        "prompt": (
            "You are a senior cybersecurity expert. In 2 sentences, explain "
            "to a beginner why reusing passwords is risky."
        ),
        "mock_response": (
            "Reusing passwords means one leaked credential unlocks many of "
            "your accounts, so a single data breach exposes your email, "
            "banking, and social media at once. Attackers automate "
            "credential-stuffing: they take a leaked password list and try it "
            "everywhere, so the same password is reused until one service "
            "breaks."
        ),
        "when": "you want expert tone, vocabulary, and depth",
    },
    {
        "name": "CHAIN-OF-THOUGHT",
        "prompt": (
            "A shop sells pens at 12 for $8. How much do 30 pens cost?\n"
            "Think step by step, then give the final answer."
        ),
        "mock_response": (
            "Step 1: per pen = 8/12 = $0.667. Step 2: 30 x 0.667 = $20. "
            "Answer: $20."
        ),
        "when": "reasoning/math problems that need intermediate steps",
    },
    {
        "name": "STRUCTURED-OUTPUT",
        "prompt": (
            'Extract the name, role, and city and reply as JSON with keys '
            'name, role, city:\n'
            '"Priya is a data scientist based in Pune."'
        ),
        "mock_response": (
            '{"name": "Priya", "role": "data scientist", "city": "Pune"}'
        ),
        "when": "your code will parse the result (JSON, table, fixed sections)",
    },
]


# ---------------------------------------------------------------------------
# Calling the LLM API (Anthropic Claude)
# ---------------------------------------------------------------------------
def call_claude(user_prompt, model=MODEL, max_tokens=MAX_TOKENS):
    """Send a user prompt to Claude and return the text reply."""
    import anthropic

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY env var
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": user_prompt}],
    )
    # The reply is a list of content blocks; collect the text:
    return "".join(block.text for block in response.content if block.type == "text")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("PROMPT ENGINEERING LAB  (Module 7, Project 3)")
    print(f"Model    : {MODEL}")
    print(f"Real API : {USE_REAL_API}")
    print("=" * 60)

    for i, lab in enumerate(LABS, start=1):
        print(f"\n{i}. {lab['name']}")
        print(f"PROMPT:  {lab['prompt']}")

        if USE_REAL_API:
            response = call_claude(lab["prompt"])
            print(f"RESPONSE (real Claude): {response}")
        else:
            print(f"RESPONSE (mock): {lab['mock_response']}")
        print(f"WHEN TO USE: {lab['when']}")

    print("\n" + "=" * 60)
    print("SUMMARY: technique -> when to use it")
    print("=" * 60)
    for lab in LABS:
        print(f"- {lab['name']:<18} {lab['when']}")

    print("\nTry it in real mode to watch each technique change Claude's "
          "actual answer.")


if __name__ == "__main__":
    main()
