"""
Project 2 - Research Assistant
Module 7: Generative AI & Prompt Engineering

Turns a topic into a structured research brief. Demonstrates
structured-output prompting (name the exact sections) plus an
anti-hallucination rule ("do NOT invent URLs, papers, or author names").

Runs OFFLINE in mock mode by default (no API key, no installs). Set
USE_REAL_API = True and add an ANTHROPIC_API_KEY to have Claude write the
brief for real. Saves the result to research_brief.md.
"""

USE_REAL_API = False
MODEL = "claude-sonnet-5"   # pick the smallest model that does the job well
MAX_TOKENS = 1024

TOPIC = "How CNNs work"

# ---------------------------------------------------------------------------
# The engineered prompt - structured output + anti-hallucination
# ---------------------------------------------------------------------------
def build_system_prompt():
    """ROLE + RULES for the whole conversation."""
    return ("You are a research assistant for engineering students. "
            "You write beginner-friendly, concise research briefs. "
            "You never invent facts, URLs, papers, or author names.")


def build_user_prompt(topic):
    """TASK + FORMAT (exact sections) + RULES."""
    return f"""Create a beginner-friendly research brief on "{topic}".

Use exactly these sections:
## 1. Overview
## 2. Key Concepts
## 3. Important Questions
## 4. Subtopics to Study Next
## 5. How to Learn More

RULES:
- Keep it concise and beginner-friendly.
- Do NOT invent URLs, papers, or author names.
- In section 5, only suggest general learning directions, not fake sources."""


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
def mock_brief(topic):
    lines = [
        f"# Research Brief: {topic}",
        "## 1. Overview",
        "A Convolutional Neural Network (CNN) is a deep learning model built "
        "for image-like data. Instead of connecting every pixel to every "
        "neuron, a CNN slides small filters over the image to detect local "
        "patterns such as edges, textures, and shapes. These patterns are "
        "stacked into layers, so later layers detect bigger structures like "
        "eyes, wheels, or faces.",
        "## 2. Key Concepts",
        "- Convolution: sliding a small filter over the image to find patterns.",
        "- Kernel / filter: the small matrix of weights that does the detecting.",
        "- Stride and padding: how fast the filter moves and how edges are handled.",
        "- Pooling: downsampling to keep the important info and shrink size.",
        "- Feature map: the output of a convolution layer.",
        "- Fully connected layer: the final classifier on top of the features.",
        "## 3. Important Questions",
        "- Why are filters shared across the image instead of per-pixel weights?",
        "- What does each layer of a CNN actually learn?",
        "- Why does pooling reduce overfitting?",
        "- How do CNNs work on non-image data like audio or text?",
        "- What is a receptive field, and why does it grow with depth?",
        "## 4. Subtopics to Study Next",
        "- Popular architectures: LeNet, AlexNet, ResNet.",
        "- Data augmentation and transfer learning.",
        "- Training tricks: batch norm, dropout, learning-rate schedules.",
        "- Applications: object detection (YOLO), segmentation (U-Net).",
        "- Visualizing what CNNs learn (feature-map visualization).",
        "## 5. How to Learn More",
        "- Work through an online course on deep learning for computer vision.",
        "- Read the original papers describing each architecture.",
        "- Build and train a small CNN from scratch on a toy dataset.",
        "- Follow reputable tutorials that show working code, and verify sources.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("RESEARCH ASSISTANT  (Module 7, Project 2)")
    print(f"Topic       : {TOPIC}")
    print(f"Model       : {MODEL}")
    print(f"Real API    : {USE_REAL_API}")
    print("=" * 60)

    system_prompt = build_system_prompt()
    user_prompt = build_user_prompt(TOPIC)

    print("\n--- SYSTEM PROMPT (role + rules) ----------------------------")
    print(system_prompt)
    print("\n--- USER PROMPT (task + format + rules) ----------------------")
    print(user_prompt)

    if USE_REAL_API:
        print("\nCalling Claude...")
        brief_text = call_claude(system_prompt, user_prompt)
    else:
        print("\nMOCK MODE: representative output shown (no API call).")
        brief_text = mock_brief(TOPIC)

    print("\n--- RESEARCH BRIEF -------------------------------------------")
    print(brief_text)
    print("---")

    with open("research_brief.md", "w", encoding="utf-8") as f:
        f.write(brief_text + "\n")
    print("\nSaved to research_brief.md")


if __name__ == "__main__":
    main()
