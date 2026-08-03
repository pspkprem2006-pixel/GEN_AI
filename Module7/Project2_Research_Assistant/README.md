# Project 2 - Research Assistant

Turns a topic into a structured research brief. This project showcases
**structured-output prompting** and the **anti-hallucination** guard.

## Files
- `research_assistant.py` - the full program
- `research_brief.md` - the generated brief (created when you run it)

## How it works
The key idea is to **name the exact sections** in the prompt so the output is
predictable enough to build on:
```
## 1. Overview            ## 2. Key Concepts
## 3. Important Questions ## 4. Subtopics to Study Next
## 5. How to Learn More
```
The rule "Do NOT invent URLs, papers, or author names" is your
anti-hallucination guard - see Module 7 notes, sections 7 and 13.

## How to run
```bash
python research_assistant.py
```
Runs OFFLINE in **mock mode** by default - no API key, no installs. It
prints the engineered prompt and a representative brief, then saves it to
`research_brief.md`.

### Real mode (optional)
1. `pip install anthropic`
2. Set your key: `set ANTHROPIC_API_KEY=sk-ant-...` (Windows) or
   `export ANTHROPIC_API_KEY=sk-ant-...` (macOS/Linux). Get one at
   https://console.anthropic.com
3. In `research_assistant.py` set `USE_REAL_API = True`
4. Run again - Claude writes the brief for real.

## Challenges
1. Change `TOPIC` to something you are studying and generate a new brief.
2. In real mode, count how many fake URLs/names Claude invents (it should
   invent none, thanks to the rule). Remove the rule and compare.
3. Ask for a 6th section (e.g. "## 6. Related Fields") and regenerate.
