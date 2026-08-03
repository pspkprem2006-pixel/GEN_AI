# Module 7 - Generative AI & Prompt Engineering

Completed tasks for Module 7 of the AI Powered Engineering Upskilling Program.

This module covers Generative AI and LLMs, the 2026 AI assistant landscape,
prompt engineering fundamentals (role, task, examples, format), the core
prompting techniques, hallucination control, RAG, LLM APIs, and responsible
use of generative AI.

## Task Files (one file per task)

| Task | Folder / File | What it does |
|---|---|---|
| Project 1 | `Project1_AI_Resume_Generator/` | AI Resume Generator - turns a person's facts into a polished resume using a 2-part engineered prompt (system role + user task/format/rules). Saves `resume.md`. |
| Project 2 | `Project2_Research_Assistant/` | Research Assistant - turns a topic into a structured research brief using structured-output prompting with fixed sections and an anti-hallucination rule. Saves `research_brief.md`. |
| Project 3 | `Project3_Prompt_Engineering_Lab/` | Prompt Engineering Lab - guided tour of the 5 core techniques: zero-shot, few-shot, role, chain-of-thought, structured output. |
| Practice | `Practice_Exercises/answers.md` | Written answers for all concept checks, example prompts for the prompting practice, cost estimates, and the self-check quiz. |

## How to run

Every project runs **OFFLINE in mock mode by default** - no API key, no
installs, no internet. It prints the engineered prompt and a representative
result so you can learn the app and the prompting immediately.

To have Claude actually respond, set `USE_REAL_API = True` inside the script,
`pip install anthropic`, and set the environment variable
`ANTHROPIC_API_KEY` (get one at https://console.anthropic.com).
Never hard-code or commit an API key.

## Key techniques demonstrated

- Role, task, examples, format - the 4 levers of a great prompt
- Zero-shot, few-shot, role, chain-of-thought, structured-output prompting
- System prompt vs user prompt
- Anti-hallucination rules ("do NOT invent URLs, papers, or author names")
- Cost reasoning: pick the smallest model that does the job well
