# Module 8: AI Agents & Automation

## Overview
This module covers AI agents and automation, teaching you how to build AI workflows, tool-using agents, and multi-agent systems.

## Projects

### 1. Email Automation (`Project1_Email_Automation/`)
An automated workflow that personalizes and "sends" an email to everyone in a list.
- **Concept**: Workflow automation (trigger → generate → deliver)
- **Run**: `python email_automation.py`

### 2. AI Agent with Tools (`Project2_AI_Agent_with_Tools/`)
A single-agent loop that demonstrates think → act → observe → answer.
- **Concept**: The single-agent loop (think → act → observe)
- **Run**: `python ai_agent.py`

### 3. Multi-Agent Workflow (`Project3_Multi_Agent_Workflow/`)
Three specialist agents collaborate to turn a topic into an article.
- **Concept**: Specialist agents collaborating
- **Run**: `python multi_agent.py`

## Practice Exercises
Complete exercises in `Practice_Exercises/practice_exercises.md` to test your understanding.

## How to Use
1. Start with **Project 1** to understand workflow automation
2. Move to **Project 2** to learn about agent loops and tools
3. Complete **Project 3** to see multi-agent collaboration
4. Work through the **Practice Exercises** to reinforce concepts

## Configuration
All projects run in **mock mode** by default (no API key required). To use real LLMs:
1. Set `USE_REAL_API = True` in the project file
2. Add your API key to the configuration
3. Run the project

## Key Concepts
- **Agent Loop**: Think → Act → Observe → Repeat
- **Tools**: Functions that give agents abilities beyond text generation
- **Workflow Automation**: Trigger → Steps → Action
- **Multi-Agent Systems**: Specialist agents collaborating with hand-offs
- **Safety**: Human-in-the-loop, loop caps, and sandboxing

## Module Structure
```
Module8/
├── Project1_Email_Automation/
│   ├── email_automation.py
│   ├── README.md
│   └── outbox/              (created when run)
├── Project2_AI_Agent_with_Tools/
│   ├── ai_agent.py
│   └── README.md
├── Project3_Multi_Agent_Workflow/
│   ├── multi_agent.py
│   ├── README.md
│   └── article_*.md        (created when run)
├── Practice_Exercises/
│   └── practice_exercises.md
└── README.md                (this file)
```

## Next Steps
After completing this module, you'll be ready for **Module 9: Deployment & Career Readiness**, where you'll turn your models and apps into shareable web apps.