# Project 3: Multi-Agent Workflow

## Overview
Three specialist agents collaborate to turn a topic into a complete article. This demonstrates the **sequential pipeline** pattern in multi-agent systems.

## The Pipeline
1. **Researcher Agent** - Gathers key points about the topic
2. **Writer Agent** - Creates a draft article from the research
3. **Editor Agent** - Polishes and formats the final article

## How It Works
Each agent has its own specialized role and system prompt. The output of one agent becomes the input of the next, creating a hand-off pattern.

```
Topic → [Researcher] → Key Points → [Writer] → Draft → [Editor] → Final Article
```

## Files
- `multi_agent.py` - Main multi-agent script
- `article_*.md` - Generated articles (created automatically)

## How to Run
```bash
python multi_agent.py
```

## Configuration
- `USE_REAL_API = False` - Set to `True` to use real LLM APIs (requires API keys)

## Sample Output
```
============================================================
MULTI-AGENT WORKFLOW: AI AGENTS
============================================================

[Agent 1: RESEARCHER] gathering key points...
   ...found point 1: AI agents are LLMs with tools and a loop
   ...found point 2: They can take actions in the world, not just generate text
   ...found point 3: The agent loop: Think → Act → Observe → Repeat
   ...found point 4: Tools give agents abilities beyond text generation
   ...found point 5: Multi-agent systems split work among specialists
   ...handing the points to the Writer...

[Agent 2: WRITER] drafting the article...
   ...handing the draft to the Editor...

[Agent 3: EDITOR] polishing and formatting...
   ...article complete!

Article saved to: article_ai_agents.md

============================================================
ARTICLE PREVIEW (first 500 characters):
============================================================
# Ai Agents

*TL;DR: A beginner-friendly overview of ai agents.*

## Introduction

Ai agents is a compelling and rapidly evolving field...
============================================================
```

## Multi-Agent Patterns
This project demonstrates the **sequential pipeline** pattern:
- Agents work in order
- Output → Input (hand-off)
- Each agent specializes in one task

Other patterns include:
- **Hierarchical**: Manager agent delegates to workers
- **Collaborative**: Agents discuss and critique each other

## Challenges
1. Add a 4th agent (Fact-Checker) to the pipeline
2. Create a hierarchical pattern with a manager agent
3. Add parallel agents for different aspects of research
4. Implement agent communication and feedback loops
5. Add quality checks between agents

## Benefits of Multi-Agent Design
- **Specialization**: Each agent is better at its narrow job
- **Separation of concerns**: Easier to build and test one agent at a time
- **Modularity**: Swap or upgrade one agent without touching others
- **Parallelism**: Independent agents can work simultaneously