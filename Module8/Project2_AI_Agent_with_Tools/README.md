# Project 2: AI Agent with Tools

## Overview
A single-agent loop that demonstrates the core agent mechanism: **think → act → observe → answer**. The agent uses tools to accomplish goals, with a rule-based brain for offline mode.

## How It Works
1. **THINK**: The agent reasons about which tool to use
2. **ACT**: It calls the appropriate tool
3. **OBSERVE**: It reads the tool's result
4. **ANSWER**: It provides the final answer

## Tools Available
- `calculator` - Perform mathematical calculations
- `clock` - Get current date and time
- `word_counter` - Count words in text
- `knowledge_lookup` - Look up information from a knowledge base

## Files
- `ai_agent.py` - Main agent script

## How to Run
```bash
python ai_agent.py
```

## Configuration
- `USE_REAL_API = False` - Set to `True` to use a real LLM API (requires API key)

## Sample Output
```
============================================================
PROJECT 2: AI AGENT WITH TOOLS
============================================================
Mode: Mock (rule-based)

GOAL: What is 15 * 23 + 100?
  THINK  : This needs the 'calculator' tool.
  ACT    : calculator({'expression': '15 * 23 + 100'})
  OBSERVE: 445
  ANSWER : The answer is: 445

GOAL: What time is it now?
  THINK  : This needs the 'clock' tool.
  ACT    : clock({})
  OBSERVE: 2026-08-04 10:30:45
  ANSWER : The answer is: 2026-08-04 10:30:45

GOAL: What is machine learning?
  THINK  : This needs the 'knowledge_lookup' tool.
  ACT    : knowledge_lookup({'topic': 'machine learning'})
  OBSERVE: Machine Learning is a subset of AI that enables systems to learn from data.
  ANSWER : The answer is: Machine Learning is a subset of AI that enables systems to learn from data.
```

## Challenges
1. Add a new tool (e.g., temperature converter)
2. Make the agent interactive with `input()`
3. Add a max-steps cap to prevent infinite loops
4. Implement a logging system for the agent's actions
5. Add more knowledge to the knowledge base

## Agent Loop Explained
The agent loop is the heart of all AI agents:
- **Think**: Reason about what to do next
- **Act**: Use a tool to take action
- **Observe**: Read the result of the action
- **Repeat/Answer**: If more steps are needed, loop; otherwise, give the final answer

This is exactly what the **ReAct** (Reason + Act) pattern describes.