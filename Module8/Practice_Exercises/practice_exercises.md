# Module 8: AI Agents & Automation - Practice Exercises

## Section 17.1: Concept Checks

### Question 1
**In one sentence, what is the difference between a chatbot and an agent?**

**Answer:**
A chatbot only *talks* (text in → text out); an **agent** can *act* — it uses tools in a loop to actually get a goal done.

---

### Question 2
**Name the four steps of the agent loop.**

**Answer:**
**Think → Act (call a tool) → Observe (read the result) → Repeat** until the goal is met (or a step cap is hit).

---

### Question 3
**Why do agents need tools? Give two examples of weaknesses tools fix.**

**Answer:**
An LLM alone can't do live or exact things. Tools fix, e.g.:
1. **No live data** → a weather/search API
2. **Bad at exact math** → a calculator
3. (Also: no memory → a database, can't act → send-email)

---

### Question 4
**What is a trigger, a step, and an action in a workflow?**

**Answer:**
- A **trigger** starts the flow (schedule, new email, webhook)
- A **step** is one unit of work in the middle (transform/decide)
- An **action** is the final effect on the world (send message, write row)

---

### Question 5
**Sketch (in words) an n8n workflow that emails you a daily weather summary.**

**Answer:**
**Schedule trigger (8 AM daily)** → **HTTP Request** node calls a weather API for your city → **Set/Function** node formats a short summary → **Email/Gmail** node sends it to you.

---

### Question 6
**When should you use multi-agent instead of a single agent?**

**Answer:**
Use multi-agent when the job splits into **distinct specialties** (research vs write vs edit) or is too big for one prompt — separate agents each do one role well and hand off.

---

### Question 7
**What are two types of agent memory, and what does each store?**

**Answer:**
1. **Short-term** (the current conversation/scratchpad — the running context)
2. **Long-term** (facts/notes saved to a file or vector DB and recalled later across sessions)

---

### Question 8
**Give three safety practices for agents that can take real actions.**

**Answer:**
1. **Human-in-the-loop** approval for consequential actions
2. **Limits/caps** (max steps, spending, allowed tools)
3. **Logging + read-only-by-default / sandboxing** so actions are auditable and reversible

---

## Section 17.2: Coding Exercises

### Question 9
**Run all three projects (mock mode) and complete one README challenge in each.**

**Answer:**
This is a hands-on task. Run each project's `.py` file in mock mode (no installs, no API key) and complete one challenge from its README:
- Project 1: Add a recipient to the list
- Project 2: Add a new goal to test
- Project 3: Add a new writing topic

---

### Question 10
**Add a new tool (e.g., temperature converter) to Project 2's agent.**

**Answer:**
```python
def temp_convert(args):
    value, to = float(args["value"]), args["to"].lower()
    if to == "f": return value * 9/5 + 32
    if to == "c": return (value - 32) * 5/9
    return "unknown unit"

TOOLS["temp_convert"] = temp_convert   # add to the agent's tool registry
# temp_convert({"value":100,"to":"f"}) -> 212.0
```

---

### Question 11
**Load Project 1's recipient list from a CSV instead of the inline list.**

**Answer:**
```python
import csv
with open("recipients.csv", newline="") as f:
    recipients = [row["email"] for row in csv.DictReader(f)]
# recipients.csv has a header row: name,email
```

---

### Question 12
**Add a 4th agent (a Fact-Checker) to Project 3's pipeline.**

**Answer:**
Insert it after Write, before Edit, so the chain becomes **Research -> Write -> Fact-Check -> Edit**:
```python
def fact_checker(draft):
    prompt = f"Check this draft for false or unsupported claims. List any, or say 'No issues'.\n\n{draft}"
    return ask_llm(prompt)          # mock or real, same as the other agents
pipeline = [researcher, writer, fact_checker, editor]
```

---

### Question 13
**Make Project 2 interactive — read the goal with `input()` in a loop.**

**Answer:**
```python
while True:
    goal = input("Goal (or 'quit'): ").strip()
    if goal.lower() in ("quit", "exit", ""):
        break
    run_agent(goal)
```

---

### Question 14
**Add a max-steps cap and a log to any project that loops.**

**Answer:**
```python
def agent_loop(goal, max_steps=5):
    log = []
    for step in range(1, max_steps + 1):
        log.append(f"step {step}: ...")
        if goal_reached():           # your done-check
            log.append("DONE"); break
    else:
        log.append("stopped: hit max_steps cap")
    return log
```

---

## Section 17.3: Applied / Design Exercises

### Question 15
**Design (on paper) an AI automation for a real chore in your life (trigger → steps → action).**

**Answer:**
**Sample automation:** *trigger* = 7 PM daily → *steps* = read today's calendar + tomorrow's todos, summarize → *action* = send me a "tomorrow at a glance" message.

---

### Question 16
**Pick a task and decide: fixed workflow, single agent, or multi-agent? Justify it.**

**Answer:**
If the steps are **fixed and predictable**, use a **workflow** (cheapest, most reliable); if it needs **judgment/branching**, use a **single agent**; only go **multi-agent** when there are clearly separate specialist roles. *Rule: use the simplest thing that works.*

---

### Question 17
**(Optional) Try n8n free and build a schedule-trigger → send-message workflow.**

**Answer:**
This is a hands-on exercise. Sign up for n8n's free tier and build a workflow with:
1. Schedule trigger (e.g., daily at 9 AM)
2. HTTP request to get data (e.g., weather API)
3. Send message (email or Slack)

---

### Question 18
**(Optional, real mode) Enable an API key and run one project with a live LLM.**

**Answer:**
This is a hands-on exercise. To run with a real LLM:
1. Get an API key from OpenAI, Anthropic, or similar
2. Set `USE_REAL_API = True` in the project file
3. Add your API key to the configuration
4. Run the project and observe real AI responses

---

## Section 17.4: Quick Self-Check Quiz

### Question 1
**An agent = an LLM + ___ + a ___.**

**Answer:**
tools; loop

---

### Question 2
**What starts a workflow?**

**Answer:**
a trigger

---

### Question 3
**What kind of tool is n8n?**

**Answer:**
visual workflow automation

---

### Question 4
**What's the risk of no loop cap?**

**Answer:**
runaway/expensive/infinite loop

---

### Question 5
**Research → Write → Edit is which pattern?**

**Answer:**
sequential/pipeline multi-agent

---

### Question 6
**Where should consequential actions get a human's OK?**

**Answer:**
human-in-the-loop

---

### Question 7
**What lets an LLM request a tool in a structured way?**

**Answer:**
function calling / tool use

---

### Question 8
**Which is simpler and more reliable — workflow or agent?**

**Answer:**
workflow

---

## Summary

These practice exercises cover:
- **Concept checks** (Questions 1-8): Understanding core concepts
- **Coding exercises** (Questions 9-14): Hands-on implementation
- **Applied/design exercises** (Questions 15-18): Real-world application
- **Quick quiz** (Questions 1-8): Quick self-assessment

**Ready for Module 9 when:** you can explain agents, tools, and workflows; describe an n8n flow; and read/modify the three projects.