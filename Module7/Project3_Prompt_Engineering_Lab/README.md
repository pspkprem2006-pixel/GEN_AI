# Project 3 - Prompt Engineering Lab

A guided tour of the five core prompting techniques - see each prompt, its
result, and *when* to use it. This is the lab where you LEARN the techniques
that Projects 1 and 2 APPLY.

## Files
- `prompt_lab.py` - the full program

## Techniques demonstrated
1. **Zero-shot** - just ask, no examples
2. **Few-shot** - show 1-3 input/output examples
3. **Role** - tell the model who to be
4. **Chain-of-thought** - "think step by step"
5. **Structured output** - ask for JSON / exact format

Each lab prints:
```
PROMPT:       the engineered prompt
RESPONSE:     a representative answer
WHEN TO USE:  the situations where the technique shines
```

## How to run
```bash
python prompt_lab.py
```
Runs OFFLINE in **mock mode** by default - no API key, no installs.

### Real mode (optional)
1. `pip install anthropic`
2. Set your key: `set ANTHROPIC_API_KEY=sk-ant-...` (Windows) or
   `export ANTHROPIC_API_KEY=sk-ant-...` (macOS/Linux). Get one at
   https://console.anthropic.com
3. In `prompt_lab.py` set `USE_REAL_API = True`
4. Run again - watch each technique change Claude's actual answer. This is
   the fastest way to *feel* why prompt engineering matters.

## Challenges
1. Add a 6th lab for **delimiters** (instructions vs data separation).
2. Run the chain-of-thought prompt in real mode, then remove "Think step by
   step" and compare the answers.
3. Change the few-shot examples and notice how the format of the reply
   follows your examples.
