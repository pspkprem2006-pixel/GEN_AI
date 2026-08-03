# Project 1 - AI Resume Generator

Turns a person's facts into a polished, achievement-focused resume. This is
the syllabus's flagship GenAI app and demonstrates **role + task + format +
rules** prompting.

## Files
- `resume_generator.py` - the full program
- `resume.md` - the generated resume (created when you run it)

## How it works
The app builds a 2-part engineered prompt:
1. **System prompt** - sets the ROLE and RULES:
   "You are an expert technical resume writer and career coach. You write
   concise, achievement-focused resumes. You never invent facts; you only
   use what you are given."
2. **User prompt** - carries the TASK, the candidate DATA, the exact FORMAT
   (section headings), and the RULES (2-3 sentence summary, action-verb
   bullets, under 250 words, no invented facts).

That structure is why the output is reliable - see Module 7 notes, section 12.

## How to run
```bash
python resume_generator.py
```
Runs OFFLINE in **mock mode** by default - no API key, no installs. It
prints the engineered prompt and a representative resume, then saves it to
`resume.md`.

### Real mode (optional)
1. `pip install anthropic`
2. Set your key: `set ANTHROPIC_API_KEY=sk-ant-...` (Windows) or
   `export ANTHROPIC_API_KEY=sk-ant-...` (macOS/Linux). Get one at
   https://console.anthropic.com
3. In `resume_generator.py` set `USE_REAL_API = True`
4. Run again - Claude writes the resume with the same prompt, smarter prose.

## Challenges
1. Put **your own details** in the `PROFILE` dict and generate your resume.
2. Add a new section (e.g. "## Education") to both the prompt and the mock.
3. Compare the mock resume with the real-mode output and list 3 differences.
