# Project 1 - AI Use Case Explorer

This is the hands-on form of the syllabus activity **"AI Use Case
Discussion."** Instead of only talking about where AI helps, you build a
tool that **catalogs and prioritizes** AI use cases - exactly how a real
AI team decides what to build first.

## Files
- `ai_use_case_explorer.py` - the full program
- `ai_use_cases.json` - your saved catalog (created when you save)

## How it works
Record AI use cases (industry, problem, AI type, **impact** 1-5,
**feasibility** 1-5). The tool then ranks them using the
Impact-vs-Feasibility framework from Module 2, section 7.3:

- `priority_label()` - a score of 4 or 5 counts as HIGH; combines the two
  axes into a quadrant: **Quick Win** (high/high), **Big Bet**
  (high impact/low feasibility), **Low Priority** (low impact/high
  feasibility), **Avoid** (low/low).
- `priority_score()` - impact x feasibility, a single number to sort by
  (a 5x4 = 20 beats a 3x3 = 9).
- `prioritize()` - sorts with `key=priority_score, reverse=True` (a
  Module 1 lambda/sorting skill) and prints the ranked table.

It comes pre-loaded with 5 example use cases (Finance, Retail, Healthcare,
Customer Service, Agriculture) and saves your catalog to
`ai_use_cases.json`.

## How to run
```bash
python ai_use_case_explorer.py
```
Menu: 1) Add a use case, 2) List all, 3) Prioritize, 4) Save, 5) Exit.

Non-interactive demo (prints the priority table of the 5 sample use cases):
```bash
python ai_use_case_explorer.py --demo
```

## Challenges
1. Run the **AI Use Case Discussion**: propose 5 use cases for an industry
   you care about, debate the Impact/Feasibility scores with a teammate,
   then prioritize and defend your top pick.
2. Add your own use case and watch its quadrant change as you re-score it.
3. Try `HIGH = 5` and see how the quadrant labels change.

## Example output (prioritize view)
```
=== AI USE CASES BY PRIORITY (highest first) ===
Rank | Score | Quadrant     | Use Case
----------------------------------------------------------------------
1    | 20    | Quick Win    | Finance - Flag fraudulent card transactions
2    | 20    | Quick Win    | Retail - Recommend products to shoppers
3    | 15    | Big Bet      | Healthcare - Detect tumors in X-ray/CT scans
4    | 15    | Low Priority | Customer Service - Answer FAQs with a chatbot
5    | 8     | Big Bet      | Agriculture - Predict crop disease from leaf photos
----------------------------------------------------------------------
```
