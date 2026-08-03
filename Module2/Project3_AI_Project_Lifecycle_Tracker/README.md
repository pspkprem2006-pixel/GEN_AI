# Project 3 - AI Project Lifecycle Tracker

This project makes the **7-stage AI Project Lifecycle** (Module 2,
section 6) tangible: you plan a real project by moving it through every
stage and watching your progress.

## Files
- `lifecycle_tracker.py` - the full program
- `ai_project.json` - the saved project (created when you save)

## How it works
Pick a project, then for each of the 7 lifecycle stages set a **status**
(Not Started / In Progress / Done) and **notes**. The tool shows a text
**progress bar** and a completion **percentage**.

- The project is a **dictionary** containing a **list of stage
  dictionaries** - nested data structures, just like real AI apps store.
- `new_project()` builds all 7 stages with a **list comprehension**.
- `completion_percent()` converts each status to a weight (Done = 1.0,
  In Progress = 0.5, Not Started = 0.0), averages them, and multiplies
  by 100.
- `progress_bar()` is pure string math: `"#" * filled` draws the
  completed part of the bar.
- The project is saved to `ai_project.json`.

## How to run
```bash
python lifecycle_tracker.py
```
Menu: 1) View project and progress, 2) Update a stage, 3) New project,
4) Save, 5) Exit.

Non-interactive demo (builds the sample "Churn Predictor" project):
```bash
python lifecycle_tracker.py --demo
```

## Challenges
1. Take the top use case from **Project 1** and map it through **all 7
   lifecycle stages** (practice exercise 16).
2. Update stages until you hit 100% and watch the bar fill up.
3. Add more stages (e.g. "Retraining" or "Ethics Review") to `STAGES`.

## Example output
```
======================================================================
PROJECT: Churn Predictor
======================================================================
#  | Stage                       | Status      | Notes
----------------------------------------------------------------------
1  | Problem Definition          | Done        | Defined goal: predict churn
2  | Data Collection             | In Progress | Pulling 2yr data
3  | Data Preparation & Cleaning | Not Started | -
...
----------------------------------------------------------------------
Overall progress: [####----------------] 21%
```

*(1 stage Done = 100% + 1 In Progress = 50%, out of 7 stages -> 1.5/7
is about 21%.)*
