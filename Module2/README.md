# Module 2 - AI & Data Science Foundations

Completed tasks for Module 2 of the AI Powered Engineering Upskilling
Program.

This module covers the foundations of AI and Data Science: what AI/ML/DL
are and how they nest, types of AI (ANI/AGI/ASI, generative vs
predictive), the three learning paradigms (supervised, unsupervised,
reinforcement), data science foundations, the 7-stage AI project
lifecycle, industry use cases and the Impact-vs-Feasibility framework,
and AI ethics.

## Task Files (one file per task)

| Task | Folder / File | What it does |
|---|---|---|
| Project 1 | `Project1_AI_Use_Case_Explorer/ai_use_case_explorer.py` | AI Use Case Explorer - catalogs AI use cases (impact/feasibility 1-5), ranks them by priority (impact x feasibility) into Quick Win / Big Bet / Low Priority / Avoid quadrants, saves to `ai_use_cases.json`. Hands-on form of the syllabus's AI Use Case Discussion. |
| Project 2 | `Project2_AI_ML_DL_Classifier_Quiz/classifier_quiz.py` | AI vs ML vs DL Classifier Quiz - 10 shuffled multiple-choice questions on AI vs ML vs DL, learning paradigms, generative vs predictive, ANI/AGI/ASI; explains each answer; final score with a rating. |
| Project 3 | `Project3_AI_Project_Lifecycle_Tracker/lifecycle_tracker.py` | AI Project Lifecycle Tracker - tracks a project through the 7 lifecycle stages with status + notes, shows a text progress bar and completion %, saves to `ai_project.json`. |
| Practice | `Practice_Exercises/answers.md` | Written answers to every practice exercise: 17.1 concept checks, 17.2 scenario classification, 17.3 applied/discussion tasks, 17.4 hands-on (incl. 2 added quiz questions), and the 17.5 self-check quiz. |

## How to run

Every project is pure Python - no external libraries, no installs. All
output is plain ASCII so it runs on any terminal.

```bash
# Interactive (menus / quiz)
python Project1_AI_Use_Case_Explorer/ai_use_case_explorer.py
python Project2_AI_ML_DL_Classifier_Quiz/classifier_quiz.py
python Project3_AI_Project_Lifecycle_Tracker/lifecycle_tracker.py

# Non-interactive demos (for quick verification)
python Project1_AI_Use_Case_Explorer/ai_use_case_explorer.py --demo
python Project2_AI_ML_DL_Classifier_Quiz/classifier_quiz.py --demo
python Project3_AI_Project_Lifecycle_Tracker/lifecycle_tracker.py --demo
```

## How the three projects fit together

Take your **top use case from Project 1**, make sure you can classify its
AI type using **Project 2's** knowledge, then **plan it stage-by-stage in
Project 3** - a complete "think like an AI engineer" workflow.
