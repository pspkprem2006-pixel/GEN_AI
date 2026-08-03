# Module 1 - Python for AI & Programming Fundamentals

Completed hands-on tasks for Module 1 of the AI Powered Engineering
Upskilling Program.

This module covers the Python foundation needed for the rest of the
program: syntax and data types, operators, control flow, functions,
collections, file handling (txt/csv/json), exception handling - and three
hands-on projects that combine everything.

## Task Files (one folder per task)

| Task | Folder / File | What it does |
|---|---|---|
| Project 1 | `Project1_Number_Guessing_Game/` | Number Guessing Game - random 1-100 game with too-high/too-low hints, attempt counter, input validation, best-score tracker and Ctrl+C exit. `number_guessing_game.py`. |
| Project 2 | `Project2_Student_Management_System/` | Student Management System - menu-driven CRUD app (Add/View/Search/Update/Delete) saving students to `students.json` via `match`-based menu. `student_management.py`. |
| Project 3 | `Project3_File_Processing/` | File Processing - reads `marks.csv`, computes statistics (total, average, highest, lowest, pass/fail), writes a report with a text bar chart to `report.txt`. `file_processing.py`. |
| Practice | `Practice_Exercises/answers.md` | Answers for all 26 practice exercises (17.1-17.6) and the 8-question self-check quiz (17.7) - every question written out with its answer. |

## How to run

Pure Python, no external libraries. Plain ASCII console output so it runs
on every computer. Each project is run from its own folder:

```bash
cd Project1_Number_Guessing_Game && python number_guessing_game.py
cd Project2_Student_Management_System && python student_management.py
cd Project3_File_Processing && python file_processing.py
```

Every project also supports a non-interactive demo mode (used for
verification): add `--demo` to the command. The programs still work fully
interactively when run normally.

## Key techniques demonstrated

- Variables, data types, operators, type conversion
- `if/elif/else`, `match`, `for`/`while`, `break`/`continue`
- Functions, parameters, return values, `if __name__ == "__main__"`
- Lists, tuples, sets, dictionaries
- File I/O: text, CSV, JSON - and `try/except` exception handling
- The "load -> manipulate in memory -> save back" CRUD pattern used by
  every data-driven AI system
