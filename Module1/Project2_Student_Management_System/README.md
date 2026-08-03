# Project 2 - Student Management System

A menu-driven console app that lets a teacher **Add**, **View**, **Search**,
**Update**, and **Delete** student records, and **saves everything to a
JSON file** so data survives between runs.

Each student is a **dictionary**; all students live in a **list** - the
"load data -> manipulate in memory -> save back" pattern is exactly what
you'll do with datasets and model results in AI work. This is a CRUD
(Create, Read, Update, Delete) application, the backbone of every
data-driven system.

This is the complete program from Module 1 notes section 13, plus the
extra touches the notes mention: a **duplicate-roll check** and a clean
**Ctrl+C** exit. Console output is plain ASCII.

**Concepts used:** dictionaries, lists, functions, loops, file handling
(JSON), exception handling, `match` pattern matching, the
`if __name__ == "__main__":` guard.

## Files
- `student_management.py` - the full application
- `students.json` - data file, created on first save

## How to run
```bash
python student_management.py
```

Menu options:
1. Add Student (roll, name, marks - numbers are validated, rolls must be unique)
2. View All Students (neat aligned table)
3. Search Student (by roll number)
4. Update Marks
5. Delete Student
6. Save & Exit (writes `students.json`)

### Auto demo (verification)
```bash
python student_management.py --demo
```
Walks through every menu option automatically (add 2 students, view,
search, update, delete, view, save & exit), then exits.

## Challenge extensions
1. Add a "Find Topper" menu option that prints the student with the
   highest marks.
2. Add a "Report Card" option that prints a pass/fail status per student
   (pass mark 40).
3. Sort the table by marks or by name before printing.
