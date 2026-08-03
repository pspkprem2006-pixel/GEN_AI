# Project 3 - File Processing

Reads a file of student marks (CSV), computes statistics (total, average,
highest, lowest, pass/fail count), and writes a clean **report** to a new
file - plus a simple text-based bar chart of grades.

This is the complete pipeline from Module 1 notes section 14, with the
extra touch the notes mention: the sample `marks.csv` is **auto-created
only when missing** so the program runs on a fresh checkout. Console
output is plain ASCII.

This project is exactly what Module 3 (Data Analysis) does with Pandas
(`df.describe()`) and Matplotlib - done by hand first, so you understand
what those libraries do for you.

**Concepts used:** `csv` module, file reading/writing, `with open`, list
comprehensions, built-ins (`sum`, `len`, `max`, `min`), dictionaries,
tuples, exception handling.

## Files
- `file_processing.py` - the full pipeline
- `marks.csv` - sample data (auto-created on first run)
- `report.txt` - the generated report (created when you run it)

## How to run
```bash
python file_processing.py
```

Prints a summary to the screen and writes `report.txt` containing the
marks bar chart and summary statistics.

## Challenge extensions
1. Read from a different CSV (pass the filename as a command-line
   argument) instead of always using `marks.csv`.
2. Add a grade column to the report (A/B/C/F) based on marks.
3. Sort the bar chart by marks, highest first.
