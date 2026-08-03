# Project 2 - Student Performance Analysis

A full EDA of student exam data with Pandas, NumPy, and Seaborn:
`load -> clean (fill missing marks) -> analyze (describe, correlation) -> 4 Seaborn charts -> report`.

The star insight: **study hours strongly correlate with marks** (correlation
about 0.87 - students who study more tend to score more).

The four statistical charts:
- **Percentage distribution** (histogram + KDE)
- **Subject averages** (bar plot)
- **Study Hours vs Percentage** (scatter + regression line)
- **Correlation heatmap** of all numeric columns

## Files

| File | What it does |
|---|---|
| `student_performance.py` | Generates 60 synthetic students, cleans missing marks, computes stats + correlation, draws the 4 Seaborn charts, saves them as a PNG. |
| `student_performance.png` | The saved 2x2 Seaborn figure (created when the script runs). |

## How to run

```bash
python student_performance.py
```

Requires `numpy`, `pandas`, `matplotlib`, `seaborn`
(`pip install numpy pandas matplotlib seaborn`). The script uses the
Matplotlib `Agg` backend, so it needs **no GUI** - it prints the student
report and saves `student_performance.png` in the same folder.

## Challenges

1. Add a `Gender` column to the synthetic data and compare average
   percentages per gender with `df.groupby("Gender")["Percentage"].mean()`.
2. Draw a **box plot** of `Percentage` by `Result` (shows medians and outliers).
3. Add a sentence of plain-English insight at the end - every EDA should
   end with written findings (section 7.8).
