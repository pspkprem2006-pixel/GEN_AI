# Module 3 - Data Analysis & Visualization

Completed tasks for Module 3 of the AI Powered Engineering Upskilling Program.

This module covers NumPy (numerical arrays), Pandas (Series & DataFrame),
data cleaning (the "80% job"), exploratory data analysis (EDA), and data
visualization with Matplotlib and Seaborn. The workflow taught is:
**clean -> analyze -> visualize**.

## Task Files (one file per task)

| Task | Folder / File | What it does |
|---|---|---|
| Project 1 | `Project1_Sales_Dashboard/` | Sales Dashboard - generates 400 synthetic orders, runs `groupby` KPI analysis, builds a 2x2 Matplotlib dashboard (bar, line, horizontal bar, pie) and saves `sales_dashboard.png`. |
| Project 2 | `Project2_Student_Performance_Analysis/` | Student Performance Analysis - 60 synthetic students, cleaning (fill missing marks with subject mean), EDA with statistics + correlation, 4 Seaborn charts (histogram, bar, regplot, heatmap) saved as `student_performance.png`. |
| Project 3 | `Project3_Data_Cleaning_Workshop/` | Data Cleaning Workshop - a deliberately messy 10-row DataFrame cleaned through the 5-step pipeline (duplicates, text, categories, types, outliers + missing values). |
| Practice | `Practice_Exercises/answers.md` | Question + answer for every practice exercise in section 17 (17.1 NumPy, 17.2 Pandas, 17.3 Data cleaning, 17.4 EDA & visualization, 17.5 Integrative, 17.6 Quiz) - answers from the 17.7 answer key. |

## How to run

Every project runs **offline** with synthetic data generated inside the
script - no CSV files, no downloads, no API keys.

```bash
# Prerequisites
pip install numpy pandas matplotlib seaborn

# Project 1 - Sales Dashboard
cd Project1_Sales_Dashboard
python sales_dashboard.py

# Project 2 - Student Performance Analysis
cd Project2_Student_Performance_Analysis
python student_performance.py

# Project 3 - Data Cleaning Workshop
cd Project3_Data_Cleaning_Workshop
python data_cleaning_workshop.py
```

Projects 1 and 2 use the Matplotlib `Agg` backend (no GUI) and save their
charts as PNG files that you open afterward. All console output is plain
ASCII.

## Key techniques demonstrated

- NumPy arrays and vectorized math (10-100x faster than Python lists)
- Pandas `groupby` + aggregation for dashboards
- The data-cleaning checklist: missing values, duplicates, types, text, categories, outliers
- EDA: `describe()`, correlation, distribution shapes, and the mini-EDA workflow
- Matplotlib subplots (2x2 dashboards) and Seaborn statistical charts
