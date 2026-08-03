# Project 3 - Data Cleaning Workshop

The most important project of the module: take a deliberately **messy**
dataset and clean it in **5 clear steps** - the "80% job" of data science.

The mess (created on purpose inside the script):
duplicate rows, missing Age/Income, messy text (`"  ravi kumar "`,
`"MUMBAI"`), inconsistent gender (`M`/`male`/`Female`), numbers as text
(`"50,000"`, `"unknown"`), and an impossible outlier (`Age = 250`).

The 5-step cleaning pipeline:
1. Remove duplicate rows (`drop_duplicates`)
2. Tidy text (`.str.strip()`, fix case, collapse double spaces)
3. Standardize categories (`.map()` on gender)
4. Fix numeric columns stored as text (`to_numeric`, `errors="coerce"`)
5. Handle outliers (Age > 100 -> NaN), then fill missing with the **median**

## Files

| File | What it does |
|---|---|
| `data_cleaning_workshop.py` | Builds the messy 10-row DataFrame in code, runs the 5-step pipeline, prints the before/after summary. |

## How to run

```bash
python data_cleaning_workshop.py
```

Requires `numpy` and `pandas` (`pip install numpy pandas`). Plain console
output only - no GUI, no CSV files to download.

## Challenges

1. Add more mess: a `Date` column with mixed formats
   (`"2024/01/05"`, `"05-01-2024"`) and clean it with `pd.to_datetime(errors="coerce")`.
2. Apply the **IQR outlier rule** (section 6.7) to `Income` instead of a fixed range.
3. Add a `Phone` column with `"unknown"` values and standardize them to `NaN`.
