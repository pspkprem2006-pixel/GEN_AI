"""
Project 3 - Data Cleaning Workshop (Module 3: Data Analysis & Visualization)

Take a deliberately MESSY dataset and clean it in 5 clear steps - the
"80% job" of data science, all in one place:

  Step 1: remove duplicate rows
  Step 2: tidy text (trim spaces, fix case, collapse double spaces)
  Step 3: standardize categories
  Step 4: fix numeric columns stored as text
  Step 5: handle outliers, then fill missing values with the median

The mess (created on purpose): duplicate rows, missing Age/Income, messy text
("  ravi kumar ", "MUMBAI"), inconsistent gender (M/male/Female), numbers as
text ("50,000", "unknown"), and an impossible outlier (Age = 250).

Run with:  python data_cleaning_workshop.py
"""

import numpy as np
import pandas as pd


def make_messy_data():
    """Build a deliberately messy 10-row dataset in code (no CSV needed)."""
    return pd.DataFrame({
        "Name":   ["  ravi kumar ", "Priya Sharma", "AMIT VERMA", "sneha patel",
                   "  ravi kumar ", "Karan Mehta", "neha gupta ", "Sanjay Rao",
                   "Divya Iyer", "Karan Mehta"],
        "Age":    [25, 32, np.nan, 28, 25, 41, 250, np.nan, 35, 41],
        "City":   ["mumbai", "Delhi", "Pune", "MUMBAI", "mumbai", "Chennai",
                   "Kolkata", "bengaluru", "Chennai", "Chennai"],
        "Gender": ["male", "Female", "M", "f", "male", "male", "Female",
                   "M", "female", "male"],
        "Income": ["50,000", "75,000", "90,000", "unknown", "50,000",
                   "120,000", "80,000", "65,000", "60,000", "120,000"],
    })


def clean_data(df):
    """The 5-step cleaning pipeline (section 14.2)."""
    # Step 1: remove duplicate rows
    before = len(df)
    df = df.drop_duplicates()
    print(f"Step 1 - duplicates : dropped {before - len(df)} duplicate row(s)")

    # Step 2: tidy text (trim spaces, fix case, collapse double spaces)
    df["Name"] = df["Name"].str.strip().str.replace(r"\s+", " ", regex=True).str.title()
    df["City"] = df["City"].str.strip().str.title()
    print("Step 2 - text       : names/cities trimmed, spaced, and titled")

    # Step 3: standardize categories
    gmap = {"m": "Male", "male": "Male", "f": "Female", "female": "Female"}
    df["Gender"] = df["Gender"].str.strip().str.lower().map(gmap)
    print("Step 3 - categories : gender standardized to Male/Female")

    # Step 4: fix numeric columns stored as text
    df["Income"] = (df["Income"].astype(str).str.replace(",", "", regex=False)
                    .replace({"unknown": np.nan, "": np.nan}))
    df["Income"] = pd.to_numeric(df["Income"], errors="coerce")
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    print("Step 4 - types      : Income/Age converted from text to numbers")

    # Step 5: handle outliers, then fill missing with the median
    df.loc[df["Age"] > 100, "Age"] = np.nan
    df["Age"] = df["Age"].fillna(df["Age"].median()).round().astype(int)
    df["Income"] = df["Income"].fillna(df["Income"].median()).astype(int)
    print("Step 5 - outliers   : Age=250 removed, missing filled with median")
    return df


def main():
    raw = make_messy_data()
    age_missing = int(raw["Age"].isna().sum())
    age_impossible = int((raw["Age"] > 100).sum())
    income_missing = int(raw["Income"].isin(["unknown", ""]).sum()
                         + raw["Income"].isna().sum())

    print("==== DATA CLEANING WORKSHOP ====")
    print()
    print("BEFORE:")
    print(raw)
    print()
    print(f"BEFORE: {len(raw)} rows, Age has {age_missing} missing + one value of "
          f"{int(raw['Age'].max())}, Income has {income_missing} missing")
    print()

    cleaned = clean_data(raw)

    print()
    print("AFTER:")
    print(cleaned)
    print()
    print(f"AFTER : {len(cleaned)} rows, "
          f"{int(cleaned.isna().sum().sum())} missing anywhere, "
          f"all text tidy, all numbers real")
    print(f"AFTER : Gender values -> {sorted(cleaned['Gender'].unique())}")
    print(f"AFTER : Age range {cleaned['Age'].min()}-{cleaned['Age'].max()}, "
          f"Income range {cleaned['Income'].min():,}-{cleaned['Income'].max():,}")


if __name__ == "__main__":
    main()
