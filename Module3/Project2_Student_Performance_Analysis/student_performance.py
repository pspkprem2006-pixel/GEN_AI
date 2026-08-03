"""
Project 2 - Student Performance Analysis (Module 3: Data Analysis & Visualization)

A full EDA of student exam data with Pandas, NumPy, and Seaborn:
    load -> clean (fill missing marks) -> analyze (describe, correlation)
    -> 4 Seaborn charts -> report

The star insight: study hours strongly correlate with marks.

The 2x2 figure of Seaborn charts is saved as 'student_performance.png'
(Matplotlib 'Agg' backend, no GUI needed).
Run with:  python student_performance.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def generate_data(seed=42):
    """Create 60 synthetic students (no external CSV needed)."""
    rng = np.random.default_rng(seed)
    n = 60

    study_hours = rng.uniform(1, 8, n)
    # Study hours drive marks -> creates the strong 0.87-style correlation.
    base = np.clip(19.6 + study_hours * 7.5 + rng.normal(0, 10.5, n), 0, 100)

    df = pd.DataFrame({
        "StudentID": [f"S{i + 1:02d}" for i in range(n)],
        "StudyHours": study_hours.round(1),
        "Math": np.clip(base + rng.normal(0, 5, n), 0, 100).round(),
        "Science": np.clip(base + rng.normal(0, 5, n), 0, 100).round(),
        "English": np.clip(base + rng.normal(0, 5, n), 0, 100).round(),
    })

    # Inject 3 missing marks (so the cleaning step has real work to do).
    missing = [(11, "Math"), (27, "Science"), (45, "English")]
    for row, subject in missing:
        df.loc[row, subject] = np.nan
    return df


def load_and_clean(df):
    """Fill missing marks with the subject average, then build features
    (section 13.2)."""
    for subject in ["Math", "Science", "English"]:
        df[subject] = df[subject].fillna(round(df[subject].mean()))  # fill blanks
    df["Total"] = df[["Math", "Science", "English"]].sum(axis=1)
    df["Percentage"] = (df["Total"] / 300 * 100).round(1)
    df["Result"] = np.where((df[["Math", "Science", "English"]] >= 40).all(axis=1),
                            "Pass", "Fail")
    return df


def analyze(df):
    """Summary stats + the correlation insight (section 13.3)."""
    study_corr = df["StudyHours"].corr(df["Percentage"])
    topper = df.loc[df["Percentage"].idxmax()]
    return {
        "study_corr": study_corr,
        "overall_average": df["Percentage"].mean(),
        "topper_id": topper["StudentID"],
        "topper_percentage": topper["Percentage"],
        "passed": int((df["Result"] == "Pass").sum()),
        "failed": int((df["Result"] == "Fail").sum()),
        "subject_avg": df[["Math", "Science", "English"]].mean().reset_index(),
    }


def make_charts(df, stats):
    """Four statistical Seaborn charts on a 2x2 grid (section 13.4)."""
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    sns.histplot(df["Percentage"], kde=True, ax=axes[0, 0])  # distribution
    axes[0, 0].set_title("Percentage Distribution")
    axes[0, 0].set_xlabel("Percentage")

    subject_avg = stats["subject_avg"]
    subject_avg.columns = ["Subject", "Average"]
    sns.barplot(data=subject_avg, x="Subject", y="Average", ax=axes[0, 1])
    axes[0, 1].set_title("Subject Averages")

    sns.regplot(data=df, x="StudyHours", y="Percentage", ax=axes[1, 0])  # relationship
    axes[1, 0].set_title("Study Hours vs Percentage")

    num_cols = df.select_dtypes(include=[np.number]).columns
    sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm", ax=axes[1, 1])  # heatmap
    axes[1, 1].set_title("Correlation Heatmap")

    fig.suptitle("Student Performance Analysis", fontsize=16, fontweight="bold")
    fig.tight_layout()
    out = "student_performance.png"
    fig.savefig(out, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    df = generate_data()
    df = load_and_clean(df)
    stats = analyze(df)

    print(f"Students        : {len(df)}")
    print(f"Overall average : {stats['overall_average']:.1f}%")
    print(f"Topper          : {stats['topper_id']} ({stats['topper_percentage']:.1f}%)")
    print(f"Passed / Failed : {stats['passed']} / {stats['failed']}")
    print(f"Study vs marks  : correlation {stats['study_corr']:.2f}")

    path = make_charts(df, stats)
    print(f"-> Open '{path}' to view the charts.")


if __name__ == "__main__":
    main()
