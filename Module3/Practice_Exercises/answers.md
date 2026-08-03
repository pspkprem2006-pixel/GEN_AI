# Module 3 - Practice Exercises & Self-Assessment (Answers)

Answers for section 17 of the Module 3 notes (Data Analysis &
Visualization). Every question is written verbatim, followed by its answer
from the 17.7 Solutions & Answer Key. All code is runnable.

Assumes:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```

## 17.1 NumPy

**1. Create a NumPy array of numbers 1-20 and print only the even ones (boolean mask).**

```python
a = np.arange(1, 21)
print(a[a % 2 == 0])                 # -> [ 2  4  6  8 10 12 14 16 18 20]
```

**2. Make a 3x3 array of random integers 1-100; print its mean, max, and the sum of each column.**

```python
m = np.random.default_rng(0).integers(1, 101, (3, 3))
print(m.mean(), m.max(), m.sum(axis=0))   # axis=0 = down each column
```

**3. Given `temps = np.array([30, 35, 28, 40, 33])`, convert all to Fahrenheit in one line.**

```python
temps = np.array([30, 35, 28, 40, 33])
print(temps * 9 / 5 + 32)            # -> [86.  95.  82.4 104.  91.4]
```

## 17.2 Pandas

**4. Create a DataFrame of 5 products with Price and Quantity; add a `Total = Price x Quantity` column.**

```python
df = pd.DataFrame({
    "Product": ["A", "B", "C", "D", "E"],
    "Price":   [10, 120, 50, 200, 30],
    "Quantity":[2, 6, 1, 8, 4],
})
df["Total"] = df["Price"] * df["Quantity"]
print(df)
```

**5. From a CSV of your choice, print `head()`, `shape`, `info()`, and `describe()`.**

```python
df2 = pd.read_csv("yourfile.csv")
print(df2.head()); print(df2.shape); df2.info(); print(df2.describe())
```

(Runnable version without a CSV - build the DataFrame in code first, then inspect it the same way:

```python
df2 = pd.DataFrame({"x": np.arange(1, 11), "y": np.arange(1, 11) * 2})
print(df2.head()); print(df2.shape); df2.info(); print(df2.describe())
```

In real practice, replace the first line with `df2 = pd.read_csv("yourfile.csv")`.)

**6. Filter the DataFrame to rows where Price > 100 **and** Quantity > 5.**

```python
print(df[(df["Price"] > 100) & (df["Quantity"] > 5)])   # use & and parentheses
```

**7. Group by a category column and compute the average of a numeric column.**

```python
df["Category"] = ["x", "y", "x", "y", "x"]
print(df.groupby("Category")["Price"].mean())
```

**8. Find the top 3 rows by a numeric column using `nlargest`.**

```python
print(df.nlargest(3, "Total"))
```

## 17.3 Data cleaning

**9. Create a small DataFrame with some `NaN` values; fill numeric ones with the median and text ones with `"Unknown"`.**

```python
d = pd.DataFrame({"age": [20, np.nan, 25], "city": ["A", None, "B"]})
d["age"] = d["age"].fillna(d["age"].median())
d["city"] = d["city"].fillna("Unknown")
print(d)
```

**10. Given a column of `["  Delhi ", "DELHI", "delhi"]`, clean it to a single consistent value.**

```python
s = pd.Series(["  Delhi ", "DELHI", "delhi"])
print(s.str.strip().str.title().unique())     # -> ['Delhi']
```

**11. Convert a column of `["1,000", "2,500", "unknown"]` into proper numbers.**

```python
n = pd.Series(["1,000", "2,500", "unknown"])
n = n.str.replace(",", "", regex=False).replace("unknown", np.nan)
print(pd.to_numeric(n, errors="coerce"))       # -> 1000.0, 2500.0, NaN
```

## 17.4 EDA & visualization

**12. Compute the correlation between two numeric columns and interpret the number.**

```python
print(df["Price"].corr(df["Total"]))
```

Interpretation: a value near **+1** means a strong positive relationship (the
two rise together); near **0** means no linear link; near **-1** means a
strong negative relationship (one rises while the other falls). And remember:
correlation is not causation (section 7.4).

**13. Draw a **bar chart** of a categorical column's `value_counts()`.**

```python
df["Category"].value_counts().plot(kind="bar"); plt.title("Counts"); plt.show()
```

**14. Draw a **histogram** of a numeric column and describe its shape.**

```python
df["Price"].plot(kind="hist", bins=10); plt.title("Price distribution"); plt.show()
```

Describe the shape with the vocabulary from section 7.7: bell-shaped
(normal), right-skewed, left-skewed, uniform, or bimodal.

**15. Draw a **scatter plot** of two related columns; add a title and axis labels.**

```python
plt.scatter(df["Price"], df["Total"])
plt.xlabel("Price"); plt.ylabel("Total"); plt.title("Price vs Total"); plt.show()
```

**16. Build a **2x2 dashboard** with four different chart types and save it as a PNG.**

```python
fig, ax = plt.subplots(2, 2, figsize=(12, 8))
df["Category"].value_counts().plot(kind="bar", ax=ax[0, 0])
df["Price"].plot(kind="hist", ax=ax[0, 1])
ax[1, 0].scatter(df["Price"], df["Total"])
df.plot(kind="line", x="Price", y="Total", ax=ax[1, 1])
fig.tight_layout(); fig.savefig("dashboard.png")
```

(Note: in a script run headlessly, add `import matplotlib; matplotlib.use("Agg")`
before importing pyplot so `savefig` works without a GUI.)

**17. Use Seaborn to draw a **correlation heatmap** of a dataset.**

```python
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.show()
```

## 17.5 Integrative

**18. Complete all three module projects and modify each with one challenge from its README.**

All three projects are complete in this folder; each project README lists
challenges. One challenge completed per project:

- **Project 1 (Sales Dashboard):** annotated the value on top of each bar in
  the region chart with `plt.text(...)` (section 8.7), e.g.
  `axes[0, 0].bar(...)` then `for bar in bars: plt.text(...)`.
- **Project 2 (Student Performance):** added a `Gender` column to the
  synthetic students and reported `df.groupby("Gender")["Percentage"].mean()`
  to check for a gender gap.
- **Project 3 (Data Cleaning):** applied the IQR outlier rule (section 6.7) to
  `Income` instead of a fixed range:
  ```python
  q1, q3 = cleaned["Income"].quantile([0.25, 0.75])
  iqr = q3 - q1
  cleaned.loc[(cleaned["Income"] < q1 - 1.5 * iqr) |
              (cleaned["Income"] > q3 + 1.5 * iqr), "Income"] = np.nan
  ```

**19. Take any public CSV (e.g., from Kaggle), and run the full pipeline: **clean -> EDA -> 3 charts -> written insight**.**

This is an open do-it task: apply the exact **clean -> EDA -> visualize**
pipeline from the module's three projects to a Kaggle CSV of your choice, and
write a one-paragraph "what I found" (section 7.8). Here is a runnable
stand-in pipeline (builds its own small dataset so it works without a
downloaded CSV - in practice replace the first line with
`df = pd.read_csv("your_kaggle_file.csv")`):

```python
# Pip install: pip install numpy pandas matplotlib seaborn
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# --- Load (replace this block with pd.read_csv("yourfile.csv")) ---
rng = np.random.default_rng(7)
n = 80
hours = rng.uniform(0, 10, n).round(1)
df = pd.DataFrame({
    "Hours": hours,
    "Score": np.clip(20 + hours * 7 + rng.normal(0, 10.5, n), 0, 100).round(1),
})

# --- Clean ---
df["Hours"] = pd.to_numeric(df["Hours"], errors="coerce")
df = df.drop_duplicates()
df["Score"] = df["Score"].fillna(df["Score"].median())
print("Missing after clean:", df.isna().sum().sum())

# --- EDA ---
print(df.describe().round(1))
print("Correlation Hours vs Score:", round(df["Hours"].corr(df["Score"]), 2))

# --- 3 charts ---
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
sns.histplot(df["Score"], kde=True, ax=ax[0]); ax[0].set_title("Score distribution")
sns.regplot(data=df, x="Hours", y="Score", ax=ax[1]); ax[1].set_title("Hours vs Score")
sns.boxplot(x=df["Score"], ax=ax[2]); ax[2].set_title("Score box plot (outliers)")
fig.tight_layout(); fig.savefig("pipeline.png")

# --- Written insight ---
print("INSIGHT: Scores are roughly bell-shaped around the mean. Hours of")
print("study is the strongest driver (correlation ~0.8): students who")
print("study more score higher. A few low outliers show up in the box plot.")
```

## 17.6 Quick self-check quiz

**1. Which is faster for math on 1M numbers: a Python list or a NumPy array?**

NumPy array (10-100x faster - whole-array operations run in optimized C
under the hood).

**2. What does `df.isna().sum()` tell you?**

The count of missing values per column (in the whole table,
`df.isna().sum().sum()` gives the total).

**3. `.loc` vs `.iloc`?**

`.loc` selects by **label**, `.iloc` by **integer position**.

**4. What does a correlation of -0.9 mean?**

A strong negative relationship: when one variable goes up, the other goes
down.

**5. Which chart shows a distribution and outliers?**

Box plot (and histogram for the distribution shape).

**6. What library is Seaborn built on?**

Matplotlib.

**7. `axis=0` aggregates in which direction?**

Down the columns (each column is reduced to one number - e.g.
`df.sum(axis=0)` sums each column).

**8. Best fill for a numeric column with outliers?**

The median (it is robust to outliers - a huge value drags the mean but
barely moves the median).
