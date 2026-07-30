import pandas as pd
data = {
    "Name": ["Aarav", "Diya", "Kabir", "Meera"],
    "Age": [20, 21, 19, 22],
    "Marks": [85, 92, 78, 90],
    "City": ["Chennai", "Delhi", "Mumbai", "Pune"],
}
df = pd.DataFrame(data)
print(df.groupby("City").size())
print(df.groupby("City")["Marks"].mean())
print(df.groupby("City")["Marks"].agg(["mean", "min", "max", "count"]))
print(df["City"].value_counts())     # how many rows of each city (great for categories)
print(df["Marks"].sum())             # total
print(df["Marks"].mean())            # average
print(df["Marks"].max())             # highest
print(df["Marks"].nlargest(3))       # top 3 values
print(df["Marks"].unique())          # the distinct values
print(df["City"].nunique()) 